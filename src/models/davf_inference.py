"""
DAVFInferenceModule: Inference wrapper for LatentDAVF/DAVF models.

Produces fixed-dimension [B, 128] feature vectors from PTM perturbation inputs.
Handles checkpoint loading with graceful degradation (zero-vector fallback).
Provides freeze/unfreeze API for fine-tuning strategies.

Supports two state_space modes:
    - "scvi_latent" (default): ODE-based LatentDAVF model in scVI latent space
    - "gene": Simple MLP encoder operating directly on gene-space inputs

Architecture flow (scvi_latent mode):
    PTM Input (gene_ids, directions, attention_mask) [B, K]
        ↓
    LatentDAVF/DAVF model (ODE integration)
        ↓
    BiPerturbEncoder condition embedding [B, hidden_dim=256]
        ↓
    DeltaProjection (trainable head)
        ↓
    DAVF Features [B, feature_dim=128]

Architecture flow (gene mode):
    PTM Input (gene_ids, directions, attention_mask) [B, K]
        ↓
    GeneMLEPEncoder (embedding + MLP, no ODE)
        ↓
    DeltaProjection (trainable head)
        ↓
    DAVF Features [B, feature_dim=128]
"""

from __future__ import annotations

import logging
import os
import pickle
import csv
import gzip
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, cast

import numpy as np

import torch
import torch.nn as nn

from src.models.latent_davf import LatentDAVF, LatentDAVFConfig
from src.models.legacy_davf import LegacyLatentDAVF
from src.models.ptm_direction_mapper import PTMDirectionMapperOutput
from src.models.gene_vocabulary import normalize_ensembl_id, normalize_gene_symbol
from src.utils.io import safe_torch_load

if TYPE_CHECKING:
    from src.integration.perturbgen.contracts import DAVFDirectionEvidence

logger = logging.getLogger(__name__)


@dataclass
class DAVFInferenceConfig:
    """Configuration for DAVFInferenceModule.

    Attributes:
            state_space: ``"scvi_latent"`` （默认，ODE-based LatentDAVF in scVI
                latent space）或 ``"gene"`` （Simple MLP encoder，直接处理
                gene-space 输入，无 ODE、无 scVI）。
            checkpoint_path: Path to model checkpoint file
            feature_dim: Output feature dimension (default 128)
            hidden_dim: BiPerturbEncoder / GeneMLEPEncoder hidden dimension (default 256)
            freeze: Whether to freeze DAVF parameters on init (default True)
            latent_dim: Latent space dimension for scvi_latent mode (default 10)
            num_genes: Number of genes in vocabulary (default 5000)
            gene_vocab_size: Vocabulary size for gene-space embedding in ``"gene"``
                mode (default 5000). Ignored when ``state_space="scvi_latent"``.
            num_steps: ODE integration steps (default 50)
            intervention_type: Optional single direction expected by this
                checkpoint (``KO``, ``KD`` or ``OE``). New modality-specific
                checkpoints should set this explicitly; ``None`` preserves
                historical feature-path compatibility.
            device: Device to use (auto-detect if None)
            scvi_model_path: Path to scVI model (null = formal scVI direction
                inference unavailable)
            embedding_asset_path: Directory of a verified PerturbGen embedding
                asset (manifest.json + vocabulary.json + gene_embeddings.safetensors).
                When set, the frozen matrix is injected into ``LatentDAVF`` as
                ``pretrained_gene_embeddings`` (schema v2; replaces the removed
                ``geneformer_path`` semantics). Asset load errors propagate.
            gene_names_path: Path to a two-column ``Ensembl ID<TAB>gene symbol``
                table used only to resolve API symbols to PerturbGen token rows.
                It is not a source of scVI decoder indices; those always come
                from the bound adapter's ordered ``gene_names``.
    """

    state_space: str = "scvi_latent"
    checkpoint_path: str = "checkpoints/latent_davf_ibd_norman/best_model.pt"
    scvi_model_path: Optional[str] = None
    embedding_asset_path: Optional[str] = None
    gene_names_path: Optional[str] = None
    feature_dim: int = 128
    hidden_dim: int = 256
    freeze: bool = True
    latent_dim: int = 10
    num_genes: int = 5000
    gene_vocab_size: int = 5000
    num_steps: int = 50
    intervention_type: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters and log warnings for null paths."""
        if self.state_space not in ("scvi_latent", "gene"):
            raise ValueError(f"state_space must be 'scvi_latent' or 'gene', got '{self.state_space}'")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {self.num_genes}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.state_space == "gene" and self.gene_vocab_size <= 0:
            raise ValueError(f"gene_vocab_size must be positive in gene mode, got {self.gene_vocab_size}")
        if self.intervention_type not in (None, "KO", "KD", "OE"):
            raise ValueError("intervention_type must be KO, KD, OE or None")

        # Warn about null paths used for full DAVF functionality
        if self.scvi_model_path is None:
            logger.warning("DAVF config: scvi_model_path is null. DAVF scVI branch will be unavailable.")
        if self.gene_names_path is None:
            logger.warning("DAVF config: gene_names_path is null. Gene name mapping will be unavailable.")


@dataclass
class DAVFInferenceOutput:
    """Output from DAVFInferenceModule.

    Attributes:
        davf_features: [B, feature_dim] tensor of DAVF feature vectors
        model_kind: Identifier for the model kind (``"davf"`` for normal,
            ``"zero_fallback"`` when checkpoint not loaded and strict mode off).
    """

    davf_features: torch.Tensor
    model_kind: str = "davf"


class DeltaProjection(nn.Module):
    """Projection head for DAVF features.

    Maps BiPerturbEncoder output [B, hidden_dim] to [B, feature_dim].
    Architecture: Linear -> LayerNorm -> GELU -> Dropout -> Linear

    Per D-12: This module is always trainable (not frozen when DAVF is frozen).
    """

    def __init__(self, hidden_dim: int = 256, feature_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),  # 256 -> 512
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, feature_dim),  # 512 -> 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project condition embedding to feature dimension.

        Args:
            x: [B, hidden_dim] condition embedding from BiPerturbEncoder

        Returns:
            [B, feature_dim] projected features
        """
        return cast(torch.Tensor, self.net(x))


class GeneMLEPEncoder(nn.Module):
    """Simple MLP encoder for gene-space DAVF inference (no ODE).

    Embeds gene_ids and directions, pools over the sequence dimension,
    then projects through an MLP to produce a condition embedding.

    Architecture:
        gene_ids  [B, K] -> Embedding -> [B, K, gene_embed_dim]
        directions [B, K] -> Embedding -> [B, K, dir_embed_dim]
        concat -> [B, K, gene_embed_dim + dir_embed_dim]
        masked mean pool over K -> [B, gene_embed_dim + dir_embed_dim]
        MLP -> [B, hidden_dim]
    """

    def __init__(
        self,
        gene_vocab_size: int = 5000,
        gene_embed_dim: int = 64,
        num_directions: int = 3,
        dir_embed_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gene_embedding = nn.Embedding(gene_vocab_size, gene_embed_dim)
        self.direction_embedding = nn.Embedding(num_directions, dir_embed_dim)
        concat_dim = gene_embed_dim + dir_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        gene_ids: torch.Tensor,
        directions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode gene-space inputs into a condition embedding.

        Args:
            gene_ids: [B, K] gene index tensor
            directions: [B, K] direction code tensor (0=KO, 1=KD, 2=OE)
            attention_mask: [B, K] validity mask (1=valid, 0=pad)

        Returns:
            [B, hidden_dim] condition embedding
        """
        gene_emb = self.gene_embedding(gene_ids)  # [B, K, gene_embed_dim]
        dir_emb = self.direction_embedding(directions)  # [B, K, dir_embed_dim]
        combined = torch.cat([gene_emb, dir_emb], dim=-1)  # [B, K, concat_dim]

        # Masked mean pool over K
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, K, 1]
        summed = (combined * mask_expanded).sum(dim=1)  # [B, concat_dim]
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = summed / counts  # [B, concat_dim]

        return cast(torch.Tensor, self.mlp(pooled))  # [B, hidden_dim]


class DAVFInferenceModule(nn.Module):
    """Inference wrapper for LatentDAVF/DAVF models.

    Loads a pretrained DAVF model and produces fixed-dimension feature vectors
    from PTM perturbation inputs. Provides freeze/unfreeze API for fine-tuning.

    Graceful degradation: returns zero features when checkpoint is missing.

    Usage:
        config = DAVFInferenceConfig(checkpoint_path="path/to/model.pt")
        module = DAVFInferenceModule(config)

        mapper_output = ptm_mapper.map_ptms(ptm_sites, gene_names)
        output = module(mapper_output)
        # output.davf_features.shape == [B, 128]
    """

    def __init__(self, config: DAVFInferenceConfig):
        super().__init__()
        self.config = config
        self._checkpoint_loaded = False
        # Only schema-v2 checkpoints with an exact current LatentDAVF state
        # dict may produce formal gene-direction evidence. Historical
        # feature-only/legacy loading remains available for compatibility.
        self._current_checkpoint_contract_valid = False
        self._checkpoint_contract: Mapping[str, Any] = {}
        self._scvi_adapter: Any | None = None

        # Determine device
        if config.device is not None:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        pretrained_gene_embeddings: Optional[torch.Tensor] = None
        self._embedding_gene_to_token: dict[str, int] = {}
        self._embedding_symbol_to_token: dict[str, int] = {}
        self._embedding_symbol_to_ensembl: dict[str, str] = {}
        self._embedding_asset: Any | None = None
        if config.state_space == "scvi_latent":
            if config.embedding_asset_path is not None:
                # Schema v2: inject the verified PerturbGen matrix into LatentDAVF.
                # Load errors (missing files, sha256 mismatch, bad schema) propagate:
                # a configured asset must resolve or the module must not start.
                from src.models.perturbgen_embedding import load_perturbgen_embedding_asset

                asset = load_perturbgen_embedding_asset(config.embedding_asset_path)
                self._embedding_asset = asset
                pretrained_gene_embeddings = asset.embeddings.to(dtype=torch.float32, device=self.device)
                self._embedding_gene_to_token = dict(asset.gene_to_token)
                if config.gene_names_path is not None:
                    self._embedding_symbol_to_ensembl = self._load_embedding_symbol_to_ensembl(
                        config.gene_names_path,
                    )
                    self._embedding_symbol_to_token = self._load_embedding_symbol_aliases(
                        config.gene_names_path,
                        asset.gene_to_token,
                    )
                logger.info(
                    "Injected PerturbGen embedding asset %s into LatentDAVF (rows=%d, dim=%d, manifest run_id=%s)",
                    config.embedding_asset_path,
                    asset.vocab_size,
                    asset.embedding_dim,
                    asset.manifest.get("run_id"),
                )

            # Create LatentDAVF model for scVI latent-space inference
            latent_config = LatentDAVFConfig(
                latent_dim=config.latent_dim,
                num_genes=config.num_genes,
                hidden_dim=config.hidden_dim,
            )
            self.latent_davf: LatentDAVF | LegacyLatentDAVF = LatentDAVF(
                latent_config,
                pretrained_gene_embeddings=pretrained_gene_embeddings,
            )
        elif config.state_space == "gene":
            # Create GeneMLEPEncoder for gene-space inference (no ODE)
            self.gene_encoder = GeneMLEPEncoder(
                gene_vocab_size=config.gene_vocab_size,
                hidden_dim=config.hidden_dim,
            )

        # Create DeltaProjection head (always trainable, per D-14)
        self.delta_projection = DeltaProjection(
            hidden_dim=config.hidden_dim,
            feature_dim=config.feature_dim,
        )

        # Keep parameters and inputs on the same runtime device.
        self.to(self.device)

        # Load checkpoint if exists
        self._load_checkpoint()

        # Apply freeze if configured
        if config.freeze:
            self.freeze_davf()

    @property
    def embedding_gene_to_token(self) -> Mapping[str, int]:
        """Return the verified gene-token mapping used by DAVF embeddings."""

        return dict(self._embedding_gene_to_token)

    @property
    def embedding_symbol_to_token(self) -> Mapping[str, int]:
        """Return the verified symbol aliases used by the DAVF mapper."""

        return dict(self._embedding_symbol_to_token)

    @property
    def embedding_symbol_to_ensembl(self) -> Mapping[str, str]:
        """Return the verified symbol→Ensembl alias asset (TD-NEW-04)."""

        return dict(self._embedding_symbol_to_ensembl)

    @staticmethod
    def _load_embedding_symbol_to_ensembl(mapping_path: str | Path) -> dict[str, str]:
        """Load a strict, unambiguous Ensembl-to-symbol table."""

        path = Path(mapping_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DAVF gene alias table not found: {path}")
        opener = gzip.open if path.suffix == ".gz" else open
        delimiter = "," if ".csv" in "".join(path.suffixes).lower() else "\t"
        symbol_to_ensembl: dict[str, str] = {}
        ambiguous_symbols: set[str] = set()
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            for row_number, row in enumerate(reader, start=1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) < 2:
                    raise ValueError(f"DAVF gene alias table row {row_number} must contain Ensembl ID and gene symbol")
                try:
                    ensembl_id = normalize_ensembl_id(row[0])
                except ValueError:
                    header_name = row[0].strip().lower().replace("-", "_")
                    if row_number == 1 and header_name in {"ensembl_id", "gene_id", "gene_id_version"}:
                        continue
                    raise ValueError(
                        f"DAVF gene alias table row {row_number} has invalid Ensembl ID: {row[0]!r}"
                    ) from None
                symbol = normalize_gene_symbol(row[1])
                previous = symbol_to_ensembl.get(symbol)
                if previous is not None and previous != ensembl_id:
                    # A public gene table can contain historical aliases or
                    # duplicated loci. Choosing one would silently change
                    # the biological target, so ambiguous symbols are made
                    # unevaluable while unique symbols remain usable.
                    ambiguous_symbols.add(symbol)
                    continue
                symbol_to_ensembl[symbol] = ensembl_id

        for symbol in ambiguous_symbols:
            symbol_to_ensembl.pop(symbol, None)
        return symbol_to_ensembl

    @staticmethod
    def _load_embedding_symbol_aliases(
        mapping_path: str | Path,
        gene_to_token: Mapping[str, int],
    ) -> dict[str, int]:
        """Project a strict Ensembl/symbol table to verified asset rows.

        The PerturbGen asset intentionally keeps its original Ensembl keys and
        compact token rows. API callers normally provide HGNC symbols, so the
        alias table is a separate boundary. No network, hash, modulo, or
        random fallback is allowed here.
        """

        symbol_to_ensembl = DAVFInferenceModule._load_embedding_symbol_to_ensembl(mapping_path)
        aliases: dict[str, int] = {}
        for symbol, ensembl_id in symbol_to_ensembl.items():
            token = gene_to_token.get(ensembl_id)
            if token is not None:
                aliases[symbol] = int(token)
        if not aliases:
            raise ValueError(
                f"DAVF gene alias table {Path(mapping_path).expanduser().resolve()} has no entries present in the verified PerturbGen vocabulary"
            )
        return aliases

    @property
    def scvi_adapter(self) -> Any | None:
        """Return the adapter explicitly bound to this DAVF instance."""

        return getattr(self, "_scvi_adapter", None)

    @property
    def current_checkpoint_contract_valid(self) -> bool:
        """Whether the loaded checkpoint is eligible for formal evidence."""

        return bool(getattr(self, "_current_checkpoint_contract_valid", False))

    def _validate_scvi_adapter(self, scvi_adapter: Any) -> None:
        """Validate the dimensions exposed by a scVI adapter.

        The latent state and decoder vocabulary are part of the DAVF model
        contract. Accepting a scVI adapter with a different schema would make
        the decoded direction evidence refer to the wrong coordinates.
        """

        if scvi_adapter is None or not callable(getattr(scvi_adapter, "decode", None)):
            raise TypeError("scvi_adapter with a decode() method is required")

        validator = getattr(scvi_adapter, "validate_compatibility", None)
        if callable(validator):
            validator(
                expected_latent_dim=self.config.latent_dim,
                expected_num_genes=self.config.num_genes,
            )
            return

        actual_latent = getattr(scvi_adapter, "n_latent", None)
        actual_genes = getattr(scvi_adapter, "n_genes", None)
        if (
            isinstance(actual_latent, bool)
            or not isinstance(actual_latent, Integral)
            or isinstance(actual_genes, bool)
            or not isinstance(actual_genes, Integral)
        ):
            raise TypeError(
                "scvi_adapter must expose validate_compatibility() or integer n_latent and n_genes metadata"
            )
        if int(actual_latent) != self.config.latent_dim:
            raise ValueError(
                "DAVF/scVI latent dimension mismatch: "
                f"DAVF expects {self.config.latent_dim}, scVI provides {int(actual_latent)}"
            )
        if int(actual_genes) != self.config.num_genes:
            raise ValueError(
                "DAVF/scVI gene dimension mismatch: "
                f"DAVF expects {self.config.num_genes}, scVI provides {int(actual_genes)}"
            )

    def _validate_current_checkpoint_scvi_binding(self, scvi_adapter: Any) -> None:
        """Revalidate the live adapter against schema-v2 scVI provenance.

        Dimensions alone do not identify a latent coordinate system.  A formal
        checkpoint therefore records the ordered decoder vocabulary and the
        scVI model path used at export; both must match the live adapter before
        direction evidence is allowed.
        """

        if not self.current_checkpoint_contract_valid:
            return
        metadata = self._checkpoint_contract.get("scvi")
        if not isinstance(metadata, Mapping):
            raise RuntimeError("schema_version=2 checkpoint is missing scVI provenance")

        expected_names = metadata.get("gene_names")
        validator = getattr(scvi_adapter, "validate_compatibility", None)
        if not callable(validator):
            raise TypeError("formal scVI binding requires validate_compatibility()")
        try:
            validator(
                expected_latent_dim=self.config.latent_dim,
                expected_num_genes=self.config.num_genes,
                expected_gene_names=expected_names,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"live scVI adapter does not match checkpoint gene vocabulary: {exc}") from exc

        adapter_config = getattr(scvi_adapter, "config", None)
        if isinstance(adapter_config, Mapping):
            actual_model_path = adapter_config.get("model_path")
        else:
            actual_model_path = getattr(adapter_config, "model_path", None)
        expected_model_path = metadata.get("model_path")
        if not isinstance(expected_model_path, str) or not expected_model_path:
            raise RuntimeError("schema_version=2 checkpoint is missing scVI model provenance")
        if not isinstance(actual_model_path, (str, Path)) or not str(actual_model_path):
            raise RuntimeError(
                "formal scVI binding requires an adapter loaded from the checkpoint's recorded model_path"
            )
        if Path(actual_model_path).expanduser().resolve() != Path(expected_model_path).expanduser().resolve():
            raise ValueError(
                "live scVI model path does not match checkpoint provenance: "
                f"checkpoint={expected_model_path!r}, live={str(actual_model_path)!r}"
            )

    def _validate_checkpoint_intervention(self, mapper_output: PTMDirectionMapperOutput) -> None:
        """Reject a KO/KD request routed to another direction checkpoint."""

        training = self._checkpoint_contract.get("training")
        if not isinstance(training, Mapping):
            training = {}
        checkpoint_type = training.get("intervention_type")
        configured_type = self.config.intervention_type
        if checkpoint_type is not None and checkpoint_type not in {"KO", "KD", "OE"}:
            raise RuntimeError("formal DAVF checkpoint has an invalid training.intervention_type")
        if configured_type is not None and checkpoint_type != configured_type:
            raise RuntimeError(
                "DAVF inference configuration intervention_type does not match the checkpoint: "
                f"config={configured_type!r}, checkpoint={checkpoint_type!r}"
            )
        expected_type = configured_type or checkpoint_type
        if expected_type is None:
            return
        expected_code = {"KO": 0, "KD": 1, "OE": 2}[expected_type]
        directions = mapper_output.directions
        mask = mapper_output.attention_mask.to(dtype=torch.bool)
        if directions.shape != mask.shape:
            raise ValueError("mapper_output directions and attention_mask must have the same shape")
        active = directions[mask]
        observed = sorted(set(int(value) for value in active.detach().cpu().tolist()))
        if observed != [expected_code]:
            raise RuntimeError(
                f"{expected_type} DAVF checkpoint cannot serve active directions {observed}; "
                f"expected direction code {expected_code}"
            )

    def bind_scvi_adapter(self, scvi_adapter: Any) -> Any:
        """Bind and validate a scVI adapter for the formal DAVF path.

        Binding is explicit so ordinary feature extraction does not load a
        heavyweight scVI model. Once bound, formal direction inference can be
        called without passing the adapter again.
        """

        if self.config.state_space != "scvi_latent":
            raise RuntimeError("a scVI adapter can only be bound when state_space='scvi_latent'")
        if self.config.scvi_model_path is None:
            raise RuntimeError("scvi_model_path is required before binding a scVI adapter")
        self._validate_scvi_adapter(scvi_adapter)
        self._validate_current_checkpoint_scvi_binding(scvi_adapter)
        self._scvi_adapter = scvi_adapter
        return scvi_adapter

    def load_scvi_adapter(self, adata: Any | None = None) -> Any:
        """Load the configured scVI model and bind it to this DAVF module."""

        if self.config.scvi_model_path is None:
            raise RuntimeError("scvi_model_path is required to load the DAVF scVI adapter")
        from src.models.scvi_adapter import ScVIAdapter, ScVIAdapterConfig

        adapter = ScVIAdapter.from_trained_model(
            self.config.scvi_model_path,
            config=ScVIAdapterConfig(
                model_path=self.config.scvi_model_path,
                n_latent=self.config.latent_dim,
                device=str(self.device),
            ),
            adata=adata,
        )
        return self.bind_scvi_adapter(adapter)

    def _validate_target_gene_alignment(
        self,
        scvi_adapter: Any,
        target_indices: Sequence[int],
        target_symbols: Sequence[str],
    ) -> None:
        """Ensure each decoder index points to the supplied gene symbol."""

        decoder_gene_names = self._resolve_scvi_decoder_gene_names(scvi_adapter, target_symbols)
        validator = getattr(scvi_adapter, "validate_target_gene_indices", None)
        if callable(validator):
            validator(target_indices, decoder_gene_names)
            return

        names = getattr(scvi_adapter, "gene_names", None)
        if not isinstance(names, (list, tuple)) or len(names) != self.config.num_genes:
            raise TypeError(
                "scvi_adapter must expose the ordered gene_names vocabulary for formal DAVF direction evidence"
            )
        for row, (index, symbol) in enumerate(zip(target_indices, decoder_gene_names, strict=True)):
            if index < 0 or index >= len(names):
                raise ValueError(f"target gene index {index} at row {row} is outside scVI vocabulary")
            if names[index] != str(symbol):
                raise ValueError(
                    "target gene index does not match scVI gene order at row "
                    f"{row}: index {index} is {names[index]!r}, not {str(symbol)!r}"
                )

    def _resolve_scvi_decoder_gene_names(
        self,
        scvi_adapter: Any,
        target_symbols: Sequence[str],
    ) -> list[str]:
        """Map API symbols to exact names in the live scVI decoder vocabulary.

        A prepared DAVF scVI model may use Ensembl IDs as ``gene_names`` while
        API callers use HGNC symbols. The alias table only bridges that naming
        boundary; the final column lookup still happens against
        ``adapter.gene_names``.
        """

        names = getattr(scvi_adapter, "gene_names", None)
        if not isinstance(names, (list, tuple)) or len(names) != self.config.num_genes:
            raise TypeError("scvi_adapter must expose the ordered gene_names vocabulary")
        name_set = {str(name) for name in names}
        aliases = self.embedding_symbol_to_ensembl
        resolved: list[str] = []
        for raw_symbol in target_symbols:
            raw = str(raw_symbol)
            if raw in name_set:
                resolved.append(raw)
                continue
            try:
                normalized = normalize_gene_symbol(raw)
            except ValueError:
                normalized = raw
            resolved.append(str(aliases.get(normalized, raw)))
        return resolved

    def _resolve_scvi_target_gene_indices(
        self,
        scvi_adapter: Any,
        target_symbols: Sequence[str],
    ) -> list[int]:
        """Resolve decoder columns from the bound adapter's live vocabulary."""

        decoder_gene_names = self._resolve_scvi_decoder_gene_names(scvi_adapter, target_symbols)
        resolver = getattr(scvi_adapter, "resolve_target_gene_indices", None)
        if callable(resolver):
            return [int(index) for index in resolver(decoder_gene_names)]

        names = getattr(scvi_adapter, "gene_names", None)
        if not isinstance(names, (list, tuple)) or len(names) != self.config.num_genes:
            raise TypeError(
                "scvi_adapter must expose resolve_target_gene_indices() or the ordered gene_names vocabulary"
            )
        index_by_name = {str(name): index for index, name in enumerate(names)}
        indices: list[int] = []
        for symbol in decoder_gene_names:
            normalized = str(symbol)
            if normalized not in index_by_name:
                raise ValueError(f"target gene symbol {normalized!r} is absent from the scVI decoder vocabulary")
            indices.append(index_by_name[normalized])
        return indices

    def build_perturbgen_direction_mapper(self, *, gene_mapper=None):
        """Build a mapper locked to the loaded PerturbGen vocabulary."""

        if not self._embedding_gene_to_token:
            raise RuntimeError("a verified PerturbGen embedding asset is required to build the strict direction mapper")
        from src.models.ptm_direction_mapper import MAX_TARGETS, PTMDirectionMapper

        mapper_gene_to_idx = dict(self._embedding_gene_to_token)
        mapper_gene_to_idx.update(self._embedding_symbol_to_token)
        return PTMDirectionMapper(
            gene_mapper=gene_mapper,
            max_targets=MAX_TARGETS,
            gene_to_idx=mapper_gene_to_idx,
        )

    @property
    def model_source(self) -> str:
        """Return the DAVF model source.

        Returns:
            'davf' if a real DAVF model checkpoint is loaded.
            'zero_fallback' if using zero-vector fallback.
        """
        return "zero_fallback" if not self._checkpoint_loaded else "davf"

    def _load_checkpoint(self) -> None:
        """Load model checkpoint with graceful fallback.

        For ``state_space="gene"``: checkpoints are not required since
        GeneMLEPEncoder is lightweight; skip loading entirely.

        For ``state_space="scvi_latent"``: tries to load checkpoint from
        config.checkpoint_path. If missing, logs warning and sets
        _checkpoint_loaded=False for graceful degradation.

        Handles both formats:
        - {"model_state_dict": state_dict, ...} (standard)
        - {state_dict directly} (legacy)
        """
        # Gene mode: encoder is self-contained, no external checkpoint needed
        if self.config.state_space == "gene":
            self._checkpoint_loaded = True
            logger.info("Gene-space mode: GeneMLEPEncoder ready (no checkpoint required)")
            return

        ckpt_path = Path(self.config.checkpoint_path)

        if not ckpt_path.exists():
            logger.warning(
                f"Checkpoint not found at {ckpt_path}. "
                "DAVFInferenceModule will return zero features. "
                "To use DAVF features, provide a valid checkpoint path."
            )
            self._checkpoint_loaded = False
            return

        # DAVF config/architecture classes embedded in legacy checkpoints.
        _DAVF_ALLOWED: set = set()
        try:
            from src.models.latent_davf import LatentDAVF, LatentDAVFConfig

            _DAVF_ALLOWED = {LatentDAVFConfig, LatentDAVF}
        except ImportError:
            pass

        # Load checkpoint with safe (weights_only=True) loading.
        # If the checkpoint is a legacy format that requires unsafe pickle
        # deserialization, this will fail gracefully with `_checkpoint_loaded=False`.
        # Users can migrate with: python scripts/tools/migrate_legacy_checkpoint.py
        try:
            checkpoint = safe_torch_load(
                ckpt_path,
                map_location=self.device,
                allowed_classes=_DAVF_ALLOWED,
            )
        except (pickle.UnpicklingError, RuntimeError) as exc:
            logger.warning(
                "Legacy checkpoint %s cannot be loaded with weights_only=True: %s. "
                "DAVFInferenceModule will return zero features. "
                "To migrate it, run: python scripts/tools/migrate_legacy_checkpoint.py "
                "--input %s --output %s.safe",
                ckpt_path,
                exc,
                ckpt_path,
                ckpt_path,
            )
            self._checkpoint_loaded = False
            return

        try:
            # Extract state dict (handle both formats per D-06)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict):
                # Direct state dict format
                state_dict = checkpoint
            else:
                raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")

            # Explicit checkpoint schema versioning (no silent partial load):
            # schema v2 is the only current formal checkpoint contract. Older
            # unversioned/schema-v1 artifacts stay on the historical feature
            # path and are never eligible for formal direction evidence.
            schema_version = checkpoint.get("schema_version") if isinstance(checkpoint, dict) else None
            if schema_version not in (None, 1, 2):
                raise ValueError(
                    f"Unsupported DAVF checkpoint schema_version={schema_version!r} "
                    f"(supported: 1, 2). Re-export or retrain the checkpoint."
                )

            if schema_version == 2:
                if self._embedding_asset is None:
                    raise ValueError(
                        "schema_version=2 current DAVF checkpoint requires a verified PerturbGen embedding asset"
                    )
                from src.models.davf_checkpoint_contract import (
                    validate_current_davf_checkpoint_payload,
                )

                from src.models.latent_davf import LatentDAVF as _LatentDAVF

                if not isinstance(self.latent_davf, _LatentDAVF):
                    raise TypeError(
                        "current schema-v2 DAVF checkpoints require a LatentDAVF module, "
                        f"got {type(self.latent_davf).__name__}"
                    )
                validate_current_davf_checkpoint_payload(
                    checkpoint,
                    asset=self._embedding_asset,
                    expected_latent_dim=self.config.latent_dim,
                    expected_num_genes=self.config.num_genes,
                    model=self.latent_davf,
                )
                self._checkpoint_contract = dict(checkpoint)
                self._current_checkpoint_contract_valid = True
                self._checkpoint_loaded = True
                logger.info(
                    "Loaded current formal DAVF checkpoint from %s (schema_version=2)",
                    ckpt_path,
                )
                return

            # Legacy-architecture checkpoints (pre-2026, ``delta_mlp`` block):
            # current LatentDAVF cannot load them; reconstruct via the compat layer.
            if LegacyLatentDAVF.is_legacy_state_dict(state_dict):
                # 注意：_load_checkpoint 在 ``self.to(self.device)`` 之后调用，
                # 新构建的 legacy 模块需显式移动到运行时设备。
                legacy = LegacyLatentDAVF().to(self.device)
                incompatible = legacy.load_state_dict(state_dict, strict=False)
                expected_keys = set(legacy.state_dict().keys())
                loaded_keys = set(state_dict.keys()) & expected_keys
                if not loaded_keys:
                    raise RuntimeError("State dict incompatible: no recognized LegacyLatentDAVF parameter keys found")
                self.latent_davf = legacy
                self._checkpoint_loaded = True
                self._current_checkpoint_contract_valid = False
                logger.info(
                    "Loaded legacy DAVF checkpoint via LegacyLatentDAVF "
                    "(loaded=%s, matched_keys=%d, missing_keys=%d, unexpected_keys=%d)",
                    self._checkpoint_loaded,
                    len(loaded_keys),
                    len(incompatible.missing_keys),
                    len(incompatible.unexpected_keys),
                )
                return

            # Load with strict=False to allow partial matches (new params, etc.)
            # Note: strict=False allows missing/extra keys, but size mismatches still raise
            try:
                incompatible = self.latent_davf.load_state_dict(state_dict, strict=False)
            except RuntimeError as load_err:
                # Size mismatch or other structural issues - treat as unloadable
                raise RuntimeError(f"State dict incompatible: {load_err}") from load_err

            expected_keys = set(self.latent_davf.state_dict().keys())
            loaded_keys = set(state_dict.keys()) & expected_keys
            if not loaded_keys:
                raise RuntimeError("State dict incompatible: no recognized LatentDAVF parameter keys found")

            self._checkpoint_loaded = True
            self._current_checkpoint_contract_valid = False

            logger.info(
                f"Loaded DAVF checkpoint from {ckpt_path} "
                f"(loaded={self._checkpoint_loaded}, matched_keys={len(loaded_keys)}, "
                f"missing_keys={len(incompatible.missing_keys)}, "
                f"unexpected_keys={len(incompatible.unexpected_keys)})"
            )

        except (OSError, RuntimeError, ValueError, ImportError, pickle.UnpicklingError) as e:
            logger.warning(
                f"Failed to load checkpoint from {ckpt_path}: {e}. DAVFInferenceModule will return zero features."
            )
            self._checkpoint_loaded = False

    def freeze_davf(self) -> None:
        """Freeze all DAVF/encoder parameters.

        Per D-15: Sets requires_grad=False for all DAVF model parameters.
        Per D-17: DeltaProjection parameters remain trainable.
        """
        if self.config.state_space == "scvi_latent":
            for param in self.latent_davf.parameters():
                param.requires_grad = False
            logger.debug("Froze LatentDAVF parameters (DeltaProjection remains trainable)")
        elif self.config.state_space == "gene":
            for param in self.gene_encoder.parameters():
                param.requires_grad = False
            logger.debug("Froze GeneMLEPEncoder parameters (DeltaProjection remains trainable)")

    def unfreeze_davf(self) -> None:
        """Re-enable gradients for all parameters.

        Per D-16: Re-enables requires_grad=True for fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.debug("Unfroze all parameters for fine-tuning")

    def forward(
        self,
        mapper_output: PTMDirectionMapperOutput,
        z_0: Optional[torch.Tensor] = None,
    ) -> DAVFInferenceOutput:
        """Extract DAVF features from PTM perturbation input.

        Args:
            mapper_output: PTMDirectionMapperOutput containing:
                - gene_ids: [B, K] gene indices
                - directions: [B, K] direction codes (0=KO, 1=KD, 2=OE)
                - attention_mask: [B, K] validity mask
            z_0: Reserved for API compatibility with latent-state DAVF workflows.
                 Only used in ``"scvi_latent"`` mode. Ignored in ``"gene"`` mode.

        Returns:
            DAVFInferenceOutput with davf_features [B, feature_dim]
        """
        B = mapper_output.gene_ids.shape[0]

        # Move inputs to the device where model parameters actually live.
        # This ensures correct behavior after model.to(device) calls on the parent.
        param_device = next(self.parameters()).device
        gene_ids = mapper_output.gene_ids.to(param_device)
        directions = mapper_output.directions.to(param_device)
        attention_mask = mapper_output.attention_mask.to(param_device)

        # Graceful degradation: return zeros if checkpoint not loaded (per D-08)
        if not self._checkpoint_loaded:
            # F-02: strict production mode — fail-fast instead of silent degradation
            strict_mode = os.environ.get("PTM2CELLNET_STRICT_MODEL_ASSETS", "").lower() in ("1", "true", "yes")
            if strict_mode:
                raise RuntimeError(
                    "DAVF checkpoint not loaded and PTM2CELLNET_STRICT_MODEL_ASSETS=1. Provide a valid checkpoint path."
                )
            logger.warning("DAVFInferenceModule.forward() called without loaded checkpoint. Returning zero features.")
            return DAVFInferenceOutput(
                davf_features=torch.zeros(
                    B,
                    self.config.feature_dim,
                    device=param_device,
                    dtype=torch.float,
                ),
                model_kind="zero_fallback",
            )

        if self.config.state_space == "gene":
            # Gene-space mode: embed + MLP, no ODE
            condition = self.gene_encoder(gene_ids, directions, attention_mask)
        else:
            # scvi_latent mode: BiPerturbEncoder condition embedding from LatentDAVF
            gene_embeddings = self.latent_davf._get_gene_embeddings(gene_ids)
            condition, _ = self.latent_davf.biperturb_encoder(
                gene_embeddings,
                directions,
                magnitudes=None,
                return_attention=False,
                attention_mask=attention_mask,
            )  # [B, hidden_dim]

        # Project to feature dimension
        davf_features = self.delta_projection(condition)

        return DAVFInferenceOutput(davf_features=davf_features, model_kind="davf")

    @torch.no_grad()
    def predict_expression_direction(
        self,
        mapper_output: PTMDirectionMapperOutput,
        z_0: torch.Tensor | np.ndarray,
        scvi_adapter: Any | None = None,
        *,
        target_gene_indices: Sequence[int] | torch.Tensor | None = None,
        target_gene_symbols: Sequence[str],
        target_ensembl_ids: Sequence[str],
        library_size: float | None = None,
        scvi_context: Any | None = None,
        direction_epsilon: float = 0.0,
        n_samples: int = 1,
    ) -> list[DAVFDirectionEvidence]:
        """Decode DAVF's latent prediction and return gene-level direction evidence.

        This is the formal DAVF entry point for the new mainline.  It is
        intentionally separate from :meth:`forward`, whose 128-dimensional
        feature output belongs to the historical PTM2CellNet fusion path.

        A complete checkpoint, scVI adapter, and verified embedding asset are
        required.  Missing assets raise immediately; zero/random fallback
        output is never converted into direction evidence.

        ``scvi_context`` may be supplied when the adapter was not built from
        the same AnnData. It carries the cell-level batch/categorical
        covariates required by scVI's decoder.
        """

        from src.integration.perturbgen.contracts import DAVFDirectionEvidence

        if self.config.state_space != "scvi_latent":
            raise RuntimeError(
                "predict_expression_direction requires state_space='scvi_latent'; "
                "gene-space DAVF has no gene-expression decoder"
            )
        if self.model_source != "davf":
            raise RuntimeError(
                "DAVF checkpoint is not loaded; expression direction evidence cannot be produced from zero_fallback"
            )
        if self.config.scvi_model_path is None:
            raise RuntimeError("scvi_model_path is required for gene-level DAVF direction evidence")
        if self.config.embedding_asset_path is None:
            raise RuntimeError("embedding_asset_path is required for formal DAVF direction evidence")
        if not self.current_checkpoint_contract_valid:
            raise RuntimeError(
                "formal DAVF direction evidence requires a schema_version=2 current LatentDAVF checkpoint; "
                "retrain/export with the current architecture"
            )
        self._validate_checkpoint_intervention(mapper_output)
        if not mapper_output.attention_mask.to(dtype=torch.bool).any(dim=1).all():
            raise ValueError(
                "each DAVF direction sample must contain at least one valid PerturbGen gene token; "
                "check the symbol/Ensembl mapping"
            )
        predict = getattr(self.latent_davf, "predict", None)
        if not callable(predict):
            raise RuntimeError(
                "the loaded DAVF checkpoint does not expose latent expression "
                "prediction; retrain with the current LatentDAVF architecture"
            )
        if scvi_adapter is None:
            scvi_adapter = getattr(self, "_scvi_adapter", None)
        scvi_adapter = self.bind_scvi_adapter(scvi_adapter)
        if direction_epsilon < 0 or not np.isfinite(direction_epsilon):
            raise ValueError("direction_epsilon must be finite and >= 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        try:
            z_0_tensor = z_0 if isinstance(z_0, torch.Tensor) else torch.as_tensor(z_0)
        except (TypeError, ValueError) as exc:
            raise ValueError("z_0 must be a numeric NumPy array or torch.Tensor") from exc
        if z_0_tensor.ndim != 2 or z_0_tensor.shape[1] != self.config.latent_dim:
            raise ValueError(f"z_0 must be [B, {self.config.latent_dim}], got {tuple(z_0_tensor.shape)}")

        batch_size = int(z_0_tensor.shape[0])
        if mapper_output.gene_ids.shape[0] != batch_size:
            raise ValueError("mapper_output batch size must match z_0")

        if target_gene_indices is None:
            supplied_target_indices: list[int] | None = None
        elif isinstance(target_gene_indices, torch.Tensor):
            supplied_target_indices = target_gene_indices.detach().cpu().tolist()
        else:
            supplied_target_indices = list(target_gene_indices)
        if supplied_target_indices is not None and len(supplied_target_indices) != batch_size:
            raise ValueError("target_gene_indices must have the same batch length as z_0")
        if not (len(target_gene_symbols) == len(target_ensembl_ids) == batch_size):
            raise ValueError("target_gene_symbols, target_ensembl_ids, and z_0 must have the same batch length")
        if supplied_target_indices is not None:
            if any(
                isinstance(index, bool) or not isinstance(index, Integral) or int(index) < 0
                for index in supplied_target_indices
            ):
                raise ValueError("target_gene_indices must contain non-negative integers")
            supplied_target_indices = [int(index) for index in supplied_target_indices]
            # Validate a caller-supplied assertion first so a stale/wrong
            # decoder index fails before it can influence any array access.
            self._validate_target_gene_alignment(
                scvi_adapter,
                supplied_target_indices,
                target_gene_symbols,
            )

        # The canonical indices always come from the adapter. A supplied list
        # is accepted only as a consistency assertion, never as the source of
        # scVI decoder columns.
        target_indices = self._resolve_scvi_target_gene_indices(
            scvi_adapter,
            target_gene_symbols,
        )
        if supplied_target_indices is not None and supplied_target_indices != target_indices:
            raise ValueError("target_gene_indices must be derived from scvi_adapter.gene_names")

        param_device = next(self.parameters()).device
        z_0_device = z_0_tensor.to(param_device, dtype=torch.float32)
        gene_ids = mapper_output.gene_ids.to(param_device)
        directions = mapper_output.directions.to(param_device)
        attention_mask = mapper_output.attention_mask.to(param_device)

        z_1 = predict(
            z_0_device,
            gene_ids=gene_ids,
            directions=directions,
            num_steps=self.config.num_steps,
            attention_mask=attention_mask,
        )
        baseline_expression = np.asarray(
            scvi_adapter.decode(
                z_0_device.detach().cpu().numpy(),
                library_size=library_size,
                n_samples=n_samples,
                adata=scvi_context,
            ),
            dtype=float,
        )
        perturbed_expression = np.asarray(
            scvi_adapter.decode(
                z_1.detach().cpu().numpy(),
                library_size=library_size,
                n_samples=n_samples,
                adata=scvi_context,
            ),
            dtype=float,
        )
        if baseline_expression.shape != perturbed_expression.shape:
            raise ValueError("scVI decoder returned inconsistent baseline and perturbed shapes")
        if baseline_expression.ndim != 2 or baseline_expression.shape[0] != batch_size:
            raise ValueError("scVI decoder must return a [B, num_genes] expression matrix")
        if not np.isfinite(baseline_expression).all() or not np.isfinite(perturbed_expression).all():
            raise ValueError("scVI decoder returned non-finite expression values")

        checkpoint_provenance = str(Path(self.config.checkpoint_path).expanduser().resolve())
        embedding_provenance = str(Path(self.config.embedding_asset_path).expanduser().resolve())
        evidence: list[DAVFDirectionEvidence] = []
        for row_index, raw_index in enumerate(target_indices):
            gene_index = int(raw_index)
            if gene_index >= baseline_expression.shape[1]:
                raise ValueError(
                    f"target gene index {gene_index} is outside decoder vocabulary [0, {baseline_expression.shape[1]})"
                )
            baseline_value = float(baseline_expression[row_index, gene_index])
            perturbed_value = float(perturbed_expression[row_index, gene_index])
            delta = perturbed_value - baseline_value
            denominator = abs(perturbed_value) + abs(baseline_value)
            confidence = 0.0 if denominator == 0.0 else abs(delta) / denominator
            if not np.isfinite(confidence):
                raise ValueError("DAVF direction confidence is non-finite")
            confidence = float(np.clip(confidence, 0.0, 1.0))
            predicted_direction: Literal["up", "down"] | None = None
            if abs(delta) > direction_epsilon:
                predicted_direction = "up" if delta > 0 else "down"
            evidence.append(
                DAVFDirectionEvidence(
                    gene_symbol=target_gene_symbols[row_index],
                    ensembl_id=target_ensembl_ids[row_index],
                    predicted_direction=predicted_direction,
                    predicted_delta=delta,
                    model_source=self.model_source,
                    checkpoint_provenance=checkpoint_provenance,
                    embedding_provenance=embedding_provenance,
                    confidence=confidence,
                )
            )
        return evidence
