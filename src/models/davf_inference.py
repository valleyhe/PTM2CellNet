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

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn as nn

from src.models.latent_davf import LatentDAVF, LatentDAVFConfig
from src.models.ptm_direction_mapper import PTMDirectionMapperOutput
from src.utils.io import safe_torch_load

logger = logging.getLogger(__name__)


@dataclass
class DAVFInferenceConfig:
    """Configuration for DAVFInferenceModule.

    Attributes:
        state_space: ``"scvi_latent"`` (default) or ``"gene"``.
            - ``"scvi_latent"``: ODE-based LatentDAVF in scVI latent space.
            - ``"gene"``: Simple MLP encoder operating directly on gene-space
              inputs (no ODE, no scVI).
        checkpoint_path: Path to model checkpoint file
        feature_dim: Output feature dimension (default 128)
        hidden_dim: BiPerturbEncoder / GeneMLEPEncoder hidden dimension (default 256)
        freeze: Whether to freeze DAVF parameters on init (default True)
        latent_dim: Latent space dimension for scvi_latent mode (default 10)
        num_genes: Number of genes in vocabulary (default 5000)
        gene_vocab_size: Vocabulary size for gene-space embedding in ``"gene"``
            mode (default 5000). Ignored when ``state_space="scvi_latent"``.
        num_steps: ODE integration steps (default 50)
        device: Device to use (auto-detect if None)
        allow_unsafe_legacy_load: Whether to allow pickle-based fallback for trusted legacy checkpoints
    """
    state_space: str = "scvi_latent"
    checkpoint_path: str = "checkpoints/latent_davf_ibd_norman/best_model.pt"
    feature_dim: int = 128
    hidden_dim: int = 256
    freeze: bool = True
    latent_dim: int = 10
    num_genes: int = 5000
    gene_vocab_size: int = 5000
    num_steps: int = 50
    device: Optional[str] = None
    allow_unsafe_legacy_load: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.state_space not in ("scvi_latent", "gene"):
            raise ValueError(
                f"state_space must be 'scvi_latent' or 'gene', got '{self.state_space}'"
            )
        if self.feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be positive, got {self.feature_dim}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {self.hidden_dim}"
            )
        if self.latent_dim <= 0:
            raise ValueError(
                f"latent_dim must be positive, got {self.latent_dim}"
            )
        if self.num_genes <= 0:
            raise ValueError(
                f"num_genes must be positive, got {self.num_genes}"
            )
        if self.num_steps <= 0:
            raise ValueError(
                f"num_steps must be positive, got {self.num_steps}"
            )
        if self.state_space == "gene" and self.gene_vocab_size <= 0:
            raise ValueError(
                f"gene_vocab_size must be positive in gene mode, got {self.gene_vocab_size}"
            )


@dataclass
class DAVFInferenceOutput:
    """Output from DAVFInferenceModule.

    Attributes:
        davf_features: [B, feature_dim] tensor of DAVF feature vectors
    """
    davf_features: torch.Tensor


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
        gene_emb = self.gene_embedding(gene_ids)        # [B, K, gene_embed_dim]
        dir_emb = self.direction_embedding(directions)   # [B, K, dir_embed_dim]
        combined = torch.cat([gene_emb, dir_emb], dim=-1)  # [B, K, concat_dim]

        # Masked mean pool over K
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, K, 1]
        summed = (combined * mask_expanded).sum(dim=1)        # [B, concat_dim]
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)      # [B, 1]
        pooled = summed / counts                              # [B, concat_dim]

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

        # Determine device
        if config.device is not None:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if config.state_space == "scvi_latent":
            # Create LatentDAVF model for scVI latent-space inference
            latent_config = LatentDAVFConfig(
                latent_dim=config.latent_dim,
                num_genes=config.num_genes,
                hidden_dim=config.hidden_dim,
            )
            self.latent_davf = LatentDAVF(latent_config)
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

        try:
            # Load checkpoint with device mapping
            try:
                checkpoint = safe_torch_load(
                    ckpt_path,
                    map_location=self.device,
                    allowed_classes=_DAVF_ALLOWED,
                )
            except (pickle.UnpicklingError, RuntimeError):
                # Legacy checkpoints containing custom config objects cannot
                # be loaded with weights_only=True.  Fall back to
                # weights_only=False for trusted checkpoints only.
                logger.warning(
                    "Checkpoint %s requires unsafe legacy loading "
                    "(weights_only=False).  Only use trusted checkpoints.",
                    ckpt_path,
                )
                checkpoint = safe_torch_load(
                    ckpt_path,
                    map_location=self.device,
                    weights_only=False,
                    enforce_safe_only=False,
                )

            # Extract state dict (handle both formats per D-06)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict):
                # Direct state dict format
                state_dict = checkpoint
            else:
                raise ValueError(
                    f"Checkpoint must be a dictionary, got {type(checkpoint)}"
                )

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
                raise RuntimeError(
                    "State dict incompatible: no recognized LatentDAVF parameter keys found"
                )

            self._checkpoint_loaded = True

            logger.info(
                f"Loaded DAVF checkpoint from {ckpt_path} "
                f"(loaded={self._checkpoint_loaded}, matched_keys={len(loaded_keys)}, "
                f"missing_keys={len(incompatible.missing_keys)}, "
                f"unexpected_keys={len(incompatible.unexpected_keys)})"
            )

        except (OSError, RuntimeError, ValueError, ImportError, pickle.UnpicklingError) as e:
            logger.warning(
                f"Failed to load checkpoint from {ckpt_path}: {e}. "
                "DAVFInferenceModule will return zero features."
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
            logger.warning(
                "DAVFInferenceModule.forward() called without loaded checkpoint. "
                "Returning zero features."
            )
            return DAVFInferenceOutput(
                davf_features=torch.zeros(
                    B, self.config.feature_dim,
                    device=param_device,
                    dtype=torch.float,
                )
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

        return DAVFInferenceOutput(davf_features=davf_features)
