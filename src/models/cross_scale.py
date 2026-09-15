"""Opt-in protein → signaling graph → perturbed-cell-state model.

This module closes the architectural gap identified in
``archive/20260808/reports/project_analysis_20260808_pre_cross_scale.md`` without pretending that unavailable external
weights or graphs are present.  It is intentionally independent from the
default ``PTM2CellNetBase`` constructor:

* :class:`MultiPLMEncoder` fuses Ankh39, ESM-2 and ProtT5 representations (or
  accepts precomputed embeddings) behind one shape contract;
* :class:`CIGNNSignalBridge` performs differentiable graph message passing and
  exposes the exact random-walk sensitivity matrix;
* :class:`CellGraphCompassHead` decodes a delta-expression spectrum and cell
  state logits with gene masking; and
* :class:`CrossScalePTM2CellNet` composes the three pieces and emits provenance.

The module does not silently construct a fully-connected biological graph.  A
research-only approximation is possible only when the caller explicitly opts
in with ``allow_graph_approximation=True`` and the returned provenance records
that choice.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
from torch.nn.parameter import UninitializedParameter
from torch.nn import functional as F

from .ptm_modules import PTMTokenAdapter
from .plm_assets import PLMAssetError, inspect_plm_asset, require_local_plm_assets
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class CrossScaleContractError(ValueError):
    """Raised when a cross-scale batch violates an explicit schema contract."""


@dataclass(frozen=True)
class CrossScaleConfig:
    """Serializable defaults for the opt-in cross-scale model."""

    protein_dim: int = 128
    signal_input_dim: int = 128
    signal_hidden_dim: int = 128
    signal_output_dim: int = 128
    cell_gene_feature_dim: int = 32
    cell_hidden_dim: int = 128
    num_cell_genes: int = 1000
    num_cell_states: int = 4
    num_ptm_types: int = 10
    max_position: int = 2048
    dropout: float = 0.1
    signal_layers: int = 2
    num_signal_edge_types: int = 0
    propagation_alpha: float = 0.85
    max_sensitivity_nodes: int = 2048
    model_version: str = "cross-scale-v1"

    def __post_init__(self) -> None:
        positive_fields = (
            "protein_dim",
            "signal_input_dim",
            "signal_hidden_dim",
            "signal_output_dim",
            "cell_gene_feature_dim",
            "cell_hidden_dim",
            "num_cell_genes",
            "num_cell_states",
            "num_ptm_types",
            "max_position",
            "signal_layers",
            "max_sensitivity_nodes",
        )
        if any(int(getattr(self, field)) <= 0 for field in positive_fields):
            raise CrossScaleContractError("cross-scale dimensions and limits must be positive")
        if self.num_signal_edge_types < 0:
            raise CrossScaleContractError("num_signal_edge_types cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise CrossScaleContractError("dropout must satisfy 0 <= dropout < 1")
        if not 0.0 <= self.propagation_alpha < 1.0:
            raise CrossScaleContractError("propagation_alpha must satisfy 0 <= alpha < 1")


def _file_sha256(path: Optional[str]) -> Optional[str]:
    """Hash an optional local provenance file without making it mandatory."""

    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    if not isinstance(edge_index, Tensor):
        raise CrossScaleContractError("edge_index must be a torch.Tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise CrossScaleContractError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype not in (torch.int64, torch.int32):
        raise CrossScaleContractError("edge_index must use an integer dtype")
    edge_index = edge_index.to(dtype=torch.long)
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise CrossScaleContractError(f"edge_index contains node ids outside [0, {num_nodes - 1}]")
    return edge_index


def _edge_weights(edge_weight: Optional[Tensor], num_edges: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if edge_weight is None:
        return torch.ones(num_edges, device=device, dtype=dtype)
    if not isinstance(edge_weight, Tensor):
        raise CrossScaleContractError("edge_weight must be a torch.Tensor")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != num_edges:
        raise CrossScaleContractError("edge_weight must have shape [num_edges]")
    weights = edge_weight.to(device=device, dtype=dtype)
    if not torch.isfinite(weights).all():
        raise CrossScaleContractError("edge_weight contains non-finite values")
    if (weights < 0).any():
        raise CrossScaleContractError("edge_weight must be non-negative")
    return weights


def normalized_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Optional[Tensor] = None,
    add_self_loops: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Build a row-normalized source→destination adjacency matrix.

    The row convention is ``A[destination, source]``.  This makes
    ``A @ node_features`` propagate source state to destination nodes and is
    also the convention used by the sensitivity matrix below.
    """

    if num_nodes <= 0:
        raise CrossScaleContractError("num_nodes must be positive")
    if not isinstance(edge_index, Tensor):
        raise CrossScaleContractError("edge_index must be a torch.Tensor")
    target_device = device or edge_index.device
    edges = _validate_edge_index(edge_index.to(target_device), num_nodes)
    weights = _edge_weights(edge_weight, edges.shape[1], target_device, dtype)
    adjacency = torch.zeros((num_nodes, num_nodes), device=target_device, dtype=dtype)
    if edges.shape[1]:
        source, destination = edges[0], edges[1]
        adjacency.index_put_((destination, source), weights, accumulate=True)
    if add_self_loops:
        adjacency = adjacency + torch.eye(num_nodes, device=target_device, dtype=dtype)
    row_sum = adjacency.sum(dim=-1, keepdim=True)
    # For the graph convention used by the sensitivity matrix self-loops are
    # always enabled.  When callers explicitly disable them, an isolated row
    # remains zero rather than unexpectedly changing the requested graph.
    if add_self_loops:
        isolated = row_sum.squeeze(-1) <= torch.finfo(dtype).eps
        if isolated.any():
            adjacency[isolated, isolated] = 1.0
            row_sum = adjacency.sum(dim=-1, keepdim=True)
    return adjacency / row_sum.clamp_min(torch.finfo(dtype).eps)


def compute_sensitivity_matrix(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Optional[Tensor] = None,
    alpha: float = 0.85,
    max_nodes: int = 2048,
) -> Tensor:
    """Compute ``S=(1-alpha)(I-alpha G')^-1`` for an explicit graph.

    ``G'`` is the row-normalized adjacency with self-loops for isolated nodes.
    The solve is differentiable with respect to edge weights.  A dense solve is
    deliberately bounded; callers with larger graphs must choose a separate,
    explicit iterative strategy instead of silently allocating an enormous
    matrix.
    """

    if not 0.0 <= alpha < 1.0:
        raise CrossScaleContractError("alpha must satisfy 0 <= alpha < 1")
    if max_nodes <= 0:
        raise CrossScaleContractError("max_nodes must be positive")
    if num_nodes <= 0:
        raise CrossScaleContractError("num_nodes must be positive")
    if num_nodes > max_nodes:
        raise CrossScaleContractError(
            f"exact sensitivity matrix is limited to {max_nodes} nodes; got {num_nodes}. "
            "Provide a sparse/iterative research implementation explicitly."
        )
    graph = normalized_adjacency(
        edge_index,
        num_nodes,
        edge_weight=edge_weight,
        add_self_loops=True,
        dtype=torch.float32,
        device=edge_index.device,
    )
    identity = torch.eye(num_nodes, device=graph.device, dtype=graph.dtype)
    system = identity - float(alpha) * graph
    inverse = torch.linalg.solve(system, identity)
    return cast(Tensor, (1.0 - float(alpha)) * inverse)


def _as_batched_nodes(features: Tensor, name: str) -> Tuple[Tensor, bool]:
    if not isinstance(features, Tensor):
        raise CrossScaleContractError(f"{name} must be a torch.Tensor")
    if features.ndim == 2:
        return features.unsqueeze(0), True
    if features.ndim != 3:
        raise CrossScaleContractError(f"{name} must have shape [B, N, D] or [N, D]")
    if features.shape[0] <= 0 or features.shape[1] <= 0:
        raise CrossScaleContractError(f"{name} must contain at least one batch item and one node")
    return features, False


class MultiPLMEncoder(nn.Module):
    """Fuse Ankh39, ESM-2 and ProtT5 representations.

    The normal research path supplies precomputed tensors using keys such as
    ``ankh39_embeddings`` / ``esm2_embeddings`` / ``prott5_embeddings``.  A
    local HuggingFace loader is provided for environments that actually possess
    the requested weights.  Missing required representations fail by default;
    ``allow_fallback=True`` is an explicit engineering/debug choice and is
    recorded in :attr:`last_provenance`.
    """

    DEFAULT_MODEL_NAMES = {
        "ankh39": "ElnaggarLab/ankh-base",
        "esm2": "facebook/esm2_t6_8M_UR50D",
        "prott5": "Rostlab/prot_t5_xl_uniref50",
    }

    def __init__(
        self,
        *,
        backbone_dims: Optional[Mapping[str, int]] = None,
        output_dim: int = 128,
        backbone_names: Optional[Sequence[str]] = None,
        required_backbones: Optional[Sequence[str]] = None,
        model_names: Optional[Mapping[str, str]] = None,
        dropout: float = 0.1,
        allow_fallback: bool = False,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise CrossScaleContractError("output_dim must be positive")
        dims = {str(name).lower(): int(dim) for name, dim in (backbone_dims or {}).items()}
        if any(dim <= 0 for dim in dims.values()):
            raise CrossScaleContractError("backbone dimensions must be positive")
        names = tuple(
            str(name).lower()
            for name in (
                backbone_names if backbone_names is not None else (tuple(dims) or ("ankh39", "esm2", "prott5"))
            )
        )
        if not names:
            raise CrossScaleContractError("at least one protein language model backbone is required")
        if len(set(names)) != len(names):
            raise CrossScaleContractError("backbone names must be unique")
        required = tuple(str(name).lower() for name in (required_backbones or names))
        unknown_required = set(required) - set(names)
        if unknown_required:
            raise CrossScaleContractError(f"required backbones are not configured: {sorted(unknown_required)}")
        unknown_dimensions = set(dims) - set(names)
        if unknown_dimensions:
            raise CrossScaleContractError(
                f"backbone dimensions are configured for unknown backbones: {sorted(unknown_dimensions)}"
            )
        configured_names = {str(name).lower(): str(value).strip() for name, value in (model_names or {}).items()}
        unknown_model_names = set(configured_names) - set(names)
        if unknown_model_names:
            raise CrossScaleContractError(
                f"model names are configured for unknown backbones: {sorted(unknown_model_names)}"
            )
        if any(not value for value in configured_names.values()):
            raise CrossScaleContractError("configured model names cannot be empty")

        self.output_dim = int(output_dim)
        self.backbone_names = names
        self.required_backbones = required
        self.allow_fallback = bool(allow_fallback)
        self.freeze_backbones = bool(freeze_backbones)
        self.backbone_dims = dims
        self.configured_model_names: Dict[str, str] = configured_names
        self.backbone_model_names: Dict[str, str] = {}
        self._frozen_backbones: set[str] = set()
        self._tokenizers: Dict[str, Any] = {}
        self.backbone_models = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.fallback_embeddings = nn.ParameterDict()
        for name in names:
            dim = dims.get(name)
            self.projections[name] = nn.Linear(dim, output_dim) if dim is not None else nn.LazyLinear(output_dim)
            self.fallback_embeddings[name] = nn.Parameter(torch.zeros(output_dim))

        score_hidden = max(8, output_dim // 2)
        self.fusion_score = nn.Sequential(
            nn.Linear(output_dim, score_hidden),
            nn.Tanh(),
            nn.Linear(score_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.last_provenance: Dict[str, Any] = {
            "available_backbones": [],
            "missing_backbones": list(names),
            "fallback_used": False,
            "fallback_allowed": bool(allow_fallback),
            "configured_model_names": dict(self.configured_model_names),
            "effective_model_names": {
                name: self.configured_model_names.get(name, self.DEFAULT_MODEL_NAMES.get(name, "")) for name in names
            },
            "model_names": {},
            "asset_status": {name: "missing" for name in names},
        }

    @staticmethod
    def _candidate_keys(name: str) -> Tuple[str, ...]:
        return (name, f"{name}_embedding", f"{name}_embeddings")

    def _collect_inputs(self, source: Mapping[str, Any]) -> Dict[str, Tensor]:
        found: Dict[str, Tensor] = {}
        for name in self.backbone_names:
            for key in self._candidate_keys(name):
                value = source.get(key)
                if isinstance(value, Tensor):
                    found[name] = value
                    break
        return found

    def _project(self, name: str, tensor: Tensor) -> Tensor:
        if tensor.ndim not in (2, 3):
            raise CrossScaleContractError(f"{name} embedding must have rank 2 or 3")
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        # ProtT5's embedding-optimized checkpoint is FP16 while the fusion
        # layers are normally FP32.  Match the projection parameter before
        # calling Linear so mixed-precision backbone outputs work on both CPU
        # and CUDA instead of failing with a matmul dtype mismatch.
        projection_weight = getattr(self.projections[name], "weight", None)
        if projection_weight is not None and not isinstance(projection_weight, UninitializedParameter):
            tensor = tensor.to(device=projection_weight.device, dtype=projection_weight.dtype)
        return cast(Tensor, self.projections[name](tensor))

    def _align_representation(self, name: str, tensor: Tensor, source: Mapping[str, Any]) -> Tensor:
        """Align backbone token states to a common residue axis when supplied.

        Different tokenizers add different special tokens and therefore cannot
        be fused by position alone.  A caller may provide
        ``residue_alignment={name: [L_backbone, L_target]}`` (or a batched
        matrix) to make that mapping explicit.  Without it, mismatched lengths
        are rejected rather than silently truncating one backbone.
        """

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        alignment_container = source.get("residue_alignment")
        if alignment_container is None:
            return tensor
        if not isinstance(alignment_container, Mapping):
            raise CrossScaleContractError("residue_alignment must be a mapping keyed by backbone name")
        raw_alignment = alignment_container.get(name)
        if raw_alignment is None:
            return tensor
        if not isinstance(raw_alignment, Tensor):
            raise CrossScaleContractError(f"residue_alignment[{name}] must be a tensor")
        alignment = raw_alignment.to(device=tensor.device, dtype=tensor.dtype)
        if alignment.ndim == 2:
            alignment = alignment.unsqueeze(0).expand(tensor.shape[0], -1, -1)
        if alignment.ndim != 3 or alignment.shape[0] != tensor.shape[0]:
            raise CrossScaleContractError(
                f"residue_alignment[{name}] must have shape [L_backbone, L_target] or [B, L_backbone, L_target]"
            )
        if alignment.shape[1] != tensor.shape[1]:
            raise CrossScaleContractError(
                f"residue_alignment[{name}] source length {alignment.shape[1]} does not match embedding length {tensor.shape[1]}"
            )
        if (alignment < 0).any() or not torch.isfinite(alignment).all():
            raise CrossScaleContractError(f"residue_alignment[{name}] must be finite and non-negative")
        if (alignment.sum(dim=1) <= torch.finfo(alignment.dtype).eps).any():
            raise CrossScaleContractError(
                f"residue_alignment[{name}] must map every target residue to at least one source token"
            )
        denominator = alignment.sum(dim=1).clamp_min(torch.finfo(alignment.dtype).eps)
        return torch.einsum("bli,blt->bti", tensor, alignment) / denominator.unsqueeze(-1)

    def _fuse_nodes(
        self,
        source: Mapping[str, Any],
        *,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        found = self._collect_inputs(source)
        if not found and not self.allow_fallback:
            raise CrossScaleContractError("no Ankh39/ESM-2/ProtT5 embeddings were supplied and fallback is disabled")
        aligned_found = {name: self._align_representation(name, tensor, source) for name, tensor in found.items()}
        if aligned_found:
            sample = next(iter(aligned_found.values()))
            sample_nodes = sample.unsqueeze(1) if sample.ndim == 2 else sample
            batch_size, seq_len = sample_nodes.shape[:2]
        else:
            batch_size = int(source.get("batch_size", 0))
            seq_len = int(source.get("sequence_length", 0))
            if batch_size <= 0 or seq_len <= 0:
                raise CrossScaleContractError(
                    "fallback-only MultiPLMEncoder needs positive batch_size and sequence_length"
                )

        projected: List[Tensor] = []
        used_names: List[str] = []
        fallback_names: List[str] = []
        missing_names: List[str] = []
        for name in self.backbone_names:
            if name in aligned_found:
                tensor = aligned_found[name]
                tensor = tensor.unsqueeze(1) if tensor.ndim == 2 else tensor
                if tensor.shape[:2] != (batch_size, seq_len):
                    raise CrossScaleContractError(
                        f"all PLM node embeddings must share [B, L]={batch_size, seq_len}; "
                        f"{name} has {tuple(tensor.shape[:2])}"
                    )
                projected.append(self._project(name, tensor))
                used_names.append(name)
                continue
            missing_names.append(name)
            if name in self.required_backbones:
                if not self.allow_fallback:
                    raise CrossScaleContractError(
                        f"required PLM embedding missing: {name}; supply precomputed weights or enable fallback explicitly"
                    )
                fallback = self.fallback_embeddings[name].view(1, 1, -1).expand(batch_size, seq_len, -1)
                projected.append(fallback)
                used_names.append(name)
                fallback_names.append(name)

        if not projected:
            raise CrossScaleContractError("no usable PLM representation remains after validation")
        stacked = torch.stack(projected, dim=1)  # [B, K, L, D]
        scores = self.fusion_score(stacked).squeeze(-1)  # [B, K, L]
        weights = torch.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        if attention_mask is not None:
            mask = attention_mask.to(device=fused.device, dtype=fused.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != (batch_size, seq_len):
                raise CrossScaleContractError("attention_mask must match fused node shape [B, L]")
            if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
                raise CrossScaleContractError("attention_mask must contain finite values in [0, 1]")
        asset_status = {
            name: ("precomputed" if name in found else "fallback" if name in fallback_names else "missing")
            for name in self.backbone_names
        }
        for name in self.backbone_models:
            asset_status[name] = "loaded"
        self.last_provenance = {
            "available_backbones": used_names,
            "provided_backbones": sorted(found),
            "missing_backbones": missing_names,
            "fallback_backbones": fallback_names,
            "fallback_used": bool(fallback_names),
            "fallback_allowed": self.allow_fallback,
            "model_names": dict(self.backbone_model_names),
            "configured_model_names": dict(self.configured_model_names),
            "effective_model_names": {
                name: self.configured_model_names.get(name, self.DEFAULT_MODEL_NAMES.get(name, ""))
                for name in self.backbone_names
            },
            "asset_status": asset_status,
            "residue_alignment_used": isinstance(source.get("residue_alignment"), Mapping),
        }
        fused = self.norm(self.dropout(fused))
        if attention_mask is not None:
            fused = fused * mask.unsqueeze(-1)
        return cast(Tensor, fused)

    def encode_nodes(
        self,
        source: Mapping[str, Any],
        *,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return fused residue/protein nodes with shape ``[B, L, D]``."""

        return self._fuse_nodes(source, attention_mask=attention_mask)

    def encode(self, source: Mapping[str, Any], *, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return masked mean pooled protein states with shape ``[B, D]``."""

        nodes = self.encode_nodes(source, attention_mask=attention_mask)
        if attention_mask is None:
            return nodes.mean(dim=1)
        mask = attention_mask.to(device=nodes.device, dtype=nodes.dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return nodes.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def forward(self, source: Mapping[str, Any], attention_mask: Optional[Tensor] = None) -> Tensor:
        return self.encode(source, attention_mask=attention_mask)

    def load_pretrained_backbone(
        self,
        name: str,
        *,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True,
        freeze: Optional[bool] = None,
    ) -> nn.Module:
        """Load one local HuggingFace backbone when the asset is available.

        Network access is opt-in through ``local_files_only=False``.  This
        method is kept separate from construction so importing the project or
        running CPU fixtures never downloads a multi-billion-parameter model.
        """

        normalized = name.lower()
        if normalized not in self.backbone_names:
            raise CrossScaleContractError(f"backbone is not configured: {name}")
        selected_name = (
            model_name or self.configured_model_names.get(normalized) or self.DEFAULT_MODEL_NAMES.get(normalized)
        )
        if not selected_name:
            raise CrossScaleContractError(f"no default model name for backbone: {name}")
        selected_path = Path(selected_name).expanduser()
        if selected_path.exists():
            asset_report = inspect_plm_asset(selected_path)
            if not asset_report["ok"]:
                raise CrossScaleContractError(
                    f"local {normalized} asset is incomplete: " + "; ".join(asset_report["errors"])
                )
            selected_name = asset_report["path"]
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            # Ankh and ProtT5 are T5 checkpoints.  AutoModel resolves them to
            # the encoder-decoder base model, whose ``last_hidden_state`` is
            # not the encoder representation intended for residue embeddings.
            # Select the encoder-only class explicitly, as recommended by the
            # upstream model cards.
            backbone_config = AutoConfig.from_pretrained(
                selected_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            model_type = str(getattr(backbone_config, "model_type", "")).lower()
            architectures = tuple(getattr(backbone_config, "architectures", ()) or ())
            is_t5_backbone = model_type == "t5" or any("t5" in str(value).lower() for value in architectures)
            if is_t5_backbone:
                from transformers import T5EncoderModel, T5Tokenizer

                model = T5EncoderModel.from_pretrained(
                    selected_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                tokenizer = T5Tokenizer.from_pretrained(
                    selected_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            else:
                model = AutoModel.from_pretrained(
                    selected_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    selected_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise CrossScaleContractError(
                f"unable to load {normalized} weights '{selected_name}' locally; "
                "provide precomputed embeddings or explicitly enable a network-backed environment"
            ) from exc

        model_config = getattr(model, "config", None) or backbone_config
        hidden_dim = getattr(model_config, "hidden_size", None)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            # T5 exposes its model width as ``d_model`` rather than
            # ``hidden_size``.
            hidden_dim = getattr(model_config, "d_model", None)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise CrossScaleContractError(f"loaded backbone {selected_name} has no valid hidden_size")
        parent_parameter = next(self.parameters(), None)
        parent_device = parent_parameter.device if parent_parameter is not None else torch.device("cpu")
        model = model.to(parent_device)
        self.projections[normalized] = nn.Linear(hidden_dim, self.output_dim).to(parent_device)
        self.backbone_dims[normalized] = hidden_dim
        self.backbone_models[normalized] = model
        self._tokenizers[normalized] = tokenizer
        self.backbone_model_names[normalized] = selected_name
        should_freeze = self.freeze_backbones if freeze is None else bool(freeze)
        if should_freeze:
            self._frozen_backbones.add(normalized)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
        else:
            self._frozen_backbones.discard(normalized)
        return cast(nn.Module, model)

    def load_pretrained_backbones(
        self,
        *,
        model_names: Optional[Mapping[str, str]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True,
        freeze: Optional[bool] = None,
    ) -> Dict[str, nn.Module]:
        """Load every configured pLM through one explicit asset gate.

        No network request occurs unless ``local_files_only=False`` is passed.
        A failure names the specific backbone and leaves already loaded
        modules available for inspection; callers should treat the operation
        as failed rather than silently proceeding with a partial fusion.
        """

        loaded: Dict[str, nn.Module] = {}
        selected_names = dict(model_names or {})
        unknown = set(selected_names) - set(self.backbone_names)
        if unknown:
            raise CrossScaleContractError(f"model names are configured for unknown backbones: {sorted(unknown)}")
        for name in self.backbone_names:
            loaded[name] = self.load_pretrained_backbone(
                name,
                model_name=selected_names.get(name),
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                freeze=freeze,
            )
        return loaded

    def load_local_pretrained_backbones(
        self,
        asset_root: str,
        *,
        model_dirs: Optional[Mapping[str, str]] = None,
        freeze: Optional[bool] = None,
    ) -> Dict[str, nn.Module]:
        """Validate and load all required backbones from one offline asset root."""

        try:
            model_names = require_local_plm_assets(
                asset_root,
                model_dirs=model_dirs,
                required_backbones=self.required_backbones,
            )
        except PLMAssetError as exc:
            raise CrossScaleContractError(str(exc)) from exc
        return self.load_pretrained_backbones(
            model_names=model_names,
            local_files_only=True,
            freeze=freeze,
        )

    @staticmethod
    def _special_token_mask(tokenizer: Any, input_ids: Tensor) -> Tensor:
        """Return a per-token special-token mask for slow or fast tokenizers."""

        try:
            rows = [
                tokenizer.get_special_tokens_mask(row.detach().cpu().tolist(), already_has_special_tokens=True)
                for row in input_ids
            ]
            return torch.tensor(rows, device=input_ids.device, dtype=torch.bool)
        except (AttributeError, TypeError, ValueError):
            return torch.zeros_like(input_ids, dtype=torch.bool)

    def _format_sequence(self, name: str, sequence: str) -> str:
        """Apply the input convention required by a configured pLM tokenizer."""

        model_name = (
            self.backbone_model_names.get(name)
            or self.configured_model_names.get(name)
            or self.DEFAULT_MODEL_NAMES.get(name, "")
        ).lower()
        if name == "prott5" or "prot_t5" in model_name or "prott5" in model_name:
            # ProtT5's SentencePiece tokenizer expects residues separated by
            # whitespace; the other configured protein tokenizers consume the
            # compact amino-acid string.
            return " ".join(sequence)
        return sequence

    def _encode_model_residues(
        self,
        name: str,
        model: nn.Module,
        tokenizer: Any,
        sequences: Sequence[str],
        *,
        max_length: int,
    ) -> List[Tensor]:
        formatted_sequences = [self._format_sequence(name, sequence) for sequence in sequences]
        try:
            encoded = tokenizer(
                formatted_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        except TypeError:
            encoded = tokenizer(
                formatted_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, Tensor):
            raise CrossScaleContractError(f"tokenizer for {name} did not return input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(attention_mask, Tensor):
            attention_mask = torch.ones_like(input_ids)
        special_mask = encoded.get("special_tokens_mask")
        if not isinstance(special_mask, Tensor):
            special_mask = self._special_token_mask(tokenizer, input_ids)
        if attention_mask.shape != input_ids.shape or special_mask.shape != input_ids.shape:
            raise CrossScaleContractError(f"tokenizer masks for {name} do not match input_ids")
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids"} and isinstance(value, Tensor)
        }
        parameters: Mapping[str, inspect.Parameter]
        try:
            parameters = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            parameters = {}
        if parameters and not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            model_inputs = {key: value for key, value in model_inputs.items() if key in parameters}
        parameter = next(model.parameters(), None)
        model_device = parameter.device if parameter is not None else input_ids.device
        model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}
        if name in self._frozen_backbones:
            model.eval()
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in model.parameters())):
            try:
                output = model(**model_inputs)
            except TypeError as exc:
                raise CrossScaleContractError(f"backbone {name} rejected its tokenizer inputs") from exc
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise CrossScaleContractError(f"backbone {name} did not return [B, T, D] last_hidden_state")
        valid = attention_mask.to(device=hidden.device, dtype=torch.bool) & ~special_mask.to(
            device=hidden.device, dtype=torch.bool
        )
        residue_states: List[Tensor] = []
        for row_index, sequence in enumerate(sequences):
            states = hidden[row_index][valid[row_index]]
            if states.shape[0] != len(sequence):
                raise CrossScaleContractError(
                    f"backbone {name} produced {states.shape[0]} residue tokens for sequence "
                    f"of length {len(sequence)}; provide precomputed embeddings and residue_alignment"
                )
            residue_states.append(states)
        return residue_states

    def encode_sequence_nodes(
        self,
        sequences: Sequence[str],
        *,
        max_length: int = 1024,
    ) -> Tuple[Tensor, Tensor]:
        """Encode raw sequences as padded residue nodes and a valid-residue mask.

        A raw sequence path is allowed only when every loaded tokenizer maps
        one non-special token to one amino-acid residue.  Ambiguous subword
        tokenization fails explicitly instead of silently turning PTM sites
        into a pooled, unaddressable protein vector.
        """

        if not sequences or any(not isinstance(sequence, str) or not sequence for sequence in sequences):
            raise CrossScaleContractError("sequences must contain non-empty strings")
        if max_length <= 0:
            raise CrossScaleContractError("max_length must be positive")
        source: Dict[str, Any] = {}
        max_sequence_length = max(len(sequence) for sequence in sequences)
        residue_mask = torch.zeros((len(sequences), max_sequence_length), dtype=torch.float32)
        for row_index, sequence in enumerate(sequences):
            residue_mask[row_index, : len(sequence)] = 1.0
        for name, model in self.backbone_models.items():
            tokenizer = self._tokenizers.get(name)
            if tokenizer is None:
                continue
            residue_states = self._encode_model_residues(
                name,
                model,
                tokenizer,
                sequences,
                max_length=max_length,
            )
            hidden_dim = residue_states[0].shape[-1]
            padded = torch.zeros(
                (len(sequences), max_sequence_length, hidden_dim),
                device=residue_states[0].device,
                dtype=residue_states[0].dtype,
            )
            for row_index, states in enumerate(residue_states):
                padded[row_index, : states.shape[0]] = states
            source[name] = padded
        source["batch_size"] = len(sequences)
        source["sequence_length"] = max_sequence_length
        nodes = self.encode_nodes(source, attention_mask=residue_mask.to(next(self.parameters()).device))
        self.last_provenance["raw_sequence_pooling"] = "residue_mean_pool"
        self.last_provenance["sequence_lengths"] = [len(sequence) for sequence in sequences]
        return nodes, residue_mask.to(device=nodes.device, dtype=nodes.dtype)

    def encode_sequences(self, sequences: Sequence[str], *, max_length: int = 1024) -> Tensor:
        """Encode raw sequences and return a valid-residue masked mean state."""

        nodes, residue_mask = self.encode_sequence_nodes(sequences, max_length=max_length)
        return nodes.sum(dim=1) / residue_mask.sum(dim=1, keepdim=True).clamp_min(1.0)


class CIGNNSignalBridge(nn.Module):
    """Differentiable signal graph bridge with explicit graph provenance."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        *,
        num_layers: int = 2,
        alpha: float = 0.85,
        dropout: float = 0.1,
        max_sensitivity_nodes: int = 2048,
        allow_graph_approximation: bool = False,
        num_edge_types: int = 0,
        require_edge_type: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise CrossScaleContractError("input_dim and hidden_dim must be positive")
        if num_layers <= 0:
            raise CrossScaleContractError("num_layers must be positive")
        if output_dim is not None and output_dim <= 0:
            raise CrossScaleContractError("output_dim must be positive")
        if not 0 <= alpha < 1:
            raise CrossScaleContractError("alpha must satisfy 0 <= alpha < 1")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim or hidden_dim)
        self.alpha = float(alpha)
        self.max_sensitivity_nodes = int(max_sensitivity_nodes)
        self.allow_graph_approximation = bool(allow_graph_approximation)
        self.num_edge_types = int(num_edge_types)
        self.require_edge_type = bool(require_edge_type)
        if self.num_edge_types < 0:
            raise CrossScaleContractError("num_edge_types cannot be negative")
        if self.require_edge_type and self.num_edge_types <= 0:
            raise CrossScaleContractError("require_edge_type needs num_edge_types > 0")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.message_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.output_projection = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_type_scale = nn.Embedding(self.num_edge_types, 1) if self.num_edge_types > 0 else None
        if self.edge_type_scale is not None:
            nn.init.zeros_(self.edge_type_scale.weight)
        self.last_provenance: Dict[str, Any] = {
            "graph_approximation_used": False,
            "alpha": self.alpha,
        }

    @staticmethod
    def _complete_graph(num_nodes: int, device: torch.device) -> Tensor:
        if num_nodes > 256:
            raise CrossScaleContractError(
                "fully-connected graph approximation is limited to 256 nodes and must be explicitly enabled"
            )
        adjacency = torch.ones((num_nodes, num_nodes), device=device, dtype=torch.bool)
        adjacency.fill_diagonal_(False)
        return adjacency.nonzero(as_tuple=False).T.contiguous()

    def _prepare_graph(
        self,
        edge_index: Optional[Tensor],
        num_nodes: int,
        edge_weight: Optional[Tensor],
        edge_type: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Tensor, Optional[Tensor], bool, Optional[Tensor]]:
        approximation_used = False
        if edge_index is None:
            missing_edges = True
        else:
            if not isinstance(edge_index, Tensor):
                raise CrossScaleContractError("edge_index must be a torch.Tensor")
            edge_index = _validate_edge_index(edge_index.to(device=device), num_nodes)
            missing_edges = edge_index.shape[1] == 0
        if missing_edges:
            if not self.allow_graph_approximation:
                raise CrossScaleContractError(
                    "signal graph edge_index is required; no fully-connected approximation is used by default"
                )
            edge_index = self._complete_graph(num_nodes, device)
            edge_weight = None
            approximation_used = True
        assert edge_index is not None
        if edge_type is None:
            if self.require_edge_type and not approximation_used:
                raise CrossScaleContractError(
                    "typed signal graph requires edge_type for every PPI/kinase-substrate edge"
                )
            return edge_index, edge_weight, approximation_used, None
        if self.edge_type_scale is None:
            raise CrossScaleContractError("edge_type was supplied but bridge was not configured with num_edge_types")
        if not isinstance(edge_type, Tensor):
            raise CrossScaleContractError("edge_type must be a torch.Tensor")
        if edge_type.dtype not in (torch.int32, torch.int64):
            raise CrossScaleContractError("edge_type must use an integer dtype")
        edge_type = edge_type.to(device=device, dtype=torch.long)
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise CrossScaleContractError("edge_type must have shape [num_edges]")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= self.num_edge_types):
            raise CrossScaleContractError(f"edge_type values must be in [0, {self.num_edge_types - 1}]")
        relation_scale = 1.0 + 0.25 * torch.tanh(self.edge_type_scale(edge_type).squeeze(-1))
        base_weight = _edge_weights(
            edge_weight,
            edge_index.shape[1],
            device,
            torch.float32,
        )
        return edge_index, base_weight * relation_scale, approximation_used, edge_type

    def forward_with_details(
        self,
        node_features: Tensor,
        edge_index: Optional[Tensor],
        *,
        edge_weight: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        nodes, _ = _as_batched_nodes(node_features, "node_features")
        if nodes.shape[-1] != self.input_dim:
            raise CrossScaleContractError(
                f"node_features last dimension must be {self.input_dim}, got {nodes.shape[-1]}"
            )
        batch_size, num_nodes, _ = nodes.shape
        edge_index, edge_weight, approximation_used, edge_type = self._prepare_graph(
            edge_index, num_nodes, edge_weight, edge_type, nodes.device
        )
        sensitivity = compute_sensitivity_matrix(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            alpha=self.alpha,
            max_nodes=self.max_sensitivity_nodes,
        ).to(dtype=nodes.dtype)
        propagated = torch.bmm(sensitivity.unsqueeze(0).expand(batch_size, -1, -1), nodes)
        hidden = self.input_projection(propagated)
        for layer, norm in zip(self.message_layers, self.norms, strict=False):
            message = torch.bmm(sensitivity.unsqueeze(0).expand(batch_size, -1, -1), hidden)
            hidden = norm(hidden + self.dropout(F.gelu(layer(message))))
        output = self.output_projection(hidden)
        if node_mask is not None:
            if not isinstance(node_mask, Tensor):
                raise CrossScaleContractError("node_mask must be a torch.Tensor")
            mask = node_mask.to(device=output.device, dtype=output.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != (batch_size, num_nodes):
                raise CrossScaleContractError("node_mask must match signal nodes [B, N]")
            if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
                raise CrossScaleContractError("node_mask must contain finite values in [0, 1]")
            output = output * mask.unsqueeze(-1)
        self.last_provenance = {
            "graph_approximation_used": approximation_used,
            "alpha": self.alpha,
            "num_nodes": num_nodes,
            "num_edges": int(edge_index.shape[1]),
            "edge_types_provided": edge_type is not None,
            "num_edge_types": self.num_edge_types,
            "sensitivity": "exact_dense_solve",
        }
        return {"node_embeddings": output, "sensitivity_matrix": sensitivity}

    def forward(
        self,
        node_features: Tensor,
        edge_index: Optional[Tensor],
        *,
        edge_weight: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward_with_details(
            node_features,
            edge_index,
            edge_weight=edge_weight,
            edge_type=edge_type,
            node_mask=node_mask,
        )["node_embeddings"]


class CellGraphCompassHead(nn.Module):
    """Decode signal features into delta expression and cell-state logits."""

    def __init__(
        self,
        signal_dim: int,
        num_genes: int,
        *,
        gene_feature_dim: int = 32,
        hidden_dim: int = 128,
        num_cell_states: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        require_graph: bool = True,
        require_signal_gene_map: bool = True,
    ) -> None:
        super().__init__()
        if min(signal_dim, num_genes, gene_feature_dim, hidden_dim, num_cell_states) <= 0:
            raise CrossScaleContractError("CellGraphCompassHead dimensions must be positive")
        if num_layers <= 0:
            raise CrossScaleContractError("CellGraphCompassHead num_layers must be positive")
        self.signal_dim = int(signal_dim)
        self.num_genes = int(num_genes)
        self.gene_feature_dim = int(gene_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_cell_states = int(num_cell_states)
        self.require_graph = bool(require_graph)
        self.require_signal_gene_map = bool(require_signal_gene_map)
        self.gene_embedding = nn.Parameter(torch.randn(num_genes, gene_feature_dim) * 0.02)
        self.gene_projection = nn.Linear(gene_feature_dim, hidden_dim)
        self.signal_projection = nn.Linear(signal_dim, hidden_dim)
        self.graph_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.graph_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.expression_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_cell_states),
        )
        self.dropout = nn.Dropout(dropout)

    def _gene_inputs(self, gene_features: Optional[Tensor], batch_size: int, device: torch.device) -> Tensor:
        if gene_features is None:
            return self.gene_embedding.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)
        features, _ = _as_batched_nodes(gene_features, "gene_features")
        if features.shape[:2] != (batch_size, self.num_genes):
            raise CrossScaleContractError(f"gene_features must have shape [B, {self.num_genes}, F]")
        if features.shape[-1] != self.gene_feature_dim:
            raise CrossScaleContractError(f"gene_features last dimension must be {self.gene_feature_dim}")
        return features.to(device=device, dtype=self.gene_embedding.dtype)

    def _signal_to_gene(self, signal_nodes: Tensor, signal_gene_map: Optional[Tensor]) -> Tensor:
        batch_size, num_signal_nodes, signal_dim = signal_nodes.shape
        if signal_gene_map is None:
            if self.require_signal_gene_map:
                raise CrossScaleContractError(
                    "signal_gene_map is required to connect signaling nodes to cell genes; "
                    "disable require_signal_gene_map only for an explicit research ablation"
                )
            return signal_nodes.mean(dim=1, keepdim=True).expand(-1, self.num_genes, -1)
        if not isinstance(signal_gene_map, Tensor):
            raise CrossScaleContractError("signal_gene_map must be a torch.Tensor")
        mapping = signal_gene_map.to(device=signal_nodes.device, dtype=signal_nodes.dtype)
        if mapping.ndim == 2:
            mapping = mapping.unsqueeze(0).expand(batch_size, -1, -1)
        if mapping.shape != (batch_size, num_signal_nodes, self.num_genes):
            raise CrossScaleContractError("signal_gene_map must have shape [N_signal, G] or [B, N_signal, G]")
        if (mapping < 0).any() or not torch.isfinite(mapping).all():
            raise CrossScaleContractError("signal_gene_map must be finite and non-negative")
        normalizer = mapping.sum(dim=1, keepdim=False).clamp_min(torch.finfo(mapping.dtype).eps)
        return torch.einsum("bnd,bng->bgd", signal_nodes, mapping) / normalizer.unsqueeze(-1)

    def forward(
        self,
        signal_nodes: Tensor,
        *,
        cell_edge_index: Optional[Tensor],
        signal_gene_map: Optional[Tensor],
        gene_features: Optional[Tensor] = None,
        gene_mask: Optional[Tensor] = None,
        cell_edge_weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        signal, _ = _as_batched_nodes(signal_nodes, "signal_nodes")
        batch_size, num_signal_nodes, _ = signal.shape
        if cell_edge_index is None:
            missing_graph = True
        else:
            if not isinstance(cell_edge_index, Tensor):
                raise CrossScaleContractError("cell_edge_index must be a torch.Tensor")
            cell_edge_index = _validate_edge_index(cell_edge_index.to(signal.device), self.num_genes)
            missing_graph = cell_edge_index.shape[1] == 0
        if missing_graph:
            if self.require_graph:
                raise CrossScaleContractError("cell_edge_index is required; the cell graph cannot be silently replaced")
            cell_edge_index = torch.arange(self.num_genes, device=signal.device).repeat(2, 1)
        assert cell_edge_index is not None
        adjacency = normalized_adjacency(
            cell_edge_index,
            self.num_genes,
            edge_weight=cell_edge_weight,
            add_self_loops=True,
            dtype=signal.dtype,
            device=signal.device,
        )
        genes = self.gene_projection(self._gene_inputs(gene_features, batch_size, signal.device))
        genes = genes + self.signal_projection(self._signal_to_gene(signal, signal_gene_map))
        for layer, norm in zip(self.graph_layers, self.graph_norms, strict=False):
            message = torch.bmm(adjacency.unsqueeze(0).expand(batch_size, -1, -1), genes)
            genes = norm(genes + self.dropout(F.gelu(layer(message))))

        if gene_mask is None:
            mask = torch.ones((batch_size, self.num_genes), device=genes.device, dtype=genes.dtype)
        else:
            if not isinstance(gene_mask, Tensor):
                raise CrossScaleContractError("gene_mask must be a torch.Tensor")
            mask = gene_mask.to(device=genes.device, dtype=genes.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != (batch_size, self.num_genes):
                raise CrossScaleContractError("gene_mask must match [B, num_genes]")
            if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
                raise CrossScaleContractError("gene_mask must contain finite values in [0, 1]")
        delta_expression = self.expression_head(genes).squeeze(-1) * mask
        pooled = (genes * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        cell_state_logits = self.state_head(pooled)
        return {
            "delta_expression": delta_expression,
            "cell_state_logits": cell_state_logits,
            "cell_state": cell_state_logits.argmax(dim=-1),
            "gene_embeddings": genes,
            "gene_mask": mask,
        }

    def compute_loss(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        expression_weight: float = 1.0,
        state_weight: float = 1.0,
    ) -> Tensor:
        """Compute masked expression loss plus state classification/regression."""

        if "delta_expression" not in targets or "cell_state" not in targets:
            raise CrossScaleContractError("targets must contain delta_expression and cell_state")
        predicted_expression = outputs["delta_expression"]
        target_expression = targets["delta_expression"].to(
            device=predicted_expression.device,
            dtype=predicted_expression.dtype,
        )
        if target_expression.shape != predicted_expression.shape:
            raise CrossScaleContractError("delta_expression target shape mismatch")
        mask = outputs.get("gene_mask")
        if mask is None:
            mask = torch.ones_like(predicted_expression)
        expression_error = F.smooth_l1_loss(predicted_expression, target_expression, reduction="none")
        expression_loss = (expression_error * mask).sum() / mask.sum().clamp_min(1.0)

        logits = outputs["cell_state_logits"]
        state_target = targets["cell_state"].to(device=logits.device)
        if state_target.ndim == 1:
            state_loss = F.cross_entropy(logits, state_target.long())
        elif state_target.shape == logits.shape:
            state_loss = F.mse_loss(logits, state_target.to(dtype=logits.dtype))
        else:
            raise CrossScaleContractError("cell_state target must be [B] or [B, num_cell_states]")
        return float(expression_weight) * expression_loss + float(state_weight) * state_loss

    @staticmethod
    def compute_metrics(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]) -> Dict[str, float]:
        predicted = outputs["delta_expression"].detach()
        target = targets["delta_expression"].to(device=predicted.device, dtype=predicted.dtype)
        mask = outputs.get("gene_mask", torch.ones_like(predicted)).to(dtype=predicted.dtype)
        rmse = torch.sqrt(((predicted - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0))
        state_target = targets["cell_state"].to(device=outputs["cell_state_logits"].device)
        if state_target.ndim == 2:
            state_target = state_target.argmax(dim=-1)
        accuracy = (outputs["cell_state"].detach() == state_target.long()).float().mean()
        return {"delta_expression_rmse": float(rmse.cpu()), "cell_state_accuracy": float(accuracy.cpu())}


class CrossScalePTM2CellNet(nn.Module):
    """Opt-in composition of protein/PTM, signal graph and cell decoders."""

    def __init__(
        self,
        *,
        config: Optional[CrossScaleConfig] = None,
        protein_encoder: Optional[MultiPLMEncoder] = None,
        plm_backbone_dims: Optional[Mapping[str, int]] = None,
        plm_model_names: Optional[Mapping[str, str]] = None,
        required_backbones: Optional[Sequence[str]] = None,
        freeze_plm_backbones: bool = True,
        allow_plm_fallback: bool = False,
        allow_graph_approximation: bool = False,
        require_signal_gene_map: bool = True,
        data_manifest: Optional[str] = None,
        data_manifest_digest: Optional[str] = None,
        signal_graph_version: str = "unversioned",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        cfg = config or CrossScaleConfig()
        if cfg.propagation_alpha < 0 or cfg.propagation_alpha >= 1:
            raise CrossScaleContractError("config.propagation_alpha must satisfy 0 <= alpha < 1")
        self.cross_scale_config = copy.deepcopy(cfg)
        self.model_version = cfg.model_version
        self.data_manifest = data_manifest
        self.data_manifest_digest = data_manifest_digest
        self.signal_graph_version = signal_graph_version
        self.seed = seed
        self.allow_graph_approximation = bool(allow_graph_approximation)
        self.protein_encoder = protein_encoder or MultiPLMEncoder(
            backbone_dims=plm_backbone_dims
            or {
                "ankh39": cfg.protein_dim,
                "esm2": cfg.protein_dim,
                "prott5": cfg.protein_dim,
            },
            backbone_names=("ankh39", "esm2", "prott5"),
            required_backbones=required_backbones if required_backbones is not None else ("ankh39", "esm2", "prott5"),
            output_dim=cfg.protein_dim,
            model_names=plm_model_names,
            allow_fallback=allow_plm_fallback,
            freeze_backbones=freeze_plm_backbones,
        )
        # The injected encoder is the source of truth for pLM asset policy;
        # otherwise a caller could run a fallback-enabled encoder while the
        # top-level model claims that fallback was disabled.
        self.allow_plm_fallback = bool(self.protein_encoder.allow_fallback)
        self.freeze_plm_backbones = bool(self.protein_encoder.freeze_backbones)
        self._last_ptm_token_embeddings: Optional[Tensor] = None
        self.ptm_adapter = PTMTokenAdapter(
            cfg.num_ptm_types,
            cfg.protein_dim,
            max_position=cfg.max_position,
            dropout=cfg.dropout,
        )
        self.ptm_fusion_norm = nn.LayerNorm(cfg.protein_dim)
        self.protein_to_signal = (
            nn.Identity()
            if cfg.protein_dim == cfg.signal_input_dim
            else nn.Linear(cfg.protein_dim, cfg.signal_input_dim)
        )
        self.signal_bridge = CIGNNSignalBridge(
            cfg.signal_input_dim,
            cfg.signal_hidden_dim,
            cfg.signal_output_dim,
            num_layers=cfg.signal_layers,
            alpha=cfg.propagation_alpha,
            dropout=cfg.dropout,
            max_sensitivity_nodes=cfg.max_sensitivity_nodes,
            allow_graph_approximation=allow_graph_approximation,
            num_edge_types=cfg.num_signal_edge_types,
            require_edge_type=cfg.num_signal_edge_types > 0,
        )
        self.cell_head = CellGraphCompassHead(
            cfg.signal_output_dim,
            cfg.num_cell_genes,
            gene_feature_dim=cfg.cell_gene_feature_dim,
            hidden_dim=cfg.cell_hidden_dim,
            num_cell_states=cfg.num_cell_states,
            dropout=cfg.dropout,
            require_signal_gene_map=require_signal_gene_map,
        )

    def _plm_source(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        source = batch.get("protein_embeddings")
        if isinstance(source, Mapping):
            return source
        source_dict: Dict[str, Any] = {}
        for name in self.protein_encoder.backbone_names:
            for key in MultiPLMEncoder._candidate_keys(name):
                if key in batch and isinstance(batch[key], Tensor):
                    source_dict[key] = batch[key]
                    break
        if source_dict:
            return source_dict
        sequences = batch.get("sequence")
        if isinstance(sequences, (list, tuple)) and sequences and all(isinstance(value, str) for value in sequences):
            return {"sequence": sequences}
        raise CrossScaleContractError("batch must contain protein_embeddings, PLM embedding keys or raw sequences")

    def _protein_nodes(self, batch: Mapping[str, Any]) -> Tensor:
        source = self._plm_source(batch)
        if "sequence" in source:
            nodes, residue_mask = self.protein_encoder.encode_sequence_nodes(cast(Sequence[str], source["sequence"]))
            valid_sequence_lengths = residue_mask.sum(dim=1).to(dtype=torch.long)
        else:
            protein_mask = batch.get("protein_attention_mask")
            if protein_mask is not None and not isinstance(protein_mask, Tensor):
                raise CrossScaleContractError("protein_attention_mask must be a torch.Tensor")
            nodes = self.protein_encoder.encode_nodes(
                cast(Mapping[str, Any], source),
                attention_mask=cast(Optional[Tensor], protein_mask),
            )
            if isinstance(protein_mask, Tensor):
                if protein_mask.ndim == 1:
                    protein_mask = protein_mask.unsqueeze(0)
                if protein_mask.shape != nodes.shape[:2]:
                    raise CrossScaleContractError("protein_attention_mask must match protein nodes [B, L]")
                if (
                    not torch.isfinite(protein_mask.float()).all()
                    or (protein_mask < 0).any()
                    or (protein_mask > 1).any()
                ):
                    raise CrossScaleContractError("protein_attention_mask must contain finite values in [0, 1]")
                if not torch.all((protein_mask == 0) | (protein_mask == 1)):
                    raise CrossScaleContractError("protein_attention_mask must be binary")
                valid_sequence_lengths = protein_mask.to(dtype=torch.float32).sum(dim=1).to(dtype=torch.long)
            else:
                valid_sequence_lengths = torch.full(
                    (nodes.shape[0],), nodes.shape[1], device=nodes.device, dtype=torch.long
                )
        ptm_types = batch.get("ptm_types")
        ptm_positions = batch.get("ptm_positions")
        if ptm_types is None and ptm_positions is None:
            self._last_ptm_token_embeddings = None
            self.ptm_adapter.last_provenance = {
                "active_site_count": 0,
                "sequence_lengths": [int(value) for value in valid_sequence_lengths.detach().cpu().tolist()],
                "position_convention": "one_based",
                "padding_id": 0,
            }
            return nodes
        if not isinstance(ptm_types, Tensor) or not isinstance(ptm_positions, Tensor):
            raise CrossScaleContractError("ptm_types and ptm_positions must be supplied together")
        try:
            ptm_tokens = self.ptm_adapter(
                ptm_types.to(device=nodes.device),
                ptm_positions.to(device=nodes.device),
                nodes.shape[1],
                ptm_mask=cast(Optional[Tensor], batch.get("ptm_mask")),
                valid_sequence_lengths=valid_sequence_lengths.to(device=nodes.device),
            )
        except ValueError as exc:
            raise CrossScaleContractError(str(exc)) from exc
        self._last_ptm_token_embeddings = ptm_tokens
        return cast(Tensor, self.ptm_fusion_norm(nodes + ptm_tokens))

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        protein_nodes = self._protein_nodes(batch)
        explicit_signal = batch.get("signal_node_features")
        if explicit_signal is not None:
            if not isinstance(explicit_signal, Tensor):
                raise CrossScaleContractError("signal_node_features must be a tensor")
            signal_input, _ = _as_batched_nodes(explicit_signal, "signal_node_features")
            signal_input = signal_input.to(device=protein_nodes.device, dtype=protein_nodes.dtype)
        else:
            signal_input = self.protein_to_signal(protein_nodes)
        bridge_details = self.signal_bridge.forward_with_details(
            signal_input,
            cast(Optional[Tensor], batch.get("signal_edge_index", batch.get("edge_index"))),
            edge_weight=cast(Optional[Tensor], batch.get("signal_edge_weight")),
            edge_type=cast(Optional[Tensor], batch.get("signal_edge_type")),
            node_mask=cast(Optional[Tensor], batch.get("signal_node_mask")),
        )
        cell_outputs = self.cell_head(
            bridge_details["node_embeddings"],
            cell_edge_index=cast(Optional[Tensor], batch.get("cell_edge_index")),
            signal_gene_map=cast(Optional[Tensor], batch.get("signal_gene_map")),
            gene_features=cast(Optional[Tensor], batch.get("cell_gene_features")),
            gene_mask=cast(Optional[Tensor], batch.get("gene_mask")),
            cell_edge_weight=cast(Optional[Tensor], batch.get("cell_edge_weight")),
        )
        cell_logits = cell_outputs["cell_state_logits"]
        manifest_path = self.data_manifest or batch.get("data_manifest")
        manifest_digest = (
            batch.get("data_manifest_digest")
            or self.data_manifest_digest
            or _file_sha256(str(manifest_path) if manifest_path is not None else None)
        )
        plm_fallback_used = bool(self.protein_encoder.last_provenance.get("fallback_used", False))
        graph_approximation_used = bool(self.signal_bridge.last_provenance.get("graph_approximation_used", False))
        provenance = {
            "model_version": self.model_version,
            "data_manifest": manifest_path,
            "data_manifest_digest": manifest_digest,
            "signal_graph_version": batch.get("signal_graph_version", self.signal_graph_version),
            "signal_graph_digest": batch.get("signal_graph_digest"),
            "seed": self.seed if self.seed is not None else batch.get("seed"),
            "protein_encoder": dict(self.protein_encoder.last_provenance),
            "ptm_adapter": dict(self.ptm_adapter.last_provenance),
            "signal_bridge": dict(self.signal_bridge.last_provenance),
            "plm_fallback_used": plm_fallback_used,
            "graph_approximation_used": graph_approximation_used,
            "fallback_used": plm_fallback_used or graph_approximation_used,
        }
        return {
            **cell_outputs,
            "protein_node_embeddings": protein_nodes,
            "ptm_token_embeddings": self._last_ptm_token_embeddings,
            "signal_node_embeddings": bridge_details["node_embeddings"],
            "sensitivity_matrix": bridge_details["sensitivity_matrix"],
            "logits": cell_logits,
            "predictions": cell_outputs["cell_state"],
            "probabilities": torch.softmax(cell_logits, dim=-1),
            "provenance": provenance,
        }

    def compute_loss(self, outputs: Mapping[str, Any], targets: Mapping[str, Tensor], **kwargs: Any) -> Tensor:
        return self.cell_head.compute_loss(outputs, targets, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        initialized_parameters = [
            parameter for parameter in self.parameters() if not isinstance(parameter, UninitializedParameter)
        ]
        total = sum(parameter.numel() for parameter in initialized_parameters)
        trainable = sum(parameter.numel() for parameter in initialized_parameters if parameter.requires_grad)
        uninitialized = sum(1 for parameter in self.parameters() if isinstance(parameter, UninitializedParameter))
        return {
            "model_version": self.model_version,
            "model_class": self.__class__.__name__,
            "config": asdict(self.cross_scale_config),
            "protein_backbones": list(self.protein_encoder.backbone_names),
            "required_backbones": list(self.protein_encoder.required_backbones),
            "configured_plm_model_names": dict(self.protein_encoder.configured_model_names),
            "effective_plm_model_names": {
                name: self.protein_encoder.configured_model_names.get(
                    name, self.protein_encoder.DEFAULT_MODEL_NAMES.get(name, "")
                )
                for name in self.protein_encoder.backbone_names
            },
            "loaded_plm_model_names": dict(self.protein_encoder.backbone_model_names),
            "allow_plm_fallback": self.allow_plm_fallback,
            "freeze_plm_backbones": self.freeze_plm_backbones,
            "allow_graph_approximation": self.allow_graph_approximation,
            "ptm_adapter": {
                "num_ptm_types": self.ptm_adapter.num_ptm_types,
                "max_position": self.ptm_adapter.max_position,
                "position_convention": "one_based",
            },
            "data_manifest": self.data_manifest,
            "data_manifest_digest": self.data_manifest_digest,
            "signal_graph_version": self.signal_graph_version,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "uninitialized_parameter_count": int(uninitialized),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CrossScalePTM2CellNet":
        model_config = config.get("cross_scale", config.get("model", {}))
        if not isinstance(model_config, Mapping):
            raise CrossScaleContractError("cross_scale config must be a mapping")
        defaults = asdict(CrossScaleConfig())
        selected = {key: model_config[key] for key in defaults if key in model_config}
        cross_config = CrossScaleConfig(**selected)
        dims = model_config.get("plm_backbone_dims")
        if dims is not None and not isinstance(dims, Mapping):
            raise CrossScaleContractError("plm_backbone_dims must be a mapping")
        model_names = model_config.get("plm_model_names")
        if model_names is not None and not isinstance(model_names, Mapping):
            raise CrossScaleContractError("plm_model_names must be a mapping")
        required_backbones = model_config.get("required_backbones")
        if required_backbones is not None:
            if isinstance(required_backbones, str) or not isinstance(required_backbones, Sequence):
                raise CrossScaleContractError("required_backbones must be a sequence of names")
            required_backbones = tuple(str(name) for name in required_backbones)
        return cls(
            config=cross_config,
            plm_backbone_dims=cast(Optional[Mapping[str, int]], dims),
            plm_model_names=cast(Optional[Mapping[str, str]], model_names),
            required_backbones=cast(Optional[Sequence[str]], required_backbones),
            freeze_plm_backbones=bool(model_config.get("freeze_plm_backbones", True)),
            allow_plm_fallback=bool(model_config.get("allow_plm_fallback", False)),
            allow_graph_approximation=bool(model_config.get("allow_graph_approximation", False)),
            require_signal_gene_map=bool(model_config.get("require_signal_gene_map", True)),
            data_manifest=cast(Optional[str], model_config.get("data_manifest")),
            data_manifest_digest=cast(Optional[str], model_config.get("data_manifest_digest")),
            signal_graph_version=str(model_config.get("signal_graph_version", "unversioned")),
            seed=cast(Optional[int], model_config.get("seed")),
        )


CrossScaleModel = CrossScalePTM2CellNet


__all__ = [
    "CrossScaleContractError",
    "CrossScaleConfig",
    "MultiPLMEncoder",
    "PTMTokenAdapter",
    "normalized_adjacency",
    "compute_sensitivity_matrix",
    "CIGNNSignalBridge",
    "CellGraphCompassHead",
    "CrossScalePTM2CellNet",
    "CrossScaleModel",
]
