"""Strict checkpoint contract for the current latent DAVF mainline.

The repository contains several historical DAVF artifacts.  Some of them have
the same ``latent_dim``/``num_genes`` labels as the current model but contain a
different ``delta_mlp`` architecture.  A formal direction result therefore
needs more than a shape check: it needs a versioned payload, the current
``LatentDAVF`` state dict, and the exact PerturbGen embedding asset used to
train it.

The contract in this module is intentionally separate from the historical
``DAVFInferenceModule.forward`` compatibility path.  Only schema version 2
payloads produced by :func:`build_current_davf_checkpoint` can pass formal
validation.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import pickle
from typing import Any, Mapping

import torch

from src.models.gene_vocabulary import normalize_ensembl_id
from src.models.latent_davf import LatentDAVF, LatentDAVFConfig
from src.models.perturbgen_embedding import PerturbGenEmbeddingAsset


CURRENT_DAVF_CHECKPOINT_SCHEMA_VERSION = 2
CURRENT_DAVF_MODEL_TYPE = "LatentDAVF"
FORMAL_DAVF_LATENT_DIM = 64
FORMAL_DAVF_NUM_GENES = 4018
DAVF_DIRECTION_CODES = {"KO": 0, "KD": 1, "OE": 2}


class DAVFCheckpointContractError(ValueError):
    """Raised when an artifact cannot be used for formal DAVF inference."""


def latent_davf_config_to_dict(config: LatentDAVFConfig) -> dict[str, Any]:
    """Serialize only public ``LatentDAVFConfig`` fields.

    The config class stores default constants as annotated class fields.  They
    are implementation constants, not architecture settings, so they are
    deliberately excluded from the checkpoint payload.
    """

    return {field.name: getattr(config, field.name) for field in fields(config) if not field.name.startswith("_")}


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DAVFCheckpointContractError(f"{name} must be a positive integer")
    return int(value)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DAVFCheckpointContractError(f"{name} must be a mapping")
    return value


def _config_from_payload(payload: Mapping[str, Any]) -> LatentDAVFConfig:
    raw_config = _require_mapping(payload.get("config"), "checkpoint.config")
    expected_fields = {field.name for field in fields(LatentDAVFConfig) if not field.name.startswith("_")}
    actual_fields = set(raw_config)
    missing = expected_fields - actual_fields
    unknown = actual_fields - expected_fields
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing={sorted(missing)}")
        if unknown:
            detail.append(f"unknown={sorted(unknown)}")
        raise DAVFCheckpointContractError(
            "checkpoint.config does not describe the current LatentDAVF config (" + ", ".join(detail) + ")"
        )
    try:
        config = LatentDAVFConfig(**{name: raw_config[name] for name in expected_fields})
    except (TypeError, ValueError) as exc:
        raise DAVFCheckpointContractError(f"checkpoint.config is not a valid LatentDAVFConfig: {exc}") from exc
    return config


def _validate_state_dict(state_dict: Any) -> Mapping[str, torch.Tensor]:
    state = _require_mapping(state_dict, "checkpoint.model_state_dict")
    if not state:
        raise DAVFCheckpointContractError("checkpoint.model_state_dict must not be empty")
    invalid = [key for key, value in state.items() if not isinstance(key, str) or not isinstance(value, torch.Tensor)]
    if invalid:
        raise DAVFCheckpointContractError("checkpoint.model_state_dict must contain only string keys and tensors")
    non_finite = [
        key
        for key, value in state.items()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise DAVFCheckpointContractError(
            "checkpoint.model_state_dict contains non-finite tensors: " + ", ".join(non_finite[:3])
        )
    return state


def _validate_embedding_metadata(
    raw_metadata: Any,
    asset: PerturbGenEmbeddingAsset,
) -> None:
    metadata = _require_mapping(raw_metadata, "checkpoint.embedding_asset")
    shape = metadata.get("embedding_shape")
    if shape != list(asset.embeddings.shape):
        raise DAVFCheckpointContractError(
            "checkpoint PerturbGen embedding shape does not match the supplied asset: "
            f"checkpoint={shape!r}, asset={list(asset.embeddings.shape)!r}"
        )
    if metadata.get("vocab_size") != asset.vocab_size:
        raise DAVFCheckpointContractError("checkpoint PerturbGen vocabulary size does not match the supplied asset")
    if metadata.get("embedding_dim") != asset.embedding_dim:
        raise DAVFCheckpointContractError("checkpoint PerturbGen embedding dimension does not match the supplied asset")
    recorded_manifest = metadata.get("manifest")
    if not isinstance(recorded_manifest, Mapping):
        raise DAVFCheckpointContractError("checkpoint.embedding_asset.manifest is required")
    if dict(recorded_manifest) != dict(asset.manifest):
        raise DAVFCheckpointContractError("checkpoint PerturbGen manifest does not match the supplied embedding asset")


def _validate_embedding_state(
    state_dict: Mapping[str, torch.Tensor],
    asset: PerturbGenEmbeddingAsset,
) -> None:
    """Require the exported DAVF token table to remain the verified asset."""

    weight = state_dict.get("gene_embed_table.weight")
    if weight is None:
        raise DAVFCheckpointContractError("current LatentDAVF checkpoint must contain gene_embed_table.weight")
    expected = asset.embeddings.detach().cpu()
    actual = weight.detach().cpu()
    if tuple(actual.shape) != tuple(expected.shape):
        raise DAVFCheckpointContractError(
            "checkpoint gene_embed_table.weight shape does not match the PerturbGen embedding asset"
        )
    if not torch.equal(actual, expected):
        raise DAVFCheckpointContractError(
            "checkpoint gene_embed_table.weight differs from the supplied PerturbGen embedding asset; "
            "formal export requires the asset to remain frozen"
        )


def _validate_scvi_metadata(
    raw_metadata: Any,
    *,
    expected_latent_dim: int,
    expected_num_genes: int,
    scvi_adapter: Any | None,
) -> None:
    metadata = _require_mapping(raw_metadata, "checkpoint.scvi")
    if metadata.get("latent_dim") != expected_latent_dim:
        raise DAVFCheckpointContractError("checkpoint scVI latent dimension does not match the formal DAVF contract")
    if metadata.get("num_genes") != expected_num_genes:
        raise DAVFCheckpointContractError("checkpoint scVI gene dimension does not match the formal DAVF contract")
    model_path = metadata.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise DAVFCheckpointContractError("checkpoint.scvi.model_path is required for latent provenance")
    names = metadata.get("gene_names")
    if (
        not isinstance(names, list)
        or len(names) != expected_num_genes
        or not all(isinstance(name, str) for name in names)
    ):
        raise DAVFCheckpointContractError("checkpoint.scvi.gene_names must be the ordered scVI decoder vocabulary")
    try:
        canonical_names = [normalize_ensembl_id(name) for name in names]
    except ValueError as exc:
        raise DAVFCheckpointContractError("checkpoint.scvi.gene_names must contain canonical ENSG identifiers") from exc
    if names != canonical_names:
        raise DAVFCheckpointContractError(
            "checkpoint.scvi.gene_names must contain canonical ENSG identifiers without version suffixes"
        )
    if len(set(canonical_names)) != len(canonical_names):
        raise DAVFCheckpointContractError("checkpoint.scvi.gene_names must not contain duplicates")
    if scvi_adapter is not None:
        validator = getattr(scvi_adapter, "validate_compatibility", None)
        if not callable(validator):
            raise DAVFCheckpointContractError(
                "scvi_adapter must expose validate_compatibility() for formal checkpoint validation"
            )
        try:
            validator(
                expected_latent_dim=expected_latent_dim,
                expected_num_genes=expected_num_genes,
                expected_gene_names=names,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise DAVFCheckpointContractError(
                f"supplied scVI adapter does not match checkpoint provenance: {exc}"
            ) from exc
        adapter_config = getattr(scvi_adapter, "config", None)
        if isinstance(adapter_config, Mapping):
            live_model_path = adapter_config.get("model_path")
        else:
            live_model_path = getattr(adapter_config, "model_path", None)
        if live_model_path is not None:
            if not isinstance(live_model_path, (str, Path)) or not str(live_model_path):
                raise DAVFCheckpointContractError("supplied scVI adapter has an invalid model_path")
            if Path(live_model_path).expanduser().resolve() != Path(model_path).expanduser().resolve():
                raise DAVFCheckpointContractError(
                    "supplied scVI adapter model_path does not match checkpoint provenance"
                )


def _validate_training_metadata(
    raw_metadata: Any,
    *,
    expected_intervention_type: str | None = None,
) -> None:
    """Validate modality provenance when a checkpoint declares it.

    Historical schema-v2 fixtures may only contain an epoch, so the modality
    fields remain optional for backward reading. Once either field is present,
    both fields are required and must agree with the canonical direction code.
    """

    if expected_intervention_type is not None and expected_intervention_type not in DAVF_DIRECTION_CODES:
        raise ValueError("expected_intervention_type must be KO, KD or OE")
    if raw_metadata is None:
        if expected_intervention_type is not None:
            raise DAVFCheckpointContractError(
                "checkpoint.training must declare intervention_type for direction-specific validation"
            )
        return
    metadata = _require_mapping(raw_metadata, "checkpoint.training")
    intervention_type = metadata.get("intervention_type")
    direction_code = metadata.get("direction_code")
    if intervention_type is None and direction_code is None:
        if expected_intervention_type is not None:
            raise DAVFCheckpointContractError(
                "checkpoint.training must declare intervention_type and direction_code for direction-specific validation"
            )
        return
    if intervention_type not in DAVF_DIRECTION_CODES:
        raise DAVFCheckpointContractError("checkpoint.training.intervention_type must be one of KO, KD, or OE")
    expected_code = DAVF_DIRECTION_CODES[intervention_type]
    if isinstance(direction_code, bool) or not isinstance(direction_code, int):
        raise DAVFCheckpointContractError("checkpoint.training.direction_code must be an integer")
    if direction_code != expected_code:
        raise DAVFCheckpointContractError("checkpoint.training.direction_code does not match intervention_type")
    if expected_intervention_type is not None and intervention_type != expected_intervention_type:
        raise DAVFCheckpointContractError(
            "checkpoint.training.intervention_type does not match the requested DAVF training direction"
        )


def validate_current_davf_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    asset: PerturbGenEmbeddingAsset,
    expected_latent_dim: int = FORMAL_DAVF_LATENT_DIM,
    expected_num_genes: int = FORMAL_DAVF_NUM_GENES,
    model: LatentDAVF | None = None,
    scvi_adapter: Any | None = None,
    expected_intervention_type: str | None = None,
) -> LatentDAVFConfig:
    """Validate a current formal DAVF payload and its exact state dict.

    ``gene_ids`` in the model state/training path refer only to rows of the
    supplied PerturbGen asset.  The scVI decoder vocabulary is validated as
    provenance here, but target indices are never derived from this payload;
    runtime callers must resolve them from ``scvi_adapter.gene_names``.
    """

    if payload.get("schema_version") != CURRENT_DAVF_CHECKPOINT_SCHEMA_VERSION:
        raise DAVFCheckpointContractError(
            "formal DAVF checkpoint requires schema_version=2; retrain/export with the current LatentDAVF architecture"
        )
    if payload.get("model_type") != CURRENT_DAVF_MODEL_TYPE:
        raise DAVFCheckpointContractError(f"formal DAVF checkpoint requires model_type={CURRENT_DAVF_MODEL_TYPE!r}")

    expected_latent_dim = _as_positive_int(expected_latent_dim, "expected_latent_dim")
    expected_num_genes = _as_positive_int(expected_num_genes, "expected_num_genes")
    if (expected_latent_dim, expected_num_genes) != (FORMAL_DAVF_LATENT_DIM, FORMAL_DAVF_NUM_GENES):
        raise DAVFCheckpointContractError("formal current DAVF validation is fixed at latent_dim=64 and num_genes=4018")
    config = _config_from_payload(payload)
    if config.latent_dim != expected_latent_dim:
        raise DAVFCheckpointContractError(
            f"checkpoint latent_dim={config.latent_dim} does not match expected {expected_latent_dim}"
        )
    if config.num_genes != expected_num_genes:
        raise DAVFCheckpointContractError(
            f"checkpoint num_genes={config.num_genes} does not match expected {expected_num_genes}"
        )
    if not config.gene_embed_frozen:
        raise DAVFCheckpointContractError("formal current DAVF checkpoint requires gene_embed_frozen=true")

    _validate_training_metadata(
        payload.get("training"),
        expected_intervention_type=expected_intervention_type,
    )
    _validate_embedding_metadata(payload.get("embedding_asset"), asset)
    _validate_scvi_metadata(
        payload.get("scvi"),
        expected_latent_dim=expected_latent_dim,
        expected_num_genes=expected_num_genes,
        scvi_adapter=scvi_adapter,
    )
    state_dict = _validate_state_dict(payload.get("model_state_dict"))
    _validate_embedding_state(state_dict, asset)

    if model is None:
        model = LatentDAVF(config, pretrained_gene_embeddings=asset.embeddings)
    elif latent_davf_config_to_dict(model.config) != latent_davf_config_to_dict(config):
        raise DAVFCheckpointContractError("runtime LatentDAVF config does not match checkpoint.config")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise DAVFCheckpointContractError(
            f"checkpoint state dict is not an exact current LatentDAVF state dict: {exc}"
        ) from exc
    return config


def build_current_davf_checkpoint(
    model: LatentDAVF,
    *,
    asset: PerturbGenEmbeddingAsset,
    embedding_asset_path: str | Path,
    scvi_model_path: str | Path,
    scvi_adapter: Any,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-describing schema v2 checkpoint from a trained model."""

    config = model.config
    if config.latent_dim != FORMAL_DAVF_LATENT_DIM or config.num_genes != FORMAL_DAVF_NUM_GENES:
        raise DAVFCheckpointContractError("formal checkpoint export requires latent_dim=64 and num_genes=4018")
    validator = getattr(scvi_adapter, "validate_compatibility", None)
    if not callable(validator):
        raise DAVFCheckpointContractError("scvi_adapter must expose validate_compatibility()")
    try:
        validator(
            expected_latent_dim=FORMAL_DAVF_LATENT_DIM,
            expected_num_genes=FORMAL_DAVF_NUM_GENES,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DAVFCheckpointContractError(f"scVI adapter is incompatible: {exc}") from exc

    gene_names = tuple(str(name) for name in getattr(scvi_adapter, "gene_names", ()))
    if len(gene_names) != FORMAL_DAVF_NUM_GENES:
        raise DAVFCheckpointContractError("scVI adapter must expose all 4018 ordered gene names")

    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    if len(state_dict) != len(model.state_dict()):
        raise DAVFCheckpointContractError("all LatentDAVF state entries must be tensors")

    payload: dict[str, Any] = {
        "schema_version": CURRENT_DAVF_CHECKPOINT_SCHEMA_VERSION,
        "model_type": CURRENT_DAVF_MODEL_TYPE,
        "config": latent_davf_config_to_dict(config),
        "model_state_dict": state_dict,
        "embedding_asset": {
            "path": str(Path(embedding_asset_path).expanduser().resolve()),
            "embedding_shape": list(asset.embeddings.shape),
            "embedding_dim": asset.embedding_dim,
            "vocab_size": asset.vocab_size,
            "manifest": dict(asset.manifest),
        },
        "scvi": {
            "model_path": str(Path(scvi_model_path).expanduser().resolve()),
            "latent_dim": FORMAL_DAVF_LATENT_DIM,
            "num_genes": FORMAL_DAVF_NUM_GENES,
            # Provenance only. Runtime target indices must come from the
            # adapter's live gene_names property.
            "gene_names": list(gene_names),
        },
        "training": dict(training),
    }
    validate_current_davf_checkpoint_payload(
        payload,
        asset=asset,
        expected_latent_dim=FORMAL_DAVF_LATENT_DIM,
        expected_num_genes=FORMAL_DAVF_NUM_GENES,
        model=model,
        scvi_adapter=scvi_adapter,
    )
    return payload


def load_current_davf_checkpoint(
    checkpoint_path: str | Path,
    *,
    asset: PerturbGenEmbeddingAsset,
    expected_latent_dim: int = FORMAL_DAVF_LATENT_DIM,
    expected_num_genes: int = FORMAL_DAVF_NUM_GENES,
    scvi_adapter: Any | None = None,
    map_location: str | torch.device = "cpu",
    expected_intervention_type: str | None = None,
) -> tuple[dict[str, Any], LatentDAVFConfig]:
    """Safely load and validate a schema v2 current DAVF checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DAVF checkpoint not found: {path}")
    try:
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError, ModuleNotFoundError) as exc:
        raise DAVFCheckpointContractError(f"could not safely load current DAVF checkpoint {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DAVFCheckpointContractError("current DAVF checkpoint must be a mapping")
    payload = dict(raw)
    config = validate_current_davf_checkpoint_payload(
        payload,
        asset=asset,
        expected_latent_dim=expected_latent_dim,
        expected_num_genes=expected_num_genes,
        scvi_adapter=scvi_adapter,
        expected_intervention_type=expected_intervention_type,
    )
    return payload, config


__all__ = [
    "CURRENT_DAVF_CHECKPOINT_SCHEMA_VERSION",
    "CURRENT_DAVF_MODEL_TYPE",
    "DAVF_DIRECTION_CODES",
    "DAVFCheckpointContractError",
    "FORMAL_DAVF_LATENT_DIM",
    "FORMAL_DAVF_NUM_GENES",
    "build_current_davf_checkpoint",
    "latent_davf_config_to_dict",
    "load_current_davf_checkpoint",
    "validate_current_davf_checkpoint_payload",
]
