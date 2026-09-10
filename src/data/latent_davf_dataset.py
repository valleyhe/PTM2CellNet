"""Validated latent-pair dataset for current ``LatentDAVF`` training.

The current DAVF flow-matching objective needs an observed pair ``(z_0, z_1)``
from the same scVI coordinate system plus perturbation inputs.  This module
defines the smallest on-disk contract for that data.  It deliberately does
not infer missing directions, fabricate latent states, or use scVI decoder
indices as perturbation token IDs.

NPZ keys
--------
Required: ``metadata_json``, ``z_0`` ``[N, 64]``, ``z_1`` ``[N, 64]``,
``gene_ids`` ``[N, K]``, ``directions`` ``[N, K]``, and ``attention_mask``
``[N, K]``.

Optional: ``magnitudes`` ``[N, K]``.

``gene_ids`` are rows in the PerturbGen vocabulary only.  They are not scVI
decoder column indices.

``metadata_json`` must be a JSON object with ``schema_version``, a ``scvi``
object (``model_path``, ``latent_dim``, ``num_genes``, ordered ``gene_names``),
and an ``embedding_asset`` object (``path``, ``vocab_size``, ``embedding_dim``,
and the verified ``manifest``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDAVFDataError(ValueError):
    """Raised when a latent-pair NPZ violates the training contract."""


LATENT_DAVF_DATA_SCHEMA_VERSION = "ptm2cellnet.latent-davf-pairs.v1"


class LatentDAVFPairDataset(Dataset[Dict[str, torch.Tensor]]):
    """In-memory tensor dataset for supervised flow matching."""

    def __init__(
        self,
        *,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        gene_ids: torch.Tensor,
        directions: torch.Tensor,
        attention_mask: torch.Tensor,
        magnitudes: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        n_samples = int(z_0.shape[0])
        for name, value in (
            ("z_1", z_1),
            ("gene_ids", gene_ids),
            ("directions", directions),
            ("attention_mask", attention_mask),
        ):
            if value.shape[0] != n_samples:
                raise LatentDAVFDataError(f"{name} has {value.shape[0]} samples, expected {n_samples}")
        if magnitudes is not None and magnitudes.shape[0] != n_samples:
            raise LatentDAVFDataError(f"magnitudes has {magnitudes.shape[0]} samples, expected {n_samples}")

        self.z_0 = z_0.contiguous()
        self.z_1 = z_1.contiguous()
        self.gene_ids = gene_ids.contiguous()
        self.directions = directions.contiguous()
        self.attention_mask = attention_mask.contiguous()
        self.magnitudes = magnitudes.contiguous() if magnitudes is not None else None
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        return int(self.z_0.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "z_0": self.z_0[index],
            "z_1": self.z_1[index],
            "gene_ids": self.gene_ids[index],
            "directions": self.directions[index],
            "attention_mask": self.attention_mask[index],
        }
        if self.magnitudes is not None:
            item["magnitudes"] = self.magnitudes[index]
        return item


def _require_array(npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in npz.files:
        raise LatentDAVFDataError(f"latent DAVF NPZ is missing required key {name!r}")
    value = np.asarray(npz[name])
    if value.dtype.kind == "O":
        raise LatentDAVFDataError(f"latent DAVF NPZ key {name!r} must not be an object array")
    return value


def _validate_float_array(value: np.ndarray, name: str, ndim: int) -> None:
    if value.ndim != ndim:
        raise LatentDAVFDataError(f"{name} must be {ndim}-D, got shape {value.shape}")
    if value.dtype.kind != "b" and not np.issubdtype(value.dtype, np.number):
        raise LatentDAVFDataError(f"{name} must be numeric")
    if not np.isfinite(value).all():
        raise LatentDAVFDataError(f"{name} contains non-finite values")


def _load_and_validate_metadata(
    npz: np.lib.npyio.NpzFile,
    *,
    latent_dim: int,
    perturbgen_vocab_size: int,
    expected_scvi_model_path: str | Path | None,
    expected_scvi_gene_names: Sequence[str] | None,
    expected_embedding_manifest: Mapping[str, Any] | None,
    expected_embedding_path: str | Path | None,
    expected_embedding_dim: int | None,
    expected_intervention_type: str | None,
) -> dict[str, Any]:
    if "metadata_json" not in npz.files:
        raise LatentDAVFDataError(
            "formal latent DAVF NPZ must contain metadata_json with scVI and PerturbGen provenance"
        )
    raw = np.asarray(npz["metadata_json"])
    if raw.ndim != 0 or raw.dtype.kind not in {"U", "S"}:
        raise LatentDAVFDataError("metadata_json must be a scalar UTF-8 JSON string")
    try:
        value = raw.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise LatentDAVFDataError("metadata_json must be a scalar UTF-8 JSON string")
        metadata = json.loads(value)
    except LatentDAVFDataError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LatentDAVFDataError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise LatentDAVFDataError("metadata_json must describe an object")
    if metadata.get("schema_version") != LATENT_DAVF_DATA_SCHEMA_VERSION:
        raise LatentDAVFDataError(f"unsupported latent DAVF data schema_version {metadata.get('schema_version')!r}")

    dataset = metadata.get("dataset")
    if dataset is not None and not isinstance(dataset, dict):
        raise LatentDAVFDataError("metadata_json.dataset must be an object when present")
    if expected_intervention_type is not None:
        if expected_intervention_type not in {"KO", "KD", "OE"}:
            raise ValueError("expected_intervention_type must be KO, KD or OE")
        if not isinstance(dataset, dict):
            raise LatentDAVFDataError(
                "formal DAVF training data must declare dataset.intervention_type"
            )
        if dataset.get("intervention_type") != expected_intervention_type:
            raise LatentDAVFDataError(
                "latent-pair intervention type does not match the requested DAVF training direction"
            )
        expected_code = {"KO": 0, "KD": 1, "OE": 2}[expected_intervention_type]
        if dataset.get("direction_code") != expected_code:
            raise LatentDAVFDataError("latent-pair direction_code does not match intervention_type")

    scvi = metadata.get("scvi")
    if not isinstance(scvi, dict):
        raise LatentDAVFDataError("metadata_json.scvi must be an object")
    if scvi.get("latent_dim") != latent_dim:
        raise LatentDAVFDataError("latent-pair metadata scVI latent_dim does not match the input arrays")
    names = scvi.get("gene_names")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise LatentDAVFDataError("metadata_json.scvi.gene_names must be a unique ordered string list")
    if scvi.get("num_genes") != len(names):
        raise LatentDAVFDataError("metadata_json.scvi.num_genes does not match gene_names")
    model_path = scvi.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise LatentDAVFDataError("metadata_json.scvi.model_path is required")

    embedding = metadata.get("embedding_asset")
    if not isinstance(embedding, dict):
        raise LatentDAVFDataError("metadata_json.embedding_asset must be an object")
    if embedding.get("vocab_size") != perturbgen_vocab_size:
        raise LatentDAVFDataError("latent-pair metadata PerturbGen vocabulary size does not match the asset")
    embedding_dim = embedding.get("embedding_dim")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise LatentDAVFDataError("metadata_json.embedding_asset.embedding_dim must be a positive integer")
    embedding_path = embedding.get("path")
    if not isinstance(embedding_path, str) or not embedding_path:
        raise LatentDAVFDataError("metadata_json.embedding_asset.path is required")
    manifest = embedding.get("manifest")
    if not isinstance(manifest, dict):
        raise LatentDAVFDataError("metadata_json.embedding_asset.manifest is required")

    if expected_scvi_model_path is not None:
        if Path(model_path).expanduser().resolve() != Path(expected_scvi_model_path).expanduser().resolve():
            raise LatentDAVFDataError("latent-pair scVI model path does not match the training adapter")
    if expected_scvi_gene_names is not None and tuple(names) != tuple(str(name) for name in expected_scvi_gene_names):
        raise LatentDAVFDataError("latent-pair scVI gene order does not match the training adapter")
    if expected_embedding_manifest is not None and manifest != dict(expected_embedding_manifest):
        raise LatentDAVFDataError("latent-pair PerturbGen manifest does not match the loaded asset")
    if expected_embedding_path is not None:
        if Path(embedding_path).expanduser().resolve() != Path(expected_embedding_path).expanduser().resolve():
            raise LatentDAVFDataError("latent-pair embedding path does not match the loaded asset")
    if expected_embedding_dim is not None and embedding_dim != expected_embedding_dim:
        raise LatentDAVFDataError("latent-pair embedding dimension does not match the loaded asset")
    return metadata


def load_latent_davf_pairs(
    path: str | Path,
    *,
    latent_dim: int,
    perturbgen_vocab_size: int,
    num_directions: int = 3,
    expected_scvi_model_path: str | Path | None = None,
    expected_scvi_gene_names: Sequence[str] | None = None,
    expected_embedding_manifest: Mapping[str, Any] | None = None,
    expected_embedding_path: str | Path | None = None,
    expected_embedding_dim: int | None = None,
    expected_intervention_type: str | None = None,
) -> LatentDAVFPairDataset:
    """Load and validate a latent-pair NPZ for current DAVF training."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"latent DAVF training data not found: {path}")
    if latent_dim <= 0 or perturbgen_vocab_size <= 0 or num_directions <= 0:
        raise ValueError("latent_dim, perturbgen_vocab_size and num_directions must be positive")

    with np.load(path, allow_pickle=False) as npz:
        metadata = _load_and_validate_metadata(
            npz,
            latent_dim=latent_dim,
            perturbgen_vocab_size=perturbgen_vocab_size,
            expected_scvi_model_path=expected_scvi_model_path,
            expected_scvi_gene_names=expected_scvi_gene_names,
            expected_embedding_manifest=expected_embedding_manifest,
            expected_embedding_path=expected_embedding_path,
            expected_embedding_dim=expected_embedding_dim,
            expected_intervention_type=expected_intervention_type,
        )
        z_0 = _require_array(npz, "z_0")
        z_1 = _require_array(npz, "z_1")
        gene_ids = _require_array(npz, "gene_ids")
        directions = _require_array(npz, "directions")
        attention_mask = _require_array(npz, "attention_mask")
        magnitudes = np.asarray(npz["magnitudes"]) if "magnitudes" in npz.files else None

    _validate_float_array(z_0, "z_0", 2)
    _validate_float_array(z_1, "z_1", 2)
    if z_0.shape != z_1.shape or z_0.shape[1] != latent_dim:
        raise LatentDAVFDataError(
            f"z_0 and z_1 must both have shape [N, {latent_dim}], got {z_0.shape} and {z_1.shape}"
        )
    if z_0.shape[0] == 0:
        raise LatentDAVFDataError("latent DAVF training data must contain at least one sample")

    if gene_ids.ndim != 2 or directions.shape != gene_ids.shape or attention_mask.shape != gene_ids.shape:
        raise LatentDAVFDataError("gene_ids, directions and attention_mask must have the same 2-D shape [N, K]")
    if gene_ids.shape[0] != z_0.shape[0]:
        raise LatentDAVFDataError("latent arrays and perturbation arrays must have the same sample count")
    if gene_ids.shape[1] == 0:
        raise LatentDAVFDataError("each sample must contain at least one perturbation slot")
    if gene_ids.dtype.kind not in "iu" or directions.dtype.kind not in "iu":
        raise LatentDAVFDataError("gene_ids and directions must use integer dtypes")
    if (gene_ids < 0).any() or (gene_ids >= perturbgen_vocab_size).any():
        raise LatentDAVFDataError(f"gene_ids must be in [0, {perturbgen_vocab_size}) for the PerturbGen vocabulary")
    if (directions < 0).any() or (directions >= num_directions).any():
        raise LatentDAVFDataError(f"directions must be in [0, {num_directions})")

    _validate_float_array(attention_mask, "attention_mask", 2)
    if not np.isin(attention_mask, (0, 1)).all():
        raise LatentDAVFDataError("attention_mask must contain only 0/1 values")
    if not attention_mask.astype(bool).any(axis=1).all():
        raise LatentDAVFDataError("each latent DAVF sample must contain at least one active perturbation target")
    if magnitudes is not None:
        _validate_float_array(magnitudes, "magnitudes", 2)
        if magnitudes.shape != gene_ids.shape:
            raise LatentDAVFDataError("magnitudes must have the same shape as gene_ids")

    return LatentDAVFPairDataset(
        z_0=torch.as_tensor(z_0, dtype=torch.float32),
        z_1=torch.as_tensor(z_1, dtype=torch.float32),
        gene_ids=torch.as_tensor(gene_ids, dtype=torch.long),
        directions=torch.as_tensor(directions, dtype=torch.long),
        attention_mask=torch.as_tensor(attention_mask, dtype=torch.float32),
        magnitudes=torch.as_tensor(magnitudes, dtype=torch.float32) if magnitudes is not None else None,
        metadata=metadata,
    )


__all__ = [
    "LATENT_DAVF_DATA_SCHEMA_VERSION",
    "LatentDAVFDataError",
    "LatentDAVFPairDataset",
    "load_latent_davf_pairs",
]
