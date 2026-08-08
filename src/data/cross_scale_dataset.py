"""Versioned dataset contract for cross-scale training and inference.

The on-disk format is a non-pickled ``.npz`` archive.  Per-sample arrays use
the leading sample axis while graph topology and signal-to-gene mappings are
shared by every sample.  The same collate function is used by training and
offline inference so shape validation cannot drift between the two paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


CROSS_SCALE_DATA_SCHEMA_VERSION = "ptm2cellnet.cross-scale.npz.v1"
DEFAULT_BACKBONES = ("ankh39", "esm2", "prott5")

_STATIC_INPUT_KEYS = (
    "signal_edge_index",
    "signal_edge_weight",
    "signal_edge_type",
    "signal_gene_map",
    "cell_edge_index",
)
_OPTIONAL_SAMPLE_INPUT_KEYS = (
    "protein_attention_mask",
    "ptm_types",
    "ptm_positions",
    "ptm_mask",
    "cell_gene_features",
    "gene_mask",
)


class CrossScaleDataContractError(ValueError):
    """Raised when a cross-scale archive violates the versioned schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_tensor(array: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(np.array(array, copy=True))
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


class CrossScaleNPZDataset(Dataset[Dict[str, Any]]):
    """Validated, in-memory view of a versioned cross-scale ``.npz`` split."""

    def __init__(
        self,
        path: str | Path,
        *,
        backbone_names: Sequence[str] = DEFAULT_BACKBONES,
        require_targets: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.backbone_names = tuple(str(name).lower() for name in backbone_names)
        self.require_targets = bool(require_targets)
        if not self.path.is_file():
            raise FileNotFoundError(f"跨尺度数据文件不存在: {self.path}")
        try:
            archive = np.load(self.path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise CrossScaleDataContractError(f"无法读取安全 NPZ 数据 {self.path}: {exc}") from exc
        with archive:
            self._arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        self._validate()
        self.digest = _sha256(self.path)

    def _validate(self) -> None:
        errors: list[str] = []
        raw_version = self._arrays.get("schema_version")
        version = str(raw_version.item()) if raw_version is not None and raw_version.ndim == 0 else None
        if version != CROSS_SCALE_DATA_SCHEMA_VERSION:
            errors.append(
                f"schema_version 必须为 {CROSS_SCALE_DATA_SCHEMA_VERSION!r}，实际为 {version!r}"
            )

        embedding_keys = [f"{name}_embeddings" for name in self.backbone_names]
        for key in embedding_keys:
            if key not in self._arrays:
                errors.append(f"缺少必需数组 {key}")
        for key in ("signal_edge_index", "signal_gene_map", "cell_edge_index"):
            if key not in self._arrays:
                errors.append(f"缺少必需图数组 {key}")
        if self.require_targets:
            for key in ("delta_expression", "cell_state"):
                if key not in self._arrays:
                    errors.append(f"缺少训练 target 数组 {key}")
        if errors:
            raise CrossScaleDataContractError("; ".join(errors))

        first_embedding = self._arrays[embedding_keys[0]]
        if first_embedding.ndim != 3 or min(first_embedding.shape) <= 0:
            errors.append(f"{embedding_keys[0]} 必须为非空 [N, L, D]")
            sample_count, sequence_length = 0, 0
        else:
            sample_count, sequence_length = first_embedding.shape[:2]
        for key in embedding_keys:
            value = self._arrays[key]
            if value.ndim != 3 or value.shape[:2] != (sample_count, sequence_length):
                errors.append(f"{key} 必须共享 [N, L]=[{sample_count}, {sequence_length}]")
            elif not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                errors.append(f"{key} 必须是有限浮点数组")

        signal_edge_index = self._arrays["signal_edge_index"]
        cell_edge_index = self._arrays["cell_edge_index"]
        signal_gene_map = self._arrays["signal_gene_map"]
        if signal_edge_index.ndim != 2 or signal_edge_index.shape[0] != 2:
            errors.append("signal_edge_index 必须为 [2, E]")
        if cell_edge_index.ndim != 2 or cell_edge_index.shape[0] != 2:
            errors.append("cell_edge_index 必须为 [2, E]")
        if not np.issubdtype(signal_edge_index.dtype, np.integer):
            errors.append("signal_edge_index 必须使用整数 dtype")
        if not np.issubdtype(cell_edge_index.dtype, np.integer):
            errors.append("cell_edge_index 必须使用整数 dtype")
        if signal_gene_map.ndim != 2 or signal_gene_map.shape[0] != sequence_length:
            errors.append(f"signal_gene_map 必须为 [{sequence_length}, G]")
            gene_count = 0
        else:
            gene_count = signal_gene_map.shape[1]
        if not np.issubdtype(signal_gene_map.dtype, np.floating) or not np.isfinite(signal_gene_map).all():
            errors.append("signal_gene_map 必须是有限浮点数组")

        if signal_edge_index.size and (
            signal_edge_index.min() < 0 or signal_edge_index.max() >= sequence_length
        ):
            errors.append("signal_edge_index 含越界节点")
        if cell_edge_index.size and (
            cell_edge_index.min() < 0 or cell_edge_index.max() >= gene_count
        ):
            errors.append("cell_edge_index 含越界节点")

        for key in _OPTIONAL_SAMPLE_INPUT_KEYS:
            optional_value = self._arrays.get(key)
            if optional_value is not None and (
                optional_value.ndim == 0 or optional_value.shape[0] != sample_count
            ):
                errors.append(f"{key} 的首维必须为样本数 {sample_count}")
        if ("ptm_types" in self._arrays) != ("ptm_positions" in self._arrays):
            errors.append("ptm_types 与 ptm_positions 必须同时存在")
        if "ptm_types" in self._arrays and self._arrays["ptm_types"].shape != self._arrays["ptm_positions"].shape:
            errors.append("ptm_types 与 ptm_positions shape 必须一致")
        mask = self._arrays.get("protein_attention_mask")
        if mask is not None and mask.shape != (sample_count, sequence_length):
            errors.append(f"protein_attention_mask 必须为 [{sample_count}, {sequence_length}]")

        if self.require_targets and "delta_expression" in self._arrays:
            target = self._arrays["delta_expression"]
            if target.shape != (sample_count, gene_count):
                errors.append(f"delta_expression 必须为 [{sample_count}, {gene_count}]")
            elif not np.issubdtype(target.dtype, np.floating) or not np.isfinite(target).all():
                errors.append("delta_expression 必须是有限浮点数组")
        if self.require_targets and "cell_state" in self._arrays:
            target = self._arrays["cell_state"]
            if target.shape != (sample_count,) or not np.issubdtype(target.dtype, np.integer):
                errors.append(f"cell_state 必须为整数 [{sample_count}]")

        sample_ids = self._arrays.get("sample_id")
        if sample_ids is not None and sample_ids.shape != (sample_count,):
            errors.append(f"sample_id 必须为 [{sample_count}]")
        if errors:
            raise CrossScaleDataContractError("; ".join(errors))
        self.sample_count = int(sample_count)
        self.sequence_length = int(sequence_length)
        self.gene_count = int(gene_count)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            f"{name}_embeddings": _as_tensor(self._arrays[f"{name}_embeddings"][index])
            for name in self.backbone_names
        }
        for key in _OPTIONAL_SAMPLE_INPUT_KEYS:
            if key in self._arrays:
                inputs[key] = _as_tensor(self._arrays[key][index])
        for key in _STATIC_INPUT_KEYS:
            if key in self._arrays:
                inputs[key] = _as_tensor(self._arrays[key])
        item: Dict[str, Any] = {"inputs": inputs}
        if self.require_targets:
            item["targets"] = {
                "delta_expression": _as_tensor(self._arrays["delta_expression"][index]),
                "cell_state": _as_tensor(self._arrays["cell_state"][index]),
            }
        if "sample_id" in self._arrays:
            item["sample_id"] = str(self._arrays["sample_id"][index])
        else:
            item["sample_id"] = str(index)
        return item

    def contract(self) -> Dict[str, Any]:
        return {
            "schema_version": CROSS_SCALE_DATA_SCHEMA_VERSION,
            "path": str(self.path),
            "sha256": self.digest,
            "sample_count": self.sample_count,
            "sequence_length": self.sequence_length,
            "gene_count": self.gene_count,
            "backbones": list(self.backbone_names),
            "backbone_dims": {
                name: int(self._arrays[f"{name}_embeddings"].shape[-1])
                for name in self.backbone_names
            },
            "targets_present": self.require_targets,
        }


def cross_scale_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate samples while preserving one shared graph instead of stacking it."""

    if not samples:
        raise CrossScaleDataContractError("cannot collate an empty cross-scale batch")
    inputs: Dict[str, Any] = {}
    first_inputs = samples[0].get("inputs")
    if not isinstance(first_inputs, Mapping):
        raise CrossScaleDataContractError("each sample must contain an inputs mapping")
    for key in first_inputs:
        values = [sample["inputs"][key] for sample in samples]
        if key in _STATIC_INPUT_KEYS:
            if not all(isinstance(value, Tensor) and torch.equal(value, values[0]) for value in values):
                raise CrossScaleDataContractError(f"batch contains inconsistent shared graph array: {key}")
            inputs[key] = values[0]
        else:
            inputs[key] = torch.stack(values)
    batch: Dict[str, Any] = {
        "inputs": inputs,
        "sample_id": [str(sample.get("sample_id", index)) for index, sample in enumerate(samples)],
    }
    if "targets" in samples[0]:
        batch["targets"] = {
            key: torch.stack([sample["targets"][key] for sample in samples])
            for key in samples[0]["targets"]
        }
    return batch


class EpochShuffleSampler(Sampler[int]):
    """Epoch-addressable deterministic sampler for exact resume ordering."""

    def __init__(self, data_source: CrossScaleNPZDataset, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class CrossScaleDataModule:
    """Train/validation/test loaders sharing the cross-scale batch contract."""

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        test_path: str | Path,
        *,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        backbone_names: Sequence[str] = DEFAULT_BACKBONES,
    ) -> None:
        if batch_size <= 0 or num_workers < 0:
            raise CrossScaleDataContractError("batch_size must be positive and num_workers non-negative")
        self.train_dataset = CrossScaleNPZDataset(train_path, backbone_names=backbone_names)
        self.val_dataset = CrossScaleNPZDataset(val_path, backbone_names=backbone_names)
        self.test_dataset = CrossScaleNPZDataset(test_path, backbone_names=backbone_names)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self._validate_split_compatibility()

    def _validate_split_compatibility(self) -> None:
        reference = self.train_dataset.contract()
        for name, dataset in (("val", self.val_dataset), ("test", self.test_dataset)):
            contract = dataset.contract()
            for field in ("sequence_length", "gene_count", "backbones"):
                if contract[field] != reference[field]:
                    raise CrossScaleDataContractError(
                        f"{name} split {field}={contract[field]!r} 与 train={reference[field]!r} 不兼容"
                    )

    def _loader(self, dataset: CrossScaleNPZDataset, *, shuffle: bool) -> DataLoader[Dict[str, Any]]:
        sampler = EpochShuffleSampler(dataset, seed=self.seed) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=cross_scale_collate,
        )

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        return self._loader(self.test_dataset, shuffle=False)

    def contracts(self) -> Dict[str, Dict[str, Any]]:
        return {
            "train": self.train_dataset.contract(),
            "val": self.val_dataset.contract(),
            "test": self.test_dataset.contract(),
        }


__all__ = [
    "CROSS_SCALE_DATA_SCHEMA_VERSION",
    "CrossScaleDataContractError",
    "CrossScaleNPZDataset",
    "cross_scale_collate",
    "EpochShuffleSampler",
    "CrossScaleDataModule",
]
