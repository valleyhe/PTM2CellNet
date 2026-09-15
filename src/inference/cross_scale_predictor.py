"""Offline inference from self-describing cross-scale artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.cross_scale_dataset import CrossScaleNPZDataset, cross_scale_collate
from ..models.cross_scale import CrossScalePTM2CellNet


ARTIFACT_SCHEMA_VERSION = "ptm2cellnet.cross-scale.artifact.v1"
CHECKPOINT_SCHEMA_VERSION = "ptm2cellnet.cross-scale.checkpoint.v1"


class CrossScaleArtifactError(ValueError):
    """Raised when an artifact is incomplete, incompatible, or unsafe to load."""


def _read_json_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise CrossScaleArtifactError(f"artifact 文件不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossScaleArtifactError(f"无法解析 artifact JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CrossScaleArtifactError(f"artifact JSON 顶层必须为对象: {path}")
    return payload


def _resolve_artifact_file(root: Path, reference: Any, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise CrossScaleArtifactError(f"artifact.{field} 必须是非空相对路径")
    relative = Path(reference)
    if relative.is_absolute():
        raise CrossScaleArtifactError(f"artifact.{field} 不允许绝对路径")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CrossScaleArtifactError(f"artifact.{field} 发生路径穿越") from exc
    if not candidate.is_file():
        raise CrossScaleArtifactError(f"artifact.{field} 文件不存在: {candidate}")
    return candidate


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_cross_scale_artifact(
    artifact_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    allow_last_checkpoint_fallback: bool = False,
) -> Tuple[CrossScalePTM2CellNet, Dict[str, Any], Dict[str, Any]]:
    """Load and cross-check artifact manifest, config, label vocabulary and weights.

    ``allow_last_checkpoint_fallback=True`` relaxes only *which* checkpoint file
    is resolved: when ``best_checkpoint`` is missing from the artifact (e.g. a
    run stopped before validation improved), ``last_checkpoint`` is used instead.
    All other cross-checks (schema, config/label consistency, strict weight
    loading) remain exactly as strict.
    """

    root = Path(artifact_dir).expanduser().resolve()
    manifest = _read_json_mapping(root / "artifact_manifest.json")
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise CrossScaleArtifactError("artifact schema 不受支持")
    if manifest.get("model_type") != "cross_scale":
        raise CrossScaleArtifactError("artifact.model_type 必须为 cross_scale")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise CrossScaleArtifactError("artifact 缺少自描述 config")
    labels = manifest.get("label_vocabulary")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels):
        raise CrossScaleArtifactError("artifact.label_vocabulary 必须是非空字符串列表")
    if len(set(labels)) != len(labels):
        raise CrossScaleArtifactError("artifact.label_vocabulary 不允许重复")

    try:
        checkpoint_path = _resolve_artifact_file(root, manifest.get("best_checkpoint"), field="best_checkpoint")
    except CrossScaleArtifactError:
        if not allow_last_checkpoint_fallback:
            raise
        checkpoint_path = _resolve_artifact_file(root, manifest.get("last_checkpoint"), field="last_checkpoint")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise CrossScaleArtifactError("请求了 CUDA，但当前环境不可用")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CrossScaleArtifactError(f"无法安全加载 checkpoint: {exc}") from exc
    if not isinstance(checkpoint, Mapping) or checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CrossScaleArtifactError("checkpoint schema 不受支持")
    checkpoint_context = checkpoint.get("artifact_context")
    if not isinstance(checkpoint_context, Mapping):
        raise CrossScaleArtifactError("checkpoint 缺少 artifact_context")
    if _canonical(checkpoint_context.get("config")) != _canonical(config):
        raise CrossScaleArtifactError("artifact manifest 与 checkpoint config 不一致")
    if checkpoint_context.get("label_vocabulary") != labels:
        raise CrossScaleArtifactError("artifact manifest 与 checkpoint 标签词表不一致")

    model = CrossScalePTM2CellNet.from_config(config)
    checkpoint_info = checkpoint.get("model_info")
    if not isinstance(checkpoint_info, Mapping):
        raise CrossScaleArtifactError("checkpoint 缺少 model_info")
    if _canonical(checkpoint_info.get("config")) != _canonical(model.get_model_info()["config"]):
        raise CrossScaleArtifactError("checkpoint 模型配置与 artifact 构建结果不一致")
    if len(labels) != model.cross_scale_config.num_cell_states:
        raise CrossScaleArtifactError(
            f"标签数 {len(labels)} 与 num_cell_states={model.cross_scale_config.num_cell_states} 不一致"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise CrossScaleArtifactError("checkpoint 缺少 model_state_dict")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CrossScaleArtifactError(f"checkpoint 权重与 config 不兼容: {exc}") from exc
    model.to(target_device).eval()
    return model, manifest, dict(checkpoint)


class CrossScalePredictor:
    """Batch predictor that reuses the training dataset and collate contract."""

    def __init__(
        self,
        model: CrossScalePTM2CellNet,
        manifest: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.manifest = dict(manifest)
        self.device = torch.device(device)
        self.labels = list(self.manifest["label_vocabulary"])

    def _validate_dataset(self, dataset: CrossScaleNPZDataset) -> None:
        if dataset.gene_count != self.model.cross_scale_config.num_cell_genes:
            raise CrossScaleArtifactError(
                f"推理 gene_count={dataset.gene_count} 与模型 {self.model.cross_scale_config.num_cell_genes} 不一致"
            )
        configured_dims = self.model.protein_encoder.backbone_dims
        dataset_dims = dataset.contract()["backbone_dims"]
        for name in self.model.protein_encoder.backbone_names:
            actual = int(dataset_dims[name])
            expected = configured_dims.get(name)
            if expected is not None and actual != expected:
                raise CrossScaleArtifactError(f"推理 {name} embedding_dim={actual} 与模型 {expected} 不一致")

    def predict_dataset(
        self,
        dataset: CrossScaleNPZDataset,
        *,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        if batch_size <= 0 or num_workers < 0:
            raise CrossScaleArtifactError("batch_size 必须为正且 num_workers 不能为负")
        self._validate_dataset(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cross_scale_collate,
        )
        sample_ids: list[str] = []
        deltas: list[Tensor] = []
        logits: list[Tensor] = []
        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    key: value.to(self.device) if isinstance(value, Tensor) else value
                    for key, value in batch["inputs"].items()
                }
                outputs = self.model(inputs)
                sample_ids.extend(batch["sample_id"])
                deltas.append(outputs["delta_expression"].detach().cpu())
                logits.append(outputs["cell_state_logits"].detach().cpu())
        delta_expression = torch.cat(deltas, dim=0)
        cell_state_logits = torch.cat(logits, dim=0)
        probabilities = torch.softmax(cell_state_logits, dim=-1)
        state_indices = probabilities.argmax(dim=-1)
        return {
            "sample_id": sample_ids,
            "delta_expression": delta_expression,
            "cell_state_logits": cell_state_logits,
            "cell_state_probabilities": probabilities,
            "cell_state_index": state_indices,
            "cell_state_label": [self.labels[int(index)] for index in state_indices.tolist()],
            "provenance": {
                "artifact_schema_version": self.manifest["artifact_schema_version"],
                "model_type": self.manifest["model_type"],
                "data_contract": dataset.contract(),
                "model_info": self.model.get_model_info(),
            },
        }


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CrossScaleArtifactError",
    "load_cross_scale_artifact",
    "CrossScalePredictor",
]
