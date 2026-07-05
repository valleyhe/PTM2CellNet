"""Shared training-artifact export helpers (P0-2 / P1-1 / P1-3).

This module is the single source of truth for the inference artifact contract
used by every training entrypoint (``scripts/train.py``,
``scripts/train_lightning.py``, ``scripts/train_pretrained.py``). Keeping the
export logic here guarantees the three scripts emit a byte-compatible set of
files that the unified inference entrypoints (``scripts/predict.py`` and the
API auto-init/``/initialize``) can consume without per-script special cases.

Artifact contract (all three files live next to each other):

* ``best_model.pt`` — bare ``state_dict`` loadable by ``PTM2CellNet.from_config``.
* ``best_model.config.yaml`` — contains ``data.cell_states`` /
  ``data.label_to_idx`` / ``model.num_classes`` plus provenance/model_card.
* ``artifact_manifest.json`` — machine-readable provenance: cell_states,
  num_classes, model_kind, data_provenance, encoder metadata, metrics, and
  release-gating fields (deployable / split_strategy / intended_use).

The functions here are intentionally defensive: they accept either a bare
``nn.Module`` or a Lightning ``LightningModule`` (wrapping the real model at
``.model``), so the same export path serves native and Lightning training.
"""

from __future__ import annotations

import datetime
import hashlib
import subprocess
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import pandas as pd
from torch import nn

from ..data.labels import derive_label_mapping
from ..utils.config import Config
from ..utils.io import save_json, save_model
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

# Config dicts carry nested YAML-derived values. ``Any`` is required here
# because callers chain ``.get()`` on nested sub-dicts (e.g. cfg["data"]["cell_states"]).
# Centralising the alias keeps the ``Any`` to this single definition.
ConfigDict = Dict[str, Any]

# Metrics dicts map metric names to numeric measurements.
MetricsDict = Dict[str, Union[float, int]]

# Metadata dicts carry encoder / dataset profile info with simple scalar values.
MetadataDict = Dict[str, Union[str, int, float, bool, List[str], None]]


class _MetricCheckResult(TypedDict):
    threshold: float
    value: Union[float, int, None]
    passed: bool


class ReleaseGateResult(TypedDict):
    passed: bool
    checked: Dict[str, _MetricCheckResult]
    missing: List[str]


# Manifest dicts are JSON-serializable records with heterogeneous value types.
# Using ``object`` avoids false-positive mypy errors from dynamic construction
# while still being more precise than ``Any`` (no implicit subtyping).
ManifestDict = Dict[str, object]


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def derive_and_apply_label_mapping(
    config: Union[Config, ConfigDict],
    train_df: pd.DataFrame,
    label_col: str = "cell_state",
) -> Tuple[List[str], Dict[str, int]]:
    """Derive ``cell_states``/``label_to_idx`` from the train split and sync config.

    Mirrors the logic already inlined in ``train.py``/``train_lightning.py`` so
    every entrypoint derives labels the same way (sorted unique labels). Also
    forces ``model.num_classes`` to match — a mismatch is a programming error,
    not something to silently coerce at model-build time.

    Returns ``(cell_states, label_to_idx)``.
    """
    cell_states, label_to_idx = derive_label_mapping(train_df, label_col=label_col)

    cfg = config if isinstance(config, Config) else Config(config)
    cfg.set("data.cell_states", cell_states)
    cfg.set("data.label_to_idx", label_to_idx)
    cfg.set("model.num_classes", len(cell_states))
    logger.info(
        "同步训练标签映射: cell_states=%s, num_classes=%d", cell_states, len(cell_states)
    )
    return cell_states, label_to_idx


# ---------------------------------------------------------------------------
# Inference artifact export (state_dict + config)
# ---------------------------------------------------------------------------

def _unwrap_model(model_or_module: Union[nn.Module, object]) -> nn.Module:
    """Return the bare ``nn.Module`` from a Lightning module or pass through."""
    # Lightning modules expose the underlying model at ``.model``.
    base = getattr(model_or_module, "model", None)
    if isinstance(base, nn.Module):
        return base
    if isinstance(model_or_module, nn.Module):
        return model_or_module
    raise TypeError(
        f"export_inference_artifact: 无法识别的模型类型 {type(model_or_module)!r}；"
        f"期望 nn.Module 或带 .model 属性的 LightningModule。"
    )


def _resolve_cell_states_for_export(
    config: Union[Config, ConfigDict],
    cell_states: Optional[Sequence[str]] = None,
) -> List[str]:
    """Pick the authoritative cell_states order for export.

    Prefers explicitly passed labels (e.g. resolved from the train dataset),
    then falls back to ``config.data.cell_states``. Raises when neither is
    available — exporting without labels would produce an un-decodable model.
    """
    if cell_states:
        return list(cell_states)
    cfg = config.to_dict() if isinstance(config, Config) else config
    cfg_states = (cfg.get("data") or {}).get("cell_states")
    if isinstance(cfg_states, list) and cfg_states:
        return [str(s) for s in cfg_states]
    raise ValueError(
        "导出推理 artifact 时无法确定 cell_states：未传入且 config 中缺少 "
        "data.cell_states。推理将无法可靠解码类别索引。"
    )


def export_inference_artifact(
    model_or_module: Union[nn.Module, object],
    config: Union[Config, ConfigDict],
    output_dir: Union[str, Path],
    cell_states: Optional[Sequence[str]] = None,
    is_demo_data: bool = False,
    data_source: str = "unknown",
    extra_encoder_metadata: Optional[MetadataDict] = None,
) -> Dict[str, str]:
    """Write the bare ``best_model.pt`` + ``best_model.config.yaml`` pair.

    Args:
        model_or_module: a bare ``nn.Module`` or a Lightning module exposing
            ``.model``. The bare model's ``state_dict`` is exported so the
            unified inference path can load it without Lightning installed.
        config: training config (``Config`` or dict). Labels are re-synced onto
            it before saving so the yaml is self-describing.
        output_dir: directory to write the artifacts into (created if missing).
        cell_states: authoritative label order; when ``None`` the config's
            ``data.cell_states`` is used.
        is_demo_data: marks the artifact ``model_kind=demo`` so downstream
            tooling can warn against biological interpretation.
        data_source: human-readable description of the training data.
        extra_encoder_metadata: encoder-specific fields (e.g. HF model name,
            ``encoder_type``, cache/source) merged into the config + manifest.

    Returns:
        ``{"checkpoint_path": ..., "config_path": ...}`` for the caller to use
        when writing the manifest.
    """
    cfg = config if isinstance(config, Config) else Config(config)
    resolved_states = _resolve_cell_states_for_export(cfg, cell_states)

    # Final consistency guard: num_classes must match the label count.
    config_num_classes = cfg.get("model.num_classes")
    if isinstance(config_num_classes, int) and config_num_classes != len(resolved_states):
        raise ValueError(
            f"export_inference_artifact: model.num_classes ({config_num_classes}) 与 "
            f"cell_states 数量 ({len(resolved_states)}) 不一致。"
        )
    cfg.set("data.cell_states", resolved_states)
    cfg.set(
        "data.label_to_idx",
        {label: idx for idx, label in enumerate(resolved_states)},
    )
    cfg.set("model.num_classes", len(resolved_states))

    model_kind = "demo" if is_demo_data else "real"
    cfg.set("model.model_kind", model_kind)
    cfg.set(
        "model_card",
        {
            "model_kind": model_kind,
            "training_data": "synthetic_random" if is_demo_data else data_source,
            "not_for_biological_use": is_demo_data,
        },
    )
    if extra_encoder_metadata:
        model_cfg = cfg.get("model", {}) or {}
        if isinstance(model_cfg, dict):
            model_cfg.update(extra_encoder_metadata)
            cfg.set("model", model_cfg)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = _unwrap_model(model_or_module)
    checkpoint_path = out_dir / "best_model.pt"
    save_model(base_model, str(checkpoint_path))

    config_path = out_dir / "best_model.config.yaml"
    cfg.save(str(config_path))

    logger.info(
        "已导出推理 artifact: %s + %s (cell_states=%s, num_classes=%d)",
        checkpoint_path, config_path, resolved_states, len(resolved_states),
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
    }


# ---------------------------------------------------------------------------
# Artifact manifest (provenance + release gating)
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Failed to get git commit hash: %s", e)
        return "unknown"


def dataset_hash(df: pd.DataFrame) -> str:
    """Stable content hash of a training dataset for manifest provenance (P1-1).

    Uses sha256 over the CSV representation so identical data yields identical
    hashes regardless of row order perturbations introduced by shuffling.
    """
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def evaluate_release_gate(
    metrics: MetricsDict,
    thresholds: Dict[str, float],
) -> ReleaseGateResult:
    """Evaluate test metrics against configurable release thresholds (P1-1).

    Each ``thresholds`` entry names a metric and its minimum acceptable value.
    A metric is considered met when its measured value is ``>=`` the threshold.
    Metrics referenced in ``thresholds`` but absent from ``metrics`` count as a
    failure (the gate cannot be evaluated without the evidence).

    Returns a dict with ``passed`` (bool), ``checked`` (per-metric pass/fail),
    and ``missing`` (threshold metrics with no measurement). Experiments below
    threshold are still saved — they are simply marked ``deployable=False``.
    """
    checked: Dict[str, _MetricCheckResult] = {}
    missing: List[str] = []
    all_passed = True
    for metric_name, threshold in thresholds.items():
        if metric_name not in metrics or metrics[metric_name] is None:
            missing.append(metric_name)
            checked[metric_name] = {
                "threshold": threshold, "value": None, "passed": False,
            }
            all_passed = False
            continue
        value = metrics[metric_name]
        try:
            ok = float(value) >= float(threshold)
        except (TypeError, ValueError):
            ok = False
        checked[metric_name] = {
            "threshold": threshold, "value": value, "passed": bool(ok),
        }
        if not ok:
            all_passed = False
    return {"passed": bool(all_passed), "checked": checked, "missing": missing}


def write_artifact_manifest(
    output_dir: Union[str, Path],
    *,
    model_class: str,
    cell_states: Sequence[str],
    config: Union[Config, ConfigDict],
    training_entrypoint: str,
    is_demo_data: bool = False,
    data_source: str = "unknown",
    train_df: Optional[pd.DataFrame] = None,
    split_strategy: Optional[str] = None,
    metrics: Optional[MetricsDict] = None,
    encoder_metadata: Optional[MetadataDict] = None,
    intended_use: Optional[str] = None,
    limitations: Optional[str] = None,
    deployable: Optional[bool] = None,
    dataset_profile: Optional[MetadataDict] = None,
    release_thresholds: Optional[Dict[str, float]] = None,
) -> ManifestDict:
    """Write ``artifact_manifest.json`` with provenance + release-gating fields.

    The manifest is the authoritative record of *what* was trained and *whether*
    it may be deployed. Callers that train on real data should pass ``metrics``
    + ``release_thresholds`` so ``deployable`` reflects a real gate rather than
    a hand-set flag (see :func:`evaluate_release_gate`).
    """
    cfg = config.to_dict() if isinstance(config, Config) else dict(config)
    out_dir = Path(output_dir)
    model_kind = "demo" if is_demo_data else "real"

    manifest: ManifestDict = {
        "checkpoint_path": str(out_dir / "best_model.pt"),
        "config_path": str(out_dir / "best_model.config.yaml"),
        "training_entrypoint": training_entrypoint,
        "model_class": model_class,
        "num_classes": (cfg.get("model") or {}).get("num_classes", len(cell_states)),
        "cell_states": list(cell_states),
        "max_sequence_length": (cfg.get("data") or {}).get("max_sequence_length", 1000),
        "ptm_types": (cfg.get("data") or {}).get("ptm_types"),
        "git_commit": _git_commit(),
        "created_at": datetime.datetime.now().isoformat(),
        # Provenance
        "model_kind": model_kind,
        "model_card": {
            "model_kind": model_kind,
            "training_data": "synthetic_random" if is_demo_data else data_source,
            "not_for_biological_use": is_demo_data,
            "description": (
                "Demo/smoke model trained on synthetic/sample data. Validates the "
                "engineering pipeline only; predictions have no biological meaning."
            )
            if is_demo_data
            else (
                "Model trained on real data. Verify dataset provenance before use."
            ),
        },
        "data_provenance": {
            "model_kind": model_kind,
            "training_data": "synthetic_random" if is_demo_data else data_source,
            "source": data_source,
            "not_for_biological_use": is_demo_data,
        },
    }

    # Encoder metadata (e.g. HF model name / encoder_type / cache) — important
    # for pretrained models whose inference needs the same tokenizer/backbone.
    if encoder_metadata:
        manifest["encoder"] = dict(encoder_metadata)

    # Dataset + split provenance (P1-1).
    if train_df is not None:
        manifest["dataset_hash"] = dataset_hash(train_df)
        manifest["dataset_stats"] = {
            "row_count": int(len(train_df)),
            "column_count": int(len(train_df.columns)),
        }
    if dataset_profile:
        manifest["dataset_profile"] = dict(dataset_profile)
    if split_strategy:
        manifest["split_strategy"] = split_strategy

    # Evaluation metrics + release gate (P1-1).
    if metrics:
        manifest["metrics"] = dict(metrics)
    manifest["intended_use"] = intended_use or (
        "Engineering pipeline validation only." if is_demo_data else
        "Cell-state prediction. Validate on held-out biological data before use."
    )
    if limitations:
        manifest["limitations"] = limitations
    # Release gate: demo is never deployable. For real data, run the configured
    # metric thresholds when provided; otherwise deployable iff metrics exist.
    gate_result: Optional[ReleaseGateResult] = None
    if not is_demo_data and metrics and release_thresholds:
        gate_result = evaluate_release_gate(metrics, release_thresholds)
        manifest["release_gate"] = gate_result

    if deployable is not None:
        manifest["deployable"] = bool(deployable)
    elif is_demo_data:
        manifest["deployable"] = False
    elif gate_result is not None:
        manifest["deployable"] = gate_result["passed"]
    else:
        manifest["deployable"] = bool(metrics)

    manifest_path = out_dir / "artifact_manifest.json"
    save_json(manifest, str(manifest_path))
    logger.info("artifact manifest 已导出: %s", manifest_path)
    if is_demo_data:
        logger.warning(
            "⚠️ 该模型在合成/示例数据上训练，已标记 model_kind=demo。"
            "产物仅用于工程链路验证，不可用于真实生物学预测。"
        )
    return manifest


__all__ = [
    "ConfigDict",
    "ManifestDict",
    "MetricsDict",
    "MetadataDict",
    "ReleaseGateResult",
    "derive_and_apply_label_mapping",
    "export_inference_artifact",
    "write_artifact_manifest",
    "evaluate_release_gate",
    "dataset_hash",
]
