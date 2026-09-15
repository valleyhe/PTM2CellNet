"""
Auto-initialization logic extracted from app.py.

Resolves cell-state labels, config, and checkpoint weights on API startup.
"""

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, cast

import torch
import yaml

from ..utils.checkpoint_utils import extract_model_state_dict, sibling_config_path
from ..utils.io import safe_torch_load
from ..utils.logging import setup_logger
from .routes.state import (
    STATE,
    initialize_model,
    initialize_pathway_mapper,
    initialize_variant_workflow,
    record_model_provenance,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# TypedDict definitions
# ---------------------------------------------------------------------------


class _DataSection(TypedDict, total=False):
    """Nested 'data' section of a training config YAML."""

    cell_states: List[str]


class _ModelSection(TypedDict, total=False):
    """Nested 'model' section of a training config YAML."""

    num_classes: int


class _TrainingConfig(TypedDict, total=False):
    """Config dict loaded from a training YAML file."""

    data: _DataSection
    model: _ModelSection
    model_kind: str
    model_card: str
    data_provenance: str


class _ArtifactManifest(TypedDict, total=False):
    """Manifest dict loaded from a sibling artifact_manifest.json."""

    cell_states: List[str]
    model_kind: str
    model_card: str
    data_provenance: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_autoinit_cell_states(
    config: _TrainingConfig,
    manifest: _ArtifactManifest,
    env_cell_states: Optional[str],
) -> Tuple[Optional[List[str]], str]:
    """Resolve cell-state labels for auto-init following a strict priority.

    Priority (P0-1):
        1. ``config["data"]["cell_states"]`` (written by the training script).
        2. ``manifest["cell_states"]`` from the sibling artifact_manifest.json.
        3. Explicit ``PTM2CELLNET_CELL_STATES`` env var (with a warning).
           A hard-coded default order is **never** used implicitly: if no source
           provides labels we return ``None`` so the caller can refuse to load.

    Returns ``(labels, source)`` where ``source`` describes which resolver won,
    for logging. ``labels`` is ``None`` when no authoritative source exists.
    """
    cfg_states = (config.get("data") or {}).get("cell_states")
    if isinstance(cfg_states, list) and cfg_states:
        return [str(s) for s in cfg_states], "config data.cell_states"

    manifest_states = manifest.get("cell_states") if isinstance(manifest, dict) else None
    if isinstance(manifest_states, list) and manifest_states:
        return [str(s) for s in manifest_states], "artifact_manifest cell_states"

    if env_cell_states:
        labels = [s.strip() for s in env_cell_states.split(",") if s.strip()]
        if labels:
            logger.warning(
                "PTM2CELLNET_CELL_STATES 被用作标签来源（config/manifest 均未声明 "
                "data.cell_states）。请确认该顺序与训练时一致，否则预测标签语义会错位。"
            )
            return labels, "PTM2CELLNET_CELL_STATES (explicit env)"

    return None, "none"


def _infer_logits_dim(state_dict: dict[str, torch.Tensor]) -> Optional[int]:
    """Best-effort: infer the classifier output dimension from checkpoint weights.

    Looks for the final classifier weight/bias by name patterns common across
    encoders (predictor head, classifier head). Returns ``None`` when nothing
    conclusive can be inferred, so callers treat it as "skip the check".
    """
    import torch as _torch

    candidates = []
    # Heuristic names ordered by specificity. The final linear layer of the
    # predictor head carries the per-class logits dim as its ``weight.shape[0]``.
    head_patterns = (
        "predictor.head.weight",
        "predictor.classifier.weight",
        "predictor.output.weight",
        "predictor.linear.weight",
        "classifier.weight",
        "head.weight",
        "output.weight",
    )
    for name in head_patterns:
        tensor = state_dict.get(name)
        if isinstance(tensor, _torch.Tensor) and tensor.ndim == 2:
            candidates.append((name, int(tensor.shape[0])))
            break

    if candidates:
        return candidates[0][1]

    # Fallback: the last *.weight whose shape[0] plausibly maps to num_classes.
    weight_keys = [k for k in state_dict if k.endswith(".weight")]
    # Bias-only classifiers are rare; also inspect the largest bias tensor.
    bias_keys = [k for k in state_dict if k.endswith(".bias")]
    best: Optional[Tuple[str, int]] = None
    for name in reversed(weight_keys):
        tensor = state_dict.get(name)
        if isinstance(tensor, _torch.Tensor) and tensor.ndim == 2:
            best = (name, int(tensor.shape[0]))
            break
    if best is None and bias_keys:
        # If no weight tensor, the bias of the final layer still encodes the
        # number of classes (one bias per logit).
        for name in reversed(bias_keys):
            tensor = state_dict.get(name)
            if isinstance(tensor, _torch.Tensor) and tensor.ndim == 1:
                best = (name, int(tensor.shape[0]))
                break
    if best is None:
        return None
    # Only trust the heuristic when the inferred dim is a plausible class count
    # (>1, and small enough to be a classifier head rather than a hidden layer).
    name, dim = best
    if 2 <= dim <= 256:
        logger.debug("推断 logits 维度: %s -> %d", name, dim)
        return dim
    return None


def _resolve_autoinit_config(checkpoint_path: str, config_path: str) -> Tuple[_TrainingConfig, _ArtifactManifest]:
    """Load config dict + sibling manifest dict for auto-init.

    Returns ``(config, manifest)``; either may be empty when the file is absent
    or unreadable. Never raises — auto-init is best-effort and the caller will
    surface label-resolution failures.
    """
    config: _TrainingConfig = {}
    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                config = cast(_TrainingConfig, yaml.safe_load(f) or {})
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("读取 config %s 失败: %s", config_path, exc)

    manifest: _ArtifactManifest = {}
    manifest_path = Path(checkpoint_path).with_name("artifact_manifest.json")
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = cast(_ArtifactManifest, json.load(f) or {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取 manifest %s 失败: %s", manifest_path, exc)

    # P1-3: merge provenance from the sibling artifact_manifest.json so the
    # authoritative model_kind (written by the training script) is honoured
    # even when the config file is regenerated without model_card fields.
    if isinstance(manifest, dict):
        for key in ("model_kind", "model_card", "data_provenance"):
            if key in manifest and key not in config:
                config[key] = manifest[key]

    return config, manifest


def _try_auto_initialize() -> None:
    """Attempt to load model from environment-configured paths on startup.

    Label semantics follow P0-1: cell_states come from the training artifact
    (config/manifest) — never from a hard-coded default. A model whose labels
    cannot be resolved is **not** loaded, because a successfully-loaded model
    with wrong label semantics is more dangerous than a model-less API.
    """
    # PTM2CELLNET_CHECKPOINT/CONFIG are the canonical variable names. MODEL_PATH
    # is accepted as a deprecated alias so existing Docker/K8s deployments keep
    # working, but we warn so operators migrate (P0-2).
    checkpoint_path = os.environ.get("PTM2CELLNET_CHECKPOINT")
    deprecated_checkpoint = os.environ.get("MODEL_PATH")
    if not checkpoint_path and deprecated_checkpoint:
        logger.warning(
            "MODEL_PATH is deprecated; use PTM2CELLNET_CHECKPOINT. Falling back to MODEL_PATH=%s for this startup.",
            deprecated_checkpoint,
        )
        checkpoint_path = deprecated_checkpoint
    if not checkpoint_path:
        checkpoint_path = ""

    # P1-1: 配置解析优先级 = 显式 PTM2CELLNET_CONFIG > checkpoint sibling config >
    # 明确报错。旧实现默认回落 configs/production.yaml——该文件是部署配置，
    # model 段只有 path/device/batch_size/max_sequence_length，缺少 encoder_type
    # 等架构字段，必然触发 PTM2CellNet.from_config 的架构校验失败
    # (architectures.py:577 -> model_utils.py:80)。
    config_path = os.environ.get("PTM2CELLNET_CONFIG", "")
    env_cell_states = os.environ.get("PTM2CELLNET_CELL_STATES")

    if not checkpoint_path:
        logger.info("PTM2CELLNET_CHECKPOINT not set — API starts without model. Load via /api/v1/initialize.")
        return

    if not Path(checkpoint_path).exists():
        logger.warning("Checkpoint not found at %s — API starts without model.", checkpoint_path)
        return

    if not config_path:
        # 与 resolve_inference_config (checkpoint_utils.py:202) 的 sibling-优先
        # 语义对齐：训练产物 best_model.config.yaml 自带模型架构字段，是唯一
        # 保证与 checkpoint 匹配的配置来源。
        sibling = sibling_config_path(checkpoint_path)
        if sibling.exists():
            config_path = str(sibling)
            logger.info("PTM2CELLNET_CONFIG 未设置，使用 checkpoint 配套配置 %s", config_path)
    if not config_path:
        logger.error(
            "自动初始化中止：未设置 PTM2CELLNET_CONFIG 且 checkpoint %s 同目录无 "
            "%s。请设置 PTM2CELLNET_CONFIG 指向训练该 checkpoint 时的 config。",
            checkpoint_path,
            sibling_config_path(checkpoint_path),
        )
        return
    if not Path(config_path).exists():
        logger.error(
            "自动初始化中止：PTM2CELLNET_CONFIG=%s 不存在。请指向训练该 checkpoint 时的 config 文件。",
            config_path,
        )
        return

    try:
        from ..models.architectures import PTM2CellNet

        config, manifest = _resolve_autoinit_config(checkpoint_path, config_path)

        # P0-1: resolve labels from the artifact first; refuse to load when no
        # authoritative source exists, instead of silently using a hard-coded
        # default that would produce semantically-wrong predictions.
        cell_states, label_source = _resolve_autoinit_cell_states(config, manifest, env_cell_states)
        if not cell_states:
            logger.error(
                "自动初始化中止：无法从 config/manifest/env 解析 cell_states "
                "(config=%s, manifest=%s, env=%s)。请提供训练产物，或显式设置 "
                "PTM2CELLNET_CELL_STATES 并确认其与训练时一致。API 将以无模型状态启动。",
                config_path,
                str(Path(checkpoint_path).with_name("artifact_manifest.json")),
                "PTM2CELLNET_CELL_STATES" if env_cell_states else "<unset>",
            )
            return

        # P0-1: 一致性校验——标签数必须与 config 声明的 num_classes 一致。
        config_num_classes = (config.get("model") or {}).get("num_classes")
        if isinstance(config_num_classes, int) and config_num_classes != len(cell_states):
            raise RuntimeError(
                f"标签数不一致：data.cell_states 有 {len(cell_states)} 个 "
                f"({cell_states})，但 config model.num_classes={config_num_classes}。"
                f"请使用与该 checkpoint 匹配的训练 config。"
            )

        # 推断 checkpoint 中 predictor 的 logits 维度，与标签数做最终一致性校验。
        raw_checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        state_dict = extract_model_state_dict(raw_checkpoint)
        inferred = _infer_logits_dim(state_dict)
        if inferred is not None and inferred != len(cell_states):
            raise RuntimeError(
                f"checkpoint logits 维度 ({inferred}) 与 cell_states 数量 "
                f"({len(cell_states)}) 不一致。标签来源={label_source}。"
                f"请使用与该 checkpoint 配套的训练 config。"
            )

        # 标签解析通过后再用 num_classes 构建模型，避免结构/标签错配。
        config.setdefault("model", {})
        if isinstance(config["model"], dict):
            config["model"]["num_classes"] = len(cell_states)

        model = PTM2CellNet.from_config(cast(dict[str, Any], config))

        # P0-2: 启动自动初始化同样使用严格加载。权重不匹配时不进入 healthy 状态，
        # 避免后续预测来自随机/部分初始化模型（这比启动失败更危险）。
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint 权重与 config 不匹配：缺失 {len(missing)} 个键，"
                f"多余 {len(unexpected)} 个键。"
                f"missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "自动初始化标签来源: %s, cell_states=%s (num_classes=%d)",
            label_source,
            cell_states,
            len(cell_states),
        )
        initialize_model(model, cell_states, model_device=device, config=cast(dict[str, Any], config))
        record_model_provenance(checkpoint_path, config_path, cast(dict[str, Any], config))

        # P1-3: warn loudly when the auto-loaded model is only a demo, so
        # operators don't ship biological predictions from a smoke model.
        if STATE.is_demo_model:
            logger.warning(
                "Loaded model '%s' is a DEMO/smoke model (model_kind=demo). "
                "It only validates the engineering pipeline and must NOT be "
                "used for real biological interpretation.",
                checkpoint_path,
            )

        # Initialize optional components
        initialize_pathway_mapper()
        try:
            initialize_variant_workflow(checkpoint_path)
        except (ImportError, ValueError, RuntimeError) as e:
            # Variant workflow is optional; continue starting the API without it
            # so that standard prediction endpoints remain available.
            logger.warning("Variant workflow initialization failed: %s", e)

        logger.info("Auto-initialized model from %s on %s", checkpoint_path, device)
    except Exception as e:
        STATE.initialization_failed = True
        logger.error(
            "Auto-initialization failed: %s. API starts without model. "
            "/health and /ready will return 503 until a model is loaded via /initialize.",
            e,
        )
