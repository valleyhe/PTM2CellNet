"""
Checkpoint 工具模块

功能概述:
    将 model checkpoint 与其训练时使用的配置绑定，避免推理时因 checkpoint 与
    config 不匹配（例如 encoder_type、num_classes、num_ptm_types 不一致）导致
    state_dict 加载失败。

设计思路:
    - 训练保存 checkpoint 时，建议在同目录写入 ``<name>.config.yaml``。
    - 推理时通过 :func:`load_checkpoint_with_config` 优先读取 checkpoint 同目录
      的 ``.config.yaml``，若不存在则回退到调用方显式传入的 ``config_path``。
    - 提供友好的错误信息，便于用户定位 checkpoint/config 不匹配问题。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .config import Config
from .logging import setup_logger
from .io import safe_torch_load

logger = setup_logger(__name__)


def extract_model_state_dict(checkpoint: Any) -> "torch.nn.modules.module._IncompatibleKeys":  # type: ignore[name-defined]
    """从任意 checkpoint 结构中提取裸模型 ``state_dict``。

    支持以下格式（统一 checkpoint 合约，对应审计报告 P0-3）：

    1. **裸 state_dict**（``state_dict`` 直接为 ``torch.Tensor`` 的字典）：
       直接返回。这是原生 ``scripts/train.py`` 通过 ``torch.save(model.state_dict())``
       产出的格式。
    2. **Lightning ``.ckpt``**：包含 ``"state_dict"`` 键，且键名带 ``model.`` 前缀
       （Lightning module 把模型挂在 ``self.model``）。剥离前缀后返回。
    3. **旧格式**：包含 ``"model_state_dict"`` 键（训练器/早期 checkpoint）。

    参数:
        checkpoint: 已通过 ``safe_torch_load`` 加载到内存的对象。

    返回:
        可直接喂给 ``model.load_state_dict`` 的 ``state_dict``。

    异常:
        TypeError: 既不是 dict，也无法识别其中的 state_dict。
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"无法识别的 checkpoint 类型 {type(checkpoint)!r}：期望 dict（state_dict）。"
        )

    # 裸 state_dict：键值为 Tensor 的字典（无外层结构）
    if not checkpoint or all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    # Lightning .ckpt：包含 state_dict，且键名可能带 model. 前缀
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        raw = checkpoint["state_dict"]
        # Lightning module 把模型挂在 self.model，剥离 "model." 前缀后即为裸权重
        return _strip_lightning_prefix(raw)

    # 旧格式：model_state_dict
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        return checkpoint["model_state_dict"]

    # 兜底：若所有值都是 Tensor，按裸 state_dict 处理
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint

    raise TypeError(
        "无法从 checkpoint 中提取 state_dict：既未找到 'state_dict'/'model_state_dict'，"
        f"也不是裸权重字典。顶层键: {sorted(checkpoint.keys())[:10]}"
    )


def _strip_lightning_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """剥离 Lightning module 的 ``model.`` 前缀。

    仅当 **所有** 键都以 ``model.`` 开头时才剥离，避免误伤本就不带前缀的权重。
    """
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("model.") for k in keys):
        return {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def diagnose_load_result(
    missing_keys: list,
    unexpected_keys: list,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """汇总 ``load_state_dict`` 的 missing/unexpected，给出可读诊断。"""
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "checkpoint_path": checkpoint_path,
    }


def sibling_config_path(checkpoint_path: str | os.PathLike) -> Path:
    """返回与 checkpoint 同目录的 ``.config.yaml`` 路径。

    例如 ``outputs/models/best_model.pt`` -> ``outputs/models/best_model.config.yaml``。
    """
    ckpt_path = Path(checkpoint_path)
    return ckpt_path.with_suffix(".config.yaml")


def load_checkpoint_with_config(
    checkpoint_path: str | os.PathLike,
    config_path: Optional[str | os.PathLike] = None,
    map_location: Any = None,
) -> Tuple[Any, Optional[Config]]:
    """加载 checkpoint 及其匹配的 Config。

    解析顺序：
        1. ``config_path`` 显式传入且文件存在 -> 使用该路径。
        2. checkpoint 同名 ``.config.yaml`` 存在 -> 使用该路径。
        3. 否则返回 ``config=None``，由调用方决定回退策略。

    参数:
        checkpoint_path: model 权重文件路径。
        config_path: 可选的显式配置文件路径，优先级最高。
        map_location: 透传给 :func:`safe_torch_load`。

    返回:
        ``(state_dict, config)`` 元组。``config`` 可能为 ``None``。

    异常:
        FileNotFoundError: checkpoint 文件不存在。
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    resolved_config_path: Optional[Path] = None
    if config_path is not None:
        candidate = Path(config_path)
        if candidate.exists():
            resolved_config_path = candidate
        else:
            logger.warning(
                "显式指定的配置文件不存在: %s，将尝试从 checkpoint 同目录加载",
                config_path,
            )

    if resolved_config_path is None:
        sibling = sibling_config_path(ckpt_path)
        if sibling.exists():
            resolved_config_path = sibling

    config_obj: Optional[Config] = None
    if resolved_config_path is not None:
        try:
            config_obj = Config.from_yaml(str(resolved_config_path))
            logger.info("已加载 checkpoint 配置: %s", resolved_config_path)
        except Exception as exc:  # pragma: no cover - 配置解析错误属于异常路径
            logger.warning("加载配置 %s 失败: %s", resolved_config_path, exc)
            config_obj = None
    else:
        logger.info(
            "未找到 checkpoint 配置 (%s)，将使用默认 config；"
            "若加载失败请确认 checkpoint 与 config 是否匹配",
            sibling_config_path(ckpt_path),
        )

    state_dict = safe_torch_load(str(ckpt_path), map_location=map_location)
    return state_dict, config_obj


def resolve_inference_config(
    checkpoint_path: str | os.PathLike,
    config_path: Optional[str | os.PathLike],
) -> Tuple[Config, str]:
    """解析推理用的 Config，优先使用 checkpoint 配套配置。

    解析顺序（与报告策略一致：sibling config 优先）：
        1. checkpoint 同目录的 ``<name>.config.yaml`` 存在 -> 使用它。
        2. 否则回退到 ``config_path``（调用方显式传入或 argparse 默认值）。

    返回 ``(config, source)``，其中 ``source`` 描述配置来源（路径），
    便于日志输出。
    """
    ckpt_path = Path(checkpoint_path)

    sibling = sibling_config_path(ckpt_path)
    if sibling.exists():
        return Config.from_yaml(str(sibling)), str(sibling)

    if config_path is not None:
        explicit = Path(config_path)
        if explicit.exists():
            return Config.from_yaml(str(explicit)), str(explicit)

    raise FileNotFoundError(
        "无法确定推理配置：既未提供 config，也未在 checkpoint 同目录找到 "
        f"{sibling.name}。请使用训练该 checkpoint 时的 config 文件。"
    )


def diagnose_state_dict_mismatch(
    model_state_keys: Dict[str, Any],
    checkpoint_state_keys: Dict[str, Any],
) -> Dict[str, list]:
    """比较模型与 checkpoint 的 state_dict 键，输出缺失/多余键。"""
    model_keys = set(model_state_keys)
    ckpt_keys = set(checkpoint_state_keys)
    return {
        "missing_keys": sorted(model_keys - ckpt_keys),
        "unexpected_keys": sorted(ckpt_keys - model_keys),
    }
