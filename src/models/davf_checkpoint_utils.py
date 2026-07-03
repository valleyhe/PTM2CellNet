"""Unified checkpoint save/load/resume utilities.

Provides save_checkpoint, load_checkpoint, and resume_training_state
to standardize checkpoint recovery across training and inference.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn

from src.utils.io import safe_torch_load

logger = logging.getLogger(__name__)


def _load_checkpoint_payload(ckpt_path: Path, device: str) -> Dict[str, Any]:
    """Load a checkpoint, falling back for trusted legacy DAVF files when needed."""
    return cast(Dict[str, Any], safe_torch_load(ckpt_path, map_location=device))


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **extras: Any,
) -> None:
    """Save a unified checkpoint dictionary.

    Args:
        path: Destination file path.
        model: Model to save state_dict from.
        optimizer: Optional optimizer to save state_dict from.
        scheduler: Optional scheduler to save state_dict from.
        epoch: Optional training epoch.
        config: Optional configuration dictionary.
        **extras: Additional keys to include in the checkpoint.
    """
    ckpt: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config if config is not None else {},
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    ckpt.update(extras)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)


def load_checkpoint(
    path: str,
    device: str = "cpu",
    strict: bool = True,
    allow_unsafe_legacy: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint dictionary from disk.

    Args:
        path: Checkpoint file path.
        device: Device to map tensors to.
        strict: If True, require 'config' and 'model_state_dict' keys.
        allow_unsafe_legacy: If True, allow pickle-based fallback for trusted legacy checkpoints.

    Returns:
        Loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If strict=True and required keys are missing.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = _load_checkpoint_payload(ckpt_path, device)
    except pickle.UnpicklingError:
        if not allow_unsafe_legacy:
            raise
        logger.warning(
            "⚠️ SECURITY RISK: weights_only=True failed for %s; "
            "retrying with weights_only=False. This allows arbitrary code "
            "execution via pickle deserialization — only use with trusted "
            "checkpoints (e.g. self-produced files). "
            "Consider using allowed_classes= to whitelist required types instead.",
            ckpt_path,
        )
        ckpt = safe_torch_load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary")

    if "config" not in ckpt:
        ckpt["config"] = {}

    required_keys = {"config", "model_state_dict"}
    if strict:
        missing = required_keys - set(ckpt.keys())
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

    return ckpt


def resume_training_state(
    checkpoint_dict: Dict[str, Any],
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> int:
    """Resume model/optimizer/scheduler state from a checkpoint dictionary.

    Args:
        checkpoint_dict: Loaded checkpoint dictionary.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.

    Returns:
        The epoch to resume from (defaults to 0 if not present).
    """
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

    epoch = checkpoint_dict.get("epoch", 0)
    config = checkpoint_dict.get("config", {})
    config_hash = ""
    if config:
        try:
            config_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:8]
        except Exception:
            logger.warning("Failed to compute config hash while resuming training state")

    logger.info(
        f"Resumed training state from epoch {epoch}. "
        f"Config hash: {config_hash or 'N/A'}"
    )
    return cast(int, epoch)
