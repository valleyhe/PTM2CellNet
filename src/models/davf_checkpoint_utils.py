"""Unified checkpoint save/load/resume utilities.

Provides save_checkpoint, load_checkpoint, and resume_training_state
to standardize checkpoint recovery across training and inference.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Union, cast, runtime_checkable

import torch
import torch.nn as nn
from typing_extensions import TypedDict

from src.utils.io import safe_torch_load

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol for objects with state_dict (optimizers, schedulers, etc.)
# ---------------------------------------------------------------------------

@runtime_checkable
class _Stateful(Protocol):
    """Protocol for objects that support state_dict / load_state_dict."""

    # state_dict returns heterogeneous tensor maps; Any is the most specific
    # type we can give without knowing the concrete optimizer/scheduler.
    def state_dict(self) -> Dict[str, Any]: ...  # noqa: TY102
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...  # noqa: TY102


# ---------------------------------------------------------------------------
# Typed structures for checkpoint dictionaries
# ---------------------------------------------------------------------------

# Type alias for the heterogeneous config dict stored in checkpoints.
# This is genuinely dynamic (YAML-loaded), so Dict[str, Any] is correct here.
_CheckpointConfig = Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]


class CheckpointMetadata(TypedDict, total=False):
    """TypedDict for checkpoint payload dictionaries.

    All fields are optional because checkpoints may contain different
    subsets depending on the training stage.
    """
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    epoch: int
    config: _CheckpointConfig  # noqa: TY102


# DAVF config/architecture classes embedded in legacy fine-tuned checkpoints.
# These are safe types that cannot be loaded with vanilla weights_only=True
# because they are custom Python objects (dataclasses / nn.Module subclasses).
# We import them lazily to avoid circular-dependency issues and only when
# actually loading a DAVF checkpoint.
_DAVF_ALLOWED_CLASSES: Optional[Set[Union[str, type]]] = None


def _get_davf_allowed_classes() -> Set[Union[str, type]]:
    global _DAVF_ALLOWED_CLASSES
    if _DAVF_ALLOWED_CLASSES is None:
        from src.models.latent_davf import LatentDAVF, LatentDAVFConfig

        _DAVF_ALLOWED_CLASSES = {
            LatentDAVFConfig,
            LatentDAVF,
        }
    return _DAVF_ALLOWED_CLASSES


def _load_checkpoint_payload(ckpt_path: Path, device: str) -> CheckpointMetadata:
    """Load a checkpoint with allowlisted DAVF types for weights_only=True."""
    return cast(
        CheckpointMetadata,
        safe_torch_load(
            ckpt_path,
            map_location=device,
            allowed_classes=_get_davf_allowed_classes(),
        ),
    )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[_Stateful] = None,
    scheduler: Optional[_Stateful] = None,
    epoch: Optional[int] = None,
    config: Optional[_CheckpointConfig] = None,
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
        "schema_version": 1,
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
    torch.save(cast(CheckpointMetadata, ckpt), out_path)


def load_checkpoint(
    path: str,
    device: str = "cpu",
    strict: bool = True,
) -> CheckpointMetadata:
    """Load a checkpoint dictionary from disk.

    Args:
        path: Checkpoint file path.
        device: Device to map tensors to.
        strict: If True, require 'config' and 'model_state_dict' keys.

    Returns:
        Loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If strict=True and required keys are missing.
        RuntimeError: If the checkpoint cannot be loaded with
            ``weights_only=True``.  Use ``scripts/tools/migrate_legacy_checkpoint.py``
            to convert legacy checkpoints.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = _load_checkpoint_payload(ckpt_path, device)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            f"Legacy checkpoint {ckpt_path} cannot be loaded with "
            f"weights_only=True. Run: python scripts/tools/"
            f"migrate_legacy_checkpoint.py --input {ckpt_path} "
            f"--output {ckpt_path}.safe to convert it."
        ) from exc
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary")

    schema_version = ckpt.get("schema_version")
    if schema_version is not None and schema_version != 1:
        raise ValueError(
            f"Unsupported DAVF checkpoint schema_version={schema_version!r} "
            f"(supported: 1). Re-export or retrain the checkpoint."
        )

    if "config" not in ckpt:
        ckpt["config"] = {}

    required_keys = {"config", "model_state_dict"}
    if strict:
        missing = required_keys - set(ckpt.keys())
        if missing:
            raise ValueError(f"Checkpoint missing required keys: {missing}")

    return ckpt


def resume_training_state(
    checkpoint_dict: CheckpointMetadata,
    model: nn.Module,
    optimizer: Optional[_Stateful] = None,
    scheduler: Optional[_Stateful] = None,
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
        except (TypeError, ValueError) as e:
            logger.warning("Failed to compute config hash while resuming training state: %s", e)

    logger.info(
        f"Resumed training state from epoch {epoch}. "
        f"Config hash: {config_hash or 'N/A'}"
    )
    return cast(int, epoch)
