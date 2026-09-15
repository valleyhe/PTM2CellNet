"""Distributed training helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

from .trainers import Trainer

logger = logging.getLogger(__name__)

_LIGHTNING_MODULE: Any = None
_LIGHTNING_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - optional dependency
    import lightning as _LIGHTNING_MODULE  # noqa: F811

    _LIGHTNING_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional dependency
    _LIGHTNING_MODULE = None
    _LIGHTNING_IMPORT_ERROR = exc


def distributed_trainer(
    model: Any,
    datamodule: Any = None,
    *,
    strict: bool = False,
    **kwargs: Any,
):
    """V2-04: build a Lightning DDP trainer when possible, else fall back safely.

    Args:
        model: The model to train.
        datamodule: Optional Lightning DataModule or compatible object.
        strict: If True, raise RuntimeError when distributed training is
            requested but unavailable (no Lightning or insufficient GPUs).
            May also be enabled globally via the
            ``PTM2CELLNET_STRICT_MODEL_ASSETS`` environment variable.
        **kwargs: Additional arguments forwarded to the Trainer constructor.

    Returns:
        A Lightning Trainer or a plain :class:`Trainer` instance.

    Raises:
        RuntimeError: If *strict* mode is enabled and the required
            infrastructure (Lightning + multi-GPU) is not available.
    """
    # Also check global strict mode env var
    _global_strict = os.environ.get("PTM2CELLNET_STRICT_MODEL_ASSETS", "").lower() in ("1", "true", "yes")
    if _global_strict:
        strict = True

    if _LIGHTNING_MODULE is not None:
        trainer_kwargs = dict(kwargs)
        devices = int(trainer_kwargs.pop("devices", 1))
        if datamodule is not None:
            trainer_kwargs["datamodule"] = datamodule
        if torch.cuda.is_available():
            available_devices = torch.cuda.device_count()
            if available_devices > 1:
                trainer_kwargs["accelerator"] = trainer_kwargs.get("accelerator", "gpu")
                trainer_kwargs["devices"] = max(devices, available_devices)
                trainer_kwargs["strategy"] = "ddp"
            elif devices > 1:
                if strict:
                    raise RuntimeError(
                        f"Requested devices={devices} but only {available_devices} GPU available. "
                        "Distributed training requires multiple GPUs. Set strict=False to allow "
                        "single-GPU fallback."
                    )
                logger.warning(
                    "Requested devices=%s but only %s GPU is available; using a normal Lightning Trainer.",
                    devices,
                    available_devices,
                )
                trainer_kwargs["devices"] = 1
        trainer = _LIGHTNING_MODULE.Trainer(**trainer_kwargs)
        return trainer

    if strict:
        raise RuntimeError(
            f"lightning is unavailable ({_LIGHTNING_IMPORT_ERROR}) and strict=True forbids "
            "fallback to plain Trainer. Install lightning or set strict=False."
        )

    logger.warning(
        "lightning is unavailable (%s); falling back to plain Trainer.",
        _LIGHTNING_IMPORT_ERROR,
    )
    return Trainer(
        model,
        # Note: datamodule is not passed to Trainer.__init__ — it should be
        # provided to Trainer.fit(datamodule=datamodule) by the caller.
        config=kwargs.get("config"),
        device=kwargs.get("device"),
    )


__all__ = [
    "distributed_trainer",
]
