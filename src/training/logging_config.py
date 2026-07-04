"""
Lightning 默认 Logger 配置 (CONF-02).

Previously each entry-point script had to manually construct a
``lightning.pytorch.loggers.TensorBoardLogger`` and pass it to ``L.Trainer``;
if it forgot, Lightning silently auto-created a ``lightning_logs/`` CSVLogger
and emitted a "no logger configured" warning.

This module provides a single library-level factory so entry points can do::

    from src.training.logging_config import configure_default_logger, build_logger

    logger = configure_default_logger(save_dir="outputs/logs", name="train")
    trainer = L.Trainer(logger=logger, ...)

This guarantees a consistent default logger everywhere and suppresses the
"no logger configured" warning.

Design:
    * Lightning is imported lazily so this module imports cleanly even when
      Lightning is unavailable (matching the optional-dependency pattern).
    * ``configure_default_logger`` returns a ready-to-use logger and falls back
      to ``CSVLogger`` if TensorBoard is unavailable.
    * ``get_default_log_dir`` centralizes the default log directory resolution
      (honors the ``PTM2CELLNET_LOG_DIR`` env var).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Environment variable overriding the default Lightning log directory.
LOG_DIR_ENV = "PTM2CELLNET_LOG_DIR"

#: Default relative log directory (used when the env var is unset).
DEFAULT_LOG_DIR = "outputs/logs"


def get_default_log_dir() -> Path:
    """Resolve the default Lightning log directory.

    Honors the ``PTM2CELLNET_LOG_DIR`` environment variable; falls back to
    ``outputs/logs`` relative to the current working directory.
    """
    env_dir = os.environ.get(LOG_DIR_ENV)
    if env_dir:
        return Path(env_dir)
    return Path(DEFAULT_LOG_DIR)


def _import_lightning():
    """Lazily import the lightning loggers module."""
    try:
        import lightning as L
        from lightning.pytorch import loggers as L_loggers
        return L, L_loggers
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Lightning is required for logger configuration but is not available: "
            f"{e}"
        ) from e


def build_logger(
    kind: str = "tensorboard",
    save_dir: Optional[str] = None,
    name: str = "lightning_logs",
    version: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Construct a single Lightning logger instance.

    Args:
        kind: Logger type — ``"tensorboard"`` (default), ``"csv"``, or ``"wandb"``.
        save_dir: Directory to write logs to (default: :func:`get_default_log_dir`).
        name: Experiment name.
        version: Version tag (``None`` = Lightning auto-increments).
        **kwargs: Passed through to the underlying logger constructor.

    Returns:
        A configured Lightning logger instance.

    Raises:
        ImportError: If Lightning (or the requested logger backend) is missing.
        ValueError: If ``kind`` is not one of the supported types.
    """
    _L, L_loggers = _import_lightning()

    resolved_save_dir = str(save_dir) if save_dir else str(get_default_log_dir())
    Path(resolved_save_dir).mkdir(parents=True, exist_ok=True)

    kind_lower = kind.lower()
    if kind_lower == "tensorboard":
        if not hasattr(L_loggers, "TensorBoardLogger"):
            logger.info(
                "TensorBoardLogger unavailable; falling back to CSVLogger."
            )
            return L_loggers.CSVLogger(
                save_dir=resolved_save_dir, name=name, version=version, **kwargs
            )
        try:
            return L_loggers.TensorBoardLogger(
                save_dir=resolved_save_dir, name=name, version=version, **kwargs
            )
        except (ModuleNotFoundError, ImportError) as e:
            # tensorboard / tensorboardX optional dependency missing.
            logger.info(
                "TensorBoardLogger construction failed (%s); falling back to "
                "CSVLogger.", e,
            )
            return L_loggers.CSVLogger(
                save_dir=resolved_save_dir, name=name, version=version, **kwargs
            )
    if kind_lower == "csv":
        return L_loggers.CSVLogger(
            save_dir=resolved_save_dir, name=name, version=version, **kwargs
        )
    if kind_lower == "wandb":
        if not hasattr(L_loggers, "WandbLogger"):
            raise ImportError(
                "WandbLogger requested but not available; install 'wandb'."
            )
        return L_loggers.WandbLogger(
            save_dir=resolved_save_dir, name=name, version=version, **kwargs
        )
    raise ValueError(
        f"Unsupported logger kind '{kind}'. Use 'tensorboard', 'csv', or 'wandb'."
    )


def configure_default_logger(
    save_dir: Optional[str] = None,
    name: str = "ptm2cellnet",
    version: Optional[Any] = None,
    kind: str = "tensorboard",
    **kwargs: Any,
) -> Any:
    """Return a ready-to-use default Lightning logger (CONF-02).

    This is the recommended entry point for training scripts::

        from src.training.logging_config import configure_default_logger
        import lightning as L

        logger = configure_default_logger(name="train")
        trainer = L.Trainer(logger=logger, default_root_dir=str(logger.save_dir))

    Using this factory guarantees a consistent default logger and suppresses
    Lightning's "no logger configured" warning.

    Args:
        save_dir: Override log directory (default: env-configurable).
        name: Experiment name.
        version: Version tag.
        kind: ``"tensorboard"`` | ``"csv"`` | ``"wandb"``.
        **kwargs: Passed to the underlying logger.

    Returns:
        A configured Lightning logger instance.
    """
    log_dir = str(save_dir) if save_dir else str(get_default_log_dir())
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    lg = build_logger(
        kind=kind, save_dir=log_dir, name=name, version=version, **kwargs
    )
    logger.info(
        "Configured default Lightning logger: kind=%s save_dir=%s name=%s",
        kind, log_dir, name,
    )
    return lg


__all__ = [
    "configure_default_logger",
    "build_logger",
    "get_default_log_dir",
    "LOG_DIR_ENV",
    "DEFAULT_LOG_DIR",
]
