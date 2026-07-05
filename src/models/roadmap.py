"""Project roadmap helpers and scope registry — compatibility/aggregation layer.

This module provides backward-compatible re-exports from the refactored
:mod:`src.project.scope` and :mod:`src.training.distributed` modules.

Functions that were previously defined here have been moved to their
canonical homes:
  - ``esm3_encoder()``  → :mod:`src.models.pretrained_encoders`
  - ``mass_spec_stream()`` and ``load_custom_ptm_database()``
                        → :mod:`src.data.loaders.legacy_loaders`
  - ``launch_gui()``    → removed (cancelled feature V2-05)
"""

from __future__ import annotations

import logging

from src.project.scope import (
    CancelledFeature,
    DeferredFeature,
    CANCELLED_FEATURES,
    DEFERRED_FEATURES,
    COMPLETED_FEATURES,
    get_cancelled_features,
    get_deferred_features,
)
from src.training.distributed import distributed_trainer

logger = logging.getLogger(__name__)

__all__ = [
    "CancelledFeature",
    "DeferredFeature",
    "CANCELLED_FEATURES",
    "DEFERRED_FEATURES",
    "COMPLETED_FEATURES",
    "get_cancelled_features",
    "get_deferred_features",
    "distributed_trainer",
]
