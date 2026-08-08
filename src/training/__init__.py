"""训练模块 - 损失函数、优化器、回调和训练器"""

from .losses import FocalLoss, DiceLoss, MultiTaskLoss
from .optimizers import configure_optimizer
from .callbacks import ModelCheckpoint, EarlyStopping, TensorBoardCallback
from .trainers import Trainer, train
from .self_supervised import MaskedPTMPrediction, pretrain_masked_ptm
from .cross_scale_trainer import CrossScaleTrainer, CrossScaleTrainingError
from .logging_config import (
    configure_default_logger,
    build_logger,
    get_default_log_dir,
)

__all__ = [
    "FocalLoss",
    "DiceLoss",
    "MultiTaskLoss",
    "configure_optimizer",
    "ModelCheckpoint",
    "EarlyStopping",
    "TensorBoardCallback",
    "Trainer",
    "train",
    "MaskedPTMPrediction",
    "pretrain_masked_ptm",
    "CrossScaleTrainer",
    "CrossScaleTrainingError",
    # CONF-02: library-level default Lightning logger configuration
    "configure_default_logger",
    "build_logger",
    "get_default_log_dir",
]
