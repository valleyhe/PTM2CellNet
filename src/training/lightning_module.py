"""
Lightning训练模块
功能概述: 封装PTM2CellNet模型，提供PyTorch Lightning训练接口
设计思路: 使用Lightning的标准训练循环，支持多种优化器和调度器

新增功能:
    - Focal Loss支持（用于类别不平衡）
    - LoRA/PEFT集成
    - AUPR指标（用于不平衡评估）
"""

import warnings
from typing import Any, Dict, Optional, cast

import random

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

# 兼容PyTorch 1.x和2.x
try:
    from torch.optim.lr_scheduler import LRScheduler as LRSchedulerClass
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRSchedulerClass

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 尝试导入torchmetrics
try:
    from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        MulticlassAUROC,
        MulticlassAveragePrecision,
        MulticlassF1Score,
    )

    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

from ..training.losses import FocalLoss
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class PTM2CellNetLightning(L.LightningModule):
    """
    Lightning封装的PTM2CellNet模型
    提供训练、验证、测试的标准接口

    新增配置选项:
        - loss_type: "cross_entropy" 或 "focal"
        - use_lora: 是否使用LoRA
        - lora_config: LoRA配置字典
        - class_weights: 类别权重（用于Focal Loss）
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ):
        """
        初始化Lightning模块

        参数:
            model: PTM2CellNet模型实例
            config: 配置字典
        """
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

        # 可复现性: 从配置中读取 seed 并设置所有 RNG，使该 Lightning 模块在被
        # 任意入口调用时也能保证可复现。注意：若已通过 L.seed_everything 全局
        # 设置种子，此处再次设置不会改变确定性。
        seed = config.get("training", {}).get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            logger.info("PTM2CellNetLightning 已从配置设置随机种子: %s", seed)

        # 从配置中提取训练参数
        training_config = config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-4)
        self.weight_decay = training_config.get("weight_decay", 0.01)
        self.optimizer_name = training_config.get("optimizer", "adamw")
        self.scheduler_name = training_config.get("scheduler", None)
        self.max_epochs = training_config.get("max_epochs", 100)

        # 损失函数配置
        self.loss_type = training_config.get("loss_type", "cross_entropy")
        self.use_lora = training_config.get("use_lora", False)
        self.lora_config = training_config.get("lora_config", None)
        self.class_weights = training_config.get("class_weights", None)

        # 初始化损失函数
        self.criterion = self._create_criterion()

        # 应用LoRA（如果需要）
        if self.use_lora:
            self._apply_lora()

        # 初始化指标
        self._setup_metrics()

    def _create_criterion(self) -> Optional[nn.Module]:
        """
        创建损失函数

        返回:
            损失函数模块，如果使用cross_entropy则返回None
        """
        if self.loss_type == "focal":
            # 解析Focal Loss参数
            gamma = self.config.get("training", {}).get("focal_gamma", 2.0)

            # 处理类别权重
            alpha = None
            if self.class_weights is not None:
                if isinstance(self.class_weights, list):
                    alpha = torch.tensor(self.class_weights, dtype=torch.float32)
                elif isinstance(self.class_weights, torch.Tensor):
                    alpha = self.class_weights
                else:
                    logger.warning("不支持的class_weights类型: %s", type(self.class_weights))

            logger.info(f"使用Focal Loss: gamma={gamma}, alpha={alpha.tolist() if alpha is not None else None}")
            return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

        else:  # cross_entropy
            logger.info("使用Cross Entropy Loss")
            return None  # 使用F.cross_entropy

    def _apply_lora(self) -> None:
        """应用LoRA到模型的编码器"""
        try:
            from .peft_config import apply_lora_to_encoder, get_lora_config

            # 获取LoRA配置
            if self.lora_config is None:
                lora_config = get_lora_config()
                logger.info("使用默认LoRA配置")
            else:
                lora_config = get_lora_config(**self.lora_config)
                logger.info("使用自定义LoRA配置: %s", self.lora_config)

            # 应用LoRA到编码器
            if hasattr(self.model, "encoder"):
                self.model.encoder = apply_lora_to_encoder(
                    self.model.encoder,
                    lora_config,
                )
                logger.info("LoRA已应用到模型编码器")
            else:
                # use_lora=True 配置错误: 显式失败，避免静默不应用 LoRA 却误导用户。
                raise AttributeError(
                    "模型缺少 encoder 属性，无法应用 LoRA；"
                    "请确保模型继承自含 encoder 的基类，或在 config 中关闭 use_lora。"
                )

        except ImportError as e:
            logger.error("应用LoRA失败: %s", e)
            raise

    def _setup_metrics(self) -> None:
        """设置评估指标"""
        self.metrics: Dict[str, Any] = {}

        if not TORCHMETRICS_AVAILABLE:
            logger.warning("torchmetrics未安装，跳过指标设置")
            return

        # 从配置获取类别数
        num_classes = self.config.get("model", {}).get("num_classes", 2)

        if num_classes == 2:
            # 二分类指标
            self.metrics["val_auroc"] = BinaryAUROC()
            self.metrics["val_aupr"] = BinaryAveragePrecision()
        else:
            # 多分类指标
            self.metrics["val_auroc"] = MulticlassAUROC(num_classes=num_classes)
            self.metrics["val_aupr"] = MulticlassAveragePrecision(num_classes=num_classes)
            self.metrics["val_f1"] = MulticlassF1Score(num_classes=num_classes, average="macro")

        logger.info("已设置验证指标: %s", list(self.metrics.keys()))

    @staticmethod
    def compute_class_weights(
        labels: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        从标签计算类别权重

        参数:
            labels: 标签张量
            num_classes: 类别数（可选）

        返回:
            类别权重张量
        """
        if num_classes is None:
            num_classes = int(labels.max().item()) + 1

        # 计算每个类别的样本数
        class_counts = torch.bincount(labels, minlength=num_classes).float()

        # 使用逆频率计算权重
        total_samples = len(labels)
        weights = total_samples / (num_classes * class_counts + 1e-8)

        # 归一化使平均权重为1
        weights = weights / weights.mean()

        return cast(torch.Tensor, weights)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            batch: 输入批次数据

        返回:
            模型输出字典
        """
        return cast(Dict[str, torch.Tensor], self.model(batch))

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤

        参数:
            batch: 输入批次数据
            batch_idx: 批次索引

        返回:
            训练损失
        """
        # 前向传播
        outputs = self.model(batch)

        # 计算损失
        if self.criterion is not None:
            # 使用自定义损失函数（如Focal Loss）
            loss = self.criterion(outputs["logits"], batch["label"])
        else:
            # 使用标准交叉熵
            loss = F.cross_entropy(outputs["logits"], batch["label"])

        # 计算准确率
        predictions = outputs["predictions"]
        accuracy = (predictions == batch["label"]).float().mean()

        # 记录指标
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_acc",
            accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return cast(torch.Tensor, loss)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        验证步骤

        参数:
            batch: 输入批次数据
            batch_idx: 批次索引

        返回:
            验证损失
        """
        # 前向传播
        outputs = self.model(batch)

        # 计算损失
        if self.criterion is not None:
            loss = self.criterion(outputs["logits"], batch["label"])
        else:
            loss = F.cross_entropy(outputs["logits"], batch["label"])

        # 计算准确率
        predictions = outputs["predictions"]
        accuracy = (predictions == batch["label"]).float().mean()

        # 记录指标
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_acc",
            accuracy,
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

        # 更新torchmetrics指标
        if TORCHMETRICS_AVAILABLE and self.metrics:
            # Ensure metrics live on the same device as the batch
            device = batch["label"].device
            for metric in self.metrics.values():
                metric.to(device)

            probs = F.softmax(outputs["logits"], dim=1)
            num_classes = outputs["logits"].shape[1]

            if num_classes == 2:
                # 二分类：使用正类概率
                self.metrics["val_auroc"].update(probs[:, 1], batch["label"])
                self.metrics["val_aupr"].update(probs[:, 1], batch["label"])
            else:
                # 多分类
                self.metrics["val_auroc"].update(probs, batch["label"])
                self.metrics["val_aupr"].update(probs, batch["label"])
                if "val_f1" in self.metrics:
                    self.metrics["val_f1"].update(predictions, batch["label"])

        return cast(torch.Tensor, loss)

    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时计算并记录指标"""
        if not TORCHMETRICS_AVAILABLE or not self.metrics:
            return

        # 计算并记录指标
        for name, metric in self.metrics.items():
            value = metric.compute()
            self.log(name, value, prog_bar=True, on_epoch=True)
            metric.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        测试步骤

        参数:
            batch: 输入批次数据
            batch_idx: 批次索引

        返回:
            测试损失
        """
        # 前向传播
        outputs = self.model(batch)

        # 计算损失
        if self.criterion is not None:
            loss = self.criterion(outputs["logits"], batch["label"])
        else:
            loss = F.cross_entropy(outputs["logits"], batch["label"])

        # 计算准确率
        predictions = outputs["predictions"]
        accuracy = (predictions == batch["label"]).float().mean()

        # 记录指标
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_acc", accuracy, on_epoch=True, logger=True)

        return cast(torch.Tensor, loss)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        返回:
            优化器或(优化器列表, 调度器配置列表)
        """
        # 创建优化器
        optimizer = self._create_optimizer()

        # 创建学习率调度器
        scheduler = self._create_scheduler(optimizer)

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def _create_optimizer(self) -> Optimizer:
        """
        创建优化器

        返回:
            优化器实例
        """
        optimizer: Optimizer
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "lion":
            try:
                from lion_pytorch import Lion

                optimizer = Lion(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            except ImportError:
                logger.warning(
                    "lion_pytorch 未安装，回退到 AdamW 优化器。"
                    "如需使用 Lion，请安装 mamba extra：pip install -e '.[mamba]'"
                )
                warnings.warn(
                    "lion_pytorch not installed, falling back to AdamW optimizer. "
                    "Install with: pip install lion-pytorch",
                    UserWarning,
                    stacklevel=2,
                )
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            momentum = self.config.get("training", {}).get("momentum", 0.9)
            nesterov = self.config.get("training", {}).get("nesterov", False)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=momentum,
                nesterov=nesterov,
            )
        else:
            logger.warning(f"未知的优化器类型: {self.optimizer_name}，使用默认AdamW")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        logger.info("创建优化器: %s, 学习率: %s", self.optimizer_name, self.learning_rate)
        return optimizer

    @staticmethod
    def configure_trainer(**trainer_kwargs: Any) -> L.Trainer:
        """Create a Lightning Trainer with default TensorBoard logger.

        Args:
            **trainer_kwargs: Keyword arguments forwarded to ``L.Trainer``.

        Returns:
            A configured ``L.Trainer`` instance.
        """
        # Default logger configuration to avoid "no logger" warning
        if "logger" not in trainer_kwargs:
            try:
                from lightning.pytorch.loggers import TensorBoardLogger

                trainer_kwargs["logger"] = TensorBoardLogger(
                    save_dir=trainer_kwargs.get("default_root_dir", "outputs/logs"),
                    name="lightning_logs",
                )
            except ImportError:
                pass
        return L.Trainer(**trainer_kwargs)

    def _create_scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerClass]:
        """
        创建学习率调度器

        参数:
            optimizer: 优化器实例

        返回:
            学习率调度器实例，或None
        """
        if self.scheduler_name is None or self.scheduler_name == "":
            return None

        if self.scheduler_name == "cosine":
            scheduler: LRSchedulerClass = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
        elif self.scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        elif self.scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            logger.warning(f"未知的调度器类型: {self.scheduler_name}，不使用调度器")
            return None

        logger.info("创建学习率调度器: %s", self.scheduler_name)
        return scheduler
