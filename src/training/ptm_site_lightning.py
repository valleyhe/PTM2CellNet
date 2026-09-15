"""
PTM位点预测Lightning模块
功能: PyTorch Lightning训练封装
"""

from typing import Dict, Any, Optional, cast
import random

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall

from ..models.ptm_site_predictor import PTMSitePredictor, create_model


class PTMSiteLightning(L.LightningModule):
    """
    PTM位点预测Lightning模块
    """

    def __init__(
        self,
        model: Optional[PTMSitePredictor] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化

        参数:
            model: PTMSitePredictor模型实例
            config: 配置字典
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.config = config or {}

        # 可复现性: 从配置中读取 seed 并设置所有 RNG。该模块接受的配置既可能是
        # 扁平结构（config['seed']）也可能是嵌套结构（config['training']['seed']），
        # 兼容两者使不同入口调用均可复现。
        _seed = self.config.get("seed")
        if _seed is None and isinstance(self.config.get("training"), dict):
            _seed = self.config["training"].get("seed")
        if _seed is not None:
            random.seed(_seed)
            np.random.seed(_seed)
            torch.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)
            torch.backends.cudnn.deterministic = True

        # 创建模型
        if model is None:
            self.model = create_model(self.config.get("model", {}))
        else:
            self.model = model

        # 损失函数
        pos_weight = self.config.get("pos_weight", None)
        if pos_weight is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 指标
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.val_precision = Precision(task="binary")
        self.test_precision = Precision(task="binary")

        self.val_recall = Recall(task="binary")
        self.test_recall = Recall(task="binary")

    def forward(self, sequence_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        return cast(Dict[str, torch.Tensor], self.model(sequence_indices))

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """单步训练/验证/测试"""
        sequence_indices = batch["sequence_indices"]
        labels = batch["label"]

        # 前向传播
        outputs = self(sequence_indices)
        logits = outputs["logits"]

        # 计算损失
        loss = self.criterion(logits, labels)

        # 预测
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        # 更新指标
        if stage == "train":
            self.train_acc(preds, labels)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", self.train_acc, prog_bar=True)
        elif stage == "val":
            self.val_acc(preds, labels)
            self.val_auroc(probs, labels)
            self.val_f1(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_auroc", self.val_auroc, prog_bar=True)
            self.log("val_f1", self.val_f1)
            self.log("val_precision", self.val_precision)
            self.log("val_recall", self.val_recall)
        elif stage == "test":
            self.test_acc(preds, labels)
            self.test_auroc(probs, labels)
            self.test_f1(preds, labels)
            self.test_precision(preds, labels)
            self.test_recall(preds, labels)

            self.log("test_loss", loss)
            self.log("test_acc", self.test_acc)
            self.log("test_auroc", self.test_auroc)
            self.log("test_f1", self.test_f1)
            self.log("test_precision", self.test_precision)
            self.log("test_recall", self.test_recall)

        return cast(torch.Tensor, loss)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        """配置优化器"""
        optimizer_name = self.config.get("optimizer", "adamw")
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            momentum = self.config.get("momentum", 0.9)
            nesterov = self.config.get("nesterov", False)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
            )
        else:
            raise ValueError(f"未知优化器: {optimizer_name}")

        # 学习率调度器
        scheduler_name = self.config.get("scheduler", "cosine")
        max_epochs = self.config.get("max_epochs", 100)

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            return optimizer
