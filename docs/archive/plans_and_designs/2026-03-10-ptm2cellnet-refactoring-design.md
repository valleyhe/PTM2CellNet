# PTM2CellNet 重构设计文档

**文档版本：** v1.0
**创建日期：** 2026-03-10
**目标范围：** 中期目标（6-12个月）
**重构策略：** 渐进式重构

---

## 1. 重构目标

### 1.1 核心目标

基于《PTM2CellNet-深度技术分析文档》的中期目标，本次重构旨在：

1. **训练框架现代化**：迁移到PyTorch Lightning，支持分布式训练和混合精度
2. **预训练模型集成**：集成ESM-2等大规模预训练模型，提升模型性能
3. **模型容量扩展**：从10M参数扩展到420M+参数
4. **保持向后兼容**：现有功能和脚本继续可用

### 1.2 非目标

- 不实现长期目标中的动态轨迹评估系统
- 不进行大规模预训练（使用现成的预训练模型）
- 不删除现有Trainer实现

---

## 2. 整体架构设计

### 2.1 目录结构演进

```
PTM2CellNet/
├── src/
│   ├── data/                          # 数据模块（保持现有）
│   │   ├── loaders.py                 # 现有
│   │   ├── preprocess.py              # 现有
│   │   ├── features.py                # 现有
│   │   ├── datasets.py                # 现有
│   │   └── lightning_datamodule.py    # 新增：Lightning数据模块
│   │
│   ├── models/                        # 模型模块（扩展）
│   │   ├── encoders.py                # 现有：CNN/Transformer/LSTM
│   │   ├── pretrained_encoders.py     # 新增：ESM-2等预训练模型
│   │   ├── ptm_modules.py             # 现有
│   │   ├── predictors.py              # 现有
│   │   └── architectures.py           # 现有（需小幅修改）
│   │
│   ├── training/                      # 训练模块（新旧并存）
│   │   ├── trainers.py                # 现有：保留
│   │   ├── lightning_module.py        # 新增：Lightning封装
│   │   ├── callbacks.py               # 现有（兼容Lightning）
│   │   ├── optimizers.py              # 现有
│   │   └── losses.py                  # 现有
│   │
│   ├── evaluation/                    # 评估模块（保持现有）
│   │   ├── evaluators.py              # 现有
│   │   ├── metrics.py                 # 现有
│   │   └── visualization.py           # 现有
│   │
│   └── utils/                         # 工具模块（保持现有）
│       ├── config.py                  # 现有
│       ├── logging.py                 # 现有
│       └── helpers.py                 # 现有
│
├── scripts/                           # 脚本（新旧并存）
│   ├── train.py                       # 现有：使用旧Trainer
│   ├── train_lightning.py             # 新增：使用Lightning
│   ├── train_pretrained.py            # 新增：预训练模型专用
│   ├── evaluate.py                    # 现有
│   └── predict.py                     # 现有
│
├── configs/                           # 配置文件（扩展）
│   ├── default.yaml                   # 现有
│   ├── lightning.yaml                 # 新增：Lightning配置
│   └── pretrained/                    # 新增：预训练模型配置
│       ├── esm2_150m.yaml
│       ├── esm2_650m.yaml
│       └── protbert.yaml
│
└── tests/                             # 测试（扩展）
    ├── unit/                          # 现有单元测试
    │   ├── test_models.py             # 扩展：测试新编码器
    │   ├── test_data.py               # 现有
    │   └── test_lightning.py          # 新增：Lightning模块测试
    └── integration/                   # 现有集成测试
        └── test_training_pipeline.py  # 扩展：测试Lightning训练
```

### 2.2 核心设计原则

1. **向后兼容**：现有代码和脚本继续工作，不破坏现有功能
2. **配置驱动**：通过配置文件选择使用旧Trainer还是Lightning
3. **接口统一**：新旧编码器实现相同接口，可互换
4. **渐进迁移**：先添加新功能，验证后再考虑移除旧代码

### 2.3 数据流对比

**现有流程（保留）：**
```
DataLoader → DataPreprocessor → FeatureExtractor → PTMDataset →
自定义Trainer → 训练循环
```

**新增流程（并行）：**
```
DataLoader → DataPreprocessor → FeatureExtractor → PTMDataset →
LightningDataModule → PTM2CellNetLightning → Lightning Trainer → 训练循环
```

---

## 3. 模块详细设计

### 3.1 数据模块扩展

#### 3.1.1 LightningDataModule实现

**文件：** `src/data/lightning_datamodule.py`

**职责：**
- 封装现有的PTMDataset，提供Lightning兼容接口
- 管理训练/验证/测试数据加载器
- 支持分布式训练的数据分片

**接口设计：**
```python
class PTMDataModule(pl.LightningDataModule):
    """Lightning数据模块，封装现有PTMDataset"""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Dict[str, Any],
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = config
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def setup(self, stage: Optional[str] = None):
        """准备数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = PTMDataset(
                self.train_df, self.feature_extractor, self.config
            )
            self.val_dataset = PTMDataset(
                self.val_df, self.feature_extractor, self.config
            )
        if stage == "test" or stage is None:
            self.test_dataset = PTMDataset(
                self.test_df, self.feature_extractor, self.config
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get("training.batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("training.num_workers", 4),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get("training.batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("training.num_workers", 4),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.get("training.batch_size", 32),
            shuffle=False,
            num_workers=self.config.get("training.num_workers", 4),
            pin_memory=True,
        )
```

**关键决策：**
- 复用现有PTMDataset，避免重复实现
- 支持多进程数据加载（num_workers）
- 使用pin_memory加速GPU传输

---

### 3.2 模型模块扩展

#### 3.2.1 预训练编码器实现

**文件：** `src/models/pretrained_encoders.py`

**职责：**
- 封装ESM-2、ProtBERT等预训练模型
- 提供与现有编码器统一的接口
- 支持冻结/微调策略

**接口设计：**
```python
from transformers import AutoModel, AutoConfig
import torch.nn as nn

class PretrainedEncoder(nn.Module):
    """预训练模型编码器基类"""

    def __init__(
        self,
        model_name: str,
        freeze: bool = False,
        use_attention_output: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        self.use_attention_output = use_attention_output

        # 获取隐藏层维度
        self.hidden_dim = self.config.hidden_size

        # 冻结参数
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            input_ids: [batch_size, seq_len] 氨基酸序列索引

        返回:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        outputs = self.model(input_ids, output_attentions=self.use_attention_output)

        # 返回最后一层隐藏状态
        return outputs.last_hidden_state


class ESM2Encoder(PretrainedEncoder):
    """ESM-2编码器"""

    def __init__(
        self,
        model_size: str = "150M",  # 150M, 650M, 2B, etc.
        freeze: bool = False,
    ):
        # ESM-2模型名称映射
        model_names = {
            "150M": "facebook/esm2_t30_150M_UR50D",
            "650M": "facebook/esm2_t33_650M_UR50D",
            "2B": "facebook/esm2_t36_3B_UR50D",
        }

        model_name = model_names.get(model_size, model_names["150M"])
        super().__init__(model_name, freeze=freeze)


class ProtBERTEncoder(PretrainedEncoder):
    """ProtBERT编码器"""

    def __init__(self, freeze: bool = False):
        super().__init__("Rostlab/prot_bert", freeze=freeze)
```

**关键决策：**
- 使用HuggingFace Transformers库加载预训练模型
- 支持冻结策略，减少训练成本
- 统一接口，与现有编码器兼容

#### 3.2.2 模型架构修改

**文件：** `src/models/architectures.py`

**修改内容：**
```python
class PTM2CellNet(nn.Module):
    """PTM2CellNet端到端模型"""

    def __init__(
        self,
        encoder_type: str = "transformer",  # 新增：esm2_150M, esm2_650M, protbert
        vocab_size: int = 20,
        embed_dim: int = 128,
        max_seq_len: int = 1000,
        num_ptm_types: int = 10,
        num_classes: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = False,  # 新增：是否冻结编码器
    ):
        super().__init__()
        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        encoder: nn.Module
        if encoder_type == "cnn":
            encoder = CNNEncoder(vocab_size, embed_dim, max_len=max_seq_len, dropout=dropout)
        elif encoder_type == "transformer":
            encoder = TransformerEncoder(
                vocab_size, embed_dim, max_len=max_seq_len,
                num_layers=num_layers, num_heads=num_heads, dropout=dropout
            )
        elif encoder_type == "lstm":
            encoder = LSTMEncoder(
                vocab_size, embed_dim, max_len=max_seq_len,
                hidden_dim=embed_dim * 2, num_layers=num_layers, dropout=dropout
            )
        # 新增：预训练编码器
        elif encoder_type.startswith("esm2"):
            model_size = encoder_type.split("_")[1] if "_" in encoder_type else "150M"
            encoder = ESM2Encoder(model_size=model_size, freeze=freeze_encoder)
            self.embed_dim = encoder.hidden_dim  # 更新embed_dim
        elif encoder_type == "protbert":
            encoder = ProtBERTEncoder(freeze=freeze_encoder)
            self.embed_dim = encoder.hidden_dim
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")

        self.encoder = encoder

        # PTM模块和预测器保持不变
        self.ptm_module = PTMModule(
            num_ptm_types=num_ptm_types,
            embed_dim=self.embed_dim,
            max_position=max_seq_len,
            num_attention_heads=max(1, num_heads),
            num_layers=max(1, num_layers),
            dropout=dropout,
        )
        self.predictor = ClassificationPredictor(
            input_dim=self.embed_dim,
            num_classes=num_classes,
            hidden_dims=[self.embed_dim * 2, self.embed_dim],
            dropout=dropout,
        )
```

**关键决策：**
- 通过encoder_type参数区分不同编码器
- 自动适配embed_dim以匹配预训练模型
- 保持PTM模块和预测器不变

---

### 3.3 训练模块扩展

#### 3.3.1 Lightning模块实现

**文件：** `src/training/lightning_module.py`

**职责：**
- 封装PTM2CellNet模型，提供Lightning训练接口
- 管理训练/验证/测试步骤
- 配置优化器和学习率调度器

**接口设计：**
```python
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class PTM2CellNetLightning(pl.LightningModule):
    """Lightning封装的PTM2CellNet模型"""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        outputs = self.model(batch)
        loss = F.cross_entropy(outputs["logits"], batch["label"])

        # 计算准确率
        acc = (outputs["predictions"] == batch["label"]).float().mean()

        # 记录指标
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        outputs = self.model(batch)
        loss = F.cross_entropy(outputs["logits"], batch["label"])
        acc = (outputs["predictions"] == batch["label"]).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """测试步骤"""
        outputs = self.model(batch)
        loss = F.cross_entropy(outputs["logits"], batch["label"])
        acc = (outputs["predictions"] == batch["label"]).float().mean()

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        training_config = self.config.get("training", {})

        # 优化器选择
        opt_name = training_config.get("optimizer", "adamw")
        lr = training_config.get("learning_rate", 1e-4)
        weight_decay = training_config.get("weight_decay", 0.01)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif opt_name == "lion":
            from lion_pytorch import Lion
            optimizer = Lion(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 学习率调度器
        scheduler_name = training_config.get("scheduler", "cosine")
        max_epochs = training_config.get("max_epochs", 100)

        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        elif scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        else:
            scheduler = None

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        return optimizer
```

**关键决策：**
- 使用Lightning的标准训练循环
- 支持多种优化器（AdamW、Lion）
- 支持学习率调度器（Cosine、ReduceLROnPlateau）
- 自动记录训练指标

---

### 3.4 配置系统扩展

#### 3.4.1 Lightning配置文件

**文件：** `configs/lightning.yaml`

```yaml
# Lightning训练配置

model:
  encoder_type: "esm2_150M"  # 可选：cnn, transformer, lstm, esm2_150M, esm2_650M, protbert
  freeze_encoder: false      # 是否冻结预训练编码器
  num_classes: 4
  dropout: 0.1

training:
  # 基础训练参数
  max_epochs: 100
  batch_size: 16            # 预训练模型需要较小batch size
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer: "adamw"        # 可选：adamw, lion, adam
  scheduler: "cosine"       # 可选：cosine, plateau

  # Lightning特定参数
  accelerator: "gpu"        # 可选：gpu, cpu, tpu
  devices: 1                # GPU数量
  strategy: "auto"          # 可选：auto, ddp, fsdp, deepspeed
  precision: "16-mixed"     # 可选：32, 16-mixed, bf16-mixed
  accumulate_grad_batches: 4  # 梯度累积
  gradient_clip_val: 1.0    # 梯度裁剪

  # 回调配置
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_loss"
    mode: "min"

  checkpoint:
    save_top_k: 3
    monitor: "val_loss"
    mode: "min"

  # 日志配置
  logger:
    type: "tensorboard"     # 可选：tensorboard, wandb, mlflow
    save_dir: "outputs/logs"

data:
  max_sequence_length: 1000
  num_workers: 4
  pin_memory: true
```

#### 3.4.2 预训练模型配置示例

**文件：** `configs/pretrained/esm2_650m.yaml`

```yaml
# ESM-2 650M模型配置

model:
  encoder_type: "esm2_650M"
  freeze_encoder: false
  num_classes: 4
  dropout: 0.1

training:
  max_epochs: 50
  batch_size: 8            # 650M模型需要更小batch size
  learning_rate: 5e-5      # 预训练模型使用更小学习率
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"

  accelerator: "gpu"
  devices: 1
  precision: "bf16-mixed"  # 使用BF16混合精度
  accumulate_grad_batches: 8  # 增加梯度累积

  early_stopping:
    enabled: true
    patience: 5
    monitor: "val_loss"

data:
  max_sequence_length: 1024  # ESM-2支持更长序列
  num_workers: 4
```

---

### 3.5 训练脚本实现

#### 3.5.1 Lightning训练脚本

**文件：** `scripts/train_lightning.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightning训练脚本
使用PyTorch Lightning进行模型训练
"""

import sys
from pathlib import Path

# 确保项目根目录在sys.path中
def _ensure_project_root():
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

_ensure_project_root()

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.utils.config import Config
from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.data.lightning_datamodule import PTMDataModule
from src.models.architectures import PTM2CellNet
from src.training.lightning_module import PTM2CellNetLightning


def parse_args():
    parser = argparse.ArgumentParser(description="PTM2CellNet Lightning训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lightning.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/sample_data.csv",
        help="训练数据路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config = Config.from_yaml(args.config)

    # 加载和预处理数据
    print("加载数据...")
    loader = DataLoader()
    df = loader.load_from_csv(args.data)

    preprocessor = DataPreprocessor(config.to_dict())
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

    # 创建数据模块
    print("创建数据模块...")
    data_module = PTMDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config.to_dict(),
    )

    # 创建模型
    print("创建模型...")
    model = PTM2CellNet.from_config(config.to_dict())
    lightning_model = PTM2CellNetLightning(model, config.to_dict())

    # 配置回调
    callbacks = []

    # 模型检查点
    checkpoint_config = config.get("training.checkpoint", {})
    if checkpoint_config.get("enabled", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.get("paths.outputs_models", "outputs/models"),
            filename="ptm2cellnet-{epoch:02d}-{val_loss:.4f}",
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_top_k=checkpoint_config.get("save_top_k", 3),
        )
        callbacks.append(checkpoint_callback)

    # 早停
    early_stopping_config = config.get("training.early_stopping", {})
    if early_stopping_config.get("enabled", True):
        early_stop_callback = EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            patience=early_stopping_config.get("patience", 10),
            mode=early_stopping_config.get("mode", "min"),
        )
        callbacks.append(early_stop_callback)

    # 学习率监控
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # 配置日志
    logger_config = config.get("training.logger", {})
    if logger_config.get("type", "tensorboard") == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=logger_config.get("save_dir", "outputs/logs"),
            name="lightning_logs",
        )
    elif logger_config.get("type") == "wandb":
        logger = pl.loggers.WandbLogger(
            project="ptm2cellnet",
            save_dir=logger_config.get("save_dir", "outputs/logs"),
        )
    else:
        logger = None

    # 创建Trainer
    trainer = pl.Trainer(
        accelerator=config.get("training.accelerator", "gpu"),
        devices=config.get("training.devices", 1),
        strategy=config.get("training.strategy", "auto"),
        precision=config.get("training.precision", "32"),
        max_epochs=config.get("training.max_epochs", 100),
        accumulate_grad_batches=config.get("training.accumulate_grad_batches", 1),
        gradient_clip_val=config.get("training.gradient_clip_val", 0.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
    )

    # 开始训练
    print("开始训练...")
    trainer.fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    # 测试
    print("测试模型...")
    trainer.test(datamodule=data_module)

    print("训练完成！")


if __name__ == "__main__":
    main()
```

---

## 4. 测试策略

### 4.1 单元测试

**目标：** 覆盖核心功能，确保代码可运行

**测试范围：**

1. **数据模块测试** (`tests/unit/test_data.py`)
   - PTMDataModule初始化
   - DataLoader创建
   - 数据批次形状验证

2. **模型模块测试** (`tests/unit/test_models.py`)
   - 预训练编码器加载
   - 模型前向传播
   - 输出形状验证

3. **Lightning模块测试** (`tests/unit/test_lightning.py`)
   - PTM2CellNetLightning初始化
   - training_step执行
   - validation_step执行
   - 优化器配置

**测试示例：**
```python
# tests/unit/test_lightning.py

import pytest
import torch
import pytorch_lightning as pl
from src.models.architectures import PTM2CellNet
from src.training.lightning_module import PTM2CellNetLightning


def test_lightning_module_init():
    """测试Lightning模块初始化"""
    config = {
        "model": {"encoder_type": "transformer", "num_classes": 4},
        "training": {"optimizer": "adamw", "learning_rate": 1e-4},
    }

    model = PTM2CellNet.from_config(config)
    lightning_model = PTM2CellNetLightning(model, config)

    assert lightning_model.model is not None
    assert lightning_model.config == config


def test_training_step():
    """测试训练步骤"""
    config = {
        "model": {"encoder_type": "transformer", "num_classes": 4},
        "training": {"optimizer": "adamw"},
    }

    model = PTM2CellNet.from_config(config)
    lightning_model = PTM2CellNetLightning(model, config)

    # 创建模拟批次
    batch = {
        "sequence": torch.randint(0, 20, (4, 100)),
        "ptm_types": torch.zeros(4, 100, dtype=torch.long),
        "ptm_mask": torch.zeros(4, 100),
        "label": torch.randint(0, 4, (4,)),
    }

    # 执行训练步骤
    loss = lightning_model.training_step(batch, 0)

    assert loss is not None
    assert loss.item() > 0
```

### 4.2 集成测试

**测试范围：**
- 端到端训练流程（1-2个epoch）
- 模型保存和加载
- 配置文件解析

---

## 5. 实施计划

### 5.1 阶段划分

**第一阶段：基础架构（2周）**
- 创建LightningDataModule
- 创建PTM2CellNetLightning
- 编写基础单元测试

**第二阶段：预训练模型集成（3周）**
- 实现PretrainedEncoder基类
- 实现ESM2Encoder
- 实现ProtBERTEncoder
- 修改PTM2CellNet架构
- 编写模型测试

**第三阶段：训练脚本和配置（2周）**
- 编写train_lightning.py
- 创建配置文件
- 测试训练流程

**第四阶段：测试和验证（3周）**
- 完善单元测试
- 集成测试
- 性能对比测试
- 文档更新

**总计：** 10周

### 5.2 里程碑

- **Week 2**: Lightning基础框架可用
- **Week 5**: 预训练模型集成完成
- **Week 7**: 训练脚本可用
- **Week 10**: 测试完成，文档更新

---

## 6. 风险和缓解措施

### 6.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 预训练模型内存占用大 | 训练失败 | 使用梯度累积、混合精度、模型并行 |
| Lightning学习曲线 | 开发延期 | 先实现简单功能，逐步深入 |
| 数据加载瓶颈 | 训练慢 | 优化num_workers，使用缓存 |
| 配置冲突 | 功能异常 | 分离新旧配置，向后兼容 |

### 6.2 项目风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 现有功能破坏 | 用户受影响 | 保留旧代码，充分测试 |
| 时间延期 | 项目延期 | 分阶段交付，优先核心功能 |
| 文档不完善 | 使用困难 | 同步更新文档和示例 |

---

## 7. 成功标准

### 7.1 功能标准

- [ ] Lightning训练脚本可以成功训练模型
- [ ] 支持ESM-2和ProtBERT预训练模型
- [ ] 现有训练脚本继续正常工作
- [ ] 配置文件可以切换不同编码器

### 7.2 性能标准

- [ ] 预训练模型性能优于现有模型（准确率提升5%+）
- [ ] 训练速度可接受（支持混合精度加速）
- [ ] 内存占用可控（单GPU可训练650M模型）

### 7.3 质量标准

- [ ] 单元测试覆盖核心功能
- [ ] 集成测试通过
- [ ] 文档更新完整

---

## 8. 后续工作

完成本次重构后，可以考虑：

1. **性能优化**：实现分布式训练（FSDP）
2. **模型扩展**：支持更多预训练模型（ProtT5、ESM-3）
3. **评估增强**：实现动态轨迹评估
4. **部署优化**：模型量化、推理加速

---

**文档结束**

*本文档将作为PTM2CellNet重构项目的核心参考。*
