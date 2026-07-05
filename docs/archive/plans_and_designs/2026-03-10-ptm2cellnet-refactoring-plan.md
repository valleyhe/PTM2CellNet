# PTM2CellNet 重构实施计划

**计划版本：** v1.0
**创建日期：** 2026-03-10
**基于设计文档：** 2026-03-10-ptm2cellnet-refactoring-design.md
**预计总时长：** 10周

---

## 执行摘要

本计划将PTM2CellNet项目从当前的轻量级架构（10M参数）升级到支持大规模预训练模型（420M+参数）的现代化架构，采用渐进式重构策略，确保现有功能持续可用。

**核心目标：**
- 迁移到PyTorch Lightning训练框架
- 集成ESM-2、ProtBERT等预训练模型
- 保持向后兼容性

**关键里程碑：**
- Week 2: Lightning基础框架可用
- Week 5: 预训练模型集成完成
- Week 7: 训练脚本可用
- Week 10: 测试完成，文档更新

---

## 阶段一：基础架构搭建（Week 1-2）

### 目标
建立Lightning训练框架的基础设施，实现最小可用版本。

### 任务列表

#### 1.1 环境准备（Day 1-2）

**任务：更新依赖包**

**文件：** `requirements.txt`

**操作：**
```bash
# 添加新依赖
transformers>=4.20.0
lion-pytorch>=0.0.7
wandb>=0.12.0  # 可选，用于实验跟踪
```

**验证：**
```bash
pip install -r requirements.txt
python -c "import transformers; import pytorch_lightning; print('依赖安装成功')"
```

**预计时间：** 2小时

---

#### 1.2 Lightning数据模块实现（Day 3-5）

**任务：创建PTMDataModule**

**文件：** `src/data/lightning_datamodule.py`（新建）

**实现步骤：**

1. 创建文件骨架
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import pandas as pd

from .datasets import PTMDataset
from .features import FeatureExtractor


class PTMDataModule(pl.LightningDataModule):
    """Lightning数据模块"""

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

        # 从配置中提取参数
        self.batch_size = config.get("training.batch_size", 32)
        self.num_workers = config.get("data.num_workers", 4)

    def setup(self, stage: Optional[str] = None):
        """准备数据集"""
        # 实现细节见设计文档
        pass

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        # 实现细节见设计文档
        pass

    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        # 实现细节见设计文档
        pass

    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        # 实现细节见设计文档
        pass
```

2. 实现setup方法
3. 实现dataloader方法
4. 添加错误处理

**验证：**
```python
# 创建测试脚本验证
from src.data.loaders import DataLoader
from src.data.preprocess import DataPreprocessor
from src.data.lightning_datamodule import PTMDataModule
from src.utils.config import Config

config = Config.from_yaml("configs/default.yaml")
loader = DataLoader()
df = loader.load_sample_data(100)
preprocessor = DataPreprocessor(config.to_dict())
train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)

datamodule = PTMDataModule(train_df, val_df, test_df, config.to_dict())
datamodule.setup()

# 验证数据加载器
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))
print(f"批次形状: {batch['sequence'].shape}")
```

**预计时间：** 8小时

---

#### 1.3 Lightning模块实现（Day 6-8）

**任务：创建PTM2CellNetLightning**

**文件：** `src/training/lightning_module.py`（新建）

**实现步骤：**

1. 创建文件骨架
```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from ..models.architectures import PTM2CellNet


class PTM2CellNetLightning(pl.LightningModule):
    """Lightning封装的PTM2CellNet模型"""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # 实现细节见设计文档
        pass

    def validation_step(self, batch, batch_idx):
        # 实现细节见设计文档
        pass

    def test_step(self, batch, batch_idx):
        # 实现细节见设计文档
        pass

    def configure_optimizers(self):
        # 实现细节见设计文档
        pass
```

2. 实现训练步骤
3. 实现优化器配置
4. 添加日志记录

**验证：**
```python
# 创建测试脚本验证
from src.models.architectures import PTM2CellNet
from src.training.lightning_module import PTM2CellNetLightning

config = {
    "model": {"encoder_type": "transformer", "num_classes": 4},
    "training": {"optimizer": "adamw", "learning_rate": 1e-4}
}

model = PTM2CellNet.from_config(config)
lightning_model = PTM2CellNetLightning(model, config)

# 模拟训练步骤
batch = {
    "sequence": torch.randint(0, 20, (4, 100)),
    "ptm_types": torch.zeros(4, 100, dtype=torch.long),
    "ptm_mask": torch.zeros(4, 100),
    "label": torch.randint(0, 4, (4,))
}

loss = lightning_model.training_step(batch, 0)
print(f"训练损失: {loss.item()}")
```

**预计时间：** 10小时

---

#### 1.4 基础单元测试（Day 9-10）

**任务：创建Lightning模块测试**

**文件：** `tests/unit/test_lightning.py`（新建）

**测试用例：**
1. test_lightning_module_init - 测试初始化
2. test_training_step - 测试训练步骤
3. test_validation_step - 测试验证步骤
4. test_configure_optimizers - 测试优化器配置

**验证：**
```bash
pytest tests/unit/test_lightning.py -v
```

**预计时间：** 6小时

---

### 阶段一验收标准

- [ ] PTMDataModule可以成功创建数据加载器
- [ ] PTM2CellNetLightning可以执行训练步骤
- [ ] 单元测试全部通过
- [ ] 代码通过语法检查（python -m compileall）

---

## 阶段二：预训练模型集成（Week 3-5）

### 目标
集成ESM-2和ProtBERT预训练模型，扩展模型架构。

### 任务列表

#### 2.1 预训练编码器基类实现（Day 11-13）

**任务：创建PretrainedEncoder基类**

**文件：** `src/models/pretrained_encoders.py`（新建）

**实现步骤：**

1. 创建基类
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
        """前向传播"""
        outputs = self.model(input_ids, output_attentions=self.use_attention_output)
        return outputs.last_hidden_state
```

2. 添加错误处理
3. 添加模型缓存支持

**验证：**
```python
# 测试预训练模型加载
from src.models.pretrained_encoders import PretrainedEncoder

encoder = PretrainedEncoder("facebook/esm2_t30_150M_UR50D")
print(f"隐藏层维度: {encoder.hidden_dim}")

# 测试前向传播
import torch
input_ids = torch.randint(0, 33, (2, 100))  # ESM-2词汇表大小为33
output = encoder(input_ids)
print(f"输出形状: {output.shape}")  # 应为 [2, 100, 640]
```

**预计时间：** 8小时

---

#### 2.2 ESM-2编码器实现（Day 14-16）

**任务：创建ESM2Encoder**

**文件：** `src/models/pretrained_encoders.py`（扩展）

**实现步骤：**

1. 添加ESM2Encoder类
```python
class ESM2Encoder(PretrainedEncoder):
    """ESM-2编码器"""

    def __init__(
        self,
        model_size: str = "150M",
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
```

2. 添加特殊处理（如序列长度限制）

**验证：**
```python
# 测试不同大小的ESM-2模型
for size in ["150M", "650M"]:
    encoder = ESM2Encoder(model_size=size)
    print(f"ESM-2 {size}: {encoder.hidden_dim} 维")
```

**预计时间：** 6小时

---

#### 2.3 ProtBERT编码器实现（Day 17-18）

**任务：创建ProtBERTEncoder**

**文件：** `src/models/pretrained_encoders.py`（扩展）

**实现步骤：**

1. 添加ProtBERTEncoder类
```python
class ProtBERTEncoder(PretrainedEncoder):
    """ProtBERT编码器"""

    def __init__(self, freeze: bool = False):
        super().__init__("Rostlab/prot_bert", freeze=freeze)
```

**验证：**
```python
# 测试ProtBERT
encoder = ProtBERTEncoder()
print(f"ProtBERT: {encoder.hidden_dim} 维")
```

**预计时间：** 4小时

---

#### 2.4 模型架构修改（Day 19-21）

**任务：修改PTM2CellNet支持预训练编码器**

**文件：** `src/models/architectures.py`（修改）

**修改内容：**

1. 在__init__方法中添加预训练编码器分支
```python
# 在现有编码器分支后添加
elif encoder_type.startswith("esm2"):
    model_size = encoder_type.split("_")[1] if "_" in encoder_type else "150M"
    encoder = ESM2Encoder(model_size=model_size, freeze=freeze_encoder)
    self.embed_dim = encoder.hidden_dim
elif encoder_type == "protbert":
    encoder = ProtBERTEncoder(freeze=freeze_encoder)
    self.embed_dim = encoder.hidden_dim
```

2. 添加freeze_encoder参数
3. 更新from_config方法

**验证：**
```python
# 测试预训练编码器集成
from src.models.architectures import PTM2CellNet

config = {
    "model": {
        "encoder_type": "esm2_150M",
        "freeze_encoder": False,
        "num_classes": 4
    }
}

model = PTM2CellNet.from_config(config)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 测试前向传播
batch = {
    "sequence": torch.randint(0, 33, (2, 100)),
    "ptm_types": torch.zeros(2, 100, dtype=torch.long),
    "ptm_mask": torch.zeros(2, 100),
}

output = model(batch)
print(f"输出形状: {output['logits'].shape}")
```

**预计时间：** 8小时

---

#### 2.5 模型测试（Day 22-25）

**任务：创建预训练模型测试**

**文件：** `tests/unit/test_models.py`（扩展）

**测试用例：**
1. test_pretrained_encoder_init - 测试预训练编码器初始化
2. test_esm2_encoder_forward - 测试ESM-2前向传播
3. test_protbert_encoder_forward - 测试ProtBERT前向传播
4. test_ptm2cellnet_with_pretrained - 测试完整模型

**验证：**
```bash
pytest tests/unit/test_models.py::test_pretrained_encoder_init -v
pytest tests/unit/test_models.py::test_esm2_encoder_forward -v
pytest tests/unit/test_models.py::test_protbert_encoder_forward -v
pytest tests/unit/test_models.py::test_ptm2cellnet_with_pretrained -v
```

**预计时间：** 10小时

---

### 阶段二验收标准

- [ ] ESM2Encoder可以成功加载预训练权重
- [ ] ProtBERTEncoder可以成功加载预训练权重
- [ ] PTM2CellNet支持预训练编码器
- [ ] 模型前向传播正常
- [ ] 单元测试全部通过

---

## 阶段三：训练脚本和配置（Week 6-7）

### 目标
创建Lightning训练脚本和配置文件，实现端到端训练流程。

### 任务列表

#### 3.1 Lightning配置文件创建（Day 26-27）

**任务：创建Lightning配置文件**

**文件：**
- `configs/lightning.yaml`（新建）
- `configs/pretrained/esm2_150m.yaml`（新建）
- `configs/pretrained/esm2_650m.yaml`（新建）
- `configs/pretrained/protbert.yaml`（新建）

**实现步骤：**

1. 创建基础Lightning配置
```yaml
# configs/lightning.yaml
model:
  encoder_type: "transformer"
  num_classes: 4
  dropout: 0.1

training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"

  accelerator: "gpu"
  devices: 1
  precision: "16-mixed"

  early_stopping:
    enabled: true
    patience: 10

  checkpoint:
    save_top_k: 3

data:
  max_sequence_length: 1000
  num_workers: 4
```

2. 创建预训练模型配置
```yaml
# configs/pretrained/esm2_650m.yaml
model:
  encoder_type: "esm2_650M"
  freeze_encoder: false
  num_classes: 4

training:
  max_epochs: 50
  batch_size: 8
  learning_rate: 5e-5
  precision: "bf16-mixed"
  accumulate_grad_batches: 8
```

**验证：**
```python
from src.utils.config import Config

config = Config.from_yaml("configs/lightning.yaml")
print(f"编码器类型: {config.get('model.encoder_type')}")
print(f"批次大小: {config.get('training.batch_size')}")
```

**预计时间：** 4小时

---

#### 3.2 Lightning训练脚本实现（Day 28-32）

**任务：创建train_lightning.py**

**文件：** `scripts/train_lightning.py`（新建）

**实现步骤：**

1. 创建脚本骨架（见设计文档）
2. 实现参数解析
3. 实现数据加载
4. 实现模型创建
5. 实现训练循环
6. 添加日志记录

**验证：**
```bash
# 使用小数据集测试训练脚本
python scripts/train_lightning.py \
    --config configs/lightning.yaml \
    --data data/raw/sample_data.csv
```

**预计时间：** 12小时

---

#### 3.3 预训练模型训练脚本（Day 33-35）

**任务：创建train_pretrained.py**

**文件：** `scripts/train_pretrained.py`（新建）

**实现步骤：**

1. 基于train_lightning.py创建
2. 添加预训练模型特定配置
3. 添加模型下载和缓存逻辑

**验证：**
```bash
# 测试ESM-2训练
python scripts/train_pretrained.py \
    --config configs/pretrained/esm2_150m.yaml \
    --data data/raw/sample_data.csv
```

**预计时间：** 8小时

---

### 阶段三验收标准

- [ ] Lightning配置文件格式正确
- [ ] train_lightning.py可以成功运行
- [ ] train_pretrained.py可以成功运行
- [ ] 训练过程可以保存检查点
- [ ] 训练日志正常记录

---

## 阶段四：测试和验证（Week 8-10）

### 目标
完善测试，验证功能，更新文档。

### 任务列表

#### 4.1 集成测试（Day 36-40）

**任务：创建集成测试**

**文件：** `tests/integration/test_lightning_pipeline.py`（新建）

**测试用例：**
1. test_end_to_end_training - 端到端训练流程
2. test_model_checkpoint - 模型保存和加载
3. test_pretrained_model_training - 预训练模型训练

**验证：**
```bash
pytest tests/integration/test_lightning_pipeline.py -v
```

**预计时间：** 12小时

---

#### 4.2 性能对比测试（Day 41-45）

**任务：对比新旧实现性能**

**测试内容：**
1. 训练速度对比
2. 内存占用对比
3. 模型性能对比（准确率）

**验证：**
```bash
# 运行性能测试脚本
python scripts/benchmark.py \
    --config-old configs/default.yaml \
    --config-new configs/lightning.yaml
```

**预计时间：** 10小时

---

#### 4.3 文档更新（Day 46-50）

**任务：更新项目文档**

**文件：**
- `README.md`（更新）
- `CLAUDE.md`（更新）
- `docs/training-guide.md`（新建）

**更新内容：**
1. 添加Lightning训练说明
2. 添加预训练模型使用指南
3. 更新架构图

**预计时间：** 8小时

---

### 阶段四验收标准

- [ ] 集成测试全部通过
- [ ] 性能对比报告完成
- [ ] 文档更新完整
- [ ] 代码通过所有质量检查

---

## 风险管理

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 | 应急计划 |
|------|------|------|----------|----------|
| 预训练模型内存不足 | 高 | 高 | 使用梯度累积、混合精度 | 减小batch size或使用更小模型 |
| Lightning版本兼容性 | 中 | 中 | 锁定版本号 | 降级到稳定版本 |
| 数据加载瓶颈 | 中 | 中 | 优化num_workers | 使用数据缓存 |
| 训练不收敛 | 低 | 高 | 调整学习率 | 使用预训练权重 |

### 项目风险

| 风险 | 概率 | 影响 | 缓解措施 | 应急计划 |
|------|------|------|----------|----------|
| 时间延期 | 中 | 中 | 分阶段交付 | 削减非核心功能 |
| 现有功能破坏 | 低 | 高 | 充分测试 | 回退到旧版本 |
| 文档不完善 | 中 | 低 | 同步更新 | 后续补充 |

---

## 资源需求

### 硬件需求

- **开发环境：** 单GPU（>=8GB显存）
- **测试环境：** 单GPU（>=16GB显存，用于650M模型）
- **存储空间：** >=20GB（预训练模型缓存）

### 软件需求

- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning 1.5+
- Transformers 4.20+

---

## 成功标准

### 功能标准

- [ ] Lightning训练脚本可以成功训练模型
- [ ] 支持ESM-2和ProtBERT预训练模型
- [ ] 现有训练脚本继续正常工作
- [ ] 配置文件可以切换不同编码器

### 性能标准

- [ ] 预训练模型性能优于现有模型（准确率提升5%+）
- [ ] 训练速度可接受（支持混合精度加速）
- [ ] 内存占用可控（单GPU可训练650M模型）

### 质量标准

- [ ] 单元测试覆盖核心功能
- [ ] 集成测试通过
- [ ] 文档更新完整
- [ ] 代码通过质量检查（flake8, pylint, mypy）

---

## 后续工作

完成本次重构后，建议进行：

1. **性能优化**：实现分布式训练（FSDP）
2. **模型扩展**：支持更多预训练模型（ProtT5、ESM-3）
3. **评估增强**：实现动态轨迹评估
4. **部署优化**：模型量化、推理加速

---

**计划结束**

*本计划将作为PTM2CellNet重构项目的执行指南。*
