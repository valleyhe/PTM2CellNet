# PTM2CellNet

基于深度学习的蛋白质翻译后修饰(PTM)分析与细胞状态预测系统。

## 目录结构

```
PTM2CellNet/
├── src/                        # 核心源码
│   ├── analysis/               # 变异解析、基因映射、通路整合、PTM工作流
│   ├── api/                    # FastAPI 推理服务
│   │   └── routes/             # predictions / state / model_info
│   ├── data/                   # 数据集、预处理、特征提取、增强、验证
│   ├── evaluation/             # 评估指标、可解释性、可视化
│   ├── integration/            # GenKI 图扰动、PTM-基因映射、虚拟扰动
│   │   └── genki/              # graph_utils / perturbation / significance
│   ├── models/                 # 编码器、DAVF、注意力、预测头、信号网络
│   ├── training/               # Lightning模块、损失函数、优化器、回调、PEFT
│   └── utils/                  # 配置、日志、IO、运行时工具
├── tests/                      # 测试
│   ├── unit/                   # 单元测试（按模块组织）
│   │   ├── analysis/           # gene_mapper / pathway / variant_parser
│   │   ├── evaluation/         # metrics / evaluators / explainers
│   │   ├── integration/        # genki / contracts / ptm_gene_mapper
│   │   └── training/           # losses / callbacks / optimizers / peft
│   ├── integration/            # 集成测试（DAVF / Lightning / Mamba / 变异工作流）
│   └── fixtures/               # 测试固件
├── scripts/                    # 生产脚本
│   ├── train*.py               # 训练入口（主/Lightning/PTM位点/多任务/二分类）
│   ├── predict*.py             # 推理入口（单条/批量/变异效应）
│   ├── evaluate.py             # 模型评估
│   ├── prepare_*.py            # 数据准备
│   └── run_*.py                # 虚拟扰动、两阶段解释
├── configs/                    # 配置文件
│   ├── default.yaml            # 默认配置
│   ├── production.yaml         # 生产配置
│   ├── lightning.yaml          # Lightning 训练配置
│   ├── mamba*.yaml             # Mamba 编码器配置
│   ├── pretrained/             # ESM-2 / ProtBERT 预训练配置
│   ├── training/               # 微调训练配置
│   └── integration/            # 虚拟扰动 / 两阶段解释配置
├── docs/                       # 技术文档与报告
├── examples/                   # 使用示例
├── .planning/                  # 开发规划文档
├── Dockerfile                  # 容器构建
├── docker-compose.yml          # 容器编排
├── requirements.txt            # Python 依赖
└── setup.py                    # 包安装
```

## 安装

```bash
pip install -r requirements.txt
```

ESM-2 等预训练模型需要 Hugging Face 访问，建议设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

### 数据准备

```bash
python scripts/prepare_data.py
python scripts/prepare_ptm_data.py
```

### 训练

```bash
# 主训练脚本
python scripts/train.py

# Lightning 训练（推荐）
python scripts/train_lightning.py

# PTM 位点预测
python scripts/train_ptm_site.py

# 多任务联合训练
python scripts/train_multitask_ptm.py
```

### 推理

```bash
# 单条预测
python scripts/predict.py

# 批量预测
python scripts/batch_predict.py

# 变异效应预测
python scripts/predict_variant_effect.py --variant "BRAF_V600E"
```

### 虚拟扰动与解释

```bash
# PTM-aware 虚拟扰动
python scripts/run_ptm_virtual_perturbation.py --gene TP53 --output outputs/

# 两阶段筛选与解释
python scripts/run_two_stage_explanation.py \
  --input data/processed/candidates.csv \
  --output outputs/
```

### API 服务

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 测试

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## 代码示例

```python
import torch
from src.models.architectures import PTM2CellNet
from src.models.pretrained_encoders import ESM2Encoder

# 创建 ESM-2 编码器
encoder = ESM2Encoder(model_size="8M", freeze=True)
encoded = encoder.tokenize(["ACDEFGHIKLMNPQRSTVWY"])

# 创建端到端模型
model = PTM2CellNet(encoder_type="esm2_8M", num_classes=4, freeze_encoder=True)
model.eval()

# 推理
batch = {
    "input_ids": encoded["input_ids"],
    "attention_mask": encoded["attention_mask"],
    "ptm_mask": torch.zeros(1, encoded["input_ids"].shape[1]),
    "ptm_types": torch.zeros(1, encoded["input_ids"].shape[1], dtype=torch.long),
}
with torch.no_grad():
    output = model(batch)

print(output["probabilities"])
```

## 支持的编码器

| 编码器 | 类型 | 配置示例 |
|--------|------|----------|
| CNN | 自定义 | `encoder_type="cnn"` |
| LSTM | 自定义 | `encoder_type="lstm"` |
| Transformer | 自定义 | `encoder_type="transformer"` |
| ESM-2 (8M~15B) | 预训练蛋白语言模型 | `encoder_type="esm2_8M"` |
| ProtBERT | 预训练蛋白语言模型 | `encoder_type="protbert"` |
| ProtT5 | 预训练蛋白语言模型 | `encoder_type="prott5"` |
| Mamba | 状态空间模型 | `encoder_type="mamba"` |

## 核心模块说明

| 模块 | 说明 |
|------|------|
| `src/models/davf*.py` | DAVF（方向感知变异融合）推理与注意力机制 |
| `src/models/ptm_modules.py` | Gated PTM Fusion、PTM 嵌入层 |
| `src/models/signaling_network.py` | 信号通路网络建模 |
| `src/models/variant_effect.py` | 变异效应预测 |
| `src/analysis/variant_workflow.py` | 变异解析完整工作流 |
| `src/integration/genki/` | GenKI 图扰动与显著性分析 |
| `src/training/lightning_module.py` | PyTorch Lightning 训练模块 |
| `src/training/peft_config.py` | LoRA/PEFT 参数高效微调配置 |

## Docker 部署

```bash
docker-compose up --build
```

## 许可证

MIT License
