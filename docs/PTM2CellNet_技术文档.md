# PTM2CellNet 技术文档

**版本**: v1.0  
**日期**: 2026-07-06  
**目标**: 从蛋白质翻译后修饰(PTM)预测细胞状态

---

## 1. 项目概述

PTM2CellNet 是一个深度学习框架，通过整合蛋白质序列信息和翻译后修饰(PTM)位点数据，预测蛋白质功能改变所导致的细胞状态变化。项目支持多种序列编码器（CNN、Transformer、LSTM、Mamba、ESM-2、ESM-3、ProtBERT、ProtT5），并提供基于 Flow Matching 的方向感知速度场模型(DAVF)用于 PTM 到细胞状态的因果推断。

### 1.1 核心能力

- 蛋白质序列编码（多种编码器选择）
- PTM 位点感知的特征提取与融合
- 细胞状态分类与回归预测
- DAVF：PTM 方向感知的细胞状态扰动模拟
- 多任务 PTM 预测
- 变体效应预测
- 外部工具集成（AlphaFold、BLAST、PSIPRED 等）
- FastAPI 生产级 HTTP API

### 1.2 技术栈

- Python 3.10+
- PyTorch 2.x
- PyTorch Lightning（可选）
- Mamba SSM
- FastAPI + Uvicorn
- scVI（可选）
- Geneformer / scGPT（可选）
- ESM-3（Evolutionary Scale）

---

## 2. 目录结构

```
PTM2CellNet/
├── configs/               # YAML 配置文件（29 个）
│   ├── model/             # 模型架构配置
│   ├── pretrained/        # 预训练编码器配置
│   ├── training/          # 训练配置
│   └── davf_integration.yaml  # DAVF 集成配置
├── src/
│   ├── data/              # 数据处理层（28 文件，~9118 行）
│   │   ├── loaders/       # 数据加载器（UniProt、PhosphoSitePlus、dbPTM）
│   │   ├── datasets.py    # PTMDataset / PTMDataModule
│   │   ├── features.py    # 特征工程（one-hot、k-mer、理化性质）
│   │   ├── preprocess.py  # 清洗、标准化、划分
│   │   ├── augmentation.py# DAVF 站点增强器
│   │   ├── validation.py  # 数据验证 + SafeUnpickler
│   │   └── aa_constants.py# 氨基酸常量与 PTM 类型归一化
│   ├── models/            # 模型层（37 文件，~12026 行）
│   │   ├── encoders.py    # CNN / Transformer / LSTM / GRU 编码器
│   │   ├── pretrained_encoders.py  # ESM-2 / ESM-3 / ProtBERT / ProtT5
│   │   ├── mamba_encoder.py       # Mamba SSM 编码器
│   │   ├── ptm_modules.py         # PTM 嵌入 / 注意力 / GatedPTMFusion
│   │   ├── architectures.py       # PTM2CellNet / PTM2CellNetLarge 架构
│   │   ├── davf.py / davf_inference.py / latent_davf.py  # DAVF 系列
│   │   ├── davf_encoder.py / davf_attention.py           # DAVF 编码器
│   │   ├── davf_losses.py / davf_velocity.py             # DAVF 损失
│   │   ├── ensemble.py          # PTM2CellNetEnsemble
│   │   ├── scvi_adapter.py      # scVI 编码-解码适配器
│   │   └── signaling_network.py # 信号通路映射器
│   ├── training/           # 训练层（13 文件，~4051 行）
│   │   ├── trainers.py     # 自定义 Trainer
│   │   ├── lightning_module.py    # Lightning 包装
│   │   ├── losses.py       # 损失函数
│   │   ├── callbacks.py    # Checkpoint / EarlyStopping
│   │   └── amp_compat.py   # AMP 兼容辅助
│   ├── evaluation/         # 评估层（5 文件，~2199 行）
│   │   ├── metrics.py      # NDCG / MAP / bootstrap CI
│   │   ├── evaluators.py   # evaluate() / cross_validate()
│   │   └── explainers.py   # LeaveOnePTMOut 解释
│   ├── api/                # API 层（12 文件，~3145 行）
│   │   ├── app.py          # FastAPI 工厂
│   │   ├── routes/         # /predict /batch_predict /initialize
│   │   ├── schemas.py      # Pydantic 模型
│   │   └── middleware.py    # 速率限制 / CORS
│   └── utils/              # 工具层（10 文件，~1984 行）
│       ├── io.py           # SafeUnpickler / safe_torch_load
│       ├── config.py       # YAML 配置管理
│       └── helpers.py      # 序列清洗 / 验证
├── scripts/                # CLI 脚本（35 个）
│   ├── train.py            # 训练入口
│   ├── predict.py          # 预测入口
│   ├── evaluate.py         # 评估入口
│   └── finetune_davf_e2e.py# DAVF E2E 微调
├── tests/                  # 测试（136 文件，1531+ 测试）
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   ├── e2e/                # 端到端测试
│   └── real_assets/        # 真实资产验收测试
├── checkpoints/            # 模型权重
│   ├── esm3/               # ESM-3 权重（~5.27 GB）
│   └── latent_davf_*/      # DAVF 检查点
├── configs/                # YAML 配置文件
└── outputs/                # 输出目录（模型、日志、结果）
```

---

## 3. 数据层架构

### 3.1 数据加载流程

```
原始数据源（UniProt / PhosphoSitePlus / dbPTM）
    │
    ▼
DataLoader（loaders.py）
    │  支持 HTTP 下载、CSV/FASTA 解析
    │  安全下载：URL 分块 + 主机白名单 + 大小防护
    ▼
DataPreprocessor（preprocess.py）
    │  序列标准化、缺失值处理、PTM 位点标注
    │  训练/验证/测试集划分（支持同源拆分）
    ▼
FeatureExtractor（features.py）
    │  One-hot、k-mer、理化性质（等电点、疏水性等）
    │  向量化运算优化
    ▼
PTMDataset（datasets.py）
    │  PyTorch Dataset，支持在线增强
    │  批量加载，内存优化
    ▼
PTMDataModule（lightning_datamodule.py）
    │  PyTorch Lightning DataModule
    │  train/val/test dataloader
```

### 3.2 支持的 PTM 类型

PTM2CellNet 支持 29 种规范化 PTM 类型，通过 `src/data/aa_constants.py` 中的 `PTM_TYPE_ALIASES` 和 `normalize_ptm_type()` 函数实现命名归一化：

| 类别 | PTM 类型 |
|------|---------|
| 核心磷酸化 | phosphorylation |
| 酰化类 | acetylation, succinylation, malonylation, glutarylation, propionylation, butyrylation, formylation |
| 泛素化类 | ubiquitination, sumoylation, neddylation |
| 甲基化 | methylation |
| 糖基化 | glycosylation, oglcnacylation |
| 氧化还原 | oxidation, nitrosylation, hydroxylation, carbonylation |
| 脂修饰 | palmitoylation, myristoylation, prenylation |
| 其他 | citrullination, deamidation, adpribosylation, sulfation, amidation, disulfidebond, lactylation, crotonylation |

### 3.3 特征工程详解

**`FeatureExtractor`** (src/data/features.py：43 符号) 支持：

1. **One-hot 编码**: 21 种氨基酸（20 标准 + gap），每个残基编码为 21 维向量
2. **k-mer 编码**: 以 k-mer 频率构建特征向量，支持 k=2,3,4
3. **理化性质**: 等电点(pI)、疏水性、分子量、极性等 10+ 种理化特征
4. **PTM 位置编码**: 在序列中标记 PTM 位点的类型和相对位置
5. **序列特征组合**: 支持多特征拼接

---

## 4. 模型架构

### 4.1 总体架构

PTM2CellNet 采用模块化设计，核心由三部分组成：

```
输入批次（序列 + PTM 位点）
    │
    ├──► [SequenceEncoder] → 序列嵌入 [batch, seq_len, embed_dim]
    │
    ├──► [PTMModule] → PTM 特征 [batch, seq_len, ptm_dim]
    │        │
    │        ▼
    │    [融合层] Gated / Attention
    │        │
    │        ▼
    ├──► [DAVF 分支]（可选） PTM → 方向感知向量
    │
    ▼
[CellStatePredictor] → 分类/回归输出
```

### 4.2 抽象基类：PTM2CellNetBase

`PTM2CellNetBase`（src/models/architectures.py:83）是所有模型变体的共享基类，提供：

- `_build_encoder()`: 根据 encoder_type 构建编码器
- `_build_predictor()`: 构建分类/回归/多任务预测器
- `forward()`: 标准前向传播
- `get_model_info()`: 模型元信息
- `from_config()`: 从 YAML 配置构建

### 4.3 序列编码器

| 编码器 | 类型 | 特点 | 位置 |
|--------|------|------|------|
| CNNEncoder | 卷积 | 多尺度卷积核（3,5,7），轻量 | encoders.py:43 |
| TransformerEncoder | 注意力 | 多头自注意力 + 位置编码 | encoders.py:89 |
| LSTMEncoder | 循环 | 双向 LSTM | encoders.py:158 |
| GRUEncoder | 循环 | 双向 GRU | encoders.py:204 |
| MambaEncoder | SSM | Mamba 状态空间模型，O(n) 复杂度 | mamba_encoder.py |
| ESM2Encoder | 预训练 | ESM-2 蛋白质语言模型（8M~650M） | pretrained_encoders.py |
| ESM3Encoder | 预训练 | ESM-3 多模态模型（序列+结构+功能） | pretrained_encoders.py:363 |
| ProtBERTEncoder | 预训练 | ProtBERT 蛋白质 BERT | pretrained_encoders.py |
| ProtT5Encoder | 预训练 | ProtT5 蛋白质 T5 | pretrained_encoders.py |

### 4.4 PTM 处理模块

**PTMEmbedding**（ptm_modules.py）：将 PTM 类型编码为稠密向量  
**PTMAttention**（ptm_modules.py）：计算 PTM 位点与序列区域的注意力权重  
**GatedPTMFusion**（ptm_modules.py）：门控机制融合序列嵌入与 PTM 特征

### 4.5 DAVF：方向感知速度场

DAVF（Direction-Aware Velocity Field）是项目的核心创新模块，用于建模 PTM 扰动如何改变细胞状态。

**核心模型层次**:

```
DAVF (davf.py)                   # 顶层 Flow Matching 模型
  ├── BiPerturbEncoder            # 扰动脉冲编码
  ├── ConditionalVelocityField    # 条件速度场网络
  └── LatentDAVF (latent_davf.py) # scVI 潜在空间 DAVF

DAVFInferenceModule (davf_inference.py)
  ├── DAVFInferenceConfig         # 推理配置
  ├── DeltaProjection             # 扰动映射投影
  └── GeneMLEPEncoder             # 轻量基因 MLP 编码器
```

**DAVF 训练流程**:

```
x_0（初始细胞状态） + x_1（扰动后目标状态）
    │
    ▼
t ~ U(0,1) 采样时间步
    │
    ▼
x_t = (1-t)*x_0 + t*x_1  （线性插值）
    │
    ▼
v(x_t, t, gene_ids, directions)  （预测速度场）
    │
    ▼
Loss = MSE(v_t, u_t) + MagLoss + DirLoss
     where u_t = x_1 - x_0  （目标速度）
```

**DAVF 推理流程**:

```
x_0 + gene_ids + directions
    │
    ▼
ODE 积分（num_steps 步欧拉法）
    │
    ▼
x_1_pred = 扰动预测
```

**DAVFLoss 组件**（davf_losses.py:17）:

| 损失 | 权重 | 作用 |
|------|------|------|
| MSE Loss | 1.0 | 确保速度向量方向正确 |
| Magnitude Loss | 0.3 | 确保预测幅度与真实幅度匹配 |
| Direction Loss | 0.1 | 符号一致性（cos 相似度） |
| 自适应权重 | — | 训练中 MSE 权重衰减，幅度/方向权重增加 |

### 4.6 预测器

**CellStatePredictor**（predictors.py）:

| 变体 | 用途 |
|------|------|
| ClassificationPredictor | 多分类细胞状态预测 |
| RegressionPredictor | 回归预测（连续值） |
| MultiTaskPredictor | 多任务联合预测 |
| DeltaPredictor | PTM 效应差值预测 |

### 4.7 模型变体

| 模型 | 说明 | 位置 |
|------|------|------|
| PTM2CellNet | 基础版，任意编码器 + PTM 融合 | architectures.py:667 |
| PTM2CellNetLarge | 扩展版，更大隐藏维度和层数 | architectures.py |
| PTM2CellNetEnsemble | 集成版，多模型投票 | ensemble.py |
| PTM2CellNetBase | 抽象基类（1168 行） | architectures.py:83 |

---

## 5. 训练层架构

### 5.1 核心 Trainer

`Trainer`（src/training/trainers.py:31）支持：
- 梯度裁剪、梯度累积
- 自动混合精度（AMP）
- 学习率 warmup
- 随机种子可复现
- Checkpoint 回调

**训练流程**:
```
for epoch in range(epochs):
    trainer.train_epoch()
        ├── 前向传播
        ├── 损失计算
        ├── 反向传播
        ├── 梯度裁剪/累积
        └── 优化器更新
    trainer.validate_epoch()
        └── 验证指标计算
    trainer.predict_epoch()
        └── 测试集推理
```

### 5.2 Lightning 训练

`LitPTM2CellNet`（lightning_module.py）提供 PyTorch Lightning 集成：
- 自动设备管理
- DDP 多 GPU 训练
- TensorBoard 日志
- Checkpoint 管理（top-k 保存）

### 5.3 自监督预训练

`SelfSupervisedPretrainer`（self_supervised.py）支持：
- Masked Language Modeling（MLM）
- PTM 位点对比学习
- 自编码去噪

### 5.4 DAVF 微调

`finetune_davf_e2e.py` 实现两阶段微调：
- **阶段 1**: 冻结 DAVF 编码器，训练 TaskHead（lr=1e-3）
- **阶段 2**: 解冻 DAVF，端到端微调（lr=1e-5）

### 5.5 Artifacts 导出

`artifacts.py` 提供训练产物导出：
- 模型权重（best_model.pt）
- 训练日志（loss 曲线）
- 配置快照（config.yaml）
- 清单文件（manifest.json）

---

## 6. 评估层架构

### 6.1 核心函数

- `evaluate()`（evaluators.py）: 单次评估
- `cross_validate()`（evaluators.py）: k-fold 交叉验证

### 6.2 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 分类 | Accuracy, Precision, Recall, F1 | 标准分类指标 |
| 分类 | AUC-ROC, AUC-PR | 曲线下面积 |
| 回归 | MAE, MSE, RMSE, R² | 标准回归指标 |
| 排序 | NDCG, MAP | 排序质量指标 |
| 置信度 | Bootstrap + t-distribution CI | 95% 置信区间 |

### 6.3 模型解释

`LeaveOnePTMOut`（explainers.py）: 逐个移除 PTM 位点，测量预测变化，量化每个 PTM 的贡献。

---

## 7. API 服务

### 7.1 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /predict | 单样本预测 |
| POST | /batch_predict | 批量预测 |
| POST | /predict/variant | 变体效应预测 |
| POST | /initialize | 初始化模型 |
| GET | /health | 健康检查 |
| GET | /live | 存活检查 |
| GET | /ready | 就绪检查 |
| GET | /model/info | 模型信息 |
| GET | /metrics | Prometheus 指标 |

### 7.2 请求格式

```json
{
  "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
  "ptm_sites": [
    {"position": 3, "type": "phosphorylation", "amino_acid": "L", "gene_symbol": "HBA1"}
  ],
  "use_davf": false
}
```

### 7.3 响应格式

```json
{
  "cell_state": "apoptosis",
  "confidence": 0.87,
  "probabilities": {
    "apoptosis": 0.87,
    "proliferation": 0.08,
    "quiescence": 0.03,
    "senescence": 0.02
  },
  "pathway_impacts": [...],
  "processing_time_ms": 45.2
}
```

### 7.4 安全特性

- 请求体大小限制（默认 10 MB）
- 速率限制（600 rpm / burst 60）
- CORS 管控
- API key 验证
- 路径穿越防护
- SafeUnpickler / safe_torch_load 分层安全反序列化
- 远程下载 URL/host/大小三层防护

---

## 8. 安装指南

### 8.1 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- CUDA 11.7+（GPU 训练）
- 至少 16 GB 内存（CPU 推理）、8 GB GPU 显存

### 8.2 安装步骤

```bash
# 克隆仓库
git clone <repo-url>
cd PTM2CellNet

# 安装核心依赖
pip install -r requirements-core.txt

# 安装预训练模型依赖（可选）
pip install -r requirements-pretrained.txt

# 安装 Mamba SSM 支持（可选）
pip install -r requirements-mamba.txt

# 安装分析工具依赖（可选）
pip install -r requirements-analysis.txt

# 安装类型检查包（开发）
pip install types-requests types-PyYAML types-setuptools
```

### 8.3 ESM-3 安装

```bash
# ESM-3 SDK (EvolutionaryScale)
pip install esm>=3.0.0

# 下载模型权重到 checkpoints/esm3/
# 权重文件（~5.27 GB）:
#   esm3_sm_open_v1.pth           (2.80 GB)
#   esm3_structure_decoder_v0.pth (1.24 GB)
#   esm3_structure_encoder_v0.pth (62.3 MB)
#   esm3_function_decoder_v0.pth  (1.30 GB)
```

### 8.4 可选依赖

| 依赖 | 用途 | 安装命令 |
|------|------|---------|
| scVI | 潜在空间 DAVF | `pip install scvi-tools` |
| Geneformer | 基因嵌入 | 手动下载权重 |
| sspa | KEGG/Reactome 通路 | `pip install sspa` |
| AlphaFold | 结构预测 | 本地部署 |

### 8.5 Docker 部署

```bash
docker-compose up -d
# API 服务: http://localhost:8000
# 健康检查: http://localhost:8000/health
```

---

## 9. 调试指南

### 9.1 常见问题

**Q: CUDA out of memory?**
```bash
# 减小 batch_size
python scripts/train.py --config configs/mamba_small_gpu.yaml --batch-size 8
# 或使用 CPU
python scripts/train.py --device cpu
```

**Q: ESM-3 导入失败?**
```python
# 检查 ESM 版本
python -c "import esm; print(esm.__version__)"
# 如无法安装，esm3_encoder() 会自动 fallback 到 ESM-2
```

**Q: 测试失败?**
```bash
# 确认使用正确的 Python 环境
which python3
# 使用 python3 -m pytest（不使用裸 pytest，可能与系统 Python 冲突）
python3 -m pytest tests/unit -q
```

### 9.2 调试命令

```bash
# 完整 CI 检查
python3 -m pytest tests/unit -q    # 单元测试
ruff check .                        # 代码风格
python3 -m mypy src --show-error-codes  # 类型检查

# 集成测试
python3 -m pytest tests/integration -q

# 端到端测试
python3 -m pytest tests/e2e -q

# 真实资产验收（需要权重文件）
PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 \\
  python3 -m pytest tests/real_assets/ -v
```

---

## 10. 配置说明

### 10.1 配置层次

```
default.yaml            # 全局默认配置
  ├── model/            # 模型架构参数
  │   ├── ptm2cellnet_base.yaml
  │   ├── ptm2cellnet_large.yaml
  │   └── ptm2cellnet_mamba.yaml
  ├── pretrained/       # 预训练编码器配置
  │   ├── esm2_8m.yaml / esm2_35m.yaml / esm2_150m.yaml / esm2_650m.yaml
  │   ├── esm3_sm_open.yaml
  │   └── protbert.yaml
  └── training/         # 训练超参数
      └── finetune_binary.yaml
```

### 10.2 核心配置项

**default.yaml**:
```yaml
model:
  encoder_type: "transformer"    # cnn / transformer / lstm / gru / mamba / esm2 / esm3 / protbert
  vocab_size: 21
  embed_dim: 128
  num_layers: 4
  num_heads: 8
  dropout: 0.1
  ptm_fusion_type: "attention"   # attention / gated / concat
  task_type: "classification"
  num_classes: 4

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  grad_clip_norm: 1.0
  use_amp: true
  seed: 42
```

**esm3_sm_open.yaml**:
```yaml
encoder_type: "esm3_small"
freeze_encoder: true
batch_size: 4
learning_rate: 1.0e-5
precision: "bf16-mixed"
accumulate_grad_batches: 8
max_epochs: 30
```

**davf_integration.yaml**:
```yaml
davf:
  state_space: "scvi_latent"    # scvi_latent 或 gene
  checkpoint_path: "checkpoints/latent_davf_ibd_norman/best_model.pt"
  feature_dim: 128
  hidden_dim: 256
  latent_dim: 10
  num_steps: 50
  freeze: true
```

---

## 11. 使用说明

### 11.1 CLI 训练

```bash
# 基础训练
python scripts/train.py --config configs/default.yaml

# 使用 Mamba 编码器
python scripts/train.py --config configs/mamba_small_gpu.yaml

# 使用预训练 ESM-2
python scripts/train_pretrained.py --model esm2_150M

# 使用 ESM-3
python scripts/train_pretrained.py --model esm3_sm_open

# Lightning 训练（支持多 GPU）
python scripts/train_lightning.py --config configs/lightning.yaml --devices 2
```

### 11.2 CLI 预测

```bash
# 单序列预测
python scripts/predict.py \\
  --checkpoint outputs/models/best_model.pt \\
  --sequence "MVLSPADKTN..." \\
  --ptm-sites '[{"position":3,"type":"phosphorylation","amino_acid":"L"}]'

# 批量预测
python scripts/batch_predict.py \\
  --checkpoint outputs/models/best_model.pt \\
  --input data/processed/ptm_integrated_human_labeled.csv \\
  --output outputs/results/predictions.csv
```

### 11.3 DAVF 微调

```bash
python scripts/finetune_davf_e2e.py \\
  --data data/processed/ptm_integrated_human_labeled.csv \\
  --checkpoint checkpoints/latent_davf_ibd_norman/best_model.pt \\
  --output outputs/davf_finetune \\
  --stage1-epochs 5 --stage2-epochs 15
```

### 11.4 API 服务

```bash
# 启动 API
python -m uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000

# 健康检查
curl http://localhost:8000/health

# 预测请求
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"sequence":"MVLSPAD...","ptm_sites":[],"use_davf":false}'
```

### 11.5 Python API

```python
from src.models.architectures import PTM2CellNet
from src.data.datasets import PTMDataset
from src.training.trainers import Trainer

# 构建模型
model = PTM2CellNet(
    encoder_type="transformer",
    embed_dim=128,
    num_classes=4,
)

# 加载数据
dataset = PTMDataset(csv_path="data/processed/train.csv")

# 训练
trainer = Trainer(model, config={"training": {"epochs": 50, "lr": 0.001}})
trainer.fit(dataset.train_dataloader(), dataset.val_dataloader())
```

---

## 12. 测试体系

### 12.1 测试分类

| 层级 | 数量 | 用途 | 运行命令 |
|------|------|------|---------|
| 单元测试 | 1531 | 模块级功能验证 | `pytest tests/unit -q` |
| 集成测试 | 67 | 模块间交互 | `pytest tests/integration -q` |
| E2E 测试 | 42 | CLI 端到端 | `pytest tests/e2e -q` |
| 真实资产 | 8 (gated) | 真实权重验收 | `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 pytest tests/real_assets/` |

### 12.2 质量保证

- **ruff**: 全项目代码风格检查（0 错误）
- **mypy**: 123 源文件类型检查（0 错误）
- **随机种子**: conftest.py 自动设置种子确保可复现
- **覆盖率**: scope.py / aa_constants / schemas 等关键模块有专用测试

---

## 13. 版本历史与规划

### 13.1 已关闭功能

| Feature | 版本 | 说明 |
|---------|------|------|
| ESM-3 集成 | v1.0 | ESM3Encoder 完整实现，支持多模态输入 |
| DAVF E2E 训练 | v1.0 | DAVFInferenceModule 两阶段微调，real-asset 通过 |
| scVI decode | v1.0 | ScVIAdapter 完整 encode→decode 管线 |
| 通路上下文映射 | v1.0 | PTMDirectionMapper 上下文感知映射 |
| PTM 类型扩展 | v1.0 | 29 种 PTM 类型，归一化支持 |

### 13.2 已取消功能

- 实时质谱流处理
- 自定义 PTM 数据库
- GUI 界面
- API key 功能扩展

---

**文档版本**: v1.0  
**最后更新**: 2026-07-06  
**维护者**: PTM2CellNet 团队
