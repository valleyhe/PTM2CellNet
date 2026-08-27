# PTM2CellNet

基于深度学习的蛋白质翻译后修饰(PTM)分析与细胞状态预测系统。

> ⚠️ **DEMO 模型说明**：仓库自带的 `outputs/models/best_model.pt` 是在**随机生成的合成数据**上训练的冒烟测试模型，仅用于验证训练/推理链路可跑通，**不代表真实生物学预测能力**（评估准确率 ≈ 随机基线）。请勿直接用于真实预测。真实数据准备方式见 [数据接入指南](docs/guides/data_integration.md)，模型性质详情见 [outputs/models/README.md](outputs/models/README.md)。

## Quick Start（快速开始）

以下命令链从零开始运行完整的训练→推理流程（约 5 分钟，CPU 可用）：

### 1. 训练（smoke 测试模式）

```bash
python scripts/train.py \
    --config configs/smoke/cnn_cpu.yaml \
    --data data/raw/sample_data.csv \
    --output outputs/quickstart
```

### 2. 单样本推理

```bash
python scripts/predict.py \
    --model outputs/quickstart/models/best_model.pt \
    --sequence "ACDEFGHIKLMNPQRSTVWY" \
    --output outputs/quickstart/pred.csv
```

### 3. 带 PTM 位点的推理

```bash
python scripts/predict.py \
    --model outputs/quickstart/models/best_model.pt \
    --sequence "ACDEFGHIKLMNPQRSTVWY" \
    --ptm-sites '[{"position":3,"type":"phosphorylation"}]' \
    --output outputs/quickstart/pred_ptm.csv
```

### 4. 批量 CSV 推理

```bash
python scripts/predict.py \
    --model outputs/quickstart/models/best_model.pt \
    --input data/raw/sample_data.csv \
    --output outputs/quickstart/batch_pred.csv
```

### 5. API 服务

```bash
# development 模式启动（无需 API key）
PTM2CELLNET_ENV=development python -c "
from src.api.app import create_app
import uvicorn
uvicorn.run(create_app(), host='0.0.0.0', port=8000)
"

# 初始化模型
curl -X POST http://localhost:8000/api/v1/initialize \
    -H "Content-Type: application/json" \
    -d '{"checkpoint_path": "outputs/quickstart/models/best_model.pt"}'

# 预测
curl -X POST http://localhost:8000/api/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'
```

### 6. 跨尺度离线 smoke（需要版本化 NPZ 数据）

跨尺度模型是 opt-in 路径，训练和批量推理共用
`ptm2cellnet.cross-scale.npz.v1` 数据契约；它不替换标准模型/API。先用
`docs/guides/real_assets_acceptance.md` 中的 pLM 与 manifest 门禁确认资产，再运行：

```bash
python scripts/train_cross_scale.py \
    --config configs/cross_scale/smoke.yaml \
    --train-data /path/to/train.npz \
    --val-data /path/to/val.npz \
    --test-data /path/to/test.npz \
    --output outputs/cross_scale_smoke

python scripts/predict_cross_scale.py \
    --artifact outputs/cross_scale_smoke \
    --data /path/to/test.npz \
    --output outputs/cross_scale_smoke/predictions.npz
```

仓库中的跨尺度测试使用 synthetic fixtures，仅证明工程契约、checkpoint
round-trip 和批量输出可运行，不代表真实扰动数据或生物学效果已验收。

### 运行全部测试

```bash
# 单元测试
python -m pytest tests/unit/ -v

# 端到端测试（约 10 分钟）
python -m pytest tests/e2e/ -v --timeout=600
```

> **提示**: 首次运行推荐使用 `configs/smoke/` 下的配置（小模型、CPU、1 epoch），验证安装无误后再切换到研究配置。

---

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
│   ├── predict*.py             # 推理入口（细胞状态推理、PTM位点推理、变异效应）
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

依赖按能力分组，默认只安装最小核心，避免在干净 CPU 环境拉取重型可选依赖（详见 [安装指南](docs/guides/installation.md)）：

```bash
# 最小核心（推荐首次安装，仅训练/推理/API 主链路，CPU 友好）
pip install -r requirements-core.txt

# 按能力叠加（在 core 之上）
pip install -r requirements-pretrained.txt   # + ESM-2 / ProtBERT
pip install -r requirements-mamba.txt        # + Mamba 编码器（需 CUDA）
pip install -r requirements-analysis.txt     # + scVI / sspa 分析

# 核心 + 预训练能力（requirements.txt 的组合，磁盘/网络充裕时）
pip install -r requirements.txt

# 需要完整可选能力时按 extra 组合安装，例如 API + Lightning/预训练模型：
pip install -e ".[api,pretrained]"
```

> 未安装某个可选依赖时，对应能力（Lion 优化器、原生 Mamba、GenKI 图扰动、scVI 基因空间工作流）会**优雅降级**而非崩溃，并在 CLI 入口打印带 `pip install -e ".[<extra>]"` 提示的警告。

> **可复现钉扎**：唯一权威钉扎源是 `requirements-lock.txt`（`pkg==version` 全量钉扎；scgpt 等互斥依赖按 TD-M05 决策不进 lock，见文件内注释）。仓库不使用 `uv.lock`（TD-N-27，2026-08-24 移除空壳文件）。lock 与各 requirements 轨的一致性由 `scripts/check_requirements_consistency.py` 在 CI `dependencies` job 中门禁。

### 外部工具（非 pip 包）

`scripts/homology_split.py` 使用 **MMseqs2** 进行蛋白质同源聚类，它是一个外部二进制工具而非 pip 包，不包含在上述 requirements 中。安装方式：

```bash
# conda（推荐，跨平台）
conda install -c conda-forge -c bioconda mmseqs2

# 或手动下载安装（Linux/macOS）
# 见 https://github.com/soedinglab/MMseqs2/releases
```

安装后确保 `mmseqs` 可执行文件在 `PATH` 中。仅在运行同源拆分脚本时需要；常规训练/推理/API 不依赖它。

ESM-2 等预训练模型需要 Hugging Face 访问，建议设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

### 数据准备

```bash
# 以下脚本生成的是随机合成数据，仅用于冒烟测试。
# 真实数据格式与采集方式见 docs/guides/data_integration.md。
python scripts/prepare_data.py
python scripts/prepare_ptm_data.py
```

> 📝 默认 `prepare_data.py` 产出 500 条随机序列 + 4 类随机标签，目的是让训练/推理链路立即可跑。要在真实数据上训练，请准备符合 [输入格式规范](docs/guides/data_integration.md#输入数据格式) 的 CSV。

### 训练

配置按用途分类（P1-3）：
- `configs/smoke/cnn_cpu.yaml`、`configs/smoke/lightning_cnn_cpu.yaml` —— **首次 E2E 验证入口**（小型 CNN、CPU、1 epoch，几十秒跑完，产物自动标记为 `model_kind=demo`）。
- `configs/research/transformer.yaml` —— 真实/研究训练（transformer、GPU、完整 epoch）。
- `configs/production.yaml` —— API/部署服务端配置。

```bash
# 首次 E2E 验证（smoke，推荐）——原生训练脚本
python scripts/train.py --config configs/smoke/cnn_cpu.yaml \
    --data data/raw/sample_data.csv --output outputs/smoke

# Lightning 路径 smoke
python scripts/train_lightning.py --config configs/smoke/lightning_cnn_cpu.yaml \
    --data data/raw/sample_data.csv

# 真实数据研究训练
python scripts/train_lightning.py --config configs/research/transformer.yaml \
    --data data/processed/your_dataset.csv
```

### 推理

```bash
# 单条细胞状态预测（消费训练产物 best_model.pt + sibling config）
python scripts/predict.py --model outputs/smoke/models/best_model.pt \
    --sequence ACDEFGHIKLMNPQRSTVWY --device cpu
```

# PTM 位点预测
python scripts/train_ptm_site.py

# 多任务联合训练
python scripts/train_multitask_ptm.py
```

### 推理

```bash
# 单条细胞状态预测
python scripts/predict.py

# PTM位点批量预测（规范脚本名）
python scripts/predict_ptm_sites.py

# 兼容旧脚本名（执行 PTM 位点预测，非细胞状态批量推理）
# 细胞状态批量推理请改用: python scripts/predict.py --input <samples.csv>
python scripts/batch_predict.py

# 变异效应预测
python scripts/predict_variant_effect.py --variant "P15056:600:V:E"
# 注: BRAF 对应 UniProt accession P15056；格式为 UniProtID:Position:RefAA:AltAA
```

### 脚本说明

- `scripts/predict.py`: PTM2CellNet 细胞状态推理入口，可处理单条序列或 CSV 输入。
- `scripts/predict_ptm_sites.py`: 批量 PTM 位点预测与 PTM 变异效应分析入口。
- `scripts/batch_predict.py`: `predict_ptm_sites.py` 的兼容包装脚本，仅做 PTM 位点预测（非细胞状态）。注意与 API `/batch_predict` 端点（细胞状态批量推理，见 `src/api/routes/predictions.py`）语义不同，不要混用。

### 虚拟扰动与解释

```bash
# PTM-aware 虚拟扰动
python scripts/run_ptm_virtual_perturbation.py --gene TP53 --output outputs/

# 两阶段筛选与解释
python scripts/run_two_stage_explanation.py \
  --input data/processed/candidates.csv \
  --output outputs/
```

> 通路数据库缓存：`PathwayDatabaseIntegration` 默认将 KEGG/Reactome
> 缓存写入项目根下的绝对路径 `.cache/pathway_cache`（与进程工作目录无关）。
> 可在构造时通过 `cache_dir` 参数覆盖。如需接入真实 PPI 拓扑，将
> STRING/BioGRID 预处理边表（列 `gene_a,gene_b,score,type`）以
> `string_edges.tsv` / `biogrid_edges.tsv` / `ppi_edges.tsv` 之一放入该目录，
> `build_pathway_graph` 即会改用真实互作边（`weight=score`）替代完全图近似。

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
