# 安装指南

## 系统要求

- Python 3.10+
- CUDA 11.8+ (GPU训练可选)
- 内存: 8GB+ (推荐16GB+)
- 存储: 10GB+ (含数据文件)

## 快速安装

### 1. 克隆仓库

```bash
git clone <repository-url>
cd PTM2CellNet
```

### 2. 安装依赖（按需选择安装范围）

PTM2CellNet 把依赖按能力分组（见 `setup.py` 的 `extras_require`）。**推荐按需安装**，避免一次性拉取全部重量级依赖：

| 安装命令 | 适用场景 | 启用的能力 |
|---|---|---|
| `pip install -e .` | **最小核心** | 训练/推理主链路（CNN/LSTM/Transformer 编码器、细胞状态预测）。不含 API、Lightning、预训练模型。 |
| `pip install -r requirements.txt` | **核心 + 大部分能力** | 上面 + FastAPI、Lightning、ESM-2/ProtBERT、scVI/DAVF、Mamba fallback。**大多数用户选这个。** |
| `pip install -e ".[api]"` | 仅补充 API 服务 | FastAPI 推理服务 |
| `pip install -e ".[lightning]"` | 仅补充 Lightning 训练 | Lightning 训练框架 |
| `pip install -e ".[pretrained]"` | 仅补充预训练编码器 | ESM-2 / ProtBERT |
| `pip install -e ".[analysis]"` | 仅补充单细胞/通路分析 | scVI / DAVF 基因空间工作流（`scvi-tools>=1.2.0` + `anndata>=0.10,<0.12`） |
| `pip install -e ".[genki]"` | 仅补充 GenKI 图扰动 | `torch-geometric` + `scanpy` |
| `pip install -e ".[mamba]"` | 仅补充原生 Mamba / Lion | `mamba-ssm` + `lion-pytorch` |
| `pip install -e ".[all]"` | **全量能力** | 上述全部（不含 dev）。磁盘/网络充裕时使用。 |

> 💡 混合安装：可以同时选多个 extra，如 `pip install -e ".[api,lightning,pretrained]"`。

**可选能力降级行为**：未安装某个 extra 时，对应能力会优雅降级而非崩溃——

| 缺失依赖 | 降级行为 | 安装命令 |
|---|---|---|
| `lion-pytorch` | LightningModule 中 Lion 优化器回退到 AdamW | `pip install -e ".[mamba]"` |
| `mamba-ssm` | MambaEncoder 使用 fallback 实现 | `pip install -e ".[mamba]"` |
| `torch-geometric` | GenKI source backend 跳过 | `pip install -e ".[genki]"` |
| `scvi-tools` | scVI / DAVF 基因空间工作流不可用 | `pip install -e ".[analysis]"` |

CLI 入口在能力不可用时会打印带安装提示的警告，不会抛晦涩堆栈。

### 3. 验证安装

```bash
# 运行单元测试
pytest tests/unit/ -v

# 检查代码质量
flake8 src/ --max-line-length=120
mypy src/ --ignore-missing-imports

# （可选）验证 scVI 链路（需要 [analysis] extra）
python -c "import scvi; from scvi.model import SCVI; print('scvi ok', scvi.__version__)"
```

## Docker安装 (推荐用于生产环境)

```bash
# 构建镜像
docker compose build

# 启动API服务
docker compose up -d

# 验证服务（注意：就绪探针是 /api/v1/ready，不是 /health）
curl http://localhost:8000/api/v1/ready
```

## 数据准备

### 下载公开数据集

```bash
# 运行数据整合脚本
python scripts/integrate_data_v2.py --max-seq-len 1000

# 或使用完整数据 (需要UniProt API访问)
python scripts/integrate_data_v2.py --max-seq-len 1000 --fetch-api
```

### 数据目录结构

```
data/
├── raw/           # 原始数据 (从公开数据库下载)
├── processed/     # 处理后数据 (整合脚本输出)
└── external/      # 外部数据源
```

## GPU配置 (可选)

```bash
# 安装PyTorch GPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 原生 Mamba / Lion 的 GPU 架构注意点 (`[mamba]` extra)

`mamba-ssm` 与 `causal-conv1d` 包含需要针对**目标 GPU 的 compute capability (sm_xx)** 编译的 CUDA kernel。官方预编译 wheel 通常只覆盖 sm_70/75/80/86/89/90。

> ⚠️ **旧架构 GPU（如 Tesla P40 = sm_61、GTX 10 系列 = sm_61）运行预编译 wheel 会出现 `CUDA error: no kernel image is available for execution on the device`。** 此时必须从源码编译并显式指定 `TORCH_CUDA_ARCH_LIST`。

查询本机 GPU 架构：

```bash
python -c "import torch; print('compute capability:', torch.cuda.get_device_capability(0))"
```

针对 sm_61（Pascal）从源码编译示例（CUDA 11.8 / torch 2.4）：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# causal-conv1d：源码里硬编码了 gencode 列表，需打开 setup.py
# 在 cc_flag 中追加 arch=compute_61,code=sm_61，然后强制源码构建：
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -v --no-build-isolation .

# mamba-ssm：同样需要追加 sm_61 gencode，然后：
MAMBA_FORCE_BUILD=TRUE pip install -v --no-build-isolation .
```

> 说明：项目内置的 `src/models/mamba_encoder.py` 是**纯 PyTorch 的 SelectiveSSM 实现**，不依赖 `mamba-ssm` 也能训练/推理；`mamba-ssm` 仅在希望使用官方 fused kernel 时才需要。缺失时 LightningModule 的 Lion 优化器会回退到 AdamW（见上表）。

## CI / 测试分层 (可选能力)

仓库测试按能力分层，默认 CI 只跑核心，允许可选能力 skip：

| 命令 | 覆盖范围 | 是否要求可选依赖 |
|---|---|---|
| `pytest -m "not slow and not gpu"` | 核心训练/推理/E2E（≈1200 项） | 否（缺失则 skip） |
| `pip install -e ".[genki]" && pytest tests/integration/test_soft_perturbation_pipeline.py tests/integration/test_two_stage_pipeline.py tests/unit/integration/` | GenKI 图扰动 / 两阶段解释 | **是**，不允许 skip |
| `pip install -e ".[mamba]" && pytest tests/unit/test_mamba_encoder.py tests/integration/test_mamba_integration.py` | Mamba 编码器 | 纯 PyTorch fallback 无需；原生 kernel 需匹配 GPU 架构 |
| `pip install -e ".[analysis]" && pytest tests/unit/analysis/` | scVI / DAVF 基因空间 | 是 |

可选能力对应的 CLI（`run_ptm_virtual_perturbation.py`、`run_two_stage_explanation.py`）在依赖缺失时会通过 `require_extras` / `validate_runtime_ready` 快速失败，并打印 `pip install -e ".[genki]"` 提示。

## 常见问题

### Hugging Face模型下载慢

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 内存不足

```bash
# 减少batch_size
python scripts/train.py --batch_size 16

# 或使用更小的模型
python scripts/train.py --config configs/cnn_gated_fusion_112k.yaml
```
