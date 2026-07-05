# PTM2CellNet 当前代码 E2E 现状分析报告

> 分析目标：判断项目是否已经满足端到端（E2E）训练与推理的要求；如未达到，详细列出主要问题，并针对阻塞性/严重问题制定解决策略。
>
> 分析时间：2026-06-30
> 分析范围：`src/`、`scripts/`、`configs/`、`tests/`、`src/api/`、`Dockerfile`、`setup.py`

---

## 1. 执行摘要

**总体结论：项目尚未达到“开箱即用”的 E2E 训练与推理要求。**

- **训练链路**：核心训练入口 `scripts/train.py` 与 `scripts/train_lightning.py` 已能在示例数据上跑通完整训练、验证、保存、评估流程，具备基础 E2E 训练能力。
- **推理链路**：**存在严重阻塞性问题**。默认的 `scripts/predict.py` 与 `scripts/evaluate.py` 因 checkpoint 与默认 config 不匹配而直接失败；FastAPI 推理服务因 schema 前向引用问题在 Pydantic v1 环境下无法通过测试，且 Docker 部署配置错误。
- **辅助/扩展链路**：多任务训练脚本 `scripts/train_multitask_ptm.py` 存在缺失导入与数据 key 不匹配，无法运行；`scripts/train_pretrained.py` 存在编码器名称处理 bug；`scripts/train_binary.py` 测试评估逻辑不完整。
- **工程化**：`setup.py` 缺少 API 运行时依赖；`Dockerfile` 启动命令与 health check 路径错误；`src/utils/io.py` 的 `load_pickle` 白名单过于严格，无法加载真实对象。

简言之：**训练能跑，但推理和部署出箱即失败；多个关键脚本/配置存在 bug 或契约不一致。**

---

## 2. 现状评估

### 2.1 端到端训练能力

| 入口脚本 | 状态 | 说明 |
|---------|------|------|
| `scripts/train.py` | ✅ 可用 | 纯 PyTorch 训练入口，已验证 1 epoch 完整跑通。 |
| `scripts/train_lightning.py` | ✅ 可用 | Lightning 训练入口，已验证可进入训练循环。 |
| `scripts/train_ptm_site.py` | ⚠️ 基本可用 | 结构正确，默认数据存在，但未完整跑全量；CPU 下 `--precision 16-mixed` 可能失败。 |
| `scripts/train_pretrained.py` | ❌ 有 bug | 编码器名称被错误替换为 `esM2_8M`，导致模型构建失败。 |
| `scripts/train_multitask_ptm.py` | ❌ 不可用 | 缺少 `defaultdict` 导入，batch key 与 collate 不匹配。 |
| `scripts/train_binary.py` | ⚠️ 训练可用，测试坏 | 测试集只取首个 batch 且 label 收集错误，未真正保存测试指标。 |

**训练模块（`src/training/`）整体实现较完整**，包含：

- `Trainer` 训练/验证/预测循环
- `FocalLoss`、`DiceLoss`、`MultiTaskLoss`
- 优化器与学习率调度
- `ModelCheckpoint`、`EarlyStopping`、`LearningRateMonitor`、`ProgressBarCallback`
- `PTM2CellNetLightning` / `PTMSiteLightning` Lightning 模块
- LoRA/PEFT 支持

### 2.2 端到端推理能力

| 入口 | 状态 | 说明 |
|------|------|------|
| `scripts/predict.py` | ❌ 默认失败 | 默认 checkpoint（CNN/11 PTMs/长度500/2类）与默认 config（transformer/4 PTMs/长度1000/4类）state_dict 不匹配。 |
| `scripts/evaluate.py` | ❌ 默认失败 | 同 `predict.py`，state_dict 不匹配。 |
| `scripts/batch_predict.py` | ⚠️ 依赖同上 | 只要模型/配置匹配，理论上可用。 |
| `scripts/predict_variant_effect.py` | ⚠️ 可用性取决于初始化 | 已能 import 并显示 help。 |

**推理模型（`src/models/`）整体实现较完整**，包含：

- `PTM2CellNet`、`PTM2CellNetLarge`
- CNN / Transformer / LSTM / GRU / Mamba 编码器
- PTM 嵌入、PTM Attention、Gated Fusion
- 分类/回归/多任务预测器
- Pooling 层、预训练编码器封装（ESM-2/ProtBERT/ProtT5）

### 2.3 API 服务能力

| 能力 | 状态 | 说明 |
|------|------|------|
| `create_app()` 构建 | ✅ 可用 | FastAPI 应用可成功创建。 |
| `/api/v1/predict` | ⚠️ Pydantic v1 下失败 | `PredictionResponse` 使用前向引用未调用 `update_forward_refs()`，导致 5 个 API 测试失败。 |
| `/api/v1/batch_predict` | ⚠️ 同上 | 受 schema 问题影响。 |
| `/api/v1/health` | ✅ 可用 | 测试通过。 |
| 生产容器启动 | ❌ 不可用 | `Dockerfile` 使用 `uvicorn src.api.app:app`，但 `app.py` 导出的是 `create_app`；health check 路径错误。 |

### 2.4 测试覆盖

- 全量 `pytest -m "not slow and not gpu"`：**1142 passed, 5 failed, 4 skipped**（Pydantic v1 下）。
- 单元测试 `tests/unit/`：**1078 passed, 2 failed, 3 skipped**。
- 失败主要集中在：API schema 前向引用、GenKI-source 集成（缺少 `torch_geometric`）。

### 2.5 部署与工程化

- `setup.py` 缺少 `fastapi`、`uvicorn`、`pydantic` 等 API 依赖。
- `Dockerfile` 与 `docker-compose.yml` 命令与路径错误。
- `src/utils/io.py` 的 `load_pickle` 安全反序列化白名单过窄，无法加载 `torch.Tensor`、`pandas.DataFrame` 等真实对象。
- 多个默认配置与 checkpoint 不一致，导致推理出箱失败。

---

## 3. 主要问题清单（按严重程度）

### 3.1 阻塞性问题（导致核心 E2E 链路失败）

| 编号 | 问题 | 位置 | 影响 |
|------|------|------|------|
| B1 | 默认 checkpoint 与默认 config 不匹配，`predict.py` / `evaluate.py` 出箱失败 | `outputs/models/best_model.pt` vs `configs/default.yaml` | 推理链路完全不可用 |
| B2 | `PredictionResponse` 前向引用未解析，API 测试失败 | `src/api/schemas.py` | API 推理不可用（Pydantic v1 环境） |
| B3 | `Dockerfile` 启动命令错误，无法启动服务 | `Dockerfile:44` | 生产部署完全不可用 |
| B4 | `setup.py` 缺少 API/Web 依赖 | `setup.py` | 安装包后无法运行 API |
| B5 | `load_pickle` 白名单过严，无法加载真实对象 | `src/utils/io.py:140-197` | 模型/数据反序列化失败 |

### 3.2 严重问题（导致重要功能不可用或结果错误）

| 编号 | 问题 | 位置 | 影响 |
|------|------|------|------|
| S1 | `scripts/train_multitask_ptm.py` 缺失 `defaultdict` 导入，batch key 与 collate 不匹配 | `scripts/train_multitask_ptm.py` | 多任务训练脚本无法运行 |
| S2 | `scripts/train_pretrained.py` 编码器名称被错误替换 | `scripts/train_pretrained.py:137` | 预训练编码器训练入口失败 |
| S3 | `scripts/train_binary.py` 测试评估逻辑错误 | `scripts/train_binary.py` | 测试指标未真正计算/保存 |
| S4 | `scripts/validate_cptac.py` 缩进错误 | `scripts/validate_cptac.py:235` | 脚本无法 import/执行 |
| S5 | Docker/docker-compose health check 路径错误 | `Dockerfile:41`, `docker-compose.yml:24` | 容器健康检查失败 |
| S6 | `load_from_uniprot` 使用 POST form data，可能不符合 UniProt API 契约 | `src/data/loaders.py` | 外部数据加载不稳定 |
| S7 | `cross_validate` 未真正训练每个 fold | `src/evaluation/evaluators.py:170-284` | 功能名不副实，结果无意义 |
| S8 | `plot_training_curves` 与 Trainer 日志 key 不一致 | `src/evaluation/visualization.py:175-217` | 训练曲线 accuracy 子图为空 |

### 3.3 中等问题（影响稳定性、可维护性或部署体验）

| 编号 | 问题 | 位置 | 影响 |
|------|------|------|------|
| M1 | `/api/v1/initialize` 端点缺失，但启动日志误导用户 | `src/api/app.py:34` | 用户无法热加载模型 |
| M2 | `src/api/routes.py` 与 `src/api/routes/` 包重复，后者阴影前者 | `src/api/routes.py` | 代码漂移，维护风险 |
| M3 | CORS 默认拒绝所有跨域请求 | `src/api/app.py:105-118` | 浏览器端无法调用 API |
| M4 | `batch_predict` 对空列表返回 500 | `src/api/routes/predictions.py:228-284` | 异常处理不友好 |
| M5 | `logging.py` 不支持 JSON 格式 | `src/utils/logging.py` | `production.yaml` 配置无效 |
| M6 | `Config.__getitem__` 缺失时返回 `None` 而非 `KeyError` | `src/utils/config.py` | 违反 dict 契约，易隐藏错误 |
| M7 | `PTM2CellNet` 与 `PTM2CellNetLarge` 大量重复代码 | `src/models/architectures.py` | 维护负担 |
| M8 | 多个依赖未安装（uvicorn、torch_geometric、scvi-tools、lion_pytorch 等） | 环境 | 功能缺失或测试失败 |

---

## 4. 阻塞性/严重问题解决策略

### B1：默认 checkpoint 与默认 config 不匹配

**现象**

```text
RuntimeError: Error(s) in loading state_dict for PTM2CellNet:
  Missing key(s): "encoder.position_emb.weight", ...
  Unexpected key(s): "encoder.conv.weight", ...
  size mismatch for ptm_module.embedding.type_embedding.weight: [11,256] vs [6,256]
  size mismatch for predictor.classifier.weight: [2,256] vs [4,256]
```

**根因**

- `outputs/models/best_model.pt` 由 CNN encoder、11 种 PTM、max_length=500、2 分类训练得到。
- `configs/default.yaml` 配置为 Transformer encoder、4 种 PTM、max_length=1000、4 分类。
- 项目缺少“config + checkpoint 对应关系”的元数据记录。

**解决策略**

1. **记录并公开默认 checkpoint 的匹配 config**
   - 在 `outputs/models/` 下新增 `best_model.config.yaml`，与 `best_model.pt` 绑定。
   - 修改 `scripts/predict.py` 与 `scripts/evaluate.py`，优先从 checkpoint 同目录加载同名 `.config.yaml`；若不存在，再回退到 `--config` 参数。

2. **统一默认配置**
   - 将 `configs/default.yaml` 调整为与 `best_model.pt` 一致（CNN、11 PTMs、长度 500、2 类），或重新生成一个与当前 `default.yaml` 匹配的 checkpoint。
   - 推荐方案：保留 `default.yaml` 不变，新增 `configs/inference_default.yaml` 用于推理默认，避免影响训练默认行为。

3. **模型保存时写入配置指纹**
   - 在 `Trainer.save_checkpoint` 与 Lightning checkpoint 中保存 `model_config.yaml` 内容或哈希，加载时校验。
   - 实现 `src/utils/checkpoint_utils.py`：

     ```python
     def load_checkpoint_with_config(checkpoint_path: str, config_path: str = None):
         config_path = config_path or Path(checkpoint_path).with_suffix(".config.yaml")
         config = Config.from_yaml(config_path) if Path(config_path).exists() else None
         state = safe_torch_load(checkpoint_path)
         return state, config
     ```

4. **预测脚本改进报错信息**
   - 当 state_dict 不匹配时，给出清晰提示："Checkpoint 与 config 不匹配，请使用训练该 checkpoint 时的 config 文件，路径通常为 outputs/models/best_model.config.yaml"。

---

### B2：`PredictionResponse` 前向引用未解析

**现象**

Pydantic v1 下运行 API 测试失败：

```text
pydantic.errors.ConfigError: field "pathway_impacts" not yet prepared so type is still a ForwardRef,
you might need to call PredictionResponse.update_forward_refs().
```

**根因**

`src/api/schemas.py` 中 `PredictionResponse` 在 `PathwayImpact` 定义之前使用了 `Optional[List["PathwayImpact"]]`，且未调用 `update_forward_refs()`。

**解决策略**

1. **显式调用前向引用解析**
   在 `src/api/schemas.py` 文件末尾添加：

   ```python
   PredictionResponse.update_forward_refs()
   ```

2. **更健壮的做法：调整定义顺序**
   将 `PathwayImpact` 类定义移到 `PredictionResponse` 之前，避免前向引用。若存在循环依赖，则优先使用方案 1。

3. **CI 覆盖 Pydantic v1/v2**
   在 `.github/workflows` 中增加两个环境矩阵（Python 3.10 + Pydantic v1，Python 3.12 + Pydantic v2），避免回归。

---

### B3/B5：Docker 启动命令与 health check 错误

**现象**

- `Dockerfile`：`CMD ["uvicorn", "src.api.app:app", ...]` 失败，因为 `app.py` 导出的是 `create_app` 工厂函数。
- Health check：`http://localhost:8000/health` 404，实际端点为 `/api/v1/health`。

**解决策略**

1. **修正 Dockerfile**

   ```dockerfile
   # 方案 A：使用工厂
   CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

   # 方案 B：在 app.py 模块级别创建 app 实例（推荐，便于单进程直接导入）
   # app = create_app()
   # CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **修正 health check 路径**

   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
     CMD curl -f http://localhost:8000/api/v1/health || exit 1
   ```

   ```yaml
   # docker-compose.yml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
   ```

3. **统一入口**
   推荐在 `src/api/app.py` 模块末尾添加：

   ```python
   app = create_app()
   ```

   然后 Dockerfile 直接使用 `uvicorn src.api.app:app`。

---

### B4：`setup.py` 缺少 API/Web 依赖

**现象**

`setup.py` 的 `install_requires` 未包含 `fastapi`、`uvicorn`、`pydantic`，仅通过 `requirements.txt` 声明，导致 `pip install -e .` 后无法运行 API。

**解决策略**

1. **拆分 extras_require**

   ```python
   setup(
       ...,
       install_requires=[
           # 核心依赖（保留现有）
           "torch>=2.0.0",
           "pandas",
           "numpy",
           "pyyaml",
           ...
       ],
       extras_require={
           "api": ["fastapi>=0.100", "uvicorn>=0.15.0", "pydantic>=1.10,<3"],
           "lightning": ["pytorch-lightning>=2.0", "torchmetrics"],
           "pretrained": ["transformers", "biopython"],
           "all": ["fastapi>=0.100", "uvicorn>=0.15.0", "pydantic>=1.10,<3",
                   "pytorch-lightning>=2.0", "torchmetrics", "transformers", "biopython"],
       },
   )
   ```

2. **同步 `requirements.txt` 与 `setup.py`**
   - 保持 `requirements.txt` 为开发全量依赖。
   - 在 README 中说明：`pip install -e .[api]` 运行 API，`pip install -e .[all]` 安装全部功能。

---

### B5：`load_pickle` 白名单过严

**现象**

`load_pickle` 使用 `_SafeUnpickler`，白名单仅包含基础 builtins 与 `numpy.dtype/ndarray`，无法加载 `torch.Tensor`、`pandas.DataFrame`、项目自定义类。

**解决策略**

1. **提供两种模式**

   ```python
   def load_pickle(file_path: str, safe: bool = True):
       if not safe:
           with open(file_path, "rb") as f:
               return pickle.load(f)
       return _safe_load(file_path)
   ```

2. **扩大安全白名单**
   在 `_SafeUnpickler.find_class` 中加入项目允许的类型：

   ```python
   allowed_modules = {
       "numpy.core.multiarray": {"scalar", "_reconstruct"},
       "numpy": {"dtype", "ndarray"},
       "torch": {"Tensor"},
       "pandas.core.frame": {"DataFrame"},
       "pandas.core.series": {"Series"},
       # 如需加载项目自定义类，显式加入
       "src.data.schemas": {"PTMSite", "PTMRecord"},
   }
   ```

3. **模型加载走 `safe_torch_load`**
   已在 `src/utils/io.py` 实现 `safe_torch_load`，确保模型加载使用它而非 `load_pickle`。

---

### S1：`scripts/train_multitask_ptm.py` 无法运行

**现象**

- 缺失 `from collections import defaultdict`。
- 数据集返回 `label`，脚本访问 `batch['labels'][ptm_type]`。
- 未使用 `collate_multitask_batch`。

**解决策略**

1. **修复导入**

   ```python
   from collections import defaultdict
   ```

2. **统一 batch key 与 collate**
   - 确认 `MultiTaskPTMDataModule` 是否支持 `collate_fn=collate_multitask_batch`。
   - 在脚本中显式传入：

     ```python
     datamodule = MultiTaskPTMDataModule(..., collate_fn=collate_multitask_batch)
     ```

   - 脚本中统一使用 `batch["labels"]`，并确保数据集返回 `labels`（dict）。

3. **为 `MultiTaskPTMDataModule` 添加 collate_fn 参数**
   若当前实现未暴露该参数，应补充：

   ```python
   def train_dataloader(self):
       return DataLoader(self.train_dataset, ..., collate_fn=self.collate_fn)
   ```

4. **回归测试**
   添加一个最小化集成测试，使用 sample 数据运行 1 step，确保多任务训练脚本可执行。

---

### S2：`scripts/train_pretrained.py` 编码器名称 bug

**现象**

第 137 行将 `esm2_8M` 处理为 `esM2_8M`（`.replace("m", "M")` 过度替换），导致 `PTM2CellNet.from_config` 报“不支持的编码器类型”。

**解决策略**

1. **修复替换逻辑**

   ```python
   # 当前错误代码
   encoder_name = args.encoder.replace("m", "M")  # 会把 esm2_8M -> esM2_8M

   # 正确做法：使用映射表
   ENCODER_NAME_MAP = {
       "esm2_8M": "esm2_8M",
       "esm2_35M": "esm2_35M",
       "esm2_150M": "esm2_150M",
       "esm2_650M": "esm2_650M",
       "protbert": "protbert",
       "prott5": "prott5",
   }
   encoder_name = ENCODER_NAME_MAP.get(args.encoder, args.encoder)
   ```

2. **统一 pretrained 配置命名**
   在 `src/models/pretrained_encoders.py` 或 `src/models/model_utils.py` 中定义标准名称常量，避免脚本层做字符串转换。

---

### S3：`scripts/train_binary.py` 测试评估逻辑错误

**现象**

测试集循环中 `break` 导致只评估第一个 batch，且 label 收集错误。

**解决策略**

1. **移除测试循环中的 `break`**
2. **正确收集所有 batch 的预测与标签**

   ```python
   all_preds, all_labels = [], []
   model.eval()
   with torch.no_grad():
       for batch in test_loader:
           logits = model(batch)
           preds = torch.argmax(logits, dim=-1)
           all_preds.extend(preds.cpu().tolist())
           all_labels.extend(batch["label"].cpu().tolist())

   test_metrics = calculate_metrics(all_labels, all_preds)
   save_json(test_metrics, output_dir / "test_metrics.json")
   ```

3. **补充回归测试**
   在 `tests/integration/` 增加一个二进制训练端到端测试（1 epoch）。

---

### S4：`scripts/validate_cptac.py` 缩进错误

**现象**

第 235 行 `from src.utils.io import safe_torch_load` 缩进错误，位于方法内部却写成模块级。

**解决策略**

1. **修正缩进**

   ```python
   def _load_models(self):
       """加载PTM预测模型"""
       import torch
       from src.utils.io import safe_torch_load
       sys.path.insert(0, str(self.model_dir.parent.parent))
       from src.models.ptm_site_predictor import PTMSitePredictor
   ```

2. **添加 import 检查到 CI**
   运行 `python -m py_compile scripts/*.py` 作为基础语法检查。

---

### S6：`load_from_uniprot` POST 契约风险

**现象**

使用 `requests.post(url, data=params)` 向 UniProt stream 提交查询，可能不符合其预期（通常应为 query string）。

**解决策略**

1. **将参数放入 query string**

   ```python
   response = requests.get(
       "https://rest.uniprot.org/uniprotkb/stream",
       params={
           "query": " OR ".join(f"accession:{acc}" for acc in accession_ids),
           "format": "json",
           "fields": "accession,sequence,gene",
       },
   )
   ```

2. **增加重试与缓存**
   使用 `requests.adapters.HTTPAdapter` 添加指数退避重试，避免网络抖动导致数据为空。

3. **明确错误日志**
   当返回非 200 时，记录 status code、response text 前 200 字符，便于排查。

---

### S7：`cross_validate` 未真正训练

**现象**

`Evaluator.cross_validate` 创建了 `train_loader` 但未使用，仅初始化模型后直接评估，名不副实。

**解决策略**

1. **两种 API 设计选择**

   - **方案 A：保留 cross_validate 名称，实现完整训练**

     ```python
     def cross_validate(self, model_class, dataset, n_splits=5, train_fn=None):
         for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
             model = model_class(...)
             train_loader = DataLoader(Subset(dataset, train_idx), ...)
             val_loader = DataLoader(Subset(dataset, val_idx), ...)
             train_fn(model, train_loader)  # 用户传入训练函数
             metrics = self.evaluate(model, val_loader)
             fold_metrics.append(metrics)
         return aggregate(fold_metrics)
     ```

   - **方案 B：改名并保留轻量评估**
     若当前行为是故意的（仅做 fold 划分评估），应改名为 `fold_evaluation` 或 `stratified_evaluation`，并移除未使用的 `train_loader`。

2. **推荐方案 A**，因为函数名强烈暗示会训练。需要引入 `train_fn` 回调，避免 `Evaluator` 引入训练依赖。

---

### S8：`plot_training_curves` 与 Trainer 日志 key 不一致

**现象**

`plot_training_curves` 期望 `train_accuracy`/`val_accuracy`，但 Trainer 只记录 `train_loss`/`val_loss`，导致 accuracy 子图空并抛出 legend 警告。

**解决策略**

1. **让 Trainer 同时记录 accuracy**
   在 `Trainer._train_epoch` 与 `Trainer._validate_epoch` 中计算并记录：

   ```python
   logs["train_accuracy"] = accuracy
   logs["val_accuracy"] = accuracy
   ```

2. **让可视化函数兼容缺失 key**

   ```python
   def plot_training_curves(logs, save_path):
       available_keys = [k for k in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"] if k in logs]
       ...
   ```

---

## 5. 依赖与环境问题

### 5.1 当前缺失的关键依赖

| 依赖 | 影响 | 建议安装方式 |
|------|------|-------------|
| `uvicorn` | API 服务无法启动 | `pip install uvicorn>=0.15.0` |
| `fastapi` / `pydantic` | API 不可用 | 加入 `setup.py` extras_require["api"] |
| `torch_geometric` | GenKI-source 集成测试失败 | 安装对应 CUDA/CPU 版本；或标记为可选依赖 |
| `scvi-tools` / `setuptools` | scVI 功能被禁用 | `pip install setuptools>=68.0 scvi-tools>=1.0.0` |
| `biopython` | FASTA 加载、UniProt 解析 | 加入 requirements |
| `mamba_ssm` | Mamba 编码器高性能实现 | 可选，否则回退到纯 Python 实现 |

### 5.2 建议行动

1. **创建 `requirements/` 目录**，拆分依赖：
   - `requirements/base.txt`
   - `requirements/api.txt`
   - `requirements/lightning.txt`
   - `requirements/pretrained.txt`
   - `requirements/dev.txt`
2. **更新 `setup.py`**，使用 `extras_require` 指向上述文件。
3. **在 README 中明确不同使用场景的pip 命令**。

---

## 6. 建议的修复优先级与验证清单

### Phase 1：打通推理链路（最高优先级）

- [ ] 为 `outputs/models/best_model.pt` 生成并提交匹配的 `best_model.config.yaml`。
- [ ] 修改 `scripts/predict.py` / `scripts/evaluate.py`，优先加载 checkpoint 同目录 config。
- [ ] 修复 `src/api/schemas.py` 前向引用问题，并增加 Pydantic v1/v2 CI 矩阵。
- [ ] 修复 `Dockerfile` 启动命令与 health check 路径。
- [ ] 在 `src/api/app.py` 模块级别创建 `app = create_app()`。

### Phase 2：修复损坏的训练脚本

- [ ] 修复 `scripts/train_multitask_ptm.py`（导入、key、collate_fn）。
- [ ] 修复 `scripts/train_pretrained.py` 编码器名称映射。
- [ ] 修复 `scripts/train_binary.py` 测试评估循环。
- [ ] 修复 `scripts/validate_cptac.py` 缩进错误。

### Phase 3：工程化与健壮性

- [ ] 放宽或分模式实现 `src/utils/io.py` 的 `load_pickle`。
- [ ] 更新 `setup.py` 依赖与 extras_require。
- [ ] 修复 `cross_validate` 行为或改名。
- [ ] 修复 `plot_training_curves` key 兼容问题。
- [ ] 改善 `load_from_uniprot` 请求方式与错误处理。
- [ ] 清理 `src/api/routes.py` 与 `src/api/routes/` 重复问题。
- [ ] 处理 CORS 默认配置与 `batch_predict` 空列表校验。

### Phase 4：环境与 CI

- [ ] 补齐缺失依赖或将其标记为可选。
- [ ] 增加 `python -m py_compile scripts/*.py` 到 CI。
- [ ] 增加最小 E2E 测试：训练 1 epoch → 预测 → 评估。

---

## 7. 结论

PTM2CellNet 在**模型架构、数据管道、核心训练器**方面已经具备较完整的能力，`scripts/train.py` 与 `scripts/train_lightning.py` 可完成端到端训练。然而，项目当前存在多个**阻塞性与严重性缺陷**，导致：

1. **默认推理出箱失败**（checkpoint/config 不匹配）。
2. **API 服务与容器部署无法直接工作**（schema 前向引用、Docker 命令错误、依赖缺失）。
3. **多个训练/评估脚本存在 bug**（多任务、预训练、二分类测试、CPTAC 验证）。
4. **工程化依赖与部署配置不一致**（`setup.py`、Docker、pickle 安全策略）。

只要按第 4 章策略依次修复 Phase 1 与 Phase 2 的问题，项目即可达到“训练 + 推理 + API 部署”的完整 E2E 可用状态。建议优先修复推理链路，因为训练已可跑通，而推理与部署是当前最明显的短板。

---

## 附录：关键文件路径速查

| 用途 | 路径 |
|------|------|
| 默认训练入口 | `scripts/train.py` |
| Lightning 训练入口 | `scripts/train_lightning.py` |
| 推理入口 | `scripts/predict.py` |
| 评估入口 | `scripts/evaluate.py` |
| 多任务训练 | `scripts/train_multitask_ptm.py` |
| 预训练编码器训练 | `scripts/train_pretrained.py` |
| 二分类训练 | `scripts/train_binary.py` |
| 主模型架构 | `src/models/architectures.py` |
| 数据模块 | `src/data/loaders.py`, `src/data/preprocess.py`, `src/data/datasets.py`, `src/data/features.py` |
| 训练模块 | `src/training/trainers.py`, `src/training/lightning_module.py` |
| 评估模块 | `src/evaluation/evaluators.py`, `src/evaluation/visualization.py` |
| API 入口 | `src/api/app.py`, `src/api/schemas.py`, `src/api/routes/predictions.py` |
| 工具模块 | `src/utils/config.py`, `src/utils/io.py`, `src/utils/logging.py` |
| 部署配置 | `Dockerfile`, `docker-compose.yml`, `configs/production.yaml` |
| 安装配置 | `setup.py`, `requirements.txt` |
