# PTM2CellNet 当前代码 E2E 训练与推理就绪性复核报告

> 分析日期：2026-07-02  
> 分析目标：基于当前工作树重新判断项目是否满足端到端（E2E）训练和推理要求；若尚未完全满足，列出主要问题，并对阻塞性/高严重问题制定解决策略。  
> 分析范围：`src/`、`scripts/`、`configs/`、`tests/`、`src/api/`、`Dockerfile`、`docker-compose.yml`、`requirements.txt`、`setup.py`、`outputs/models/best_model.pt`。  
> 取证方式：静态代码核对 + 默认 CLI 推理/评估冒烟测试 + E2E pytest + 全量非 slow/gpu pytest + API readiness 检查 + 可选依赖探测。

---

## 1. 总体结论

**当前代码已经满足“核心细胞状态模型”的基础 E2E 训练、CLI 推理和默认评估要求，但尚未达到完整产品级 E2E 就绪。**

可以认为已经达标的部分：

- `scripts/train.py` 能完成数据加载、预处理、标签映射持久化、模型训练、checkpoint/config/manifest 保存、测试集评估和训练曲线输出。
- `scripts/predict.py` 能自动使用 checkpoint 同目录的 `best_model.config.yaml`，避免模型权重和默认配置漂移；单样本和 CSV 批量推理均具备真实 batch 推理路径。
- `scripts/evaluate.py --model outputs/models/best_model.pt` 当前已可跑通，能生成 `evaluation_metrics.json`、ROC、PR 和混淆矩阵图。
- 仓库内默认发布模型 `outputs/models/best_model.pt` 已更新为 4 类 demo 模型，并绑定了 `outputs/models/best_model.config.yaml`，默认推理/默认评估不再复现旧的 2 类/4 类标签不一致问题。
- `tests/e2e/test_train_predict_cli.py` 全部通过，覆盖原生训练到推理、Lightning checkpoint 合约、PTM 位点输入、API 初始化后预测等关键闭环。

尚未完全达标的部分：

- 全量非 slow/gpu 测试仍未绿灯：`1181 passed / 3 failed / 11 skipped`。失败集中在 API 变异预测状态隔离和 `scripts/predict.py` 内部 helper 签名兼容性。
- Docker 部署配置仍存在严重问题：应用自动加载模型读取 `PTM2CELLNET_CHECKPOINT`，但 `Dockerfile` 和 `docker-compose.yml` 设置的是 `MODEL_PATH`；容器默认不会自动加载模型。`Dockerfile` healthcheck 使用 `/api/v1/ready` 会因此失败，`docker-compose.yml` healthcheck 使用 `/api/v1/health` 又会把未加载模型的实例误判为健康。
- GenKI source / 图扰动能力在当前环境不可运行：`torch_geometric: False`。相关测试现在会 skip，不再导致 CI 失败，但这只证明“优雅降级”，不证明该能力 E2E 可用。
- `mamba_ssm` 和 `lion_pytorch` 当前环境缺失；Mamba/Lion 属于可选能力，当前核心链路不依赖它们，但不能宣称全功能就绪。
- 默认模型是随机合成数据训练出的 demo/smoke 模型，只能证明工程链路可跑，不具备真实生物学预测能力。

因此，本项目当前状态应表述为：

> **核心 E2E 训练与 CLI 推理链路已基本可用；API 在线部署、变异预测状态管理、容器自动加载与可选高级分析能力仍存在阻塞或高风险问题。**

---

## 2. 当前验证证据

### 2.1 默认 CLI 单样本推理

执行：

```bash
python scripts/predict.py \
  --model outputs/models/best_model.pt \
  --sequence ACDEFGHIKLMNPQRSTVWY \
  --output /tmp/ptm2cellnet_predict_check.csv \
  --device cpu
```

结果：

- 成功自动使用 `outputs/models/best_model.config.yaml`。
- 成功加载 `PTM2CellNet` 模型。
- 输出 4 类概率：`apoptosis`、`differentiation`、`proliferation`、`quiescence`。
- 成功写出 `/tmp/ptm2cellnet_predict_check.csv`，文件大小 268 bytes。

结论：**默认发布模型的 CLI 单样本推理可用。**

### 2.2 默认模型评估

执行：

```bash
python scripts/evaluate.py \
  --model outputs/models/best_model.pt \
  --output /tmp/ptm2cellnet_eval_check \
  --device cpu
```

结果：

- 成功自动使用 `outputs/models/best_model.config.yaml`。
- 成功生成 200 条示例数据并完成预处理/划分。
- 成功加载模型并评估。
- 输出指标示例：`accuracy=0.2667`、`auc_roc=0.5528`、`auc_pr=0.3669`。
- 成功写出：
  - `/tmp/ptm2cellnet_eval_check/evaluation_metrics.json`
  - `/tmp/ptm2cellnet_eval_check/roc_curve.png`
  - `/tmp/ptm2cellnet_eval_check/pr_curve.png`
  - `/tmp/ptm2cellnet_eval_check/confusion_matrix.png`

结论：**旧报告中的“默认发布模型评估失败”问题当前已修复。**

### 2.3 训练到推理 E2E 测试

执行：

```bash
python -m pytest tests/e2e/test_train_predict_cli.py -q
```

结果：

```text
15 passed, 2 warnings
```

覆盖要点：

- 原生 `scripts/train.py` 训练小模型并生成 `best_model.pt`。
- 训练产物生成 `best_model.config.yaml`，并持久化 `data.cell_states` 和 `data.label_to_idx`。
- `scripts/predict.py` 不依赖 hardcoded demo fallback。
- 单样本和批量推理都输出 `prob_<state>` 概率列。
- `--ptm-sites` 和 CSV `ptm_sites` 列会被解析并进入推理预处理。
- Lightning checkpoint 可剥离 `model.` 前缀并供裸模型加载。
- API `/api/v1/initialize` 后可以调用 `/api/v1/predict`。

结论：**核心 E2E 训练 -> 推理闭环已有自动化覆盖且通过。**

### 2.4 全量非 slow/gpu 测试

执行：

```bash
python -m pytest -m "not slow and not gpu" -q
```

结果：

```text
3 failed, 1181 passed, 11 skipped, 40 warnings
```

失败项：

1. `tests/unit/test_api.py::TestVariantAPI::test_predict_variant_not_initialized`
2. `tests/unit/test_api_routes.py::TestVariantPredictRoute::test_variant_not_initialized_returns_503`
3. `tests/unit/test_scripts.py::TestPredictScript::test_predict_in_batches`

结论：**核心训练/推理 E2E 通过，但全仓库质量门禁未通过，不能宣称产品级就绪。**

### 2.5 API 未加载模型时的状态

执行：

```python
from fastapi.testclient import TestClient
from src.api.app import create_app

client = TestClient(create_app())
for path in ["/api/v1/health", "/api/v1/live", "/api/v1/ready", "/api/v1/model/info"]:
    r = client.get(path)
    print(path, r.status_code, r.text[:200])
```

结果：

```text
/api/v1/health 200 {"status":"unhealthy","version":"1.0.0","model_loaded":false,...}
/api/v1/live   200 {"status":"alive"}
/api/v1/ready  503 {"detail":"模型未初始化"}
/api/v1/model/info 503 {"detail":"模型未初始化"}
```

结论：

- API 端点语义本身是清晰的：`/live` 判断进程存活，`/ready` 判断模型是否已加载，`/health` 是兼容端点。
- 但部署配置没有完全按该语义落地，详见 P0-2。

### 2.6 可选依赖现状

当前环境探测：

```text
torch_geometric: False
scvi: True
pkg_resources: True
mamba_ssm: False
lion_pytorch: False
lightning: True
transformers: True
fastapi: True
```

结论：

- scVI 相关导入风险较旧版本已有改善，`pkg_resources` 当前可用。
- GenKI source backend、原生 Mamba、Lion optimizer 仍不可作为当前环境已验证能力。

---

## 3. 当前能力矩阵

| 能力 | 当前状态 | 证据 | 结论 |
|---|---:|---|---|
| 原生训练 CLI | 通过 | E2E 测试训练小模型并生成产物 | 核心可用 |
| 训练标签映射持久化 | 通过 | `best_model.config.yaml` / E2E 测试 | 核心可用 |
| 默认单样本 CLI 推理 | 通过 | 实测 `scripts/predict.py` | 核心可用 |
| 默认 CSV 批量 CLI 推理 | 通过 | E2E 测试覆盖 | 核心可用 |
| 默认模型评估 | 通过 | 实测 `scripts/evaluate.py` | 核心可用 |
| Lightning checkpoint 推理合约 | 通过 | E2E 测试覆盖 | 基础可用 |
| API `/initialize` 后 `/predict` | 通过 | E2E 测试覆盖 | 基础可用 |
| API 未加载模型 readiness | 通过 | `/ready` 返回 503 | 端点语义正确 |
| Docker 自动模型加载 | 失败/高风险 | env 名不匹配 | 阻塞部署就绪 |
| docker-compose 健康检查 | 失败/高风险 | 使用 `/health`，未加载模型也 200 | 阻塞部署就绪 |
| 变异预测未初始化行为 | 失败 | 2 个测试失败 | 高风险 |
| 全量非 slow/gpu pytest | 失败 | 3 failed | 未达产品级质量门禁 |
| GenKI source backend | 当前环境不可用 | `torch_geometric: False`，相关测试 skip | 可选能力未就绪 |
| Mamba/Lion | 当前环境不可用 | `mamba_ssm: False`、`lion_pytorch: False` | 可选能力未就绪 |
| 真实生物学模型能力 | 未证明 | 默认模型 README 声明为 demo | 不能宣称业务预测就绪 |

---

## 4. 阻塞性与高严重问题

### P0-1：全量非 slow/gpu 测试仍失败

严重程度：P0  
影响范围：质量门禁、CI、发布可信度  
当前状态：未解决

证据：

```text
3 failed, 1181 passed, 11 skipped, 40 warnings
```

失败包括：

- 变异预测未初始化时应返回 503，但实际返回 200。
- `scripts.predict._predict_in_batches()` 的测试仍按旧签名接收 2 个返回值，当前实现要求额外 `cell_states` 参数并返回 3 个值。

影响：

- 项目不能宣称全量测试通过。
- API 变异预测状态存在跨测试/跨请求污染风险。
- 内部 helper 接口变更没有同步测试或兼容层，说明脚本 API 合约仍不稳定。

解决策略：

1. 修复 API 全局状态隔离：
   - 在 `src/api/routes/state.py` 增加显式 `reset_state()` 或 `clear_variant_workflow()`，供测试和初始化流程使用。
   - 在 `initialize_model()` 中不要隐式保留旧的 `STATE.variant_workflow`，除非调用方明确要求保留；更安全的默认是加载新模型时清空变异 workflow 和 pathway mapper。
   - 在 `tests/unit/test_api.py` 增加 `autouse` fixture，像 `tests/unit/test_api_routes.py` 一样完整保存/恢复 `STATE`，避免测试顺序影响结果。

2. 修复变异预测未初始化语义：
   - `predict_variant()` 当前代码已经在 `STATE.variant_workflow is None` 时返回 503；失败更可能来自全局 `STATE.variant_workflow` 被前序测试或 fixture 残留。
   - 验收时需要单独运行失败用例和全量测试，确保不是只在单测隔离下通过。

3. 修复 `_predict_in_batches()` 合约漂移：
   - 方案 A（推荐）：保留当前生产需要的 `probabilities_per_row` 返回值，同时给 `cell_states` 设置默认值，例如缺省时生成 `class_0...class_n`，并保持返回三元组；同步更新旧测试。
   - 方案 B：拆分为两个函数：低层 `_predict_indices_in_batches()` 返回 `(predictions, confidences)`，高层 `_predict_with_prob_columns_in_batches()` 返回三元组。这样测试和生产输出职责更清楚。
   - 方案 C：只更新测试为当前三元组签名。该方案成本最低，但不能解决内部 API 兼容性问题。

推荐验收命令：

```bash
python -m pytest \
  tests/unit/test_api.py::TestVariantAPI::test_predict_variant_not_initialized \
  tests/unit/test_api_routes.py::TestVariantPredictRoute::test_variant_not_initialized_returns_503 \
  tests/unit/test_scripts.py::TestPredictScript::test_predict_in_batches -q

python -m pytest -m "not slow and not gpu" -q
```

---

### P0-2：Docker 部署无法可靠进入 ready 状态

严重程度：P0  
影响范围：容器部署、服务健康检查、线上流量接入  
当前状态：未解决

证据：

- `src/api/app.py` 自动加载模型读取：

```python
checkpoint_path = os.environ.get("PTM2CELLNET_CHECKPOINT", "")
config_path = os.environ.get("PTM2CELLNET_CONFIG", "configs/production.yaml")
```

- `Dockerfile` 设置的是：

```dockerfile
ENV ... MODEL_PATH=/app/outputs/models/best_model.pt
```

- `docker-compose.yml` 设置的是：

```yaml
- MODEL_PATH=/app/outputs/models/best_model.pt
```

- `Dockerfile` healthcheck 使用 `/api/v1/ready`，未加载模型时返回 503。
- `docker-compose.yml` healthcheck 使用 `/api/v1/health`，未加载模型时仍 HTTP 200，只在 body 中标记 `unhealthy`。

影响：

- 直接用 `Dockerfile` 构建运行时，容器很可能因为没有 `PTM2CELLNET_CHECKPOINT` 而不加载模型，`/ready` 一直 503，healthcheck 失败。
- 用 `docker-compose.yml` 时，未加载模型的实例会被 `/health` 误判为健康，流量可能打到不可推理服务。
- 文档中部署变量若继续使用 `MODEL_PATH`，会与代码真实读取变量不一致。

解决策略：

1. 统一环境变量命名：
   - 推荐保留代码中的 `PTM2CELLNET_CHECKPOINT` / `PTM2CELLNET_CONFIG` 作为正式变量。
   - 修改 `Dockerfile` 和 `docker-compose.yml`：

```yaml
PTM2CELLNET_CHECKPOINT=/app/outputs/models/best_model.pt
PTM2CELLNET_CONFIG=/app/outputs/models/best_model.config.yaml
```

2. 增加向后兼容：
   - 在 `_try_auto_initialize()` 中可兼容读取 `MODEL_PATH`，但日志应提示该变量已弃用：

```python
checkpoint_path = (
    os.environ.get("PTM2CELLNET_CHECKPOINT")
    or os.environ.get("MODEL_PATH", "")
)
```

3. 统一健康检查端点：
   - `Dockerfile` 和 `docker-compose.yml` 都应使用 `/api/v1/ready`。
   - `/api/v1/health` 保留兼容用途，但不应作为 readiness。

4. 修正容器内权重复制/挂载：
   - `Dockerfile` 当前只复制 `src/`、`configs/`、`scripts/`，不复制 `outputs/models`。如果不挂载模型目录，镜像内没有默认模型。
   - 生产建议不把权重 bake 到镜像，而是明确要求挂载模型目录，并在 compose/K8s 中设置 `PTM2CELLNET_CHECKPOINT`。
   - 若要 demo 镜像开箱即用，则需要复制 `outputs/models/best_model.pt` 和 `best_model.config.yaml`，并明确该镜像只用于 demo。

推荐验收命令：

```bash
docker compose build api
docker compose up -d api
docker compose ps
curl -f http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/model/info
```

---

### P1-1：变异预测 workflow 与模型状态生命周期耦合不清

严重程度：P1  
影响范围：API 变异预测、热加载、多模型切换  
当前状态：部分实现但语义不稳

证据：

- `/api/v1/initialize` 在模型加载成功后尝试 `initialize_variant_workflow(str(ckpt_path))`。
- `initialize_variant_workflow()` 失败时会将 `STATE.variant_workflow = None`，但测试失败说明全局状态仍可能在不同测试/请求间残留。
- `STATE` 是模块级全局单例，所有 TestClient 和路由共享。

影响：

- 热加载新模型后，旧的 variant workflow 可能与新模型不一致。
- 测试顺序可能影响 API 行为。
- 多 worker 部署时每个进程有独立 `STATE`，如果初始化请求只打到一个 worker，其它 worker 仍未初始化。

解决策略：

1. 将模型状态和 variant workflow 状态视为同一生命周期：
   - 新模型初始化开始前先进入 `loading` 状态。
   - 模型加载失败时不覆盖已有 healthy 模型，或明确清空并返回 failed。
   - 模型加载成功但 variant workflow 失败时，`/predict` 可 ready，`/predict/variant` 不 ready。

2. 为每个能力增加独立 readiness：
   - `/api/v1/ready`：核心模型 ready。
   - `/api/v1/ready/variant`：variant workflow ready。
   - `/api/v1/model/info` 返回 `variant_workflow_loaded`。

3. 测试层面：
   - 所有 API 测试文件统一使用 `autouse` fixture 重置 `STATE`。
   - 新增测试：初始化模型后 variant 失败时 `/predict` 返回 200，`/predict/variant` 返回 503。

---

### P1-2：GenKI source backend 当前环境不可运行

严重程度：P1（若为交付范围）；P2（若仅为可选扩展）  
影响范围：软扰动、两阶段解释、图神经网络分析  
当前状态：优雅降级，未运行验证

证据：

```text
torch_geometric: False
```

相关测试当前结果：

```text
60 passed, 5 skipped
```

影响：

- 现在测试不会失败，但 `genki_source` backend 相关能力没有在当前环境真实跑通。
- 如果文档或交付物声称 GenKI source 可用，这一声明不成立。

解决策略：

1. 明确能力分级：
   - 核心模型训练/推理：必选。
   - GenKI source / graph backend：可选 extra。

2. 安装和 CI 分层：
   - 默认 CI 跑核心测试，允许 GenKI source 测试 skip。
   - 增加独立 job：`pip install -e ".[genki]"` 后运行 GenKI source 测试，不允许 skip。

3. CLI 入口加 preflight：
   - `scripts/run_ptm_virtual_perturbation.py` 和 `scripts/run_two_stage_explanation.py` 在用户选择 `genki_source` 时调用 `require_extras(["torch_geometric"], feature="GenKI source backend")`。
   - 报错信息应给出 `pip install -e ".[genki]"`。

推荐验收命令：

```bash
pip install -e ".[genki]"
python -m pytest \
  tests/integration/test_soft_perturbation_pipeline.py \
  tests/integration/test_two_stage_pipeline.py \
  tests/unit/integration/test_genki_adapter.py \
  tests/unit/integration/test_genki_reference_data.py -q
```

---

### P1-3：默认模型只能证明链路，不证明真实预测能力

严重程度：P1  
影响范围：产品声明、用户预期、科研可复现性  
当前状态：文档已有警告，但能力边界仍需贯穿 CLI/API 输出

证据：

- `README.md` 和 `outputs/models/README.md` 明确说明 `best_model.pt` 是随机合成数据训练的 demo 模型。
- 默认评估 accuracy 约随机基线。

影响：

- 工程 E2E 通过不等于真实生物学任务可用。
- 如果 API 默认加载 demo 模型，对外响应应避免被误解为真实预测。

解决策略：

1. 在 `artifact_manifest.json` 和 `best_model.config.yaml` 中加入 `model_card` / `data_provenance` 字段：
   - `model_kind: demo`
   - `training_data: synthetic_random`
   - `not_for_biological_use: true`

2. API `/model/info` 返回模型性质：
   - `model_kind`
   - `training_data_version`
   - `is_demo_model`

3. CLI 推理加载 demo 模型时输出 warning：
   - 只在检测到 manifest 标记 demo 时输出。
   - 不影响 smoke test，但避免误用。

4. 真实模型发布门槛：
   - 固定训练/验证/测试数据版本。
   - 导出完整 metrics、模型卡、配置、随机种子、代码 commit。
   - 提供至少一个真实数据 E2E 回归测试或小型 fixture。

---

## 5. 已修复或明显改善的历史问题

### 已修复：checkpoint/config 漂移

当前状态：

- `scripts/train.py` 保存：
  - `best_model.config.yaml`
  - `best_model_last.config.yaml`
  - `artifact_manifest.json`
- `scripts/predict.py` 和 `scripts/evaluate.py` 使用 `resolve_inference_config()`，优先读取 checkpoint 同目录 `.config.yaml`。

结论：旧报告中“新训练模型离开显式 `--config` 后会漂移”的主要风险已经缓解。

### 已修复：默认发布模型评估失败

当前状态：

- `outputs/models/best_model.config.yaml` 是 4 类配置。
- `DataLoader.load_sample_data()` 会读取配置中的 `data.cell_states` 和 `data.ptm_types`。
- 默认 `scripts/evaluate.py --model outputs/models/best_model.pt` 实测通过。

结论：旧报告中的默认评估 P0 问题已经解决。

### 已改善：GenKI 缺依赖不再导致默认测试失败

当前状态：

- 相关测试在缺少 `torch_geometric` 时 skip。
- 目标测试子集 `60 passed / 5 skipped`。

结论：默认 CI 稳定性改善，但 GenKI source 本身仍未在当前环境证明可运行。

---

## 6. 推荐修复路线图

### 阶段 1：恢复全量测试绿灯（必须）

目标：`python -m pytest -m "not slow and not gpu" -q` 通过。

任务：

1. 修复 `STATE.variant_workflow` 测试隔离/生命周期问题。
2. 修复或兼容 `_predict_in_batches()` 签名变更。
3. 单独验证 3 个失败测试。
4. 重新跑全量非 slow/gpu 测试。

验收：

```bash
python -m pytest -m "not slow and not gpu" -q
```

### 阶段 2：修复 Docker/API 部署就绪（必须）

目标：容器启动后能自动加载模型并通过 `/api/v1/ready`。

任务：

1. 统一环境变量为 `PTM2CELLNET_CHECKPOINT` / `PTM2CELLNET_CONFIG`。
2. `docker-compose.yml` healthcheck 改为 `/api/v1/ready`。
3. 决定 demo 模型是否随镜像复制；若不复制，则文档和 compose 必须要求挂载模型目录。
4. 增加部署级 smoke test。

验收：

```bash
docker compose up --build -d api
curl -f http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/model/info
```

### 阶段 3：明确可选能力边界（建议）

目标：核心能力和高级扩展能力的安装、测试、文档边界一致。

任务：

1. 为 GenKI/Mamba/Lion 建立独立 CI profile。
2. 文档明确 `pip install -e ".[genki]"`、`pip install -e ".[mamba]"` 的适用场景。
3. CLI 在选择可选 backend 时 fail-fast，并给出 extra 安装提示。

验收：

```bash
python -m pytest tests/unit/test_dependency_check.py -q
python -m pytest tests/integration/test_soft_perturbation_pipeline.py -q
```

### 阶段 4：真实数据 E2E 和模型卡（科研/产品发布前必须）

目标：从“工程 smoke 可用”升级为“真实任务可用”。

任务：

1. 准备一个可公开或可脱敏的小型真实数据 fixture。
2. 增加真实数据训练/评估 smoke，不要求高指标，但要求数据契约真实。
3. 为真实模型输出 model card、data card、metrics、artifact manifest。
4. API/CLI 输出模型 provenance。

验收：

```bash
python scripts/train.py --config <real_config> --data <real_fixture.csv> --epochs 1 --output /tmp/real_e2e
python scripts/evaluate.py --model /tmp/real_e2e/models/best_model.pt --data <real_fixture.csv> --output /tmp/real_eval
python scripts/predict.py --model /tmp/real_e2e/models/best_model.pt --input <real_infer_fixture.csv> --output /tmp/real_pred.csv
```

---

## 7. 当前最终判断

### 是否满足 E2E 训练和推理要求？

分层回答：

1. **核心工程链路：基本满足。**  
   原生训练、训练产物配置绑定、默认 CLI 推理、默认评估、E2E 测试均已通过。

2. **API 服务部署链路：尚未满足。**  
   API 路由本身基本可用，但 Docker/compose 自动加载和 readiness 配置仍有阻塞问题。

3. **全仓库质量门禁：尚未满足。**  
   全量非 slow/gpu pytest 仍有 3 个失败。

4. **高级分析/可选能力：尚未满足全功能 E2E。**  
   GenKI source、Mamba、Lion 等可选能力在当前环境未安装或未真实验证。

5. **真实生物学预测能力：未证明。**  
   默认模型是 demo/smoke 模型，只证明链路，不证明业务有效性。

### 发布建议

- 若发布目标是“开发版 / smoke 版”：可以声明核心训练和 CLI 推理链路已基本跑通，但必须附带 demo 模型免责声明。
- 若发布目标是“产品级 API 服务”：不建议发布，至少需要先完成 P0-1 和 P0-2。
- 若发布目标是“科研可用模型”：不建议发布，需要真实数据 E2E、模型卡、数据版本和可靠评估指标。

