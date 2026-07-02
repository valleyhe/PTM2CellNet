# PTM2CellNet E2E 训练与推理现状分析报告

**分析日期**：2026-06-30  
**分析范围**：`scripts/train.py`、`scripts/predict.py`、`scripts/evaluate.py`、FastAPI 推理接口、核心数据/模型/训练模块及全量测试套件。  
**执行人**：Kimi Code CLI

---

## 1. 总体结论

| 维度 | 结论 |
|---|---|
| **E2E 训练** | **已满足基础要求**。`scripts/train.py` 可完整跑通：数据加载 → 预处理 → 模型构建 → 训练 → 评估 → 保存指标与可视化。 |
| **E2E 推理（CLI）** | **已修复原阻塞性问题**。修复前 `scripts/predict.py` 因缺失 batch 维度直接崩溃；修复后单样本与批量 CSV 预测均验证通过。 |
| **E2E 推理（API）** | **可用**。FastAPI 端点 `/predict`、`/batch_predict` 测试通过，推理链路正常。 |
| **测试健康度** | 全量测试 `920 passed / 9 failed / 3 skipped`，核心训练/推理链路测试全部通过。 |

---

## 2. 验证方法与执行结果

### 2.1 全量测试套件

```bash
python -m pytest tests/ -q
```

**结果**：`920 passed, 9 failed, 3 skipped in 319.52s`

核心通过项：
- `tests/integration/test_integration.py`：8/8 passed
- `tests/integration/test_lightning_pipeline.py`：9/9 passed
- `tests/integration/test_davf_pipeline.py`：12/12 passed（1 skipped）
- `tests/unit/test_api.py`：21/21 passed
- `tests/unit/test_data.py`、`tests/unit/test_models.py`、`tests/unit/training/test_trainers.py`：74/74 passed
- `tests/unit/test_training_inference_consistency.py`：15/18 passed（2 skipped，1 failed为回归契约问题）

### 2.2 E2E 冒烟测试（实际执行）

| 命令 | 结果 |
|---|---|
| `python scripts/train.py --config configs/default.yaml --epochs 1 --output outputs/smoke_test` | ✅ 成功，生成 `best_model.pt`、训练曲线、测试指标 |
| `python scripts/predict.py --model outputs/smoke_test/models/best_model.pt --config configs/default.yaml --sequence "ACDEFGHIKLMNPQRSTVWY"` | ✅ 成功（修复后） |
| `python scripts/predict.py --model outputs/smoke_test/models/best_model.pt --config configs/default.yaml --input /tmp/predict_input.csv --output /tmp/predict_output.csv` | ✅ 成功（修复后） |
| `python scripts/evaluate.py --model outputs/smoke_test/models/best_model.pt --config configs/default.yaml --output outputs/smoke_test/eval` | ✅ 成功，生成 ROC/PR/混淆矩阵 |

---

## 3. 问题详细分析

### 3.1 已修复：阻塞性问题

#### `scripts/predict.py` 单样本/批量推理缺失 batch 维度

- **严重程度**：🔴 阻塞性
- **现象**：
  ```
  ValueError: not enough values to unpack (expected 2, got 1)
  File "src/models/encoders.py", line 97, in forward
      batch_size, seq_len = sequences.shape
  ```
- **根因**：`src/api/routes/predictions.py::preprocess_request()` 返回的是 1D 张量 `(max_seq_len,)`，而 `scripts/predict.py` 未像 API 端点那样添加 `unsqueeze(0)`，直接将 1D 输入送入模型，导致编码器解析 shape 失败。
- **修复位置**：
  - `scripts/predict.py` 第 72 行（单样本分支）
  - `scripts/predict.py` 第 118 行（批量 CSV 分支）
- **修复内容**：在 `preprocess_request()` 后统一执行：
  ```python
  batch = {key: val.unsqueeze(0) for key, val in batch.items()}
  ```
- **验证**：修复后单样本与批量 CSV 预测均成功输出 `predicted_cell_state` 与概率分布。

---

### 3.2 中高严重程度（非阻塞，但需处理）

#### 问题 1：回归任务输出契约不一致

- **严重程度**：🟠 中高
- **涉及测试**：
  - `tests/unit/test_architecture_boundaries.py::TestRegressionPredictor::test_regression_output_format`
  - `tests/unit/test_model_enhancements.py::TestPTM2CellNetRegression::test_ptm2cellnet_regression`
  - `tests/unit/test_training_inference_consistency.py::TestAPIConsistency::test_regression_task_consistency`
- **现象**：
  ```python
  AssertionError: assert "logits" not in output
  # output = {"logits": tensor(...), "predictions": tensor(...)}
  ```
- **根因**：`src/models/predictors.py::RegressionPredictor.forward()` 同时返回 `"logits"` 与 `"predictions"`，而测试与预期契约认为回归任务只应包含 `"predictions"`。
- **解决策略**：
  1. 优先方案：修改 `RegressionPredictor.forward()`，仅返回 `{"predictions": predictions}`。
  2. 兼容性方案：若下游有依赖 `logits` 的代码，可在返回字典中保留但将主键统一为 `predictions`，并更新测试预期。
  3. 同步检查 `Evaluator`、Lightning Module 中对回归输出的解析逻辑，确保修改后无回归。

#### 问题 2：GenKI / 虚拟扰动流程缺失 `torch_geometric`

- **严重程度**：🟠 中高（若解释/扰动是产品能力）
- **涉及测试**（5 个失败）：
  - `tests/integration/test_soft_perturbation_pipeline.py::test_soft_perturbation_cli_accepts_genki_source_backend`
  - `tests/integration/test_soft_perturbation_pipeline.py::test_soft_perturbation_cli_accepts_latent_vgae_scoring`
  - `tests/integration/test_two_stage_pipeline.py::test_two_stage_cli_accepts_genki_source_backend`
  - `tests/unit/integration/test_genki_adapter.py::TestBackendDetection::test_get_backend_info_genki_source`
  - `tests/unit/integration/test_genki_reference_data.py::TestReferenceDataErrors::test_load_reference_data_genki_source`
- **现象**：
  ```
  RuntimeError: Backend genki_source is not ready; missing or broken dependencies: torch_geometric
  ```
- **根因**：`src/integration/genki/reference_data.py` 的 `genki_source` backend 依赖 `torch_geometric`，但当前环境未安装，且代码未做优雅降级。
- **解决策略**：
  1. **补齐依赖**：在 `requirements.txt` 中加入 `torch_geometric` 及其兼容版本，并补充安装文档（CUDA 版本需与 PyTorch 对齐）。
  2. **优雅降级**：在 `genki_source` backend 不可用时，提供 mock/reference 数据回退或显式跳过，避免整个 CLI 流程崩溃。
  3. **测试隔离**：对依赖可选后端的测试增加 `pytest.importorskip("torch_geometric")` 或类似环境检测，避免在基础 CI 环境中强制失败。

#### 问题 3：DAVF 旧版 checkpoint 警告文案不匹配

- **严重程度**：🟡 中
- **涉及测试**：
  - `tests/unit/test_davf_inference.py::TestCheckpointLoading::test_legacy_pickle_checkpoint_requires_explicit_opt_in`
- **现象**：
  ```python
  AssertionError: assert "UNSAFE" in warning_message
  # 实际 warning: "Checkpoint %s requires unsafe legacy loading (weights_only=False). ..."
  ```
- **根因**：`src/models/davf_checkpoint_utils.py` 中的警告信息不含 `"UNSAFE"` 大写关键字，与测试断言不一致。
- **解决策略**：
  1. 将警告文案中的 `unsafe` 改为 `UNSAFE` 或加入 `"UNSAFE"` 关键字。
  2. 或更新测试断言，使其匹配当前实际文案。

---

### 3.3 低严重程度（建议项）

| 问题 | 说明 | 建议 |
|---|---|---|
| `scripts/predict.py` 批量模式逐条推理 | 当前 `for row in df.iterrows()` 单条前向，效率低 | 改为将多行堆叠成 batch 后统一推理，与 API `/batch_predict` 实现对齐 |
| `torch.load(weights_only=False)` 告警 | 多处旧式加载，PyTorch 未来将默认 `True` | 核心推理/评估脚本逐步显式传参或封装安全加载函数 |
| 训练曲线 legend 警告 | 单 epoch 时 `No artists with labels found` | 仅在多 epoch 时绘制验证曲线 legend，或添加空值保护 |

---

## 4. 已修改文件

| 文件 | 修改内容 |
|---|---|
| `scripts/predict.py` | 在单样本与批量 CSV 分支中为 `preprocess_request()` 输出添加 `unsqueeze(0)`，修复 CLI 推理阻塞性 bug |

---

## 5. 建议下一步行动

1. **验证真实输入格式**：确认本次 `predict.py` 修复能覆盖你们的实际 CSV 列名（如 `sequence`、`ptm_sites`、`gene_symbol` 等）与 PTM 数据格式。
2. **处理回归契约问题**：若项目需要支持回归任务，优先统一 `RegressionPredictor` 输出字典，并同步修复 3 处失败测试。
3. **补齐 GenKI 依赖**：若虚拟扰动 / 两阶段解释是产品能力，需补充 `torch_geometric` 依赖与安装文档，并考虑添加优雅降级。
4. **完善 CLI 批量推理性能**：将 `predict.py` 的批量模式改为真正的 batch 推理，提升吞吐。
5. **安全加载迁移**：逐步将 `torch.load(..., weights_only=False)` 替换为显式参数或安全加载封装，减少未来 PyTorch 版本升级风险。
