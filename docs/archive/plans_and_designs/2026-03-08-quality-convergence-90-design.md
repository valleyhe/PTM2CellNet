# PTM2CellNet 质量收敛设计（90%+ 覆盖率冲刺）

日期：2026-03-08  
状态：已评审（进入实施计划阶段）

## 1. 背景与目标

当前基线：
- 全量测试：`105 passed`
- 类型检查：`mypy src = 0 errors`
- 代码规范：`flake8 src tests scripts` 通过
- 覆盖率：`84.40%`

本轮目标：
- 在不改变业务行为前提下，将 `src` 总覆盖率提升到 **90%+**。
- 保持现有质量门禁全绿（`mypy`、`flake8`、测试回归）。

## 2. 设计原则

1. **测试优先，不改业务逻辑**：以补测试覆盖分支为主。  
2. **高 ROI 优先**：优先处理对总覆盖率贡献最大的低覆盖模块。  
3. **确定性测试**：使用小样本、固定输入、避免随机波动。  
4. **小步快跑**：每批次补测后立即全量回归，防止回归扩散。

## 3. 方案比较

### 方案 A（推荐）：Training 分支优先

先补：
- `src/training/callbacks.py`
- `src/training/trainers.py`
- `src/training/optimizers.py`

再补：
- `src/evaluation/visualization.py`
- `src/data/features.py`

优点：
- 对总覆盖率拉升最直接；
- 分支可控、验证路径明确。

风险：
- 训练相关测试构造稍复杂，但可用 tiny dataset 降低成本。

### 方案 B：可视化+特征优先

优点：函数式代码易测；  
风险：单独推进未必能快速冲到 90%。

### 方案 C：并行混合

优点：覆盖面大；  
风险：变更面和调试成本更高。

## 4. 覆盖率目标映射（测试矩阵）

### 4.1 `src/training/callbacks.py`（当前约 76%）

目标分支：
- `monitor` 缺失告警分支；
- `save_last=True` 文件输出分支；
- `save_best_only=True` 的 improve / no-improve 双分支；
- `mode=min/max` 比较器分支；
- EarlyStopping 的 `wait` 递增与 `should_stop` 触发分支。

### 4.2 `src/training/trainers.py`（当前约 74%）

目标分支：
- 未 compile 调用训练/验证异常分支；
- 非法 batch、缺失 label/logits/predictions 分支；
- `val_loader` 有无双路径；
- callback 触发 `early_stop` 路径；
- scheduler 的 `ReduceLROnPlateau` / 非 plateau 双分支；
- `predict` 非 dict 输出异常分支。

### 4.3 `src/training/optimizers.py`（当前约 73%）

目标分支：
- optimizer：`adam` / `adamw` / `sgd` / invalid；
- scheduler：`cosine` / `plateau` / `step` / invalid；
- scheduler 为空分支。

### 4.4 `src/evaluation/visualization.py`（当前约 70%）

目标分支：
- ROC/PR：二分类 vs 多分类；
- confusion matrix：`normalize=True/False`；
- training curves：键缺失组合分支；
- feature importance 与 attention heatmap 参数分支。

### 4.5 `src/data/features.py`（当前约 75%）

目标分支：
- 编码模式：`onehot/kmer/both`；
- 非法字符、空序列、短序列；
- PTM 解析：非法 JSON、越界位置、不支持类型；
- `include_ptm_features` 开关；
- `combine_features` 1D/2D 拼接分支。

## 5. 实施顺序与门禁

实施顺序：
1. `training/callbacks.py`
2. `training/trainers.py`
3. `training/optimizers.py`
4. `evaluation/visualization.py`
5. `data/features.py`
6. 全量回归

每批次门禁（必须全部满足）：
- 新增测试先局部通过；
- 全量 `pytest tests/unit tests/integration --cov=src --cov-fail-under=80` 通过；
- `python -m mypy src` 维持 0 error；
- `flake8 src tests scripts` 通过。

## 6. 风险与回滚策略

风险：
- 训练相关测试可能出现不稳定。  

应对：
- 使用固定输入、微型 dataset、减少 epoch/batch；
- 优先行为断言，避免过度依赖浮点精度；
- 若某策略不稳，回退该测试并改用等价分支触发方式。

## 7. 验收标准（DoD）

硬标准：
- 覆盖率 `>= 90%`；
- 全量测试通过；
- `mypy` 全绿；
- `flake8` 全绿。

交付物：
- 新增/扩展测试代码；
- 覆盖率提升前后对比；
- 剩余盲区与后续建议（若存在）。

## 8. 结论

采用 **方案 A（Training 分支优先）**，再用 Visualization/Features 补顶，预计可在最小生产代码改动下把覆盖率从 `84.40%` 推升到 `90%+`。

## 9. 实施结果附录（2026-03-08）

- 覆盖率（Before）：`84.40%`
- 覆盖率（After）：`90.60%`（`TOTAL 1756 stmts, 165 miss`）
- 测试计数（After）：`140 passed`
- 质量门禁：
  - `python -m mypy src`：`Success: no issues found in 29 source files`
  - `flake8 src tests scripts`：通过（无输出）
  - `python -m pytest -q tests/unit tests/integration --cov=src --cov-report=term-missing --cov-fail-under=80`：通过
- 残余盲区（主要模块）：
  - `src/data/preprocess.py`：79%
  - `src/evaluation/evaluators.py`：83%
  - `src/utils/helpers.py`：83%
  - `src/api/routes.py`：86%
