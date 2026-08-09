# PTM2CellNet Current Status

**Last updated: 2026-08-08（最终回归、续训与发布 artifact 复核后）**

当前权威分析是
[`E2E训练与推理现状分析_2026-08-08.md`](E2E训练与推理现状分析_2026-08-08.md)，
R-01～R-03 的组件修复结论见
[`r01_r03_systematic_repair_report_20260808.md`](r01_r03_systematic_repair_report_20260808.md)，
标准训练修复的 v2 记录见
[`E2E训练与推理代码修复报告_2026-08-08_v2.md`](E2E训练与推理代码修复报告_2026-08-08_v2.md)。
修复前综合分析和 2026-08-04 报告均已作为历史快照归档到
[`archive/20260808/`](https://github.com/valleyhe/PTM2CellNET/tree/main/archive/20260808)。

## Quick Reference

- **验证环境**：Python 3.12.13、PyTorch 2.4.1+cu118、NumPy 2.4.3、CUDA 可用；真实资产测试未启用网络/GPU/外部服务验收门禁。
- **测试基线**：完整 `pytest -q` 为 **1897 passed、13 skipped、28 warnings**；默认 coverage 门禁为 **1843 passed、5 skipped、branch coverage 75.63%**，`fail_under=74`。
- **静态质量**：`ruff check src scripts tests`、`mypy src --show-error-codes`（133 个源文件，0 errors）和 `compileall` 均通过。
- **当前实现**：标准模型离线训练/推理/API 工程链路可运行；跨尺度模型已有版本化 NPZ DataModule、独立训练/批量推理 CLI、完整 checkpoint 和 artifact manifest，但仍是 opt-in 离线路径，未接入标准 API。
- **R-01～R-03 状态**：`MultiPLMEncoder`、`PTMTokenAdapter`、`CIGNNSignalBridge`/敏感度矩阵和 `CellGraphCompassHead` 的工程契约与 synthetic batch 测试通过；本地三路 pLM 资产结构门禁通过，真实图/扰动训练和科学验收仍未验证。
- **剩余需求差距**：跨尺度所需 9 个受控数据集仍未登记本地快照，`cross_scale_training`/`cross_scale_inference` profile 正确 fail-fast；跨尺度 checkpoint 尚未被 API 初始化路由支持，真实数据 train→val→test 和科学效果仍未验收。
- **覆盖率门禁**：默认 CI 使用 branch coverage，`fail_under=74`；下一轮优先补 `training/self_supervised.py`、初始化路由和 `data/validation.py`。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)
- [E2E 训练与推理现状分析](E2E训练与推理现状分析_2026-08-08.md)、[测试覆盖率治理](TEST_COVERAGE.md)
- [真实资产验收](guides/real_assets_acceptance.md)
- [归档清单](https://github.com/valleyhe/PTM2CellNET/blob/main/archive/20260808/MANIFEST.md)
- [历史系统性复核 v19](https://github.com/valleyhe/PTM2CellNET/blob/main/archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome 和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本次 8 项真实资产测试均按设计跳过，
因此“测试全绿”不等价于真实生物学效果或外部服务已验收。
