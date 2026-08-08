# PTM2CellNet Current Status

**Last updated: 2026-08-08（覆盖率治理与 E2E 现状复核完成后）**

当前权威分析是
[`E2E训练与推理现状分析_2026-08-08.md`](E2E训练与推理现状分析_2026-08-08.md)，
R-01～R-03 的组件修复结论见
[`r01_r03_systematic_repair_report_20260808.md`](r01_r03_systematic_repair_report_20260808.md)。
修复前综合分析和 2026-08-04 报告均已作为历史快照归档到
[`archive/20260808/`](../archive/20260808/)。

## Quick Reference

- **验证环境**：Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用；本次未启用真实资产/外部网络验收门禁。
- **测试基线**：核心 coverage 套件 **1787 passed, 5 skipped，branch coverage 75.02%**；其余 E2E/CI/real-assets 套件 **54 passed, 8 skipped**，合计 **1841 passed, 13 skipped**。
- **静态质量**：`python -m ruff check src scripts tests` 通过；`python -m mypy src --show-error-codes` 在 **128 个源文件中 0 errors**；`python -m compileall -q src scripts tests` 通过。
- **当前实现**：标准模型离线训练/推理/API 工程链路可运行；跨尺度模型已完成组件和 synthetic batch 契约，但没有训练/预测 CLI 或 API 接入。
- **R-01～R-03 状态**：已提供 opt-in `MultiPLMEncoder`、`PTMTokenAdapter`、`CIGNNSignalBridge`/敏感度矩阵和 `CellGraphCompassHead` 工程契约；真实 pLM 权重、真实信号图与生物学验收仍未验证，详见修复报告。
- **剩余需求差距**：真实扰动训练、受控数据/图/pLM release、跨尺度 checkpoint artifact 与部署链路仍未闭环；不能把工程 fixture 结果当作科学结论。
- **覆盖率门禁**：默认 CI 使用 branch coverage，`fail_under=74`；下一轮优先补 `self_supervised`、初始化路由、数据验证和训练回调。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)
- [E2E 训练与推理现状分析](E2E训练与推理现状分析_2026-08-08.md)、[测试覆盖率治理](TEST_COVERAGE.md)
- [真实资产验收](guides/real_assets_acceptance.md)
- [归档清单](../archive/20260808/MANIFEST.md)
- [历史系统性复核 v19](archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实 DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome 和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本次 8 项真实资产测试均按设计跳过，
因此“测试全绿”不等价于真实生物学效果或外部服务已验收。
