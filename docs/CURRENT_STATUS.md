# PTM2CellNet Current Status

**Last updated: 2026-08-16（Norman/Adamson 导入管线合并与文档归档后）**

当前权威分析是仓库根目录的
[`project_analysis_20260816.md`](https://github.com/valleyhe/PTM2CellNET/blob/main/project_analysis_20260816.md)。
2026-08-08/09 的 E2E 现状分析、R-01～R-03 修复报告与修复报告 v2 已作为历史快照
归档到 [`archive/20260816/`](https://github.com/valleyhe/PTM2CellNET/tree/main/archive/20260816)
（原路径留有重定向入口）；更早的快照位于
[`archive/20260808/`](https://github.com/valleyhe/PTM2CellNET/tree/main/archive/20260808)。

## Quick Reference

- **验证环境**：Python 3.12.13、PyTorch 2.4.1+cu118、NumPy 2.4.3、CUDA 可用；真实资产测试未启用网络/GPU/外部服务验收门禁。
- **测试基线**：完整 `pytest tests/unit -q` 为 **1716 passed、4 skipped**（2026-08-16，79.4s，4 个 skip 均为环境守卫：2×scvi 反向守卫、1×onnx 缺失、1×pickle 限制）；上一轮 coverage 门禁基线为 branch coverage 75.27%（2026-08-09 数据，`fail_under=74`，待下一轮复测刷新）。
- **静态/构建质量**：`ruff check src scripts tests`、`mypy src --show-error-codes`（133 个源文件，0 errors）、`compileall`、`setup.py check`、wheel 构建和 Sphinx strict 均通过；`pip check` 仍有两个可选依赖冲突。
- **当前实现**：标准模型离线训练/推理/API 工程链路可运行；跨尺度模型已有版本化 NPZ DataModule、独立训练/批量推理 CLI、完整 checkpoint 和 artifact manifest，但仍是 opt-in 离线路径，未接入标准 API。callback 状态/`global_step` 恢复和 `cell_edge_weight` 契约已闭环。
- **R-01～R-03 状态**：`MultiPLMEncoder`、`PTMTokenAdapter`、`CIGNNSignalBridge`/敏感度矩阵和 `CellGraphCompassHead` 的工程契约与 synthetic batch 测试通过；本地三路 pLM 资产结构门禁通过，真实图/扰动训练和科学验收仍未验证。
- **剩余需求差距**：`32ded55`（2026-08-16）已登记 norman_adamson（GSE133344 完整导入，GSE90546 待解析）、kinase_substrate（OmniPath 41,506 条）、string、bioplex、regnetwork 快照，`cross_scale_training` 9 个 required 数据集仅剩 **scperturb（已下载未登记）、replogle、scgenescope** 3 项 controlled 且正确 fail-fast；跨尺度 checkpoint 尚未被 API 初始化路由支持，TD-H02 embedding 预计算器、信号图原始格式导入与真实数据科学验收仍未实现（详见 project_analysis_20260816.md 任务 1）。
- **覆盖率门禁**：默认 CI 使用 branch coverage，`fail_under=74`；下一轮优先补 `training/self_supervised.py`、初始化路由、`data/validation.py`、跨尺度推理和 pLM 资产异常路径。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)
- [项目代码与文档综合分析（2026-08-09）](https://github.com/valleyhe/PTM2CellNET/blob/main/project_analysis_20260809.md)、[E2E 训练与推理历史快照](E2E训练与推理现状分析_2026-08-08.md)、[测试覆盖率治理](TEST_COVERAGE.md)
- [真实资产验收](guides/real_assets_acceptance.md)
- [归档清单（2026-08-09）](https://github.com/valleyhe/PTM2CellNET/blob/main/archive/20260809/MANIFEST.md)
- [历史系统性复核 v19](https://github.com/valleyhe/PTM2CellNET/blob/main/archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome 和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本次 8 项真实资产测试均按设计跳过，
因此“测试全绿”不等价于真实生物学效果或外部服务已验收。
