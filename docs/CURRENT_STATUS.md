# PTM2CellNet Current Status

**Last updated: 2026-08-08（系统性代码/需求复核与归档完成后）**

当前权威结论是根目录的
[`project_analysis_20260808.md`](../project_analysis_20260808.md)。本文件只保留
可快速核对的状态摘要；2026-08-04 的修复与 E2E 报告已经作为历史快照归档到
[`archive/20260808/`](../archive/20260808/)。

## Quick Reference

- **验证环境**：Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用；本次未启用真实资产/外部网络验收门禁。
- **测试基线**：`tests/unit` **1558 passed, 4 skipped**；`tests/integration` **67 passed, 1 skipped**；`tests/e2e` **42 passed**；`tests/real_assets` **8 skipped**（共 1680 项默认/门禁测试记录）。
- **静态质量**：`python -m ruff check src scripts tests` 通过；`python -m mypy src --show-error-codes` 在 **124 个源文件中 0 errors**；`python -m compileall -q src scripts tests` 通过。
- **当前实现**：核心离线训练/推理/API 链路可运行；DAVF 支持缺失 checkpoint 的显式零特征降级和旧 checkpoint 兼容加载；`use_ptm_module=false` 支持仅序列消融。
- **需求差距**：原始需求中的 Ankh39 三模型融合、CIGNN/PPI 因果桥、Cell-Graph-Compass 解码器和多项指定数据源尚未实现；详见综合报告第 4 节。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)
- [真实资产验收](guides/real_assets_acceptance.md)
- [归档清单](../archive/20260808/MANIFEST.md)
- [历史系统性复核 v19](archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实 DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome 和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本次 8 项真实资产测试均按设计跳过，
因此“测试全绿”不等价于真实生物学效果或外部服务已验收。
