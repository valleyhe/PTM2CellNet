# PTM2CellNet Current Status

**Last updated: 2026-08-22（PerturbGen 双路径集成工程主干入库 + 第四轮综合分析后）**

当前权威分析是仓库根目录的
[`project_analysis_20260822.md`](./project_analysis_20260822.md)。
2026-08-21 的综合分析已作为历史快照归档到
[`archive/20260822/`](./archive/20260822/MANIFEST.md)（原路径留有重定向入口）；
更早批次见 `archive/20260820/`、`archive/20260816/`、`archive/20260809/`、`archive/20260808/`。

## Quick Reference

- **验证环境**：Python 3.12.13（SSH_unit env）、PyTorch 2.4.1+cu118、CUDA 可用；真实资产测试未启用网络/GPU/外部服务验收门禁。
- **测试基线**：全量 `pytest tests -m "not gpu and not real_assets"` 于**网络命名空间隔离（`unshare -rn`）+ HF 离线缓存（`HF_HUB_OFFLINE=1`）**下 **2310 passed、15 skipped、1 deselected、exit 0（496.35s，2026-08-22 实测）**；该口径含 PerturbGen 新增约 112 例。注意：默认联网环境下套件存在网络隐性依赖（见下"测试基础设施"）。
- **静态/构建质量**：`compileall` 通过；`ruff check src scripts tests` 通过；`mypy src` 实测 **23 errors / 8 files**（默认 env，mypy 2.1.0）= 0821 基线 5 处（predictions.py/cross_scale.py）+ PerturbGen 新增 18 处（TD-N-11）；base env（mypy 1.20.0）38/18；`pip check` 仅 1 条 PyNaCl 平台告警（F-10 族）。
- **测试基础设施（本轮关键发现）**：外部包 `UniProtMapper` 的 HTTP 调用无超时（被 `src/analysis/gene_mapper.py` 优先选用），代理/网络半死时集成测试实测挂死 21+ 分钟、生产 `/predict`（含 `gene_symbol`）链路同样可无限挂起——登记 **TD-N-24（高）**；另有 49 项测试依赖 HF 在线检查（TD-N-25，建议 CI 固化 `HF_HUB_OFFLINE=1` + 隔离跑法，复现命令见权威报告 §7.3）。
- **PerturbGen 双路径集成（本轮入库 `ddbf571`）**：`src/integration/perturbgen/`（contracts/data_prep/env_guard/config_builder/runner/results/dual_path/reports/embedding_export）+ `src/models/{gene_vocabulary,perturbgen_embedding}.py` + 5 个 CLI（run_perturbgen_pipeline / export_perturbgen_gene_embeddings / evaluate_perturbgen_dual_path / benchmark_perturbgen / check_perturbgen_release_evidence）+ `configs/integration/perturbgen.yaml` + `.github/workflows/perturbgen-real-assets.yml`。按设计方案（`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`）：**M1–M3 工程门全部通过（mocked）；M0/Gate-0 因外部资产缺失 BLOCKED（encoder 权重、Python 3.11 独立环境、≥3 donor 合规 cohort）；M4（DAVF runtime 注入）/M6（科学验收）/M7（收口）按 Gate-0 阻断未开始**。API 不暴露 PerturbGen 端点为方案 §9 有意决策。
- **R-01～R-03 状态**：`MultiPLMEncoder`、`PTMTokenAdapter`、`CIGNNSignalBridge`/敏感度矩阵和 `CellGraphCompassHead` 的工程契约与 synthetic batch 测试通过；本地三路 pLM 资产结构门禁通过，真实图/扰动训练和科学验收仍未验证。
- **剩余需求差距**：REQUIREMENTS v2.2 全部 24 项 Complete（2026-08-22 代码侧独立复核成立）；`cross_scale_training` required 数据集仅剩 replogle、scgenescope 2 项 controlled 且正确 fail-fast；跨尺度 API（`/cross-scale/*`）已接入。缺口集中于真实数据科学验收（F-04：`tests/real_assets/` 门禁内全部 skip）与 PerturbGen Gate-0 阻断链（含 **CI analysis job 缺失**——方案 §4.5/§5.3 定义但未开发，属被遗漏交付物）。
- **覆盖率门禁**：默认 CI branch coverage `fail_under=74`；上一轮基线 75.27%（2026-08-09 数据），待 CI/独立环境复测刷新（PerturbGen 112 例增量未计入）。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)、[真实资产验收](guides/real_assets_acceptance.md)
- [项目代码与文档综合分析（2026-08-22，当前权威）](../project_analysis_20260822.md)、[归档清单（2026-08-22）](../archive/20260822/MANIFEST.md)
- [PerturbGen 双路径整合方案（v2.0，现行需求基线）](DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
- [测试覆盖率治理](TEST_COVERAGE.md)
- [历史系统性复核 v19](https://github.com/valleyhe/PTM2CellNET/blob/main/archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome、PerturbGen 外部环境和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本次真实资产测试均按设计跳过，
因此"测试全绿"不等价于真实生物学效果或外部服务已验收。此外，本机代理/外网异常时
请使用权威报告 §7.3 的隔离跑法，否则含 UniProt/HF 在线检查的测试可能挂起（TD-N-24/25）。
