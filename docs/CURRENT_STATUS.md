# PTM2CellNet Current Status

**Last updated: 2026-08-24（PerturbGen M4 提交入库 + TD-N-10/TD-N-24 闭环确认 + 归档批次 20260824）**

当前权威分析是仓库根目录的
[`project_analysis_20260824.md`](../project_analysis_20260824.md)。
2026-08-23 及更早的综合分析已归档：`archive/20260824/reports/`、`archive/20260822/reports/`；
各批次清单见对应 `archive/YYYYMMDD/MANIFEST.md`。

## Quick Reference

- **验证环境**：主进程 Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用；PerturbGen 独立环境 conda `perturbgen`（Python 3.11.15）已于 2026-08-23 完全建立并通过 GPU 实测（evidence 见 `outputs/perturbgen/env_evidence_20260823.json`，复现入口 `scripts/setup_perturbgen_env.sh` + `scripts/download_perturbgen_wheels.sh`）。
- **测试基线（2026-08-24 更新）**：全量 `pytest tests`（`unshare -rn` 网络隔离 + HF 离线）**2328 passed / 16 skipped / exit 0（551.86s）**，零失败。
- **静态/构建质量**：`compileall` 通过；`ruff check src scripts tests` 全绿；`mypy src` **23 errors / 8 files**（= TD-N-06 基线 5 处 + TD-N-11 PerturbGen 18 处，无新增）；`pip check` 仅 1 条 PyNaCl 平台告警（F-10 族）。
- **技术债闭环（本轮确认，提交 `63ebf75`）**：
  - **TD-N-24（高）已修复**：`src/analysis/gene_mapper.py:150-186` 新增 `_call_external_mapper`——外部 `UniProtMapper` 调用包裹 daemon 线程 + `_EXTERNAL_MAPPER_TIMEOUT_S` 硬超时，超时转既有异常分支；生产 `/predict` 挂死风险解除。回归锚点 `tests/unit/analysis/test_gene_mapper.py`（TD-N-33 合并后单一入口）。
  - **TD-N-10（高）已修复**：`.github/workflows/perturbgen-real-assets.yml:44-46` 安装步骤追加 `pip install -r requirements-analysis.txt`，Gate-4 门禁 anndata 缺口闭环。
  - **「CI analysis job 缺失」表述作废**：`.github/workflows/ci.yml:59-84` 已存在 `analysis` job（显式跑 data_prep/results/pipeline_mocked 三文件并装 analysis 依赖）；残留语义偏差（方案 §4.5 字面「单测必跑」仅部分满足）降级登记为 U-15（低），见权威分析 §任务1。
- **PerturbGen 双路径集成**：工程代码 M1–M3 全绿（mocked）；Gate-0 已闭环 ①~⑤（独立环境、真实 ckpt、token 对齐、离线初始化、真实 perturb smoke `outputs/perturbgen/spike/20260823_m0_smoke/evidence.json`）+ M2 train→perturb 工程闭环（`20260823_m2_trainmask/evidence.json`，288/288 张量映射验证）；**剩余唯一硬阻断 = M0⑥ ≥3 donor 合规 cohort**（本地 30 文件审计 0 合规候选，`20260823_donor_audit/evidence.json`）。M4 代码前置 6/8 落地（schema v2 注入链路 + checkpoint 版本化），余 2/8（`ptm_direction_mapper.py` 公共 resolver、`davf.py` Protocol 化）按方案 §7.3 有意延后至 Gate-0 通过。
- **R-01～R-03 状态**：工程契约与 synthetic batch 测试通过；三路 pLM 权重已本地化；真实图/扰动训练和科学验收仍未验证（依赖 replogle/scgenescope 数据供给）。
- **剩余需求差距**：REQUIREMENTS v2.2 全部 24 项 Complete 维持成立；缺口集中于真实数据科学验收（F-04 门禁内 skip）与 Gate-0 外部数据链，逐项清单（U-01~U-15）见权威分析 §任务1。
- **覆盖率门禁**：默认 CI branch coverage `fail_under=74`；上一轮实测基线 75.27%（2026-08-09），待 CI/独立环境复测刷新（受 TD-N-09 pytest-cov 冲突阻塞）。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)、[真实资产验收](guides/real_assets_acceptance.md)
- [项目代码与文档综合分析（2026-08-24，当前权威）](../project_analysis_20260824.md)、[归档清单（2026-08-24）](../archive/20260824/MANIFEST.md)
- [PerturbGen 双路径整合方案（v2.0，现行需求基线）](DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
- [测试覆盖率治理](TEST_COVERAGE.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome、PerturbGen 外部环境和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本次真实资产测试均按设计跳过，
因此"测试全绿"不等价于真实生物学效果或外部服务已验收。网络隔离 + HF 离线跑法
（`unshare -rn env HF_HUB_OFFLINE=1 ...`）为本轮门禁口径；本地默认跑法固化仍待 TD-N-25 清偿。
