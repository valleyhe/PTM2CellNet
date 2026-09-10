# PTM2CellNet Current Status

**Last updated: 2026-09-03（Gate-0 本地 donor 队列复审）**

当前权威分析是仓库根目录的
[`project_analysis_20260901.md`](../project_analysis_20260901.md)。
2026-08-24 及更早的综合分析和点时报告已归档：`archive/20260827/reports/`、
`archive/20260824/reports/`、`archive/20260822/reports/`；
各批次清单见对应 `archive/YYYYMMDD/MANIFEST.md`。

## Quick Reference

- **验证环境**：主进程 Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用；PerturbGen 独立环境 conda `perturbgen`（Python 3.11.15）已于 2026-08-23 完全建立并通过 GPU 实测（evidence 见 `outputs/perturbgen/env_evidence_20260823.json`，复现入口 `scripts/setup_perturbgen_env.sh` + `scripts/download_perturbgen_wheels.sh`）。
- **测试基线（2026-09-02 更新）**：本轮 DAVF/PerturbGen 聚焦回归 **157 passed / 2 skipped / 16 warnings**；真实资产 CUDA 桥接 **3 passed / 7 warnings**。2026-09-01 全量离线基线仍为 **2357 passed / 16 skipped / 47 warnings / exit 0（536.65s）**。
- **静态/构建质量**：`compileall` 通过；`ruff check src scripts tests` 全绿；`mypy src` **23 errors / 8 files**（= TD-N-06 基线 5 处 + TD-N-11 PerturbGen 18 处，无新增）；`python -m pip check` 当前有 3 个环境冲突（ptm2cellnet/NumPy、scgpt/scvi-tools、ssh-unit/torchaudio），详见权威分析 TD-N-34 和验证章节。
- **技术债闭环（本轮确认，提交 `63ebf75`）**：
  - **TD-N-24（高）已修复**：`src/analysis/gene_mapper.py:150-186` 新增 `_call_external_mapper`——外部 `UniProtMapper` 调用包裹 daemon 线程 + `_EXTERNAL_MAPPER_TIMEOUT_S` 硬超时，超时转既有异常分支；生产 `/predict` 挂死风险解除。回归锚点 `tests/unit/analysis/test_gene_mapper.py`（TD-N-33 合并后单一入口）。
  - **TD-N-10（高）已修复**：`.github/workflows/perturbgen-real-assets.yml:44-46` 安装步骤追加 `pip install -r requirements-analysis.txt`，Gate-4 门禁 anndata 缺口闭环。
  - **「CI analysis job 缺失」表述作废**：`.github/workflows/ci.yml:59-84` 已存在 `analysis` job，且安装 `requirements-analysis.txt` 后显式运行 `data_prep/results/pipeline_mocked` 三组测试；旧报告的缺失与静默 skip 判断已关闭。
- **DAVF 正式连接**：Norman 旧主线已有当前 schema-v2 checkpoint；另外已用真实 scPerturb KO/KD 数据分别训练 `checkpoints/davf/davf_ko_dixit/best_model.pt` 与 `checkpoints/davf/davf_kd_nadig/best_model.pt`，两者均为 `latent_dim=64`、`num_genes=4018`，并通过 checkpoint contract、真实 scVI 解码和 token/decoder index 分离测试。正式生物学方向准确率尚未宣称。
- **PerturbGen 双路径集成**：已在官方 foundation checkpoint 上完成可复现的本地六阶段 smoke adaptation（Datlinger 2021 M0，LCK，227×2001 预测矩阵和 embedding asset 均已导出）；这证明代码链和 checkpoint 可运行，不等于多 donor 生物学验收。`run_davf_perturbgen_e2e.py` 已自动执行 DAVF→三方方向 gate→独立双路径 stage plan，≥3 donor 合规 cohort 仍待补齐。
- **DAVF × PerturbGen E2E 接线（2026-09-02）**：新增 `src/integration/perturbgen/orchestrator.py` 与 `scripts/run_davf_perturbgen_e2e.py`。真实 scVI context → route-specific DAVF → 三方方向 gate → 独立 source/within-state PerturbGen stage plans 已接通；KO/KD 结果可由 `scripts/merge_davf_perturbgen_reports.py` 按 canonical Ensembl ID 合并。默认只生成真实方向证据清单，必须显式加 `--run-perturbgen` 才执行外部六阶段，避免把不完整效用证据误报为 PASS。
- **Gate-0 复审（2026-09-03）**：重新审计本机 30 个 scPerturb H5AD，26 个可读、0 个满足正式 `normal/disease + raw counts + explicit donor + ≥3 shared donors + Ensembl` 契约；证据为 `outputs/perturbgen/spike/20260903_donor_audit/evidence.json`。DatlingerBock2021 的真实 preflight 因缺少 `state`、`donor` 被拒绝，不能作为正式效用数据。
- **R-01～R-03 状态**：工程契约与 synthetic batch 测试通过；三路 pLM 权重已本地化；真实图/扰动训练和科学验收仍未验证（依赖 replogle/scgenescope 数据供给）。
- **剩余需求差距**：正式 REQUIREMENTS v2.1/v2.2 共 24 项 Complete 维持成立；另有 2026-08-21 PerturbGen 后续方案仍未完成合规 donor cohort、held-out DAVF 方向指标和 formal evidence，逐项清单见权威分析 §4。
- **覆盖率门禁**：默认 CI branch coverage `fail_under=74`；本轮只执行了不带 `--cov` 的全量回归，当前覆盖率不宣称为历史 75.27% 数值；刷新任务仍依赖一致的 CI/独立环境。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)、[真实资产验收](guides/real_assets_acceptance.md)
- [项目代码与文档综合分析（2026-09-01，当前权威）](../project_analysis_20260901.md)、[归档清单（2026-09-01）](../archive/20260901/MANIFEST.md)
- [PerturbGen 双路径整合方案（v2.0，现行需求基线）](DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
- [测试覆盖率治理](TEST_COVERAGE.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome、PerturbGen 外部环境和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本轮真实资产测试已验证本地 KO/KD
DAVF→scVI→PerturbGen bridge，但这不等价于真实生物学效果或外部服务已验收。网络隔离 + HF 离线跑法
（`unshare -rn env HF_HUB_OFFLINE=1 ...`）为本轮门禁口径；本地默认跑法固化仍待 TD-N-25 清偿。
