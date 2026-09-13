# PTM2CellNet Current Status

**Last updated: 2026-09-13（追加本轮真实修复状态；2026-09-10 历史检查基线保留）**

当前权威分析是仓库根目录的
[`project_analysis_20260910.md`](../project_analysis_20260910.md)。
2026-09-01 及更早的综合分析、docs/ 下 7 个 2026-08 初点时报告已归档：
`archive/20260910/reports/`（含 20260901 权威报告）、`archive/20260901/reports/`、
`archive/20260827/reports/` 等；各批次清单见对应 `archive/YYYYMMDD/MANIFEST.md`。

## Quick Reference

- **验证环境**：主进程 Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用；PerturbGen 独立环境 conda `perturbgen`（Python 3.11.15）已于 2026-08-23 完全建立并通过 GPU 实测（evidence 见 `outputs/perturbgen/env_evidence_20260823.json`，复现入口 `scripts/setup_perturbgen_env.sh` + `scripts/download_perturbgen_wheels.sh`）。
- **测试基线（2026-09-10 更新）**：最终离线回归 **2573 passed / 15 skipped / 7 deselected / 55 warnings / exit 0（805.68s）**（命令排除 `slow`、`gpu`、`real_assets`；含 9 月批次和本次 G-1/G-2/G-3/N-05/技术债回归）。此前 2026-09-02 聚焦回归 157 passed、真实资产 CUDA 桥接 3 passed 的记录不变。
- **静态/构建质量（2026-09-10）**：`compileall` 通过；`ruff check src scripts tests` 全绿；`mypy src` **0 errors / 161 files**；requirements consistency 通过；全仓 `ruff format --check` 仍有 329 个文件需要格式化，未在本轮批量改写；`python -m pip check` 当前有 3 个环境冲突（ptm2cellnet/NumPy、scgpt/scvi-tools、ssh-unit/torchaudio），详见权威分析和验证章节。
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
- **2026-09-10 收口修复**：G-1/N-05 matched-null 与 E2E 组装、G-2 DAVF confidence/davf_score、G-3 gate hard-fail/report binding、目标 optional dependency 边界和 PMADS iterrows 回归已在代码与离线测试闭合；真实 cohort、Gate-E ≥200 benchmark 和 T4/Gate-5 evidence 仍未执行。
- **覆盖率门禁**：默认 CI branch coverage `fail_under=74`；本轮只执行了不带 `--cov` 的全量回归，当前覆盖率不宣称为历史 75.27% 数值；刷新任务仍依赖一致的 CI/独立环境。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## 2026-09-13 本轮修复状态

本节只记录 2026-09-13 的源码审计、修复和检查；上方 2026-09-10 的回归数字与
历史检查日期不因本轮未运行真实资产而改写。详细记录见
[`project_repair_report_20260913.md`](../project_repair_report_20260913.md)。

- `scripts/run_perturbgen_pipeline.py` 现要求通过的 E2E report invocation 与当前
  base YAML 的 gene、mode、声明的 Ensembl ID/route、`pipeline.random_seed` 和授权
  path 子集绑定；报告中的显式 `perturbgen_config_path` 若存在也必须指向当前 base
  config。`paths` 仍按 `PerturbGenInvocation` 的授权双路径集合解释，不把它误当成
  materialized stage YAML。
- `scripts/run_davf_perturbgen_e2e.py` Gate-0 继续在内存副本执行准备校验，并对原始
  tokenise 输入的 Ensembl 列执行 canonical 硬检查；已登记外部 tokeniser 实际读取
  原始 `h5ad_path`。登记 `GF_tokenisation.py:223-226,238,301-305` 已静态核对同一
  原始文件的 counts→`X`→恢复 counts→重算 `n_counts` 接线；prepared copy 的新增
  `n_counts` 不需 materialize。本轮未运行真实 tokenisation，也没有新增 X/layer 转换。
- 双路径评价已要求显式 KO/KD route：KO/up 才要求 `mask,pad,delete`，KD/up 只要求
  `mask`，down 的 `overexpress` 与 route 独立；Volta 已同步 frozen replay 调用。
  其 verifier 还要求 manifest、删除默认 mask fallback，并在任一 verify 失败时 exit 1。
- N-05/evaluator 的 replay 记录现包含实际提取列名、阈值、donor/top-k/bootstrap
  参数、manifest lineage、runner `outputs` 已登记的 `result_h5ad.sha256` 和显式
  bootstrap seed；seed 从对应 perturb artifact 的 manifest seed 派生。assembler
  只复制该既有 hash，不新算或猜 hash。candidate p 仍由外部输入，候选级多
  path/mode/seed empirical-p 聚合规则尚未定义，`uniform_candidate_pvalue` 不能作为
  正式科学验收。
- Volta 首轮 targeted 检查为 38 passed；修正 CSV 解析并更新既有契约后，4 文件现有
  回归 `test_results`、`test_eval_assembly`、`test_evaluate`、`test_frozen` 合计
  `52 passed`、exit 0，4 个旧 frozen fixture `14 passed`、exit 0。首次 10/4 失败
  保留为迭代记录，未弱化生产契约。该结果不等于真实 M6/OE biology PASS。
- Volta 首次真实生成链复核发现 frozen replay 读取 DEG CSV 时 `fdr` 为字符串，
  与 `results` 的数值筛选契约不符；改为 `pd.read_csv`/JSON DataFrame 后，明确 synthetic
  AnnData/runner metadata 的 assembler→evaluator 实际生成 JSON 经 frozen 独立重算无
  mismatch、`independent=True`、正常 CLI verify exit 0，篡改生成 verdict 后 exit 1。
  这是工程/synthetic 结果，不是正式 M6 biology 验收。
- 本 worker 最终补跑此前未纳入 89 项集合的三个既有入口，共 `25 passed, 8 warnings`、
  exit 0；仅更新缺显式 `pipeline.random_seed` 的既有 fixture，没有增加生产 fallback。
- 本轮未验证真实合规 cohort、训练-only/held-out donor split、Gate-E ≥200、正式
  多 seed/null 或 M6/OE；这些仍是外部资产/协调依赖，不把 9/10 的工程回归数字扩写
  成生物学 PASS。

## 文档入口

- [安装指南](guides/installation.md)、[数据接入指南](guides/data_integration.md)、[训练指南](guides/training.md)、[部署指南](guides/deployment.md)、[真实资产验收](guides/real_assets_acceptance.md)
- [项目代码与文档综合分析（2026-09-10，当前权威）](../project_analysis_20260910.md)、[归档清单（2026-09-01）](../archive/20260901/MANIFEST.md)
- [PerturbGen 双路径整合方案（v2.0，现行需求基线）](DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
- [测试覆盖率治理](TEST_COVERAGE.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome、PerturbGen 外部环境和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本轮真实资产测试已验证本地 KO/KD
DAVF→scVI→PerturbGen bridge，但这不等价于真实生物学效果或外部服务已验收。网络隔离 + HF 离线跑法
（`unshare -rn env HF_HUB_OFFLINE=1 ...`）为本轮门禁口径；本地默认跑法固化仍待 TD-N-25 清偿。
