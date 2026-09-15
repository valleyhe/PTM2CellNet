# PTM2CellNet Current Status

**Last updated: 2026-09-15（第八轮后续：落地 `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §6.2 的代码契约层——阶段 0 冻结研究设计 config、§4.1/§4.2/§4.3/§4.4/§4.6 全部标准表契约、有符号网络传播、per-cell-type Ensembl 交集、E2E 兼容 candidate spec 生成 + downstream target-set sidecar/评估（5 个源码模块 + 4 个 CLI + 84 个新测试，合成数据契约级验证）；真实 PTM/网络/benchmark 资产仍是外部输入（方案 §10）；第八轮 M6 冻结记录保留；同日后续完成全仓文档同步至新指南——README、`.planning/{task_plan,STATE,ROADMAP,PROJECT,REQUIREMENTS,progress,findings}`、`index.rst`、E2E/bridge 指南交叉引用、2026-08-21 方案注记与 TEST_COVERAGE 失效链接修复）；2026-09-15 批次 A/B/C 当前事实记录见 Quick Reference 首条）**

当前权威分析是仓库根目录的
[`project_analysis_20260914.md`](../project_analysis_20260914.md)。
2026-09-13 综合分析、2026-09-10 综合分析与修复日志、2026-09-01 及更早报告、根目录 2026-08 stub 已归档：
`archive/20260914/`（含 20260913 权威报告与 task_plan）、`archive/20260913/`（含 20260910 权威报告）、`archive/20260910/reports/`（含 20260901）、
`archive/20260901/reports/`、`archive/20260827/reports/` 等；各批次清单见对应
`archive/YYYYMMDD/MANIFEST.md` 或 `ARCHIVE_MANIFEST.md`。

## Quick Reference

- **2026-09-15 当前事实（PTM activity A/B/C 批次）**：批次 A 新增 `src/analysis/ad_deg_table.py` 与 `scripts/build_ad_deg_table.py`，产出 aggregate 八列和 donor-level 长表，复用 donor log2 normalization/Welch/BH，manifest 显式记录 normal reference、effect scale 与 output contracts，并新增 A 核心/CLI/双消费者测试；批次 B 使 `scripts/run_davf_perturbgen_e2e.py` 显式接入 `--downstream-target-sidecar`，对 gated candidate 的 `result.h5ad` 以 `pred_counts` 为 baseline、`X` 为 perturbed 计算 donor-level delta，写入 payload/lineage，`matches_predicted` 与 `matches_observed` 分开记录，source 三方 gate 行为不变，下游 lineage 已接线；批次 C 为 `PropagationConfig.max_paths_per_seed` 增加正整数校验，超限 hard fail 且不截断，记录 per-seed diagnostics/manifest。相关定向测试 **81 passed、8 warnings**；ruff check、目标文件 format check、mypy src（172 files、0 errors）、requirements consistency、compileall、git diff --check 均通过；全量非 slow/gpu 为 **2780 passed / 22 skipped**，另有 1 个已知既有 real DAVF checkpoint failure；全仓 format check 仍有 **325 个既有待格式化文件**；真实冻结资产测试 **1 passed**，GPU probe 为 Tesla P40 / CUDA 11.8 / torch CUDA available。当前无冻结 `ptm_research_config`，且 KSTAR/PhosR/activity benchmark 等外部资产缺失，不能执行真实 AD CLI/D1 六阶段；不宣称 biology PASS。
- **PTM activity → AD 交集主线契约层（2026-09-14，第八轮后续）**：`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §6.2 全部落地——`src/analysis/ptm_research_config.py`（阶段 0 冻结研究设计：objective/axis/contrast/method/network release/cell types/DEG 阈值/传播参数/七字段 semantic_context 模板，fail-fast 校验）、`src/analysis/ptm_activity.py`（§4.1 输入契约 + §5.1 预处理（显式重复位点策略、site/total 归一化缺失保持 NaN、显式 gene map、无 donor 行标 exploratory）+ §4.2 activity 表解析）、`src/analysis/signed_network.py`（§4.3 边契约：unsigned 只进 coverage、平行边同号取强、异号整体剔除；§5.3 有符号简单路径传播：`activity × Π(sign×confidence) × decay^length`，gene_edge_types 终止，n_paths/path_length_min/coverage/degree_normalized 全记录；不调用 `signaling_network.py` 硬编码路径）、`src/analysis/ptm_gene_score.py`（§4.4 per (source,target) score 表、`prediction_status=direction_only` 无 PTM 侧显著性；§5.4 交集 membership∈concordant/discordant/PTM_only/AD_only × tier∈formal/exploratory，`I_c`=concordant∧formal，`n_cell_types_supported` 注明非独立证据）、`src/integration/perturbgen/downstream_target_evaluation.py`（sidecar 契约 + per-target delta 的 matches_predicted/matches_observed 分字段一致性）、CLI ×4（`run_ptm_activity` / `build_ptm_global_gene_scores` / `build_ptm_ad_intersections` / `build_celltype_candidate_specs`：产出 E2E 可直接消费的 candidate-spec/v1、真实 per-cell-type context_cell_index、source 无 DEG 只进 exploratory、`--embedding-vocab` 下 source 无 token 硬失败）。测试：5 个单元文件 + `tests/integration/test_ptm_activity_pipeline.py` 四阶段全链共 **84 个新测试全绿**；mypy 171 文件 0 errors。指南：`docs/guides/ptm_activity_pipeline.md`。**合成数据只证明契约与管线接通；真实 PTM 定量、signed 网络资产、activity benchmark 仍是外部输入（方案 §10），无 PTM→RNA 模型训练，Gate-E 不触发（§6.4）。**
- **第八轮 M6 冻结与 GPU 前置（2026-09-14）**：按 `project_repair_report_20260914.md` §4 推进——① 第七轮报告 #3（高）闭合：GSE174367 EX 完成 between_donor M6 数据契约冻结（`outputs/perturbgen/frozen/20260914_gse174367_ex/`：train 12 / held-out 6 分层 seed=2 性别平衡 split、5 候选 APP/PSEN1/BACE1/MAPT/APOE（KO，AD 三通路）、90-run acceptance plan、`_validate_frozen_cohort_asset` 真实数据 PASS 6,369 cells；ENSG 经 cohort var 与 embedding 词表双重查证）；② 第七轮报告 #1 前置闭合：5×99 matched-null selection manifest 真实生成（EX 表达特征 + token_rank，fc 显式 unavailable/default），并在此过程中发现并修复 `null_selection._token_values` 拒绝正式词表特殊 token（仅 `<cls>`/`<eos>`/`<mask>`/`<pad>`）的真实缺口，其他非法 key 仍硬失败；③ 新增 `tests/real_assets/test_real_frozen_cohort.py` 冻结漂移守护（gate 开启真实跑 18.25s PASS）与 bridge 指南 GPU runbook（六阶段 + Python API null 批跑 + candidate_spec 研究输入说明）。**M6 冻结是数据契约，不是生物学 PASS**；六阶段 GPU 执行、matched-null 批跑与 Gate-E 仍未执行。当前 `nvidia-smi`/CUDA 探针成功，未进行驱动修复。
- **验证环境**：主进程 Python 3.12.13、PyTorch 2.4.1+cu118、CUDA 可用；PerturbGen 独立环境 conda `perturbgen`（Python 3.11.15）已于 2026-08-23 完全建立并通过 GPU 实测（evidence 见 `outputs/perturbgen/env_evidence_20260823.json`，复现入口 `scripts/setup_perturbgen_env.sh` + `scripts/download_perturbgen_wheels.sh`）。
- **第六轮综合处理（2026-09-14）**：第四/五轮未提交修改按 3 个语义提交落 main（提交前确认与 `origin/main` 同步于 `219b81f`）；6 处 living docs 反向漂移修复（README 研究路径与格式、bridge §1、方案 §11 复审注记、CHANGELOG [Unreleased]、REQUIREMENTS A-01/A-05/A-06）；`project_analysis_20260913.md` 与 `task_plan.md` 归档至 `archive/20260914/`（`project_repair_report_20260913.md` 核验仍准确、保留）。4 个并行子代理对抗审查确认 F-01/F-02/F-03 真实闭合，新识别 F-10～F-16（高：F-10 frozen 验收路径未传播 `pairing`、F-14 AD 审计脚本口径过时；中：F-11 统计 lineage 缺 pairing、F-12 held-out state 覆盖无约束、F-13 standardize Diagnosis per-sample 校验缺口；低：F-15/F-16）及 TD-14-01～06，策略见 `project_analysis_20260914.md` §6。第五轮全量回归 **2646 passed / 1 failed（已知真实资产基线）/ 21 skipped / 773.33s**、mypy 166 文件 0 errors；第六轮 compileall 通过、聚焦回归 267 passed（本轮仅文档与归档，未触碰代码路径）。
- **当前验证结果（2026-09-13 第四轮，本轮修复）**：全量命令 `python -m pytest -m "not slow and not gpu" --timeout=300` 结果为 **2639 passed / 1 failed / 21 skipped / 67 warnings / 1412.90s**；唯一失败仍为 `tests/integration/test_scvi_davf_connection.py::test_real_current_davf_direction_keeps_token_and_decoder_indices_separate`（本地旧 checkpoint `latent_davf_perturbgen_4018` 的 `scvi.gene_names` 非 canonical ENSG，contract hard-fail；与第三轮基线同一已知真实资产失败，本轮 +16 个新测试全过、无新增失败）。同轮 `ruff check src scripts tests` 通过、`mypy src/ --ignore-missing-imports` 为 **166 个源文件、0 errors**、requirements consistency 通过（274 lock pins）；本轮触碰文件已 `ruff format`，全仓 format 债维持"一次性独立 PR"取舍未批量改写。
- **第四轮修复落地（F 系列）**：F-01 E2E 统计接续（`run_davf_perturbgen_e2e.py --assemble-statistical-evidence --deg-table --null-distribution-manifest`，串接质量提取→formal 评估输入→empirical-p 聚合→BH-FDR→dual-path AND，lineage 写回 `statistical_evidence`；重放核心抽离至 `src/integration/perturbgen/replay_evaluation.py`）；F-02 跨候选共享准备（`orchestrator.build_shared_prepare_plans` 每 route 公共执行一次 tokenise/train_mask/train_decoder 于 `<root>/<route>/_prepare/`，候选只执行 perturb/export/report，`@artifact` 引用解析到共享产物）；F-03 生成端 donor 行绑定（`build_scperturb_latent_pairs` + `build_davf_scperturb_pairs.py --donor-obs-column --train-donors --held-out-donors`，NPZ 写 `target_donors` 与 `dataset.donor_rows`）；F-09 边界文档化（runner `StagePlan` 属工程执行层，formal invocation wrapper 是唯一公开正式入口）。GPU matched-null 批跑、真实 cohort 与合规 checkpoint 重训仍未执行（A-01/A-05 资产边界不变）。
- **当前验证结果（2026-09-13 第三轮，提交 `96ee544`，历史记录保留）**：全量命令结果为 **2623 passed / 1 failed / 21 skipped / 67 warnings / 933.65s**；失败原因同上（旧 checkpoint 非 canonical ENSG）。同轮 ruff check / mypy（165 文件 0 错误）/ compileall / requirements（274 pins）全部通过；format check 仍为 339 文件待格式化。`96ee544` 已 push，`main...origin/main` 为 `0 0`（remote 已更名 `PTM2CellNet` 并更新本地 URL）。
- **历史测试基线（仅背景）**：2026-09-10 的 2573/2578 通过记录、2026-09-02 的 157 项聚焦回归和 3 项 CUDA bridge 记录均不作为当前验收结果；旧的 2610 通过数字不再作为当前结果。
- **静态/构建质量（2026-09-13 当前记录）**：`compileall` 通过；`ruff check src scripts tests` 通过；`mypy src/ --ignore-missing-imports` 为 **165 个源文件、0 errors**；requirements consistency 通过且为 **274 lock pins**；`ruff format --check src scripts tests` 退出 1，**339 文件需格式化、124 文件已格式化**，未批量改写。
- **技术债闭环（本轮确认，提交 `63ebf75`）**：
  - **TD-N-24（高）已修复**：`src/analysis/gene_mapper.py:150-186` 新增 `_call_external_mapper`——外部 `UniProtMapper` 调用包裹 daemon 线程 + `_EXTERNAL_MAPPER_TIMEOUT_S` 硬超时，超时转既有异常分支；生产 `/predict` 挂死风险解除。回归锚点 `tests/unit/analysis/test_gene_mapper.py`（TD-N-33 合并后单一入口）。
  - **TD-N-10（高）已修复**：`.github/workflows/perturbgen-real-assets.yml:44-46` 安装步骤追加 `pip install -r requirements-analysis.txt`，Gate-4 门禁 anndata 缺口闭环。
  - **「CI analysis job 缺失」表述作废**：`.github/workflows/ci.yml:59-84` 已存在 `analysis` job，且安装 `requirements-analysis.txt` 后显式运行 `data_prep/results/pipeline_mocked` 三组测试；旧报告的缺失与静默 skip 判断已关闭。
- **DAVF 正式连接**：Norman 旧主线已有当前 schema-v2 checkpoint；另外已用真实 scPerturb KO/KD 数据分别训练 `checkpoints/davf/davf_ko_dixit/best_model.pt` 与 `checkpoints/davf/davf_kd_nadig/best_model.pt`，两者均为 `latent_dim=64`、`num_genes=4018`，并通过 checkpoint contract、真实 scVI 解码和 token/decoder index 分离测试。正式生物学方向准确率尚未宣称。PTM classifier 只预测 site presence；E2E 的候选方向来自用户假设或逐 site override，不是原始位点自动推断。
- **PerturbGen 双路径集成**：已在官方 foundation checkpoint 上完成可复现的本地六阶段 smoke adaptation（Datlinger 2021 M0，LCK，227×2001 预测矩阵和 embedding asset 均已导出）；这证明代码链和 checkpoint 可运行，不等于多 donor 生物学验收。六阶段为 `tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report`；自第四轮起每条 route 的准备三阶段在 `_prepare/` 公共执行一次、候选循环只执行两路 perturb 与 export/report（F-02）。
- **DAVF × PerturbGen E2E 接线（2026-09-02）**：新增 `src/integration/perturbgen/orchestrator.py` 与 `scripts/run_davf_perturbgen_e2e.py`。候选准入链为 `candidate_spec → scVI/PTMDirectionMapper → DAVF decode → 三方方向 gate → CandidateEvidence → PerturbGenInvocation`；默认只生成 gate/invocation，必须显式加 `--run-perturbgen` 才执行外部六阶段。`source_intervention=[src]` 与 `within_state=[tgt]+pert_tps` 是两个实验场景，正式 dual-path 结论保留 AND。
- **Gate-0 复审（2026-09-13）**：当前正式 cohort=0；重新审计本机 30 个 scPerturb H5AD，26 个可读、0 个满足正式 `normal/disease + raw counts + explicit donor + ≥3 shared donors + Ensembl` 契约；证据为 `outputs/perturbgen/spike/20260913_donor_audit/evidence.json`。DatlingerBock2021 的真实 preflight 因缺少 `state`、`donor` 被拒绝，不能作为正式效用数据。
- **AD 队列与 between_donor Gate-0（2026-09-14，第五轮）**：`data/AD` 四个 GEO 脑队列审计结论 `GATE0_BLOCKED_SEMANTICS_AND_LABELS`（`outputs/perturbgen/spike/20260914_ad_cohort_audit/evidence.json`：原 shared-donor 语义与 AD case-control 结构冲突，且 3/4 数据集缺 donor 标签）。按用户选项 B 决策，`PerturbGenDataSpec.pairing` 新增 `between_donor`（两组 donor 不相交、各 ≥3；默认 `within_donor` 行为不变），E2E 以 `--perturbgen-cohort-pairing` 显式声明。GSE174367 已标准化为 `data/AD/standardized/GSE174367_ad_cohort.h5ad`（61,472 cells × 58,676 canonical ENSG；donor 推导=SampleID×唯一供体协变量向量 18/18，运行时重验；45 条版本化/`_PAR_Y` ENSG 按同基因求和合并），between_donor preflight 7/7 细胞类型 PASS（`outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json`）。这是数据契约验收，不是生物学 PASS；GSE157827/GSE188545/GSE147528 仍缺 donor/cell 注释或 cell calling，四队列未合并。
- **R-01～R-03 状态**：工程契约与 synthetic batch 测试通过；三路 pLM 权重已本地化；真实图/扰动训练和科学验收仍未验证（依赖 replogle/scgenescope 数据供给）。
- **剩余需求差距**：正式 REQUIREMENTS v2.1/v2.2 共 24 项 Complete 维持成立；U-01～U-05 等既有接口只代表代码底座和边界复核，不是本轮新增或正式科学结果。真实合规 donor cohort、held-out DAVF 方向指标、自动统计接续和 formal evidence 仍未完成，逐项清单见权威分析 §4。
- **2026-09-10 收口修复**：G-1/N-05 matched-null 与 E2E 组装、G-2 DAVF confidence/davf_score、G-3 gate hard-fail/report binding、目标 optional dependency 边界和 PMADS iterrows 回归已在代码与离线测试闭合；真实 cohort、Gate-E ≥200 benchmark 和 T4/Gate-5 evidence 仍未执行。
- **覆盖率门禁**：默认 CI branch coverage `fail_under=74`；本轮只执行了不带 `--cov` 的全量回归，当前覆盖率不宣称为历史 75.27% 数值；刷新任务仍依赖一致的 CI/独立环境。
- **范围裁剪**：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展已取消；兼容代码可保留，不得重新列为 roadmap 目标。

## DAVF × PerturbGen 研究边界（当前）

当前流程要把四个终点分开记录（第 0 项为 2026-09-14 新增上游主线）：

0. **上游 PTM activity 主线**：`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`（v1.0）与 `docs/guides/ptm_activity_pipeline.md` 定义的 `真实 PTM（全局）→ KSTAR/PhosR activity（外部环境）→ signed network 传播 → global gene score → AD per-cell-type DEG 同方向交集 → gene-level intervention candidate spec`。契约层已落地（合成数据验证）；方向字段分字段记录、source/target 不合并、无独立 null 前 `direction_only` 不写 PTM 侧 q 值。方案 §10 六项外部输入就绪前，交集与候选只能是合成契约证据。
1. **候选准入**是推理桥接。`PTM proposal/candidate_spec → scVI/PTMDirectionMapper → DAVF decode 方向 → 三方 gate → CandidateEvidence → PerturbGenInvocation` 只决定候选能否进入下游；`PerturbGenInvocation` 要求 gate 通过。普通 CLI 选择 `perturb`/`--path` 时还必须提供 `--e2e-gate-report` 并绑定通过的 invocation，底层 runner 只执行 `StagePlan`。上游主线阶段 5 产出的 `candidate-spec/v1` 走同一入口，E2E 行为不变。
2. **准备与运行**是 PerturbGen 六阶段。`source_intervention=[src]` 是状态转移前干预，`within_state=[tgt]+pert_tps` 是目标状态内干预，二者是实验场景，不是两条独立工作链。自第四轮起，每条 route 的 `tokenise/train_mask/train_decoder` 在 `<root>/<route>/_prepare/` 公共执行一次，候选循环只执行两路径 `perturb`、`export_gene_embeddings/report`（F-02 已落地）。
3. **统计验收**需要把真实 null、候选 empirical-p/q、未扰动质量和 dual-path 结果接续起来。自第四轮起 E2E 支持 `--assemble-statistical-evidence` 自动串接这组统计（F-01 代码落地）；自第五轮起 AD case-control 队列可经 `--perturbgen-cohort-pairing between_donor` 进入 Gate-0（GSE174367 已过数据契约 preflight）；真实 GPU matched-null 运行与完整 E2E 队列执行仍缺（A-05），smoke/synthetic 结果不等于 biology PASS。
4. **资产生命周期与索引**：Workflow B 为 `encoder → 冻结 embedding asset → LatentDAVF 重训 → Gate-E` 的独立资产生命周期；基础 encoder export 不消费候选 perturb 结果，也不回灌当前 DAVF。正式合并使用 canonical Ensembl ID，PerturbGen token index 与 scVI decoder index 不得混用。

观测方向是 donor-level disease−normal，DAVF 方向是 `decode(z_intervened)−decode(z_context)`。当前 gate 直接比较 up/down，统一参考轴仍是待完成的研究任务；每次正式分析必须记录 context、intervention、比较基准和研究目标。病程一致性、状态逆转与治疗因果性不能混称，也不能用全局同号/取反替代参考轴。三路同号只说明当前输入下的一致，不能单凭来源复用把它写成三个独立证据；独立表达观测与 DAVF 训练 cohort/donor 隔离必须有可追溯证明。正式 PASS 仍要求真实 `normal/disease` raw counts、canonical Ensembl、显式 donor、至少 3 个可评估 donor（`within_donor` 跨态共享或 `between_donor` 两组各 ≥3，按 cohort 显式声明）、冻结 scVI gene order/embedding manifest，以及真实 null/质量/效用统计；synthetic、smoke 和 bridge 只证明工程契约。

## 2026-09-13 本轮修复状态

本节只记录 2026-09-13 的源码审计、修复和检查；上方 2026-09-10 的回归数字与
历史检查日期不因本轮未运行真实资产而改写。修复细节见
[`project_repair_report_20260913.md`](../project_repair_report_20260913.md)；
综合对抗分析见 [`project_analysis_20260914.md`](../project_analysis_20260914.md)。

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
- 本轮 formal candidate/invocation 现在必须携带相同的七字段 `semantic_context`
  （`context`、`intervention`、`comparison_baseline`、`reference_axis`、
  `research_objective`、`evidence_source`、`cohort`）；`research_objective` 仅允许
  `association`、`replication`、`reversal`。缺失、空值、非法 objective 或
  intervention 与 KO/KD route 不一致都会 fail-fast。`checkpoint.scvi.gene_names`
  同时执行 canonical ENSG、无版本后缀、无重复的 strict check；因此本地旧
  `checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt` 已被暴露为不合规资产，
  不是可用的正式 checkpoint。
- 本轮 E2E 的 `statistical_evidence` 只明确为 `inconclusive`，并写出
  `scientific_acceptance=false`；不会自动接续或生成 null、未扰动质量、候选 p/q 或
  双路径科学结论。正式 cohort、GPU null/多 seed、Gate-E benchmark 和统一 evidence
  lineage 仍未完成；mock、synthetic、smoke、bridge 与工程回归均不等于 biology PASS。
- 双路径评价已要求显式 KO/KD route：KO/up 才要求 `mask,pad,delete`，KD/up 只要求
  `mask`，down 的 `overexpress` 与 route 独立；Volta 已同步 frozen replay 调用。
  其 verifier 还要求 manifest、删除默认 mask fallback，并在任一 verify 失败时 exit 1。
- N-05/evaluator 的 replay 记录现包含实际提取列名、阈值、donor/top-k/bootstrap
  参数、manifest lineage、runner `outputs` 已登记的 `result_h5ad.sha256` 和显式
  bootstrap seed；seed 从对应 perturb artifact 的 manifest seed 派生。assembler
  只复制该既有 hash，不新算或猜 hash。
- **同日后续一轮（U-01～U-07）**：formal 评估拒绝 uniform/手填 p；候选 p 的
  `conservative_max_required_runs` 聚合已接线；匹配 null 有
  `run_matched_null_stages` 生成端；未扰动质量可从 h5ad 提取；训练/E2E 可绑定
  `train_donors`/`held_out_donors`。详情见
  [`project_repair_report_20260913.md`](../project_repair_report_20260913.md)。
  这些是代码契约，不是 Gate-0 合规 cohort 或正式 GPU 99-null 生物学 PASS。
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
- [PTM activity → AD 交集执行方案（v1.0，2026-09-14 当前主线）](PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md)、[PTM activity 管线指南](guides/ptm_activity_pipeline.md)
- [项目代码与文档综合分析（2026-09-14，当前权威）](../project_analysis_20260914.md)、[归档清单（2026-09-14）](../archive/20260914/ARCHIVE_MANIFEST.md)、[归档清单（2026-09-13）](../archive/20260913/ARCHIVE_MANIFEST.md)
- [PerturbGen 双路径整合方案（v2.0，现行需求基线）](DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
- [测试覆盖率治理](TEST_COVERAGE.md)

## 解释测试结果时的边界

默认 unit/integration/E2E 测试大量使用合成数据、mock 或本地 fixture；真实跨尺度图/扰动数据、DAVF、ESM-3、CPTAC、
外部 UniProt/KEGG/Reactome、PerturbGen 外部环境和多 GPU DDP 验收必须显式设置
`PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 并提供相应资源。本轮真实资产测试已验证本地 KO/KD
DAVF→scVI→PerturbGen bridge，但这不等价于真实生物学效果或外部服务已验收。网络隔离 + HF 离线跑法
（`unshare -rn env HF_HUB_OFFLINE=1 ...`）为本轮门禁口径；本地默认跑法固化仍待 TD-N-25 清偿。
