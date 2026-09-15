# PTM2CellNet 修复与验收报告（2026-09-15）

## 结论先行

本轮代码修复项已完成：PTM global score 已绑定冻结 `config.contrast`，source DEG 已执行正式方向/FDR/donor gate；frozen formal replay 已强制 formal evidence、独立 empirical p-value、eval-input/report lineage 绑定，并保留 route 级共享 `prepare_root`。这些结论属于代码契约与聚焦回归层，不能上升为正式生物学结论。

最新分析文件仍是 [`project_analysis_20260914.md`](/home/scu/PTM2CellNet/project_analysis_20260914.md:201)，仓库不存在 `project_analysis_20260915.md`；但前者 §7 的标题已经记录 2026-09-15 当前事实。当前正式 biology PASS = **0**：没有运行 formal GPU 六阶段、真实 matched-null、正式统计复算或 Workflow B Gate-E。冻结 cohort/manifest 的契约通过、synthetic/engineering 测试和 CUDA preflight 都不等于 biology PASS。

## 1. 本轮已完成的修复

### 1.1 PTM contrast 与 source gate

- `build_ptm_global_gene_scores.py` 从冻结研究配置读取 `config.contrast`，拒绝 CLI contrast 与配置不一致的输入，并将同一 contrast 写入 activity 过滤和 manifest lineage；见 [`build_ptm_global_gene_scores.py`](/home/scu/PTM2CellNet/scripts/build_ptm_global_gene_scores.py:76) 与 manifest 写入处 [`build_ptm_global_gene_scores.py`](/home/scu/PTM2CellNet/scripts/build_ptm_global_gene_scores.py:116)；对应管线契约见 [`ptm_activity_pipeline.md` §4](/home/scu/PTM2CellNet/docs/guides/ptm_activity_pipeline.md:82)。
- `build_celltype_candidate_specs.py` 不再因 target-set 有交集就直接生成正式 source 候选：source 必须有本 cell type 自己的 DEG 行，且 `observed_direction` 合法、FDR 不超过冻结阈值、normal/disease donor 数均达到配置要求；失败项只进入 exploratory；见 [`build_celltype_candidate_specs.py`](/home/scu/PTM2CellNet/scripts/build_celltype_candidate_specs.py:218)、[`build_celltype_candidate_specs.py`](/home/scu/PTM2CellNet/scripts/build_celltype_candidate_specs.py:228) 和 [`build_celltype_candidate_specs.py`](/home/scu/PTM2CellNet/scripts/build_celltype_candidate_specs.py:241)。指南对该边界的说明见 [`ptm_activity_pipeline.md` §6](/home/scu/PTM2CellNet/docs/guides/ptm_activity_pipeline.md:143)。

### 1.2 Frozen formal replay、独立 empirical p 与 lineage

- formal eval input 现在必须声明 `evaluation_mode=formal`、`evidence_class=empirical_null`、独立 `empirical_*` p-value source、可读取的 `input_json`、匹配的 64 位内容摘要和 contract；formal 输入不能携带手填 `candidate_pvalue`；见 [`frozen_cohort.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/frozen_cohort.py:486) 和 [`frozen_cohort.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/frozen_cohort.py:491)。
- report manifest 与 eval input 的路径、内容摘要、contract、evaluation mode、evidence class、p-value source 必须一致，缺失或篡改直接形成 binding mismatch；正式验收入口读取这些声明后再决定 replay verdict，见 [`frozen_cohort.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/frozen_cohort.py:570) 与 [`run_frozen_acceptance.py`](/home/scu/PTM2CellNet/scripts/run_frozen_acceptance.py:133)。
- candidate empirical p 不再信任已发布的候选层数值；正式统计先从每个 path/mode/seed 的独立结果聚合，再统一执行候选层 BH。正式 assembly 同时拒绝 uniform/外部手填候选 p，并要求从 h5ad 提取未扰动质量；见 [`eval_assembly.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/eval_assembly.py:397) 和 [`eval_assembly.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/eval_assembly.py:459)。

### 1.3 Route 共享 prepare_root

每条 KO/KD route 的 `tokenise → train_mask → train_decoder` 由 shared prepare plan 执行一次；候选循环只复用已登记 artifact 并运行两路 `perturb → export_gene_embeddings → report`。缺少 `@artifact` 或 `prepare_root` 是硬错误，不会静默重建或猜测资产。实现见 [`orchestrator.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/orchestrator.py:535)、[`orchestrator.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/orchestrator.py:566)、[`run_davf_perturbgen_e2e.py`](/home/scu/PTM2CellNet/scripts/run_davf_perturbgen_e2e.py:914) 和 [`run_davf_perturbgen_e2e.py`](/home/scu/PTM2CellNet/scripts/run_davf_perturbgen_e2e.py:674)。

## 2. E2E 全链审阅

当前主线是：

`真实 PTM（全局） → KSTAR/PhosR activity → signed network 传播 → global gene score → AD 每 cell type donor-level DEG 交集 → gene-level candidate spec → scVI context / route-specific DAVF → 独立表达方向 → 三方 gate → PerturbGen invocation → shared prepare → source_intervention + within_state → matched-null/质量/empirical p/BH → dual-path AND → frozen verify`。

1. **PTM 上游与方向语义。** PTM 阶段不按 cell type 运行；KSTAR/PhosR 和 OmniPath/signed network 是外部标准表资产，当前仓库只消费其契约输出。`ptm_site_direction`、`activity_direction`、`predicted_gene_direction`、`observed_direction`、`davf_predicted_direction` 和 `davf_action` 必须分开记录，不能用全局同号或取反代替参考轴。依据见 [`ptm_activity_pipeline.md` §0](/home/scu/PTM2CellNet/docs/guides/ptm_activity_pipeline.md:7) 与 [`PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §3.1](/home/scu/PTM2CellNet/docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md:59)。

2. **Candidate spec 与 source/target 分离。** source PTM 蛋白/激酶是 gene-level intervention 或 surrogate；交集 DEG 是下游 target/evaluation set，不能把 target DEG 改写成 source 的 observed direction。阶段 5 CLI 可生成 `candidate-spec/v1` 和 target sidecar，但文献 PTM site、外部方向假设和七字段 `semantic_context` 仍是研究输入，不由代码臆造；见 [`ptm_activity_pipeline.md` §5–§6](/home/scu/PTM2CellNet/docs/guides/ptm_activity_pipeline.md:129) 与 [`PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §3.3](/home/scu/PTM2CellNet/docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md:84)。

3. **scVI/DAVF 三方准入。** E2E 使用真实 context 行、当前 route 的 DAVF/scVI 坐标和 verified PerturbGen vocabulary；proposal direction、DAVF decode delta、独立 donor-level observed direction 共同决定是否创建 invocation。当前 gate 是工程准入检查，尚未统一 disease-minus-normal 与干预后 decode delta 的科学参考轴，不能宣称因果方向已验证；见 [`davf_perturbgen_e2e.md` §候选清单](/home/scu/PTM2CellNet/docs/guides/davf_perturbgen_e2e.md:63) 与 [`davf_perturbgen_e2e.md` §候选方向边界](/home/scu/PTM2CellNet/docs/guides/davf_perturbgen_e2e.md:82)。

4. **Gate-0 与冻结资产。** 进入外部 PerturbGen stage 前必须验证 raw counts、canonical Ensembl、normal/disease、显式 donor、pairing 和 context gene order；`within_donor` 要求跨态共享 donor，`between_donor` 要求两组 donor 不相交且各至少 3 个。已存在的 frozen cohort/acceptance plan 只能证明契约资产冻结，不能替代真实 formal run 的 stage、null 和统计产物；见 [`davf_perturbgen_e2e.md` §Gate-0](/home/scu/PTM2CellNet/docs/guides/davf_perturbgen_e2e.md:90) 与 [`perturbgen_bridge.md` §3](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:84)。

5. **六阶段与路径身份。** 六阶段固定为 `tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report`。shared prepare 完成后，候选分别执行 `source_intervention=[src]` 和 `within_state=[tgt]+pert_tps`；单路结果不能外推为双路径正式结论。底层 `PerturbGenRunner` 只执行 `StagePlan`，不自行重查 DAVF gate；正式边界仍由绑定通过 invocation 的 E2E/CLI wrapper 保证，见 [`perturbgen_bridge.md` §4](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:267)。

6. **统计闭环。** 只有真实六阶段结果、未扰动 h5ad 质量、真实 matched-null distribution manifest、donor-level DEG 和完整 seed/path/mode 覆盖齐备后，才能用 `--resume --assemble-statistical-evidence` 组装 formal eval input，聚合独立 empirical p、执行 BH-FDR、保留双路径 AND 并写回 E2E lineage。缺任一输入只能是 inconclusive 或硬失败；mock、synthetic、smoke、bridge 不等于 biology PASS，见 [`davf_perturbgen_e2e.md` §统计接续](/home/scu/PTM2CellNet/docs/guides/davf_perturbgen_e2e.md:122) 和 [`perturbgen_bridge.md` §6](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:397)。

7. **Downstream target sidecar。** sidecar 只补充 source→target 的 donor-level delta 一致性，`matches_predicted` 与 `matches_observed` 分字段记录；一致计数是 concordance，不改变 source 三方 gate，也不是因果验证。当前这条 downstream lineage 已接线，但必须随真实 result h5ad 一起执行，见 [`run_davf_perturbgen_e2e.py`](/home/scu/PTM2CellNet/scripts/run_davf_perturbgen_e2e.py:650) 和 [`ptm_activity_pipeline.md` §7](/home/scu/PTM2CellNet/docs/guides/ptm_activity_pipeline.md:158)。

8. **Workflow B。** `encoder → 冻结 embedding asset → LatentDAVF 重训 → Gate-E` 是独立资产生命周期，不消费本次候选结果，也不回灌当前 DAVF。该链和 A 的双路径效用验收不能合并记账；见 [`perturbgen_bridge.md` §5](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:285) 及最新分析 [`project_analysis_20260914.md` §3.1](/home/scu/PTM2CellNet/project_analysis_20260914.md:135)。

## 3. 未解决根因与边界

- **研究输入尚未齐备。** 当前缺正式冻结的 `ptm_research_config`、可直接消费的真实 candidate spec/source proposal、donor-level AD DEG、真实 PTM 定量、KSTAR/PhosR activity、signed network release/map 和 activity benchmark。尤其 `position + ptm_type`、`proposed_direction`、七字段 semantic context 和 source 自身 DEG 证据不可由程序安全推断；这是外部研究输入缺失，不是用默认值修补的问题。依据 [`PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §4](/home/scu/PTM2CellNet/docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md:98) 与 [`ptm_activity_pipeline.md` §1](/home/scu/PTM2CellNet/docs/guides/ptm_activity_pipeline.md:21)。
- **正式运行产物尚未产生。** 缺真实 formal cohort 在本次运行中的可追溯输入/输出、六阶段 result h5ad、rescue 记录、每个 candidate/path/mode/seed 至少 99 个有限 null 值的 distribution manifest、未扰动质量 JSON、eval input/report manifest 和 `--verify` 结果。已冻结的 cohort contract 与 preflight 不是这些科学运行产物。
- **GPU 只是可用，不是已验收。** P40/CUDA 探针成功只说明设备入口可用；没有执行训练、扰动、matched-null 或生物学统计，故不能据此把工程 PASS 写成 biology PASS。
- **科学参考轴仍需研究设计冻结。** 当前三方 gate 的实现比较 up/down 同号，但 observed 是 donor-level disease−normal，DAVF 是 intervention decode−current context decode，proposal 是外部假设。没有明确 association/replication/reversal 与 reference axis，不能把同号解释成治疗因果性。
- **Workflow B Gate-E 未完成。** 尚未完成冻结 embedding asset、LatentDAVF 重训、独立 benchmark 与 ≥200 真实 benchmark 的 Gate-E 证据；不能把 Workflow A 的候选运行结果当作 B 的训练结果。
- **验证债务仍有两项。** changed Python files 的 format check 已通过，但全目录 format check 仍失败；全量非 slow/gpu pytest 被 timeout 截止且无最终摘要。因此不能从中推断“全量测试通过”，也不能把格式债务或 timeout 伪装成代码缺陷/生物学失败。

## 4. 验证矩阵

| 检查 | 已确认结果 | 结论边界 |
|---|---:|---|
| frozen formal replay 聚焦回归 | **66 passed / 8 warnings** | formal replay、lineage、独立聚合路径的聚焦工程证据 |
| PTM contrast/source gate 聚焦回归 | **6 passed / 8 warnings** | contrast 绑定与 source formal gate 的契约证据 |
| dual-path 聚焦回归 | **8 passed / 15 warnings** | 双路径接口/判定的工程证据 |
| `ruff check src scripts tests` | 通过 | 静态规则通过 |
| `mypy src/ --ignore-missing-imports` | **172 files，0 errors** | 类型检查通过；不代替真实运行 |
| requirements consistency | 通过，**274 pins** | 依赖声明一致 |
| `compileall` | 通过 | Python 编译级检查通过 |
| `git diff --check` | 通过 | diff 空白错误检查通过 |
| changed Python format check | **42 files already formatted** | 触碰文件已格式化 |
| whole-repo format check | **exit 1；324 reformat，168 formatted** | 既有全仓格式债，不能写成全仓通过 |
| full non-slow/gpu pytest | **timeout 420，exit 124；collected 2820** | 停在 `tests/integration/test_davf_pipeline.py::TestDAVFPipelineIntegration::test_mapper_to_inference_flow`，无 final summary，不能推断全量 PASS/FAIL |
| GPU preflight | Tesla **P40**；driver **580.178.04**；CUDA **13**；Torch **2.4.1+cu118**；`cuda true`；1 device | 仅证明设备/运行时入口可见 |
| formal GPU / biology | **未运行；biology PASS=0** | 缺 candidate spec、DEG、null/perturb 产物、formal real cohort/manifest、外部资产与 Workflow B Gate-E |

验证口径与仓库边界可对照 [`CURRENT_STATUS.md` 当前 Quick Reference](/home/scu/PTM2CellNet/docs/CURRENT_STATUS.md:12)、[`lessons.md` L-2026-0915-01](/home/scu/PTM2CellNet/lessons.md:638) 和 [`project_analysis_20260914.md` §7](/home/scu/PTM2CellNet/project_analysis_20260914.md:201)。其中旧文档中的历史测试数字仅作背景，本矩阵以本轮已确认快照为准。

## 5. D0–D5 收口计划（预计 3–5 个 GPU 日）

| 日程 | 最小动作 | 完成条件 |
|---|---|---|
| **D0（CPU，半天）** | 冻结 `ptm_research_config` 的 contrast/reference axis/objective/pairing/阈值；取得真实 PTM、activity、signed network、AD donor DEG、文献 source proposal、candidate spec、sidecar、embedding vocab 和 cohort manifest；运行 Gate-0 与 frozen asset 守护。 | 所有输入有绝对路径、版本/manifest lineage；source/target、donor split、Ensembl/scVI order 和七字段 semantic context 可追溯；不依赖默认值补齐。 |
| **D1（GPU，第 1 日）** | 按 [`perturbgen_bridge.md` §3 runbook](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:139) 先执行不组装统计的真实 E2E；每条 route 先做一次 shared prepare，再运行 gated candidate 的 `source_intervention` 与 `within_state`。 | 六阶段 stage manifest、`prepare_root`、两路结果 h5ad、seed 与 donor split 完整；缺引用/缺资产直接失败。 |
| **D2（GPU，第 2 日）** | 运行真实 matched-null；覆盖 5 candidate × 2 path × 3 seed × 3 mode 的冻结矩阵，并为每个组合写实际 null distribution manifest。 | 每个组合满足正式 null 数量和有限值要求；rescue extractor 绑定真实 h5ad/ donor；不使用 skeleton、synthetic 或手填分数。 |
| **D3（GPU，第 3 日，必要时延长至第 4 日）** | 使用同一 output root `--resume`，加入 `--assemble-statistical-evidence --deg-table --null-distribution-manifest`；提取未扰动质量，独立聚合 empirical p，统一 BH，计算 dual-path AND，写回完整 report lineage；需要时加入 downstream sidecar。 | `statistical_evidence.status=assembled`，formal evidence/source/quality/null/seed/path lineage 齐全，候选结果可被 replay。 |
| **D4（CPU，半天）** | 用 E2E 报告中登记的 eval input/report manifest 执行 `run_frozen_acceptance.py --verify`，检查 donor 泄漏、candidate/path/mode/seed 覆盖、输入绑定和独立 h5ad 重算。 | verify/replay 非零即停止正式宣称；只有独立重算复现且契约全部通过，才可报告计算效用结论。 |
| **D5（GPU/CPU，0.5–2 日，Workflow B）** | 在 A 验收后单独导出并冻结 encoder embedding，重训当前 LatentDAVF，执行 Gate-E benchmark；整理 canonical Ensembl、scVI decoder order、embedding manifest、benchmark 与 release evidence。 | Gate-E 的独立资产链和 ≥200 真实 benchmark 证据齐全；不把 A 的候选结果回灌 B，也不把单路结果写成普遍疗效。 |

D1–D3 是约 3 个 GPU 日的最小主线；设备排队、null 批量和 B 的重训/benchmark 可能扩展到 **3–5 个 GPU 日**。在 D0 输入未齐之前不启动 formal GPU，以免产生不可解释或无法复核的中间资产。

## 6. 发布口径

本轮可发布的是“代码契约修复完成、聚焦工程验证通过、全量检查有明确边界”的修复报告；不可发布正式生物学效用、治疗/临床因果结论、真实 null 显著性或 Gate-E 通过结论。正式下一步只按 D0→D1→D2→D3→D4 推进，Workflow B 的 D5 单独记账。

## 9. 引用索引

### 本地文档

- `project_analysis_20260914.md` §1–§7
- `docs/CURRENT_STATUS.md`
- `lessons.md`：`L-2026-0914-01`、`L-2026-0914-02`、`L-2026-0914-03`、`L-2026-0914-04`、`L-2026-0914-05`、`L-2026-0915-01`
- `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §4–§13
- `docs/guides/ptm_activity_pipeline.md`

### 代码与测试

- `scripts/build_ptm_global_gene_scores.py`
- `scripts/build_celltype_candidate_specs.py`
- `src/integration/perturbgen/frozen_cohort.py`
- `src/integration/perturbgen/replay_evaluation.py`
- `src/integration/perturbgen/eval_assembly.py`
- `scripts/run_frozen_acceptance.py`
- `scripts/run_davf_perturbgen_e2e.py`
- `src/integration/perturbgen/orchestrator.py`
- `tests/unit/scripts/test_build_ptm_global_gene_scores.py`
- `tests/unit/scripts/test_build_celltype_candidate_specs.py`
- `tests/unit/integration/perturbgen/test_frozen_cohort.py`
- `tests/unit/integration/perturbgen/test_replay_evaluation.py`
- `tests/unit/integration/perturbgen/test_eval_assembly.py`
- `tests/unit/integration/perturbgen/test_orchestrator.py`
- `tests/unit/scripts/test_run_davf_perturbgen_e2e.py`
- `tests/real_assets/test_real_frozen_cohort.py`
