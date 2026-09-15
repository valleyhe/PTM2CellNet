# PTM activity → AD 交集主线：代码现状分析与后续执行方案

日期：2026-09-14
性质：工程现状盘点与执行规划。本文不改变任何已冻结契约；依据为代码与测试
的实现事实，对照 `docs/guides/ptm_activity_pipeline.md`（指南）与
`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`（方案 v1.0）。
口径不变：synthetic / 契约 PASS ≠ 生物学 PASS。

---

## 1. 结论

上游阶段 0–5 契约层（5 模块 + 4 CLI + 84 个测试项）与下游 E2E / 统计 / M6
冻结基础设施均已就绪。真正卡住主线推进的是 8 个缺口，其中只有 1 个
（AD donor-level DEG 表生成器）是当前无外部依赖、可立即开工的代码工作；
其余分别是 E2E 接线、外部资产等待与 GPU 真实执行。

| 层 | 状态 | 证据 |
|---|---|---|
| 阶段 0 研究设计冻结 | 已落地 | `src/analysis/ptm_research_config.py`（schema `ptm2cellnet.ptm-research-config/v1`，七字段 semantic_context 模板） |
| 阶段 1 PTM 输入标准化 | 已落地 | `src/analysis/ptm_activity.py` + `scripts/run_ptm_activity.py`（13 列契约、replicate policy、总蛋白归一化） |
| 阶段 2 Activity inference | 外部边界（按设计） | KSTAR/PhosR 独立环境；本仓库只有 `ptm_activity.tsv` 读取校验 |
| 阶段 3 signed 传播 | 已落地 | `src/analysis/signed_network.py` + `scripts/build_ptm_global_gene_scores.py`（有符号简单路径、平行边冲突剔除、`decay^length`） |
| 阶段 4 AD 交集 | 已落地 | `src/analysis/ptm_gene_score.py` + `scripts/build_ptm_ad_intersections.py`（membership/evidence_tier/`I_c`） |
| 阶段 5 候选 spec | 已落地 | `scripts/build_celltype_candidate_specs.py`（candidate-spec/v1 + sidecar + build_summary） |
| 阶段 6 target-set 评估 | 库接口有、未接线 | `downstream_target_evaluation.evaluate_target_set_deltas` 无 CLI、E2E 零引用 |
| E2E 六阶段 + 统计 | 已落地（CPU 契约层） | `run_davf_perturbgen_e2e.py` + `--assemble-statistical-evidence` 链 |
| GSE174367 数据契约 | 已冻结 | 7/7 cell type preflight PASS、M6（EX，train 12 / held-out 6，90-run，99 nulls） |
| GPU 真实执行 | 未运行 | runbook `docs/guides/perturbgen_bridge.md` §3 步骤 0–5 尚未真实走通 |

**执行顺序**：批次 A（DEG 生成器）→ 批次 B（downstream 接线）→ 批次 D1
（M6 五候选 GPU 真实执行，**不依赖 PTM 外部资产，可与 A/B 并行**）→
批次 C（外部资产接入，等待用户提供期间完成代码侧防护）→ 批次 D2（主线
交集候选完整 E2E）→ 批次 E（正式验收）。

---

## 2. 代码现状盘点（事实）

### 2.1 上游阶段 0–5：契约层已落地

| 阶段 | 模块 / CLI | 关键硬失败（raise / exit 1） | 关键降级（exploratory，不进正式输出） |
|---|---|---|---|
| 0 | `ptm_research_config.py` | schema/objective/reference_axis/pairing/threshold/propagation 非法；semantic_context 七字段缺一或 objective 不一致 | — |
| 1 | `ptm_activity.py` + `run_ptm_activity.py` | 缺列、`ptm_qvalue` 越界、`value_scale` 混用、总蛋白 ≤ 0、`replicate_policy=fail` 撞重复、gene_map 冲突 | 无 donor 行标记 cohort-level；无总蛋白归一化列保持 NaN；unmapped 蛋白保留原值计数 |
| 2 | `ptm_activity.py`（读取侧） | 方向与 score 符号矛盾、p/q 越界、(regulator, contrast, method) 重复、零 activity 禁止传播 | — |
| 3 | `signed_network.py` + `build_ptm_global_gene_scores.py` | confidence 越界、seed activity 为 0 | 无符号边只进 coverage；自环剔除；平行边同号取强、**异号整键剔除并计数**（不静默选边）；seed 无网络节点记 `seeds_without_node` |
| 4 | `ptm_gene_score.py` + `build_ptm_ad_intersections.py` | DEG 缺 donor 计数、direction 与符号矛盾、(cell_type, gene) 重复、frozen cell type 无 DEG | donor/FDR 不达标的 concordant 行 tier=exploratory；无 null 前 `prediction_status=direction_only`，**永不写 PTM 侧 q 值** |
| 5 | `build_celltype_candidate_specs.py` | proposal 契约失败、`--embedding-vocab` 下 source 无 token（正式运行硬失败） | source gene 该 cell type 无 DEG 行 → `exploratory_sources`，不生成候选行；manifest 缺 cell type 同理 |

测试：84 个测试项（`tests/unit/analysis/test_ptm_*.py` 61 + 
`tests/unit/integration/perturbgen/test_downstream_target_evaluation.py` 12 +
`tests/integration/test_ptm_activity_pipeline.py` 3 个 CLI 全链），全部合成数据。

实现要点（与方案 §5 逐条对应，已验证）：

- 传播是显式有符号简单路径枚举：每条路径贡献
  `activity × Π(sign_e × confidence_e) × decay^length`，`math.fsum` 按
  (seed, target) 求和；路径在 `gene_edge_types` 边终止于基因。
- score 表一行一个 (source_activity, target_gene) 对，不聚合 per-gene；
  `network_coverage` = 该 seed 全部 gene 路径中到达该 target 的比例；
  `degree_normalized` = gene_score / n_paths。
- `context_cell_index` 绑定该 cell type 在 cohort 的第一个 cell 的位置索引
 （集成测试断言 `== 1`，非默认第 0 行）。
- candidate spec 与权威 `build_candidate_spec_payload` 同形，覆盖全部 13 个
  必填字段；新增 `semantic_context`（E2E 强校验七字段）与 `source_activity_id`
  （关联 sidecar）；不写权威版可选字段 `low_donor_support`（E2E 不要求，
  DEG 表强制 donor 计数使其恒可判定，低风险差异，仅记录）。
- 9 个目标文件无 TODO/FIXME；仅 `PTM_INPUT_SCHEMA_VERSION` 与
  `GENE_SCORE_SCHEMA_VERSION` 两个常量已定义未被任何输出消费（manifest 用
  各自的 `*-manifest/v1` 字面量，仅记录，不行动）。

### 2.2 下游 E2E 与统计基础设施：已就绪

- **E2E 入口** `scripts/run_davf_perturbgen_e2e.py`（814 行）：`--candidate-spec`
  消费上阶段产物（不校验 schema_version，校验 context_h5ad/candidates/
  context_cell_index/semantic_context）；report 顶层含 `candidates`（每项
  `{intervention_type, status, davf_evidence, direction_gate, candidate,
  invocation}`）、`merged_gated_routes`、`perturbgen_runs`、
  `statistical_evidence`、`perturbgen_gate0`、`donor_split`、
  `perturbgen_prepare`。
- **统计组装**（`--assemble-statistical-evidence` + `--deg-table` +
  `--null-distribution-manifest`）：`_assemble_statistical_evidence` 串接
  `resolve_run_artifacts`（rglob stage_manifest 绑定 result_h5ad）→
  `extract_unperturbed_quality_from_h5ad` → `build_eval_input_payload` →
  `replay_dual_path_evaluation`，产出候选级 empirical p / BH q / dual-path
  verdict 写回 report lineage。
- **deg 表消费契约**：`replay_evaluation.load_deg_table`（CSV/JSON →
  DataFrame），列名经 `--deg-donor-column/--deg-gene-column/
  --deg-effect-column/--deg-fdr-column` 配置；
  `summarize_rescue_by_donor` → `build_held_out_signature(deg_table,
  held_out_donor=donor, ...)` 逐 donor 留出打分——即 E2E 期望的是
  **per (donor, gene) 的 donor-level 长表**，不是聚合表。
- **delta 矩阵原料已存在**：`results._aggregate_donor_expression_from_h5ad
  (output_h5ad, donor_obs_column, var_gene_column)` 返回
  `(baseline_by_donor, perturbed_by_donor)` 两个 `dict[gene][donor]->float`
  （baseline=`pred_counts` 层，perturbed=`X` 层）。这是构建
  `evaluate_target_set_deltas` 所需 gene × donor delta DataFrame 的最近接口，
  但当前未导出成表、E2E report 中亦无该矩阵。
- **direction_gate**（`src/integration/perturbgen/direction_gate.py`，178 行）：
  三方 = `proposal.proposed_direction` × `davf_evidence.predicted_direction`
  × candidate 的 `observed_direction`（含 observed_fdr 与 provenance 校验）；
  pass 时 `corrective_action = ko/oe`。**与 target-set 一致性
  （sidecar 的 matches_predicted/matches_observed）零代码关联**——
  driver–target gate contract 待办 ② 的现状基线。
- **invocation 边界**：`PerturbGenInvocation.__post_init__` 硬校验
  `direction_gate_status=="pass"`、方向一致、provenance 非空、
  `semantic_context.intervention == intervention_type`；共享 prepare 经
  `build_shared_prepare_plans` / `resolve_prepare_artifact_references`
  （`@artifact:stage:name` 引用缺失即硬错误）。
- **冻结资产**：`outputs/perturbgen/frozen/20260914_gse174367_ex/`
  （M6：EX，between_donor，train 12 / held-out 6 donor，5 候选
  APP/PSEN1/BACE1/MAPT/APOE，90-run plan，每 run 99 matched nulls，token
  indices 已在词表验证）；`embedding_asset_20260822/`（18967 词表含特殊
  token，768 dim safetensors + manifest）。
- **数据资产**：`data/AD/standardized/GSE174367_ad_cohort.h5ad`（61,472
  cells、58,676 genes、18 donor = AD 11 / Control 7，obs 含
  `donor/state/cell_type/sample/batch/age/sex/pmi/rin/tangle_stage/
  plaque_stage`，var index = version-less ENSG，layers["counts"] 原始计数）。
- **GPU runbook**：`docs/guides/perturbgen_bridge.md` §3 步骤 0–5
  （前置校验 → candidate_spec → 六阶段 → matched-null 批跑 → `--resume`
  统计组装 → `run_frozen_acceptance.py --verify` 独立复算）。
- **依赖分层**：core 无 scanpy/anndata（分析层在
  `requirements-analysis.txt`）；DEG 生成若走 scipy Welch + 自实现 BH 可留
  在 core，若用 scanpy 需 analysis 环境。

### 2.3 AD DEG 生成现状（重点核对）

- 全仓**未找到**生成 per-cell-type donor-level DEG 长表的工具
  （`rank_genes_groups`/`pseudobulk`/`wilcoxon` 均 0 命中）。
- 最接近的实现：`src/data/gse_normal_disease.py::
  summarize_normal_disease_directions`（CLI
  `scripts/summarize_gse_directions.py`）——donor 级 log2 归一化均值 →
  disease−normal 聚合 delta + Welch `ttest_ind` + BH；但输出是按
  (cell_type, ensembl_id) 聚合的 direction evidence CSV（有
  `normal_donors/disease_donors` 计数列、**无 donor 维度行**），不能直接作
  E2E `--deg-table`。
- 现存唯一 direction evidence 产物是 IBD 队列
  （`outputs/gse214695_hc_cd/direction_evidence.csv`）；**GSE174367 (AD) 的
  DEG 表未生成**。
- 结论：同一"AD DEG"存在两种消费契约——主线阶段 4/5 的八列**聚合 TSV**
  （`load_deg_table`：`cell_type/ensembl_id/gene_symbol/log2fc/fdr/
  observed_direction/n_normal_donors/n_disease_donors`）与 E2E 统计组装的
  **donor-level 长表**——生成器必须同时覆盖两端。

---

## 3. 差距清单

| 编号 | 差距 | 现状证据 | 阻塞什么 | 对应批次 |
|---|---|---|---|---|
| GAP-1 | AD donor-level DEG 表生成器缺失 | §2.3；`gse_normal_disease.py` 输出聚合表无 donor 行 | 阶段 4/5 真实输入、E2E `--deg-table`、null_selection fc 特征、rescue 评估 | A |
| GAP-2 | 双 DEG 消费契约（聚合 TSV / donor 长表）无统一生成口径 | `ptm_gene_score.load_deg_table` vs `replay_evaluation.load_deg_table` + `build_held_out_signature` | 同 GAP-1，且口径不冻结会导致两表不一致 | A |
| GAP-3 | `downstream_target_evaluation` 无 CLI、E2E 零接线 | 全仓唯一 import 是 `build_celltype_candidate_specs.py` 取常量 | 阶段 6 target-set delta 评估无法产出 | B |
| GAP-4 | E2E report 无 per-target delta 矩阵；原料未导出 | `_aggregate_donor_expression_from_h5ad` 存在但无导出/消费方 | GAP-3 的数据源 | B |
| GAP-5 | driver–target gate contract 未显式化 | `direction_gate.py` 与 sidecar 一致性零关联；方案 §5.5 要求显式边界 | target-set 一致性的正式定位 | B（第一版只文档化+并排记录） |
| GAP-6 | 方案 §10 六项外部输入缺失（第 6 项 DEG 由 GAP-1 解决） | data/ 下无真实 PTM 定量表、ptm_activity.tsv、signed_network.tsv、benchmark | 主线真实数据全链 | C（等待用户资产）+ A |
| GAP-7 | 传播无规模防护（仅 `max_depth`） | `propagate_signed_scores` 无路径数/节点上限 | 真实 OmniPath 级网络路径爆炸风险（可达：§10 接入即触发） | C |
| GAP-8 | GPU 真实执行未运行 | M6 冻结后六阶段 / matched-null / `--verify` 均未跑 | 第一份真实 formal E2E report | D1 / D2 |

另记录（不列行动项）：`PTM_INPUT_SCHEMA_VERSION`/`GENE_SCORE_SCHEMA_VERSION`
定义未消费；PTM 版 candidate spec 不写 `low_donor_support`。

---

## 4. 后续执行方案

### 4.0 批次总览与依赖

| 批次 | 内容 | 前置 | 外部依赖 | 可开工 |
|---|---|---|---|---|
| A | AD donor-level DEG 表生成器 | 无 | 无（h5ad 已有） | 立即 |
| B | downstream target 评估写入 E2E lineage + driver–target gate contract | 无（合成 result_h5ad 即可测） | 无 | 立即（与 A 并行） |
| D1 | M6 五候选 GPU 真实执行 + 统计组装 + `--verify` | runbook；建议 A 先行（DEG 长表供统计组装） | GPU（脱沙箱） | A 完成后 |
| C | §10 外部资产接入 + 传播规模防护 | 防护部分无前置；接入部分待资产 | 用户提供五项资产 | 防护立即；接入待资产 |
| D2 | 主线交集候选完整 E2E（含 downstream 评估） | A + B + C + 真实 proposal | GPU + PTM 资产 | C 后 |
| E | 正式验收与报告语义 | D1 / D2 | — | 最后 |

关键顺序洞察：**D1 用 M6 已冻结的 5 个文献锚点候选（APP/PSEN1/BACE1/MAPT/
APOE），不依赖 PTM activity 主线的任何外部资产**。GPU 真实执行不必等
§10；主线候选（来自交集）才需要等批次 C。

### 4.1 批次 A：AD donor-level DEG 表生成器

**目标**：从 `GSE174367_ad_cohort.h5ad` 一次性生成两种契约的 DEG 产物，
冻结生成口径，同时解锁主线阶段 4/5 真实输入与 E2E 统计组装。

**决策点（实现前冻结，写进测试）**：

1. per-donor effect 口径（E2E 长表）：建议每行为
   `(cell_type, donor, ensembl_id)` 的 disease-donor 视角 log2fc——
   donor 级 log2 归一化表达相对 **normal 组 donor 均值** 的 log2 比值，
   仅 disease donor 生成行（rescue 语义 = 扰动把 disease donor 拉回 normal
   参考；`build_held_out_signature` 逐 disease donor 留出、其余训练）。
   若实现时发现 rescue 需要 normal donor 行，以代码契约为准显式修正本节。
2. 方法学复用 `gse_normal_disease.py` 的既有实现（donor 级 log2 归一化
   均值、Welch `ttest_ind`、自实现 `_bh_adjust`），函数级复用优先于重写；
   落在 core 依赖（scipy/pandas），不引入 scanpy。
3. 阈值归属：`deg_max_fdr` / `min_donors_per_state` 只在阶段 4 交集消费，
   生成器输出全量行 + donor 计数，**不在生成侧过滤**（方案 §8：不得在
   AD 结果上反调 PTM 侧阈值）。

**文件级改动**：

| 文件 | 动作 | 内容 |
|---|---|---|
| `src/analysis/ad_deg_table.py` | 新增 | donor 级聚合、Welch + BH、聚合表与长表两个 builder、manifest（记录口径、normal 参考定义、cell type 列表、donor 计数审计） |
| `scripts/build_ad_deg_table.py` | 新增 | CLI：`--cohort-h5ad`（默认 config.cohort_h5ad）、`--config`（cell_types/pairing 来源）、`--cell-type-obs-column`、`--state-obs-column`、`--donor-obs-column`、`--output-tsv`（聚合八列）、`--donor-level-output`（E2E 长表 CSV）、`--manifest-output` |
| `src/data/gse_normal_disease.py` | 视复用方式小改 | 若提取共用函数则同步原调用方；不改其公开行为 |

**输出契约**：

- 聚合表 `ad_deg.tsv`（主线八列）：`observed_direction ∈ up/down/neutral`
  与 log2fc 符号一致（0 → neutral），donor 计数非空——直接通过
  `ptm_gene_score.load_deg_table` 全部校验。
- 长表 `ad_deg_donor_level.csv`（E2E）：列名与 `--deg-donor-column` 等
  四参数对齐（默认 `donor/gene_symbol/log2fc/fdr`），通过
  `replay_evaluation.load_deg_table` + `build_held_out_signature` 最小
  样例验证（held-out donor 可留出、训练 donor ≥ 2）。
- 两表同源同 run 生成，manifest 登记 sha256 与口径，防止两表漂移。

**测试计划**：合成小 h5ad（≥2 cell type × 4 donor × 数基因）手算期望
log2fc/fdr/donor 计数（延续 L-2026-0914-03/05：期望值手算，不复用被测
代码输出）；两表分别过上述两个消费方校验；CLI 集成测试（缺 obs 列、
cell type 无 donor、between_donor 泄漏 donor 检出即硬失败）。

**验收**：真实 `GSE174367_ad_cohort.h5ad` 生成两表；阶段 4 CLI 用真实
聚合表 + 合成 gene score 走通契约；E2E `--deg-table` 指向长表可组装统计
（可与 D1 步骤 4 合并验收）。产出口径写入 `lessons.md`。

**边界**：不做 PTM 侧阈值联动；不改 M6 冻结资产（null_selection 的 fc
特征当前 `"unavailable/default"`——重新生成 selection 属于重开 M6 冻结，
须显式决策并重走冻结记录，默认不动）。

### 4.2 批次 B：downstream target 评估写入 E2E lineage

**目标**：把已实现的库接口接入 E2E，使 report 携带 per-source 的
target-set delta 评估，并显式化 driver–target gate contract 第一版。

**设计**：

1. delta 矩阵构建：对每个 gated 候选的 perturb `result_h5ad`，用
   `_aggregate_donor_expression_from_h5ad`（baseline=`pred_counts`，
   perturbed=`X`）得 `dict[gene][donor]`，delta = perturbed − baseline，
   组装为行 = canonical ENSG、列 = donor 的 DataFrame，喂
   `evaluate_target_set_deltas`。
2. E2E 新参数 `--downstream-target-sidecar`（显式路径，不自动发现、不猜
   同目录）：提供时执行评估；评估前提为对应 result_h5ad 可解析（即
   `--run-perturbgen` 已跑或 `--resume` 命中）。sidecar 中找不到某候选
   ensembl 或候选与 sidecar cell_type 不配套 → **硬失败**（spec 与
   sidecar 必须同 build 产出，契约不匹配立即暴露）。
3. report 新增顶层字段 `downstream_target_evaluation`：payload
   （`ptm2cellnet.target-set-evaluation/v1`）+ lineage（sidecar 路径 +
   sha256、per 候选的 result_h5ad 路径、baseline/perturbed 层名）。
   未提供参数时字段缺省，report 其余行为不变。
4. driver–target gate contract 第一版：**不改 `direction_gate.py` 判定
   逻辑**。source 三方 gate 仍是唯一 pass/fail 判据；target-set 一致性
   （matches_predicted/matches_observed）作为并排补充 evidence 记录，
   payload 不含 verdict 语言。在指南 §7 与 `AGENTS.md` 主线约束节写明：
   升级为正式 gate（参与 pass/fail）需另行批准并单独实现。

**文件级改动**：

| 文件 | 动作 | 内容 |
|---|---|---|
| `scripts/run_davf_perturbgen_e2e.py` | 修改 | 新参数、评估组装函数（delta 矩阵构建 + 调库 + lineage 写入 report） |
| `src/integration/perturbgen/results.py` | 视情况 | 若 `_aggregate_donor_expression_from_h5ad` 需公开（去下划线或加薄包装），保持原调用方不变 |
| `docs/guides/ptm_activity_pipeline.md` §7、`docs/guides/davf_perturbgen_e2e.md` | 修改 | 删除"属于后续工作，本轮未接"表述，登记新参数与 report 字段、gate contract 边界 |

**测试计划**：合成 result_h5ad + sidecar → report 含评估与 lineage；
missing target 显式列出；matches_predicted/matches_observed 分字段；sidecar
与 spec 不配套 → 硬失败；不提供参数 → 字段缺省且其余断言不变。

**验收**：D1 完成后，同参数 `--resume` 追加 `--downstream-target-sidecar`
（M6 候选与主线 sidecar 结构一致，可用合成 sidecar 或 M6 候选手工构造）
得到含 `downstream_target_evaluation` 的 report。

### 4.3 批次 C：外部资产接入与传播防护

**外部资产（§10 六项）接入表**——资产由用户提供，代码侧只做入口校验
（均已存在）：

| # | 资产 | 入口 | 校验落点 |
|---|---|---|---|
| 1 | 真实 PTM 位点定量表 | `run_ptm_activity.py` | 13 列契约、replicate、归一化、manifest |
| 2 | KSTAR/PhosR activity 表（独立环境产出 `ptm_activity.tsv`） | `build_ptm_global_gene_scores.py --activity-tsv` | 方向/符号/p-q/唯一性/零 activity |
| 3 | OmniPath signed 网络（`signed_network.tsv` + `network_id_map.tsv`） | `build_ptm_global_gene_scores.py --network-tsv/--network-id-map` | 九列/三列契约、符号冲突、confidence |
| 4 | activity benchmark（已知 perturbation） | 阶段 2 完成条件 | **缺口**：仓库无 benchmark 对比工具。最小做法：接入时以表对表对比脚本或人工记录写入 manifest；不预先开发 |
| 5 | 组织/疾病/物种/比较轴兼容性说明 | config + manifest | `ptm_research_config.yaml` 冻结字段 |
| 6 | AD donor-level DEG 表 | 批次 A 产出 | 两端消费方校验 |

- signed 网络导出脚本（独立环境用 OmniPath 客户端产出九列 TSV + id map）
  为可选工作项：若用户自带导出则不需要；若需要，新建
  `scripts/export_signed_network.py` 独立运行、不进 core。
- sensitivity 方法（PhosR）单独跑一遍阶段 3 CLI 再对比，不平均（既有行为）。

**传播规模防护（代码侧，可立即做）**：

- `PropagationConfig` 增加可选 `max_paths_per_seed`（默认 `None` =
  无上限）；`propagate_signed_scores` 逐 seed 计数路径，超限
  `raise SignedNetworkContractError`（提示降低 `max_depth` 或收紧网络
  过滤），**不静默截断**；manifest 记录 per-seed 路径数分布。
- 防护理由：真实 OmniPath 级网络 + 数百 kinase seed 在 `max_depth=3` 下
  路径数可达爆炸量级，这是 §10 接入即触发的可达场景，不是臆想边界。
- 测试：上限触发 raise、默认行为不变、manifest 字段。

**验收**：六项资产登记进 `data/manifests/datasets.yaml`；真实资产走通
阶段 1→5 全链 CLI（此时仍只证明管线接通，生物学结论待 D2/E）。

### 4.4 批次 D：GPU 真实执行（脱离沙箱）

**D1：M6 五候选（先行，不等 PTM 资产）**

按 `docs/guides/perturbgen_bridge.md` §3 runbook 步骤 0–5 完整执行：

| 步 | 动作 | 说明 |
|---|---|---|
| 0 | `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 pytest tests/real_assets/test_real_frozen_cohort.py -v`（CPU） | 冻结资产前置校验 |
| 1 | candidate_spec | M6 已有 5 候选与 token indices 验证；DEG 表列名与 `--deg-*-column` 对齐（批次 A 长表） |
| 2 | 六阶段 GPU（不传统计参数） | train/held-out 列表与冻结 manifest 逐字一致；GPU 锁；共享 prepare 复用 |
| 3 | matched-null 批跑 | 5 候选 × 2 path × 3 seed × 3 mode × 99 nulls；CLI 仅 dry-run，执行走 Python API |
| 4 | 同参数 `--resume` 组装统计 | `--assemble-statistical-evidence` + 批次 A 长表 + null manifest |
| 5 | `run_frozen_acceptance.py --verify` | 独立复算 M6 |

产出：第一份真实 formal E2E report（含 empirical p/q、dual-path verdict、
质量证据）。可顺带验收批次 B（追加 sidecar 参数 resume 一轮）。

**D2：主线交集候选（依赖 A + B + C）**

1. 冻结正式 `ptm_research_config.yaml`（真实 cell type 列表、network
   release、PTM cohort、semantic_context 模板）；
2. 阶段 1→5 真实数据全链（§10 资产 + 批次 A DEG 表 + 文献锚点
   `source_proposals.tsv`）；
3. 阶段 5 产出的 per-cell-type candidate_spec + sidecar 逐个走 E2E：
   `--run-perturbgen` 六阶段 → matched-null → 统计组装 →
   `--downstream-target-sidecar` 评估；
4. 真实 proposal 的 matched-null 与统计证据按 M6 同构口径冻结。

**边界**：GPU 全部脱离沙箱执行；`source_intervention` 与 `within_state`
双场景 AND；单路结果只标对应场景。

### 4.5 批次 E：正式验收与报告语义

- Gate-4/5：真实 cohort 生物学验收口径（raw counts、显式 donor、canonical
  Ensembl、scVI gene order、冻结 embedding、真实 matched null、未扰动质量、
  双路径统计）逐项核对，写正式验收记录。
- 结论措辞：association 目标下只写"PTM/activity/network 预测与该 cell
  type 的 AD DEG 方向一致"（方案 §12）；concordance ≠ causal validation。
- 文档同步：按 L-2026-0914-06 教训，落地后同轮同步 CURRENT_STATUS /
  project_analysis / `.planning/` 全层 / CHANGELOG / lessons，避免单点
  更新造成反向漂移。

---

## 5. 全局边界（不做清单）

1. 不改 `direction_gate.py` 判定逻辑（批次 B 第一版）；target-set 一致性
   不参与 gate pass/fail。
2. 不把 KSTAR/PhosR/omnipath/anndata 重型依赖加入 requirements-core。
3. 不做 activity-conditioned DAVF / Gate-E（≥200 条可追溯真实 PTM→gene
   benchmark 前提不满足）。
4. 不把 synthetic、契约、bridge、规划写成生物学 PASS；不把多 cell type
   重复当独立 donor 证据。
5. 交集 target gene 不自动变成干预 token；source/target 角色分离不变。
6. 不动 M6 冻结资产（含 null_selection fc 特征），重开冻结须显式决策。

---

## 6. 与既有文档的关系

- 本文是执行规划，不替代方案 v1.0 与指南；冲突时以代码、测试与方案为准。
- 每个批次落地后按 §4.5 第 3 条同步 living docs，并把口径决策追加进
  `lessons.md`。
