# PTM Activity → AD 交集管线指南

对应方案：`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`（v1.0）。
本指南只登记已实现的命令契约；没有真实 PTM 资产时，这些命令只能用合成
数据验证工程契约，不代表生物学结果。

## 0. 边界（先读）

- PTM 阶段不区分 cell type，输出全局 activity 和 gene score；AD 阶段保留
  cell type，交集按 cell type 与 canonical Ensembl 独立生成（方案 §1）。
- KSTAR/PhosR 在独立环境运行，主环境只消费其标准表输出（`ptm_activity.tsv`），
  不把重型依赖加入 core（方案 §6.2）。执行合同见
  [`kstar_activity_plan.md`](kstar_activity_plan.md)。截至 2026-09-21，
  `/home/scu/anaconda3/envs/kstar` 已钉 Python 3.12.14 + `kstar==1.2.0` 及
  PhosphoSitePlus 衍生资源 hash；adapter 可写出十三列 `ptm_activity.tsv` 且
  `regulator_id` 映射到 canonical Ensembl；kinase+TF 网已另冻为
  `omnipath-kinase+tf-2026-09-21`（**未改** 2026-09-16 TF-only 文件）。正式
  KSTAR analysis 仍需匹配 `unique_reference_id` 的 ST/Y network。这不等于
  biology PASS。
- 工程入口已固定为 `bash scripts/setup_kstar_env.sh` 与
  `scripts/run_kstar_activity.py`。主环境不 import KSTAR；CLI 的 `mapping`/`analysis`
  都必须在 `kstar` 环境运行并显式提供存在的 `--network-dir`。`--mode analysis` 还必须
  提供 `--ensembl-mapping`。setup 或 analysis 缺少与冻结 `unique_reference_id` 一致的
  network 时硬失败，不伪造 ST/Y 资产。
- PTM smoke（`scripts/generate_ptm_smoke.py`）只证明阶段 1/3/4/5 表契约。
  `method=KSTAR` 且 `method_version=smoke-stub-*` **不是** KSTAR 运行结果，
  不得写入 formal lineage，也不是 biology PASS。
- signed network 是外部冻结资产（OmniPath signed signaling / TF regulon 导出）；
  本管线只做读取、过滤、传播。`src/models/signaling_network.py` 的硬编码路径
  不参与正式传播（方案 §5.3）。
- source（PTM 源蛋白/激酶）与 target（交集 DEG）角色分离；candidate spec 承载
  source，sidecar 承载一对多 source→target 关系（方案 §3.3/§5.5）。
- gene score 无独立 null 或外部 benchmark 校准前不写 PTM 侧显著性，
  `prediction_status=direction_only`（方案 §4.4）。

## 1. 阶段 0：冻结研究设计

`ptm_research_config.yaml` 契约（`src/analysis/ptm_research_config.py`）：

```yaml
schema_version: ptm2cellnet.ptm-research-config/v1
research_objective: association      # association | replication | reversal
reference_axis: disease_minus_normal # 或 contrast_specific；禁止全局取反
contrast: "disease-minus-normal"
primary_activity_method: KSTAR       # sensitivity 方法另行传播，不平均
sensitivity_activity_method: PhosR
network_release: "omnipath-kinase+tf-2026-09-21"   # 2026-09-21 起绑定 combined kinase+TF release
network_release_manifest: data/manifests/kstar_signed_network_release_20260921.json
                                    # 可选；设置后阶段 3 启动时执行 verify_network_release_binding
                                    # （release 名 + combined sha256/行数三向一致，漂移硬失败）
cell_types: [EX, IN]                 # 预先冻结的主要 cell type 列表
cohort_h5ad: data/AD/standardized/GSE174367_ad_cohort.h5ad
cohort_pairing: between_donor
species: "9606"
ptm_cohort: CPTAC_AD_BRAIN
mode: exploratory                    # exploratory | formal（2026-09-21 新增）
                                    # formal 拒绝一切 PENDING_* 占位资产；exploratory 保持 may_enter_lineage=false
activity_admission:                 # 2026-09-21 新增：传播前 activity 准入（方案 §4.4/§5.2 预冻结阈值）
  max_activity_qvalue: 1.0          # 当前无独立 benchmark 校准，登记宽松值；一经冻结不得事后调整
  min_substrates: 0
  min_network_coverage: 0.0
deg_max_fdr: 0.05
min_donors_per_state: 3
replicate_policy: mean               # mean | fail（重复位点登记规则）
deg_donor_aggregation: pseudobulk_counts  # per_cell_log2_mean | pseudobulk_counts（2026-09-16 冻结）| pseudobulk_counts_centered
                                    # centered 为多队列合并口径：每个 (cell_type, cohort) 减各自 normal donor 基线后再检验，
                                    # 需在 CLI 提供 --cohort-column（如 dataset）；单队列时与 pseudobulk_counts 数值一致
propagation:
  max_depth: 3                       # 简单路径深度上限
  decay: 0.5                         # 每跳权重衰减
  gene_edge_types: [tf_regulation, "kinase_substrate:signaling"]   # 终止于基因的边类型
                                    # 2026-09-21 绑定 combined release 后 kinase→substrate 边亦终止于基因
  max_paths_per_seed: null           # 可选；设置时须为正整数的逐 seed 上限
semantic_context:                    # 七字段模板，{cell_type} 占位符逐类型替换
  context: "GSE174367 {cell_type} cells"
  intervention: "KO"
  comparison_baseline: "donor-level disease vs normal"
  reference_axis: "disease_minus_normal"
  research_objective: association
  evidence_source: "PTM activity network concordance + donor DEG"
  cohort: "GSE174367"
```

## 2. 阶段 1：PTM 输入标准化

输入 `ptm_site_quantification.tsv`（方案 §4.1 十三列，TSV）：每行一个
sample × donor × 位点 × PTM 类型的定量记录。`value_scale` 全表必须唯一
（`linear`/`log2`）；`donor_id` 可为空（该行标记 cohort-level exploratory）；
`total_protein_value` 可为空（归一化列保持 NaN，不回填原始值）。

```bash
python scripts/run_ptm_activity.py \
  --config ptm_research_config.yaml \
  --input-tsv ptm_site_quantification.tsv \
  --gene-map gene_map.tsv \
  --output-tsv standardized_ptm.tsv \
  --manifest-output ptm_input_manifest.json
```

`gene_map.tsv` 三列（`protein_id/gene_symbol/ensembl_id`）：未映射蛋白保留
原值并计数（无 canonical Ensembl 的行不进传播）。重复位点由
`replicate_policy` 处理：`fail` 硬失败，`mean` 聚合并记录。

无真实 phosphoproteome 时，用确定性 smoke 资产先测阶段 1 契约（以及跳过
KSTAR 后的 3/4/5）：

```bash
python scripts/generate_ptm_smoke.py \
  --output-dir outputs/ptm_activity/20260918_ptm_smoke \
  --run-pipeline
```

Smoke 自带一张 `release=smoke-2026-09-18` 的 kinase→TF→gene 小网，**不是**
`omnipath-2026-09-16`（后者只有 `tf_regulation`，激酶 activity 无法在其上传播）。
生成器另写 `kstar_evidence_from_sites.tsv`，形状给未来 KSTAR adapter 用，
本身不是 KSTAR 输出。

## 3. 阶段 2：Activity inference（独立 KSTAR/PhosR 环境）

KSTAR/PhosR 在独立环境运行，产出 `ptm_activity.tsv`（方案 §4.2 十三列）。
该表在本管线由阶段 3 的读取点校验（方向与分数符号一致、q 值范围、
per (regulator, contrast, method) 唯一）。本仓库现有 KSTAR 入口如下：

```bash
bash scripts/setup_kstar_env.sh

# 从项目根目录运行；以下参数值必须由实际 manifest / network asset 提供
CONDA_DEFAULT_ENV=kstar \
CONDA_PREFIX=/home/scu/anaconda3/envs/kstar \
/home/scu/anaconda3/envs/kstar/bin/python scripts/run_kstar_activity.py \
  --mode mapping \
  --standardized-ptm <standardized_ptm.tsv> \
  --input-manifest <ptm_input_manifest.json> \
  --output <mapped.tsv> \
  --adapter-manifest <adapter_manifest.json> \
  --case-condition disease \
  --reference-condition normal \
  --contrast disease-minus-normal \
  --network-dir <existing_kstar_network_dir> \
  --kstar-output-dir <kstar_output_dir> \
  --phospho-type <ST_or_Y>

# analysis 另需 --ensembl-mapping，写出十三列 ptm_activity.tsv
# CONDA_DEFAULT_ENV=kstar CONDA_PREFIX=... python scripts/run_kstar_activity.py \
#   --mode analysis --ensembl-mapping data/processed/gene_symbol_ensembl_map.tsv ...
```

`--mode analysis` 使用同一组必填参数，另外由 CLI 分别执行 increased/decreased
官方 KSTAR analysis，并写出标准 `ptm_activity.tsv`（`regulator_id` 为 canonical
Ensembl）；`mapping` 只验证并写出 ExperimentMapper handover，不推断 kinase activity。
CLI 在隔离环境、钉包、资源 hash 校验和显式 `--network-dir` 检查通过后才 import
KSTAR。缺 environment、解释器不匹配、hash 漂移、或缺匹配 `unique_reference_id`
的 network 均硬失败。当前已核实环境钉包、PhosphoSitePlus 衍生资源 hash、官方
ExperimentMapper smoke（7 mapped rows）以及 kinase+TF 新 release；KSTAR activity、
formal lineage 和 biology PASS 尚未完成。

KSTAR 1.2.0 handover 的 post-fix 临时 synthetic smoke 中，两路结果首行已分别为
`KSTAR_KINASE` 加对应 directional data 列，CLI 内部两次 `from_kstar` 均成功；但 test-only
synthetic 网络的两路 p 值没有方向性激酶证据，CLI 按预期以
`paired KSTAR outputs contain no directional kinase evidence` 退出 1，未生成
`ptm_activity.tsv`。这是当前硬失败契约，不是完整 CLI 十三列表 smoke 通过，不是正式 biology PASS，
也不添加 fallback。

独立环境、二元位点证据、PhosphoSitePlus freeze、kinase–substrate 传播网缺口
与验收顺序见 [`kstar_activity_plan.md`](kstar_activity_plan.md)。在该方案
§7 benchmark 过线前，`may_enter_lineage` 必须为 false。禁止把 smoke stub
的 `method=KSTAR` 回标成真 KSTAR。

## 4. 阶段 3：Signed network 传播与 global gene score

```bash
python scripts/build_ptm_global_gene_scores.py \
  --config ptm_research_config.yaml \
  --activity-tsv ptm_activity.tsv \
  --network-tsv signed_network.tsv \
  --network-id-map network_id_map.tsv \
  --output-tsv ptm_global_gene_scores.tsv \
  --manifest-output gene_score_manifest.json
```

- `signed_network.tsv`（方案 §4.3 九列）：`effect_sign` 为 `+1`/`-1`；
  无符号边只进 coverage 统计、不进传播；符号冲突的平行边整体剔除并计数；
  `confidence` 缺省按 1.0 计并登记。kinase activity 传播必须用
  `omnipath-kinase+tf-2026-09-21`（或后续新 id），**不得**把
  `omnipath-2026-09-16` TF-only 文件原地加点边。
- 传播：有符号简单路径枚举（`max_depth` 内），路径在 `gene_edge_types`
  边终止于基因；每条路径贡献 `activity × Π(sign×confidence) × decay^length`，
  per (source, target) 求和（方案 §5.3）。
- `max_paths_per_seed` 可选；非 `null` 时必须为正整数，作为每个 seed 的路径上限。
  超限直接硬失败（`SignedNetworkContractError`），不截断；manifest 记录
  `per_seed_path_counts` 和 `path_count_distribution` diagnostics。
- `network_id_map.tsv` 三列（`network_id/gene_symbol/ensembl_id`）：网络 id
  到 canonical Ensembl 的显式映射，作者负责；未映射 target 不进正式 score 表。
- **Activity 准入（2026-09-21 起硬门禁）**：传播前每个 activity 行必须通过 config
  `activity_admission` 冻结阈值（`max_activity_qvalue`/`min_substrates`/
  `min_network_coverage`，方案 §4.4/§5.2 预冻结）；被拒行带逐行原因进
  manifest `sources.activity_table.admission.rejected`，policy hash 一并记录。
  旧 `activities_for_propagation`（method/contrast 之外全选）已删除，不存在绕过
  admission 的传播入口。无独立 benchmark 校准前 admitted 结果仍只有 exploratory
  效力（manifest `benchmark_gate.available=false`），`may_enter_lineage=false`。
- **独立 benchmark 评估（2026-09-22 起 opt-in）**：`--activity-benchmark <tsv>`
  提供已知 kinase perturbation phosphoproteomics 表（两列契约
  `regulator_id` + signed `perturbation_effect`，regulator 唯一）时，CLI 执行
  `evaluate_activity_benchmark`（paired 集方向 concordance + Spearman 秩相关
  + bootstrap CI），并把 manifest `benchmark_gate` 升为 `available=true` 与
  真实 PASS/FAIL verdict。判据阈值必须先在 config `activity_benchmark` 段
  预注册（三个必填：`min_paired_regulators`/`min_direction_concordance`/
  `min_abs_spearman`，可选 `bootstrap_iterations`/`ci_level`/`seed`；方案
  §8.7 不得事后挑选）——未注册而给 `--activity-benchmark` 直接硬失败。
  FAIL 不阻断输出：gene score 照常生成但只有 exploratory 效力；formal
  mode 下 FAIL 硬失败。评估范围是 primary method + frozen contrast 的全量
  activity 行（校准方法本身，与 admission 阈值正交）。
- config 设置 `network_release_manifest` 时，启动即执行
  `verify_network_release_binding`：release 名、manifest 记录与实际
  `--network-tsv` 的 sha256/行数三向一致，任何漂移硬失败。
- 输出 `ptm_global_gene_scores.tsv`（方案 §4.4 十二列）：一行一个
  (source_activity, target_gene) 对，不聚合成 per-gene 单值。
  sensitivity 方法需要单独跑一遍本命令再对比，不得平均。

## 4a. AD DEG 表生成（阶段 4 输入准备）

```bash
python scripts/build_ad_deg_table.py \
  --config ptm_research_config.yaml \
  --cohort-h5ad data/AD/standardized/GSE174367_ad_cohort.h5ad \
  --output-tsv ad_deg.tsv \
  --donor-level-output ad_deg_donor_level.tsv \
  --manifest-output ad_deg_manifest.json
```

estimand 由 config 的 `deg_donor_aggregation` 冻结。多队列合并时用
`pseudobulk_counts_centered` 并显式提供队列列：

```bash
python scripts/build_ad_deg_table.py \
  --config ptm_research_config_centered.yaml \
  --cohort-h5ad data/AD/standardized/GSE174367_GSE157827_combined.h5ad \
  --cohort-column dataset \
  --output-tsv ad_deg_centered_aggregate.tsv \
  --donor-level-output ad_deg_centered_donor_level.tsv \
  --manifest-output ad_deg_centered_manifest.json
```

centering 语义：每个 (cell_type, cohort) 的 normal donor pseudobulk 基线被从
该组所有 donor 值中减去，disease-vs-normal Welch t 在 cohort 内偏移去除后执行；
manifest audit 记录 `cohort_column`、per cell type 的 cohort 清单与
`donor_counts_by_cohort`。任一 (cell_type, cohort) 缺 normal donor 即硬失败。
实测（2026-09-16，GSE174367+GSE157827 合并）：centered 口径 min FDR
EX 0.5254 / INH 0.7908，仍无 FDR≤0.05 行——多队列合并不改变 observed gate
本地不可达的结论，该口径的价值是为后续新队列提供无 cohort 偏移的合并工具。

## 5. 阶段 4：AD cell type 交集

```bash
python scripts/build_ptm_ad_intersections.py \
  --config ptm_research_config.yaml \
  --gene-scores-tsv ptm_global_gene_scores.tsv \
  --deg-table ad_deg.tsv \
  --output-dir intersections/
```

`ad_deg.tsv`（方案 §4.5 八列）：`cell_type/ensembl_id/gene_symbol/log2fc/fdr/
observed_direction/n_normal_donors/n_disease_donors`，donor 计数每行必填。

输出 per cell type：`ptm_ad_intersection_<cell_type>.tsv`（membership ∈
concordant/discordant/PTM_only/AD_only，evidence_tier ∈ formal/exploratory/
ptm_only/ad_only；`I_c` = concordant ∧ formal）、`intersection_summary.json`
（含 `n_cell_types_supported`，注明不是独立 donor 证据）和
`target_set_manifest.json`（阶段 5 输入）。AD FDR/donor 阈值来自冻结 config，
不得在 AD 结果上反调 PTM 侧阈值（方案 §8）。

## 6. 阶段 5：候选 spec 与 target-set sidecar

```bash
python scripts/build_celltype_candidate_specs.py \
  --config ptm_research_config.yaml \
  --source-proposals-tsv source_proposals.tsv \
  --deg-table ad_deg.tsv \
  --target-set-manifest intersections/target_set_manifest.json \
  --context-h5ad data/AD/standardized/GSE174367_ad_cohort.h5ad \
  --cell-type-obs-column cell_type \
  --embedding-vocab embedding_vocab.json \
  --output-dir specs/
```

- `source_proposals.tsv` 九列（研究输入，不可自动生成）：
  `protein_id/position/ptm_type/gene_symbol/ensembl_id/source_activity_id/
  proposed_direction/site_probability/provenance`（可选 `aa`）。方向假设与
  proposal 置信由文献锚点给出（方案 §5.5 完成条件）。
- source gene 在该 cell type 必须有自己的 DEG 行（三方 gate 第三方）；
  没有的 source 记录为 exploratory，不生成正式候选行——target-set 一致性
  不能替代 source gene 的三方 gate（方案 §5.5）。
- `context_cell_index` 绑定该 cell type 在 cohort 中的第一个 cell（真实行号，
  不用默认第 0 行，方案 §4.6）。
- 提供 `--embedding-vocab`（JSON gene→token）时，source gene 无 verified
  token 直接硬失败（方案 §5.5）；正式运行必须提供。
- 输出：`candidate_spec_<cell_type>.json`（现有 `ptm2cellnet.candidate-spec/v1`，
  可直接被 `scripts/run_davf_perturbgen_e2e.py --candidate-spec` 消费）、
  `downstream_targets_<cell_type>.json` sidecar 和 `build_summary.json`。

## 7. 阶段 6：DAVF/PerturbGen 与 target-set 评估

### 7.1 scVI context 准备（E2E 前置，2026-09-16 起）

E2E 的 `context_h5ad` 必须与 route 的冻结 scVI checkpoint 4018 基因轴完全一致并
携带 `davf_batch` 列。唯一准备入口是 `scripts/prepare_scvi_context.py`
（`src/data/scvi_context.py`）；缺基因 policy（`zero_fill`/`fail`）、`davf_batch`
绑定值（必须在该 route scVI batch registry 内）和绑定 rationale 都是显式必填
输入并逐字记入 manifest——不能用"首个 batch"或隐式默认替代研究决策：

```bash
python scripts/prepare_scvi_context.py \
  --cohort-h5ad data/AD/standardized/GSE174367_ad_cohort.h5ad \
  --scvi-model checkpoints/scvi/davf_ko_dixit \
  --gene-aliases data/processed/davf_scperturb/ko/prepared.gene_aliases.tsv \
  --davf-route ko \
  --davf-batch "DixitRegev2016_K562_TFs_7_days:168" \
  --missing-gene-policy zero_fill \
  --batch-binding-rationale "<verbatim research rationale>" \
  --output-h5ad context_ko.h5ad \
  --manifest-output context_ko.manifest.json
```

输出保留 cohort obs（附加绑定的 batch 列），基因轴替换为 adapter
`gene_names` 顺序（缺失轴基因按 policy 显式处理，队列外基因丢弃并在 manifest
计数）。route 候选轴覆盖先用 `scripts/audit_davf_axis_coverage.py` 审计
（KO 当前仅 APOE；APP/PSEN1/BACE1/MAPT 在本地 scPerturb 数据集中无扰动数据）。

### 7.2 E2E 与 target-set 评估

E2E 运行方式不变，见 `docs/guides/davf_perturbgen_e2e.md` 与
`docs/guides/perturbgen_bridge.md`。target-set 下游 delta 评估用库接口
（`src/integration/perturbgen/downstream_target_evaluation.py`）：

```python
from src.integration.perturbgen.downstream_target_evaluation import (
    evaluate_driver_target_gate, evaluate_target_set_deltas,
    driver_target_gate_to_payload, evaluation_to_payload, load_downstream_target_sidecar,
)

sidecar = load_downstream_target_sidecar("specs/downstream_targets_EX.json")
# delta_frame: 行=canonical ENSG 基因，列=评价单位（donor/seed/mode），
# 值=上游产出的带符号表达 delta（DAVF decode delta 或 PerturbGen 预测差分）。
evaluations = evaluate_target_set_deltas(delta_frame, sidecar)
payload = evaluation_to_payload(evaluations, cell_type="EX", delta_matrix_source="...")

# driver–target gate（方案 §5.5）：source gate 证据与 target-set concordance
# 分字段合并成补充 lineage 证据；source_gate 取 E2E report 的 direction_gate 段。
evidence = evaluate_driver_target_gate(
    report["candidates"][0]["direction_gate"],
    evaluations["ENSG00000082701"],
    semantic_context={"cohort": "GSE174367"},
)
gate_payload = driver_target_gate_to_payload(evidence)
```

E2E 只有在显式提供 `--downstream-target-sidecar path/to/downstream_targets_EX.json`
并与 `--run-perturbgen` 同用时才读取 sidecar；它不自动发现或默认启用，也不能用于
`--dry-run`。此时 E2E 对每个 source 三方 gate 通过的候选解析其 PerturbGen
`result.h5ad`，按 donor 聚合 `pred_counts` baseline 与 `X` perturbed，计算
`X - pred_counts` 的 donor-level delta，并把 `downstream_target_evaluation` 及其
lineage 写入报告。

per-target 的 `matches_predicted` / `matches_observed` 分字段记录，不合并；
missing target 显式列出，不填零；一致计数只是方向一致性（concordance），
是与 source 三方 gate 互补的方向一致性证据，不改变 source 三方 gate 的
pass/fail，也不是因果验证或生物学 PASS（方案 §8.6）。

提供 sidecar 时，E2E 报告同时写入 `downstream_target_evaluation.driver_target_gate`
（per source，schema `ptm2cellnet.driver-target-gate/v1`）：source 段原样记录
source 三方 gate 的 status/reasons/三方方向与 `source_has_own_deg`，target 段
记录分字段 concordance 计数与比例，`driver_target_status` 是无阈值的事实分类
（`not_evaluable` / `all_concordant_observed` / `mixed` / `none_concordant_observed`，
参照 target 自身的 `observed_direction`）。该记录不产生 pass/fail 决策：
source 三方 gate 仍是 formal 候选的唯一准入门（方案 §5.5）；source 无自身
DEG 的候选仍只进 exploratory 清单，本接口只让这类候选的方向一致性可被
显式记录与复核，不改变准入边界。

### 7.3 外部扰动模型证据源（当前策略 B：GEARS + Geneformer，2026-09-17 起）

方向证据允许引入外部模型作为**并列、分字段记录的补充来源**。当前登记的
两类资产均使用 `external-perturbation-prediction/v1` manifest，由
[`src/integration/perturbgen/external_perturbation_evidence.py`](../../src/integration/perturbgen/external_perturbation_evidence.py)
校验并产出 `ptm2cellnet.external-perturbation-evidence/v1` payload；命令与环境见
[bridge guide §6b](perturbgen_bridge.md)。GEARS 声明 `go_extrapolation`，Geneformer
声明 `network_counterfactual`；二者都不是疾病 context 的因果干预，也不与
DAVF/PerturbGen 证据字段合并，不构成 pass/fail 或 biology PASS。

2026-09-17 的 APOE 预注册锚点回测 `verdict=fail`（GEARS top-100 方向一致率
0.25、Spearman -0.0428），因此两源资产当前均**不接入主线 lineage**。早期 LPM
路径保留为历史脚本/文档记录，已被 bridge guide §6b 的策略 B 取代。

### 7.4 五候选分流（方案 §6，2026-09-18）

`scripts/route_ad_candidates.py` 把方案第 6 章的主路线/备选路线编成可重复的
路由报告。它复用已有轴审计与 DEG 表，不重新发明覆盖口径，也**不**生成
E2E invocation 或改写 fail 的 GEARS/Geneformer 资产。

```bash
python scripts/route_ad_candidates.py \
  --axis-audit-tsv outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv \
  --deg-table outputs/ptm_activity/20260916_d2/ad_deg_centered_aggregate.tsv \
  --anchor-backtest outputs/external_evidence/apoe_anchor_backtest.json \
  --output-dir outputs/ptm_activity/20260918_research_decision
```

当前真实跑通结论（工程分流，不是 biology PASS）：

- APOE KO → `APOE_KO_ENGINEERING` / `engineering_verification`
- APP/PSEN1/BACE1/MAPT KO → `B2` / `direction_only`（无同机制公共 inventory）
- KD 不再单独输出（2026-09-18：`kd_policy=merged_into_ko_out_of_scope`）
- observed **显著**行 = 0（min FDR = 0.52538，报告阈值仍为 0.05），但 signed
  方向准入可以通过；`n_formal_invocations = 0`
- `--public-inventory-tsv` 在冻结决策下硬失败；不要再找公共 Perturb-seq

冻结决策写在 `observed_gate_decision.json`
（schema `ptm2cellnet.ad-research-decision/v1`）。可选 `--grn-support-tsv`
只在有可追溯 TF 路径时把 KO 候选升到 B4；缺失因子不会被填成高分。
`--contract-to-apoe-ko` 启用 B6 收缩，且不得再声称覆盖五候选。

## 8. 验证与测试

- 单元契约：`tests/unit/analysis/test_ptm_research_config.py`、
  `test_ptm_activity.py`、`test_signed_network.py`、`test_ptm_gene_score.py`、
  `test_ad_candidate_routing.py`、
  `tests/unit/integration/perturbgen/test_downstream_target_evaluation.py`、
  `tests/unit/scripts/test_route_ad_candidates.py`。
- 四阶段 CLI 全链（合成数据）：`tests/integration/test_ptm_activity_pipeline.py`。
- PTM smoke 生成器与 CLI：`tests/unit/analysis/test_ptm_smoke.py`、
  `tests/unit/scripts/test_generate_ptm_smoke.py`。
- 合成数据与 smoke 只证明契约与管线接通，不构成生物学 PASS（方案 §2.3/§12）。
