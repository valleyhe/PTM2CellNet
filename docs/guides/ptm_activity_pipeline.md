# PTM Activity → AD 交集管线指南

对应方案：`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`（v1.0）。
本指南只登记已实现的命令契约；没有真实 PTM 资产时，这些命令只能用合成
数据验证工程契约，不代表生物学结果。

## 0. 边界（先读）

- PTM 阶段不区分 cell type，输出全局 activity 和 gene score；AD 阶段保留
  cell type，交集按 cell type 与 canonical Ensembl 独立生成（方案 §1）。
- KSTAR/PhosR 在独立环境运行，主环境只消费其标准表输出（`ptm_activity.tsv`），
  不把重型依赖加入 core（方案 §6.2）。
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
network_release: "2026-08"
cell_types: [EX, IN]                 # 预先冻结的主要 cell type 列表
cohort_h5ad: data/AD/standardized/GSE174367_ad_cohort.h5ad
cohort_pairing: between_donor
species: "9606"
ptm_cohort: CPTAC_AD_BRAIN
deg_max_fdr: 0.05
min_donors_per_state: 3
replicate_policy: mean               # mean | fail（重复位点登记规则）
propagation:
  max_depth: 3                       # 简单路径深度上限
  decay: 0.5                         # 每跳权重衰减
  gene_edge_types: [tf_regulation]   # 终止于基因的边类型
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

## 3. 阶段 2：Activity inference（外部）

KSTAR/PhosR 在独立环境运行，产出 `ptm_activity.tsv`（方案 §4.2 十三列）。
该表在本管线由阶段 3 的读取点校验（方向与分数符号一致、q 值范围、
per (regulator, contrast, method) 唯一）。本仓库没有运行 KSTAR/PhosR 的入口。

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
  `confidence` 缺省按 1.0 计并登记。
- 传播：有符号简单路径枚举（`max_depth` 内），路径在 `gene_edge_types`
  边终止于基因；每条路径贡献 `activity × Π(sign×confidence) × decay^length`，
  per (source, target) 求和（方案 §5.3）。
- `max_paths_per_seed` 可选；非 `null` 时必须为正整数，作为每个 seed 的路径上限。
  超限直接硬失败（`SignedNetworkContractError`），不截断；manifest 记录
  `per_seed_path_counts` 和 `path_count_distribution` diagnostics。
- `network_id_map.tsv` 三列（`network_id/gene_symbol/ensembl_id`）：网络 id
  到 canonical Ensembl 的显式映射，作者负责；未映射 target 不进正式 score 表。
- 输出 `ptm_global_gene_scores.tsv`（方案 §4.4 十二列）：一行一个
  (source_activity, target_gene) 对，不聚合成 per-gene 单值。
  sensitivity 方法需要单独跑一遍本命令再对比，不得平均。

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

E2E 运行方式不变，见 `docs/guides/davf_perturbgen_e2e.md` 与
`docs/guides/perturbgen_bridge.md`。target-set 下游 delta 评估用库接口
（`src/integration/perturbgen/downstream_target_evaluation.py`）：

```python
from src.integration.perturbgen.downstream_target_evaluation import (
    evaluate_target_set_deltas, evaluation_to_payload, load_downstream_target_sidecar,
)

sidecar = load_downstream_target_sidecar("specs/downstream_targets_EX.json")
# delta_frame: 行=canonical ENSG 基因，列=评价单位（donor/seed/mode），
# 值=上游产出的带符号表达 delta（DAVF decode delta 或 PerturbGen 预测差分）。
evaluations = evaluate_target_set_deltas(delta_frame, sidecar)
payload = evaluation_to_payload(evaluations, cell_type="EX", delta_matrix_source="...")
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

## 8. 验证与测试

- 单元契约：`tests/unit/analysis/test_ptm_research_config.py`、
  `test_ptm_activity.py`、`test_signed_network.py`、`test_ptm_gene_score.py`、
  `tests/unit/integration/perturbgen/test_downstream_target_evaluation.py`。
- 四阶段 CLI 全链（合成数据）：`tests/integration/test_ptm_activity_pipeline.py`。
- 合成数据只证明契约与管线接通，不构成生物学 PASS（方案 §2.3/§12）。
