# 全局 PTM Activity 与 AD 细胞类型交集驱动 DAVF/PerturbGen 执行方案

版本：v1.0（执行方案）
日期：2026-09-14
状态：待真实 PTM 资产就绪后执行；本文件不代表生物学结果已经通过验收。
实现注记（2026-09-14）：§6.2 所列最小文件与阶段 0/1/3/4/5 的 CLI 契约层已落地
（`src/analysis/ptm_research_config.py`、`ptm_activity.py`、`signed_network.py`、
`ptm_gene_score.py`、`src/integration/perturbgen/downstream_target_evaluation.py` 及
四个 `scripts/` CLI；命令契约见 `docs/guides/ptm_activity_pipeline.md`）。阶段 2
（KSTAR/PhosR）与阶段 6（DAVF/PerturbGen 运行）仍按本方案边界执行。

## 1. 结论与冻结决策

在缺少可靠配对 PTM–RNA 数据的条件下，本项目不训练无标签的直接 PTM→RNA 神经网络，也不把 activity vector 直接加入现有 LatentDAVF。第一阶段采用可解释、可追溯的两层流程：

    真实 PTM（全局，不分 cell type）
        -> KSTAR/PhosR 等 activity inference
        -> signed OmniPath + TF regulon network propagation
        -> global PTM gene score / predicted expression direction
        -> 分别与 AD 每个主要 cell type 的 donor-level DEG 做同方向交集
        -> 每个 cell type 形成 downstream target set
        -> 以 PTM 源蛋白/激酶作为 gene-level intervention 或 surrogate
        -> DAVF/PerturbGen 计算下游表达变化
        -> 以交集基因作为 target/evaluation set

冻结决策：

- PTM 阶段不区分 cell type，输出全局 activity 和 gene score；
- AD 阶段保留 cell type，交集按 cell_type 与 canonical Ensembl 独立生成；
- PTM 源蛋白/激酶是干预来源；交集 DEG 是下游观测和评价目标，不能自动变成干预 token；
- PTM site direction、activity direction、gene score direction、AD disease-normal direction、DAVF decode delta 和 PerturbGen KO/KD/OE action 必须分字段记录；
- 没有配对 PTM–RNA 时，输出是机制先验驱动的候选和方向一致性证据，不是训练得到的 PTM→RNA 因果模型。

## 2. 研究目标与非目标

### 2.1 研究目标

判断真实 PTM 改变经过 kinase/PTM activity 和带符号的 signaling/TF regulon 网络后，能否得到全局、可解释的下游基因方向，并判断该方向是否在 AD 主要细胞类型中有同方向的 donor-level 表达证据。对通过筛选的 PTM 源基因，使用现有 DAVF/PerturbGen 估计 gene-level surrogate intervention 的下游效应。

### 2.2 本阶段交付物

1. 带 provenance 的全局 PTM activity 表；
2. 带 signed edge、网络版本和 coverage 的 global gene score 表；
3. 每个 AD 主要 cell type 的同方向 PTM–DEG 交集表；
4. 每个 cell type 的 candidate spec 和 downstream target-set sidecar；
5. 源基因的 DAVF/PerturbGen gene-level surrogate 分析；
6. 分离记录 proposal、AD 观测、DAVF decode、PerturbGen 统计和科学验收。

### 2.3 非目标

- 不使用 data/AD 单独训练 PTM→RNA 预测器；
- 不把 global PTM score 伪装成 cell-level PTM；
- 不把网络传播结果称为真实表达量或实验 ground truth；
- 不把 AD DEG 交集基因自动当成 PTM 源基因；
- 不把 PTM 类型到 KO/KD/OE 的静态 mapper 当成生物学方向推断；
- 不修改当前 LatentDAVF checkpoint 使其声称支持 activity-conditioned inference；
- 不用 smoke、synthetic、bridge 或工程契约 PASS 代替生物学 PASS。

## 3. 语义与干预边界

### 3.1 方向字段

| 字段 | 含义 | 来源 | 能否直接作为干预动作 |
|---|---|---|---|
| ptm_site_direction | 位点修饰量相对变化 | 真实 PTM | 否 |
| activity_direction | kinase/PTM activity 相对变化 | KSTAR/PhosR | 否 |
| predicted_gene_direction | signed network 推断的下游方向 | 网络传播 | 否 |
| observed_direction | AD donor-level disease - normal | AD DEG | 否 |
| davf_predicted_direction | DAVF decode 后减当前 context | DAVF | 否 |
| davf_action | KO/KD/OE 等 gene-level action | mapper/gate | 是，但只表示模型动作 |

不得用一个全局取反规则转换这些字段。正式分析必须填写现有 semantic_context 的七个字段：context、intervention、comparison_baseline、reference_axis、research_objective、evidence_source、cohort。

### 3.2 研究目标

运行前冻结以下目标之一：

- association：判断 PTM/network 方向是否与 AD 表达状态一致；
- replication：用 gene-level surrogate intervention 复现指定疾病相关方向；
- reversal：评估相反干预是否减少疾病相关方向。

本方案第一轮默认 association。PTM/network 预测基因 G 上调且 AD 某细胞类型中 G 的 disease-normal log2FC 大于零，只能说明方向一致，不自动说明抑制该 PTM 可以治疗 AD。

### 3.3 源基因和下游基因

每条候选必须保留以下关系：

    source_ptm_site -> source_gene/kinase -> network_path -> target_gene(s)

推荐执行模式：

- source_gene 或 kinase：进入 DAVF/PerturbGen 的 gene-level intervention 或 surrogate；
- target_gene(s)：PTM gene score 与 AD DEG 的交集，作为 DAVF/PerturbGen 下游 target/evaluation set；
- source_gene 和 target_gene 不得合并成一个字段。

如果直接扰动 target_gene，研究问题就变成直接干预下游基因，而不是 PTM 源改变导致的下游效应。该模式可以另行命名为 downstream_target_perturbation，但不属于本方案正式主路径。

## 4. 数据与产物契约

### 4.1 真实 PTM 输入

建议标准化为 ptm_site_quantification.tsv，最小字段为：

    sample_id  donor_id  condition  protein_id  gene_symbol  residue
    ptm_type  ptm_value  value_scale  ptm_qvalue  total_protein_value
    species  source_dataset

要求：

- 每行是一个样本、供体、位点和 PTM 类型的定量记录；
- protein、位点和 PTM 类型必须可追溯；
- condition 至少能区分两组或时间点；
- 正式 donor-level 比较必须有显式 donor_id；没有 donor 只能标为 cohort-level exploratory；
- 有总蛋白量时优先进行 site/total-protein 归一化；没有时在 manifest 中写明；
- PTM 没有 cell type 是允许的，但不得伪造 cell-level activity；
- 物种、来源、处理方法、网络版本和许可信息必须登记。

### 4.2 Activity 输出

建议产物为 ptm_activity.tsv：

    activity_unit  regulator_id  regulator_type  condition_or_contrast
    activity_score  activity_direction  activity_pvalue  activity_qvalue
    n_substrates  network_coverage  method  method_version  input_manifest

activity_score 是方法内部的 activity 量纲，不能直接当作 log2FC。磷酸化数据使用 KSTAR 或 PhosR；同时运行时一个作为 primary，另一个作为 sensitivity，不能简单平均成真值。非磷酸化 PTM 需要相应 enzyme–substrate/site 先验。

### 4.3 Signed network 输出

网络边至少包含：

    source_id  target_id  edge_type  effect_sign  site
    species  evidence  confidence  release

至少覆盖：

    enzyme/kinase -- activation/inhibition --> substrate/site
    protein       -- activation/inhibition --> protein
    TF            -- activation/repression --> gene

没有 effect_sign 的边只能用于无符号 coverage 统计，不能用于正式方向传播。OmniPath 作为先验时必须冻结物种、关系类型、证据过滤和 release。

### 4.4 Global gene score 输出

建议产物为 ptm_global_gene_scores.tsv：

    source_activity_id  target_ensembl_id  target_gene_symbol
    gene_score  predicted_gene_direction  n_paths  path_length_min
    network_coverage  degree_normalized  ptm_cohort
    prediction_status  provenance

gene_score 只在网络传播内部尺度上解释。只有独立 null 或外部 benchmark 校准后才能增加 score_qvalue；否则禁止写成 PTM 侧显著。

### 4.5 AD DEG 输入

继续使用当前标准化队列：

    data/AD/standardized/GSE174367_ad_cohort.h5ad

该队列用于 scVI/PerturbGen context 和 cell-type-specific DEG，不用于单独训练 PTM→RNA 映射。AD DEG 至少保留：

    cell_type  ensembl_id  gene_symbol  log2fc  fdr
    observed_direction  n_normal_donors  n_disease_donors

正式 between_donor 口径要求 normal 与 disease donor 不相交，且每组至少 3 个 donor；以现有 frozen manifest 和 PerturbGenDataSpec.pairing=between_donor 为准。

### 4.6 交集和 cell-type 产物

对每个预先冻结的主要 cell type 生成：

    ptm_ad_intersection_<cell_type>.tsv

字段建议：

    cell_type  target_ensembl_id  target_gene_symbol  gene_score
    predicted_gene_direction  observed_log2fc  observed_fdr
    observed_direction  n_normal_donors  n_disease_donors
    direction_match  network_coverage  evidence_tier

交集定义：

    I_c = { g |
           g 在 PTM global score 中，
           g 在 cell_type=c 的 AD DEG 中，
           predicted_gene_direction == observed_direction，
           AD FDR 和 donor support 通过 }

PTM score 侧使用预先冻结的 rank、activity q-value、network coverage 或独立 null 阈值。没有校准阈值时只能输出 direction_match 和 rank_concordance，不能称为双侧显著。

每个 cell type 生成独立 candidate spec，并为该 cell type 绑定真实 context_cell_index。不能把多个 cell type 合成一个 DEG，也不能长期使用默认第 0 行作为 context。

## 5. 算法执行

### 5.1 PTM 预处理

1. 验证 sample、condition、protein、site、PTM 类型和数值；
2. 按预先登记规则处理重复位点；
3. 优先进行总蛋白校正并记录缺失状态；
4. 将 protein/UniProt/基因映射到 canonical Ensembl；
5. 明确比较轴后计算 disease-normal 或指定 contrast；
6. 输出 ptm_input_manifest.json，记录来源、参数、物种、样本和 donor。

### 5.2 Activity inference

1. 读取标准化 phosphosite 矩阵；
2. 使用对应版本的 kinase–substrate network；
3. 输出 activity、方向、p/q、coverage 和 supporting substrate 数；
4. 用已知 kinase perturbation phosphoproteomics 或公开 benchmark 验证 activity 排名；
5. benchmark 未通过时只保留 exploratory 输出，不进入正式 candidate proposal。

KSTAR/PhosR 输出的是上游 activity 证据，不是下游表达标签。KSTAR 参考：https://www.nature.com/articles/s41467-022-32017-5；PhosR 参考：https://pmc.ncbi.nlm.nih.gov/articles/PMC8190506/。

### 5.3 Signed network propagation

传播使用实际 edge sign 和证据权重：

    activity(source)
        × signed_edge(source, target)
        × evidence/confidence_weight
        -> target activity
        -> TF activity
        -> gene_score

记录最短路径、路径数量、coverage 和 degree normalization。高连接度基因不能只因边多就获得高排名。当前 src/models/signaling_network.py 的硬编码 pathway、output gene 和固定影响因子不能用于正式传播。

### 5.4 PTM–AD 交集

1. 预先冻结主要 cell type 列表；
2. 以 canonical Ensembl 为唯一连接键；
3. 每个 cell type 独立 join；
4. 要求 predicted_gene_direction 与 observed_direction 同向；
5. 对 PTM score 使用已校准的 rank、q-value 或 null 规则；
6. 保留 PTM_only、AD_only、concordant 和 discordant；
7. 记录 n_cell_types_supported，但不把跨 cell type 重复当作独立 donor 证据。

### 5.5 DAVF/PerturbGen

对每个 cell type：

- source_gene/kinase 作为当前 gene-level intervention 或 surrogate；
- intersection target set 作为 downstream target/evaluation set；
- DAVF 输出源干预后的全基因或 target-set expression delta；
- PerturbGen 继续执行 source_intervention 和 within_state 两个场景；
- target-set 评价按 canonical Ensembl 提取结果，不改变 token index 和 scVI decoder index；
- source gene 无 verified PerturbGen token 时直接硬失败。

现有 candidate_spec.py 和 orchestrator.py 仍是正式 proposal 与 invocation 接口。第一版用 downstream target-set sidecar 承载一对多 source–target 关系，避免把旧 candidate schema 硬改成另一种语义。

当前 direction_gate 仍然比较同一个 proposal gene 的 proposal direction、DAVF decode direction 和 AD observed direction。若 source_gene 本身没有对应的 AD DEG，target-set 的方向一致性不能替代 source gene 的既有三方 gate；该候选只能作为 exploratory，除非另行批准并实现 driver–target gate contract。不得为了让候选进入 runner 而把 target_gene 的 DEG 改写成 source_gene 的 observed_direction。

## 6. 代码和资产调整

### 6.1 直接复用

| 文件或资产 | 用法 |
|---|---|
| scripts/import_kinase_substrate.py | OmniPath enzyme–substrate 导入起点；不改变现有 enzsub 语义 |
| data/manifests/datasets.yaml | 登记 PTM、网络 release、来源和许可 |
| data/AD/standardized/GSE174367_ad_cohort.h5ad | scVI/PerturbGen context、cell type、donor 和 DEG |
| src/integration/perturbgen/candidate_spec.py | 生成每个 cell type 的正式 candidate spec |
| src/integration/perturbgen/orchestrator.py | proposal→DAVF→gate→invocation |
| src/integration/perturbgen/direction_gate.py | 既有 proposal、DAVF、AD observed 三方 gate |
| src/models/davf_inference.py | gene-level intervention 的 scVI latent decode |
| src/models/ptm_direction_mapper.py | 只用于 action/token mapping |
| scripts/run_davf_perturbgen_e2e.py | 现有 formal E2E 入口 |

### 6.2 待新增的最小文件

以下是代码边界建议，不表示已经实现。实现注记（2026-09-14）：以下清单中的
模块与 CLI 已按本节边界落地（含阶段 0 的 `ptm_research_config` 契约，作为
`src/analysis/ptm_research_config.py` 补充；传播算法为有符号简单路径枚举，
无硬编码 output genes 或固定影响因子）：

    src/analysis/ptm_activity.py
        PTM 输入校验、activity 工具输出解析和 manifest

    src/analysis/signed_network.py
        signed edge 读取、过滤、传播和 coverage

    src/analysis/ptm_gene_score.py
        global gene score、方向、rank/null 状态和 provenance

    scripts/run_ptm_activity.py                         # 已实现（2026-09-14）
    scripts/build_ptm_global_gene_scores.py             # 已实现（2026-09-14）
    scripts/build_ptm_ad_intersections.py               # 已实现（2026-09-14）
    scripts/build_celltype_candidate_specs.py           # 已实现（2026-09-14）

    src/integration/perturbgen/downstream_target_evaluation.py
        按 target-set 提取 DAVF/PerturbGen 下游 delta

第一版不新增 PTM-conditioned DAVF、不新增无标签表达神经网络、不把 KSTAR/PhosR 重型依赖加入 requirements-core.txt。若 PhosR 需要 R 环境，使用独立分析环境，以标准表和 manifest 交接。

### 6.3 当前文件的处理边界

- src/models/signaling_network.py：保留为探索性/历史路径，正式 PTM score 不调用；
- src/models/ptm_direction_mapper.py：保留 action code，禁止将其结果命名为表达方向；
- src/models/latent_davf.py 与 src/models/davf_inference.py：第一阶段不改 condition encoder；
- src/integration/perturbgen/candidate_spec.py：优先用 target-set sidecar，避免扩大旧 contract；
- src/integration/perturbgen/direction_gate.py：网络下游一致性是补充 evidence，不替代既有三方 gate；
- PerturbGenRunner：不自行重查 gate，正式入口仍是绑定通过 invocation 的 E2E wrapper。

### 6.4 Gate-E 边界

本方案属于 Workflow A 的 PTM proposal、方向 gate 和 PerturbGen 效用分析，不自动触发 Workflow B 的 Gate-E。只有在未来要把 activity condition 真正加入 LatentDAVF，并拥有新的训练输入、冻结 feature order、重训 checkpoint 和独立的至少 200 条可追溯真实 PTM→gene benchmark 时，才启动 Gate-E。当前 global gene score 与 AD DEG 交集不能充当 Gate-E benchmark。

## 7. 阶段化执行计划

### 阶段 0：冻结研究设计

输入：研究目标、PTM 数据候选和 AD 主要 cell type 列表。
动作：冻结 reference axis、contrast、research objective、primary activity method、network release、AD DEG 阈值和 PTM score 过滤策略。
产物：ptm_research_config.yaml（契约已实现：src/analysis/ptm_research_config.py；正式 yaml 由真实研究运行时冻结）和研究设计记录。
完成条件：cell type、donor 口径和方向解释固定。

### 阶段 1：准备真实 PTM

输入：真实 phosphosite 或其他 PTM 定量矩阵。
动作：标准化、位点/蛋白映射、总蛋白检查、condition/donor 审计。
产物：标准 PTM 表、ptm_input_manifest.json 和审计报告。
停止条件：缺 site 定量、物种不符、比较轴不明或 donor 语义不可解释。

### 阶段 2：Activity inference

输入：标准 PTM 表、KSTAR/PhosR 网络和独立 benchmark。
动作：运行 primary 方法，另一方法可做 sensitivity。
产物：ptm_activity.tsv、method log 和 activity manifest。
完成条件：输出可重放，网络版本和输入范围可追溯。

### 阶段 3：Signed network gene score

输入：activity 表、OmniPath enzyme–substrate/signaling 和 TF regulon。
动作：传播到 TF 和 gene，执行 degree normalization，输出路径和 coverage。
产物：ptm_global_gene_scores.tsv、network manifest 和必要的 degree-matched null。
完成条件：没有硬编码 output genes 或固定影响因子；无 signed edge 的结果不进入正式方向表。

### 阶段 4：AD cell type 交集

输入：global gene score、AD donor-level DEG 和冻结的主要 cell type 列表。
动作：按 Ensembl 分 cell type join，要求方向一致，保留未匹配和反向结果。
产物：每个 cell type 的 intersection 表、统计摘要和 target-set manifest。
完成条件：AD FDR/donor 口径通过；PTM 侧没有 null 时不写显著性。

### 阶段 5：形成候选和运行输入

输入：PTM source proposal、intersection target-set、scVI/embedding asset。
动作：每个 cell type 生成 candidate spec；source gene 作为干预对象，target set 作为 sidecar；补齐 semantic context 和 provenance。
产物：candidate_spec_<cell_type>.json、downstream_targets_<cell_type>.json 和 proposal lineage。
完成条件：source gene 的 Ensembl、site position、PTM type、verified token 和真实 context_cell_index 有效。

### 阶段 6：DAVF/PerturbGen 与统计

输入：通过 gate 的 candidate spec、冻结 cohort、DAVF checkpoint、embedding asset 和 PerturbGen 独立环境。
动作：使用现有 scripts/run_davf_perturbgen_e2e.py；只有显式 run-perturbgen 才执行六阶段；沿用 shared prepare/reuse、donor split、null、quality、dual-path 和统计组装契约。
产物：E2E report、两个场景结果、target-set delta、matched-null、empirical p/q、quality 和双路径证据。
完成条件：invocation 绑定通过 gate，所有路径、donor、seed、null、质量和 target-set lineage 可复核。

现有命令和参数以 docs/guides/davf_perturbgen_e2e.md、docs/guides/perturbgen_bridge.md 为准。新增 PTM 脚本的命令接口在实现前不得写成已存在命令。

## 8. 数据泄漏和统计边界

1. PTM activity 和 network gene score 阶段不得读取 AD DEG；
2. AD DEG 可以作为 PTM proposal 的独立表达一致性 gate，但同一批 DEG 参与筛选后不能再次称为 DAVF 的独立测试集；
3. DAVF/PerturbGen train donor 与 held-out donor 按现有 donor split manifest 绑定；
4. 随机按 cell 切分不能替代 donor-level split；
5. 多 cell type 中同一基因重复出现不是多个独立实验；
6. network target 与 AD DEG 同方向只能叫 concordance，不能叫 causal validation；
7. PTM score 显著性需要独立 permutation、degree-matched null 或外部 benchmark；没有 null 时只报告 rank/coverage；
8. DAVF predicted delta 是 intervention decode 减 context decode，AD log2FC 是 disease 减 normal，二者不可直接合成同一个效应量。

## 9. 验收标准

### 9.1 PTM/activity 层

- 位点、样本、物种和处理过程可追溯；
- activity 输出可重放，网络和版本已冻结；
- 已知 kinase/PTM perturbation benchmark 有明确 activity 评估；
- 缺少总蛋白、site mapping 或 donor 时显式标为 exploratory。

### 9.2 Network/gene score 层

- activation、inhibition、repression 符号显式记录；
- degree bias、路径数量和 coverage 有报告；
- gene score 只解释为相对方向或 rank；
- PTM 阈值不是在 AD DEG 结果上调出来的。

### 9.3 AD 交集层

- 使用 canonical Ensembl；
- 每个主要 cell type 独立计算；
- 同方向匹配、FDR 和 donor support 可追溯；
- target-set 与 source intervention 角色分离。

### 9.4 DAVF/PerturbGen 层

- DAVF 使用有效 checkpoint、scVI gene order 和冻结 embedding asset；
- candidate、DAVF delta、AD observed direction 和 action code 分字段；
- 正式 PerturbGen 只能从通过 invocation 的 E2E wrapper 进入；
- raw counts、canonical Ensembl、显式 donor、held-out split、matched null、未扰动质量和双路径统计满足现有契约；
- smoke、synthetic、bridge 和工程测试只能证明执行或契约，不能证明生物学 PASS。

## 10. 外部输入与阻塞项

必须取得：

1. 至少一份真实、定量、位点级 PTM 数据；
2. 对应物种和 PTM 类型的 enzyme–substrate/site 网络；
3. 带 activation/inhibition/repression 语义的 signaling 和 TF regulon 网络；
4. activity inference 所需的独立 benchmark 或已知 perturbation 结果；
5. PTM 数据与 AD 队列在组织、疾病、物种和比较轴上的兼容性说明；
6. 文献 PTM site 锚点、candidate proposal、donor-level DEG、冻结 cohort 和 PerturbGen 资产。

没有配对 PTM–RNA 时：

- 同疾病/组织/物种但不同 donor：结果为 cohort-level mechanistic concordance；
- 条件不匹配：只能 exploratory；
- 不能做 donor-level paired prediction；
- 只有出现足够的匹配 PTM–RNA/perturbation 训练和 held-out 数据，才考虑监督式表达预测器或 activity-conditioned DAVF。

## 11. 最小落地路径

1. 冻结研究目标、方向轴、主要 cell type 和 PTM 数据兼容性；
2. 实现 PTM 输入契约和 activity 输出解析；
3. 复用 scripts/import_kinase_substrate.py，补齐 signed signaling/TF 网络资产；
4. 实现 signed propagation 和 global gene score；
5. 实现 global score 与每个 AD cell type DEG 的 Ensembl 同方向交集；
6. 每个 cell type 生成 source candidate spec 和 downstream target-set sidecar；
7. 不改 LatentDAVF，先运行现有 DAVF/PerturbGen 主线；
8. activity、network、交集和 surrogate utility 均可复核后，再评估是否需要新 contract；
9. 未出现真实训练输入和独立 benchmark 前，不升级为 activity-conditioned DAVF。

## 12. 完成定义

本方案完成不等于 PTM→RNA 模型训练完成。完成时应具备：

- 真实 PTM 输入和完整 provenance；
- 可重放的 activity 输出；
- 带符号、版本和 coverage 的 network gene score；
- 每个主要 AD cell type 的同方向交集和 target-set；
- source intervention 与 downstream target evaluation 的角色分离；
- 现有 DAVF/PerturbGen 的 invocation、donor split、null、quality 和双路径 lineage；
- 所有结果有 association、replication 或 reversal 的明确语义；
- 明确区分机制一致性、模型输出和正式生物学验收。

在此之前，任何“PTM 导致了某 cell type 的基因表达变化”都应改为“PTM/activity/network 预测与该 cell type 的 AD DEG 方向一致”。

## 13. 当前项目依据与参考

项目文件：

- docs/CURRENT_STATUS.md
- project_analysis_20260917.md
- docs/guides/davf_perturbgen_e2e.md
- docs/guides/perturbgen_bridge.md
- docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md
- scripts/import_kinase_substrate.py
- src/integration/perturbgen/candidate_spec.py
- src/integration/perturbgen/orchestrator.py
- src/integration/perturbgen/direction_gate.py
- src/models/ptm_direction_mapper.py
- src/models/davf_inference.py
- src/models/signaling_network.py

文献与资源：

- KSTAR：https://www.nature.com/articles/s41467-022-32017-5
- PhosR：https://pmc.ncbi.nlm.nih.gov/articles/PMC8190506/
- OmniPath：https://academic.oup.com/nar/article/54/D1/D652/8326458
- phosphoproteomics 到转录表达的网络模型：https://pmc.ncbi.nlm.nih.gov/articles/PMC4216050/
- LEMBAS：https://www.nature.com/articles/s41467-022-30684-y
