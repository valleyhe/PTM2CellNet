# observed gate 与无真实 KO/KD 扰动时的预测执行方案

**版本**：v1.0
**日期**：2026-09-18
**适用范围**：PTM activity 主线、AD observed gate、DAVF 方向证据和
PerturbGen 下游效用评估
**当前科学状态**：正式 biology PASS = 0

**2026-09-18 研究决策冻结（已写入 `ptm2cellnet.ad-research-decision/v1`）**

1. observed gate 选分支 3：修订 *准入* 语义为 signed disease−normal 方向一致。
   donor-level Welch+BH 与 `deg_max_fdr=0.05` 仍是报告/显著性标签；不得把
   FDR>0.05 改标成显著，也不得下调 FDR 或改 gene universe。
2. KD 不再单独做；本 AD 五候选只走 KO loss-of-function。DAVF 标签 0/1 仍分离；
   不得从 KO 取反或复用 KO checkpoint 当 KD。
3. 不再检索公共 Perturb-seq；B1/B3 transfer 为 out of scope。阶段 2 资产盘点取消。

> 本文是可执行的研究与工程方案，不是当前结果报告。文中“可预测”只表示
> 可以生成可审计、可校准、用于候选排序的方向假设；没有 target-held-out
> 验证时，不承诺准确率，也不能把结果写成真实 KO/KD 效应、DAVF ground truth、
> 因果结论或正式 biology PASS。

## 目录

- [1. 摘要与决策结论](#1-摘要与决策结论)
- [2. 范围、明确不做事项与正式验收边界](#2-范围明确不做事项与正式验收边界)
- [3. 当前已证实问题与根因](#3-当前已证实问题与根因)
- [4. 预测目标与语义冻结](#4-预测目标与语义冻结)
- [5. 无真实本地 KO/KD 时的证据分层](#5-无真实本地-kokd-时的证据分层)
- [6. 推荐主路线、备选路线与候选分流](#6-推荐主路线备选路线与候选分流)
- [7. 数据、manifest 与 schema 契约](#7-数据manifest-与-schema-契约)
- [8. 评估设计与可校准性](#8-评估设计与可校准性)
- [9. 端到端阶段计划](#9-端到端阶段计划)
- [10. 当前仓库的最短落地路径](#10-当前仓库的最短落地路径)
- [11. 失败、阻塞与禁止降级语义](#11-失败阻塞与禁止降级语义)
- [12. 资源、工期与风险](#12-资源工期与风险)
- [13. Formal biology acceptance checklist](#13-formal-biology-acceptance-checklist)
- [14. 参考资料与本地证据](#14-参考资料与本地证据)

## 1. 摘要与决策结论

当前有两个独立阻塞：

1. **observed gate 无显著行**：当前 AD 分析以 donor 为独立样本单位，
   使用 <code>between_donor</code>、donor-level pseudobulk、Welch 检验和
   BH-FDR。细胞数很多不等于 donor 数足够。GSE174367 的 EX/INH 都只有
   normal 7 vs disease 11 个 donor；两队列 centered 合并后为 16 vs 23
   个 donor，最小 FDR 仍为 EX 0.5254、INH 0.7908。
2. **DAVF 训练轴/KD 对当前候选缺失**：当前五个 AD 候选都在
   PerturbGen vocabulary 中，但轴审计显示只有 APOE 进入候选 KO 轴，
   KD 对五个候选为空；其余 APP、PSEN1、BACE1、MAPT 没有本地 target-specific
   KO/KD 扰动训练输入。KD 模型和 checkpoint 存在，不等于它覆盖这些候选。

本方案采取两条并行但不混淆的路径：

- **探索性预测路径**：在没有本地真实 KO/KD 时，优先寻找 context 尽可能匹配的
  公共 Perturb-seq/CRISPRi 数据，采用 target-held-out 评估的 GEARS/CPA 或
  同类迁移模型；用 AD-specific GRN/CellOracle、foundation model 和 PTM signed
  network 作为分字段的补充证据。结果最多写成
  <code>direction_only</code>、<code>exploratory</code> 或候选排序。
- **正式主线路径**：只有真实 target-specific KO/KD 扰动数据产生
  同一 scVI 坐标系的 <code>z0/z1</code> latent pair，并通过 held-out 方向验证后，
  才能训练或消费 route-specific LatentDAVF；随后仍必须通过
  PTM proposal + DAVF + observed expression 三方 gate，才能进入 PerturbGen。

推荐顺序是：

1. 先冻结语义、候选、参考轴、数据来源和拆分；
2. 单独审计 observed gate，并由研究负责人选择增加 donor、预注册基因面板或修订
   gate 语义；
3. 按 target × context × intervention 盘点公共扰动数据；
4. 先跑 no-op/mean/linear 基线，再跑 GEARS/CPA；
5. 在有适用 TF/ATAC 数据时补 CellOracle/GRN，在没有方向语义时只用
   Geneformer/scGPT 做影响排序；
6. 形成探索性 evidence sidecar，不接入 formal lineage；
7. 只有 A、B 两侧都满足正式条件后，才执行 E2E/PerturbGen biology acceptance。

## 2. 范围、明确不做事项与正式验收边界

### 2.1 本方案范围

本方案覆盖以下候选和主线：

<div align="center">

<code>PTM site → activity → signed network → AD observed → candidate → DAVF → 三方 gate → PerturbGen</code>

</div>

候选为：

| 候选 | Ensembl ID | 研究对象 |
|---|---|---|
| APOE | ENSG00000130203 | 当前有候选 KO 轴；KD 仍缺失 |
| APP | ENSG00000142192 | 当前无候选 KO/KD 训练轴 |
| PSEN1 | ENSG00000080815 | 当前无候选 KO/KD 训练轴 |
| BACE1 | ENSG00000186318 | 当前无候选 KO/KD 训练轴 |
| MAPT | ENSG00000186868 | 当前无候选 KO/KD 训练轴 |

本方案可以产出：

- 公共扰动模型的 post-perturbation expression delta；
- 目标基因和 readout gene 的方向、效应量、区间或模型不确定性；
- target-held-out、context-held-out 和 time/dose-held-out 的评估结果；
- PTM、observed、网络、模型之间的字段分离证据；
- 可被人工审阅的 exploratory candidate ranking；
- 在真实 target-specific pair 到位后，进入现有 LatentDAVF/E2E 体系的实施顺序。

### 2.2 明确不做事项

以下行为禁止写入方案实现，也禁止在报告中暗示已完成：

1. 不把 observed 的 AD <code>disease - normal</code> 方向当作 KO/KD 干预
   ground truth。
2. 不把 vocabulary/token 命中当作 target-specific perturbation coverage。
3. 不用 APOE KO checkpoint 替代其他候选，也不把 KO 全局取反伪造成 KD。
4. 不用 gene-space delta、均值差、zero-fill、随机 token 或随机 latent pair
   伪造 DAVF 训练输入。
5. 不把 Geneformer/scGPT 的 embedding 或 influence score 直接命名为
   up/down expression effect。
6. 不把 PTM signed propagation、文献关联、自然扰动或遗传关联当成干预效应。
7. 不因现有队列无显著行而放宽 FDR、改用 raw p、事后缩小 gene universe 或
   改写 donor 统计单位。
8. 不在 APOE anchor 回测失败时把 GEARS/Geneformer 接入正式 gate 或 lineage。
9. 不把单一路径的 <code>source_intervention</code> 或
   <code>within_state</code> 结果外推为普遍治疗效用。
10. 不因 smoke、synthetic、mock、bridge 或 schema 通过而宣称
    formal biology PASS。

### 2.3 Formal 与 exploratory 的边界

| 结果类型 | 允许的输入 | 允许的结论 | 禁止的结论 |
|---|---|---|---|
| formal DAVF | 真实 target-specific KO/KD、同 route scVI、真实 z0/z1 pair、held-out 验证 | route-specific DAVF direction evidence | 用外部模型代替 DAVF 训练轴 |
| exploratory trained-response | 公共扰动数据、明确干预语义、target-held-out 评估 | 可校准的方向假设/排序 | AD-specific 因果效应、formal gate pass |
| exploratory network-counterfactual | AD context、可审计 GRN/TF/ATAC 或 foundation model | 网络支持的方向或影响排序 | 定量 KO/KD 表达效应 |
| direction_only | PTM/文献/自然扰动/无可校准模型 | 研究假设和后续实验优先级 | 置信度冒充统计显著性 |
| formal biology acceptance | 以上所有必要证据，加真实 observed gate、null、质量、双路径统计 | 项目定义下的正式验收结论 | 任何缺失条件下的 PASS |

探索性预测不是 formal DAVF 的降级模式。它是独立的 supplementary evidence
生命周期，不能让 invocation 绕过 observed gate 或 DAVF gate。

## 3. 当前已证实问题与根因

### 3.1 observed gate 无显著行的根因

当前代码路径是 donor-level 分析，不是把细胞当独立样本：

1. 根据显式 <code>between_donor</code> 语义区分 normal 和 disease donor；
2. 对每个 cell type、donor 和 gene 聚合表达；
3. 计算 donor-level disease-minus-normal effect；
4. 做 Welch 检验和 BH-FDR；
5. candidate builder 对 <code>fdr &gt; 0.05</code> 行不生成 formal candidate。

证据和当前数值：

| 证据 | 当前事实 |
|---|---|
| d1 输入 | <code>outputs/ptm_activity/20260916_d1/ad_deg_manifest.json</code> |
| d1 donor 数 | EX/INH 均为 normal 7、disease 11 |
| d1 最小 FDR | EX 0.915381，INH 1.0 |
| d2 输入 | <code>outputs/ptm_activity/20260916_d2/ad_deg_centered_manifest.json</code> |
| d2 donor 数 | EX/INH 均为 normal 16、disease 23，按 dataset centered |
| d2 最小 FDR | EX 0.525380，INH 0.790793 |
| 固定阈值 | <code>deg_max_fdr=0.05</code> |

主要统计原因是有效样本量小、donor 间表达异质性和文库深度差异扩大方差；
跨队列 baseline shift 是放大因素。两队列 centered 后有所改善但仍离 0.05 很远，
因此不能把问题归结为一个实现过滤错误，也不能通过更换统计口径后挑出有利结果。

### 3.2 DAVF 训练轴和 KD 缺失的根因

当前轴审计文件
outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv
给出的候选覆盖为：

| 候选 | embedding vocabulary | KO candidate axis | KD candidate axis | 实际扰动证据 |
|---|---:|---:|---:|---|
| APOE | true | true | false | FrangiehIzar2021 KO |
| APP | true | false | false | 当前本地候选审计无 |
| PSEN1 | true | false | false | 当前本地候选审计无 |
| BACE1 | true | false | false | 当前本地候选审计无 |
| MAPT | true | false | false | 当前本地候选审计无 |

这里的 KD 不是“没有 KD 代码”。当前 KD route 有独立 checkpoint 和训练目标，
但它没有这五个候选的 target-specific intervention response。正式 DAVF 训练要求：

<div align="center">

<code>真实 target cell → matched control/perturbed pair → scVI z0/z1 → token target + intervention type → LatentDAVF</code>

</div>

词表只回答“这个基因能否作为 token/context 被编码”；它不回答“这个基因干预后
表达如何变化”。因此 vocabulary 为 true、KD axis 为 false 可以同时成立。

当前本地已有的外部证据也不能补上该缺口：

- GEARS 资产的 context 是 K562/Norman，属于 <code>go_extrapolation</code>；
- Geneformer 资产是 GSE174367 EX disease-cell 的 in-silico influence spectrum，
  没有 up/down expression direction；
- APOE 预注册 anchor 为 <code>verdict=fail</code>，GEARS top-100 方向一致率
  0.25（阈值 0.59），Spearman -0.0428；两源均未进入 lineage。

### 3.3 两个阻塞的逻辑关系

~~~mermaid
flowchart LR
    A[PTM proposal / activity / signed network] --> B[候选 spec]
    B --> C[DAVF intervention direction]
    B --> D[observed donor-level disease-normal]
    C --> E{三方 gate}
    D --> E
    A --> E
    E -->|pass| F[PerturbGen source_intervention + within_state]
    E -->|inconclusive/blocked| X[停止 formal invocation]
    G[公共模型/GRN/文献探索证据] -. supplementary only .-> C
    G -. 不能替代 .-> E
~~~

修复 observed gate 不会自动产生 KD latent pair；补公共预测模型也不会提高 observed
FDR。两侧必须分别解决，不能合并成一个“模型同意”分数。

## 4. 预测目标与语义冻结

### 4.1 每次运行必须冻结的七项语义

每个候选和每个证据源都必须显式记录：

| 字段 | 允许内容 | 说明 |
|---|---|---|
| <code>context</code> | 例如 GSE174367 EX disease、K562 Norman | 细胞类型、状态、队列、培养条件的完整上下文 |
| <code>intervention</code> | KO、KD、OE | 真实机制或模型反事实，不能省略 |
| <code>comparison_baseline</code> | matched control、当前 context、normal donor baseline | 预测差值的基准 |
| <code>reference_axis</code> | disease_minus_normal、intervention_minus_control 等 | 只说明比较轴，不做全局同号转换 |
| <code>research_objective</code> | association、replication、reversal | 不能把 association 写成 causal treatment |
| <code>evidence_source</code> | PTM、observed、DAVF、GEARS、CPA、GRN 等 | 证据来源必须分栏 |
| <code>cohort</code> | 真实数据/公共数据集名称及版本 | 用于 provenance 和可比性审计 |

### 4.2 KO、KD 与近似干预的语义

- **KO**：目标基因完全或近似完全失活；只有原始实验明确为 KO，或模型在
  KO 数据上训练，才能写 <code>intervention=KO</code>。
- **KD**：目标基因的部分抑制/CRISPRi/剂量降低；必须有 CRISPRi、partial
  knockdown、剂量序列或明确的 KD 训练语义。
- **OE**：过表达，不能作为 KO/KD 的取反。
- **in_silico deletion**：foundation model 的删除反事实，单独写
  <code>evidence_kind=network_counterfactual</code>，不改名为 KO。
- **GO extrapolation**：GEARS 的未知目标外推，单独写
  <code>evidence_kind=go_extrapolation</code>，不改名为真实 KO/KD。

项目中的方向编码仍按现有 DAVF 契约：KO=0、KD=1、OE=2。route、checkpoint、
scVI 坐标系和训练标签必须一致；不能用一个 route 的 checkpoint 处理另一个
intervention type。

### 4.3 预测输出的最小语义

每个 target × context × intervention × readout gene 至少输出：

| 字段 | 含义 |
|---|---|
| <code>predicted_delta</code> | 预测的 post-intervention 减 baseline 效应量；明确单位/变换 |
| <code>predicted_direction</code> | predicted_delta 的 up/down/neutral；低于方向分辨率时为 null |
| <code>uncertainty</code> | seed、bootstrap、ensemble 或模型区间；没有就写 unavailable |
| <code>confidence_status</code> | direction_only、exploratory_supported、formal_davf |
| <code>evidence_kind</code> | trained_response、go_extrapolation、network_counterfactual 等 |
| <code>validation_status</code> | not_evaluated、heldout_pass、heldout_fail、inconclusive |
| <code>context_similarity</code> | 与目标 AD context 的可比性证据，不可默认为高 |
| <code>model_version</code> | 模型、代码 commit、数据版本和 seed |

target 自身的 delta 与 readout gene delta 分开记录。若模型只能输出 influence
score，则只输出 <code>influence_score</code>，不得填入
<code>predicted_delta</code> 或 <code>predicted_direction</code>。

### 4.4 两种 observed/干预方向不能混写

| 来源 | 差值 | 语义 |
|---|---|---|
| observed | donor-level disease - normal | AD 队列中的状态关联 |
| DAVF | decode(z_intervened) - decode(z_context) | route-specific intervention response |
| PerturbGen | perturbed - unperturbed，按 source 或 target 场景 | 下游效用预测 |
| 公共模型 | model-specific post-perturbation - baseline | 公共 context 中的模型预测 |

即使四种差值符号相同，也只能说明方向一致性，不能说明它们是三个独立实验，
更不能把同号改写成因果验证。

### 4.5 source_intervention 与 within_state

- <code>source_intervention=[src]</code>：状态转移前的 source 场景；
  目标是评估对进入目标状态的影响。
- <code>within_state=[tgt]+pert_tps</code>：目标状态内部的扰动场景；
  目标是评估目标状态内的效用。

正式候选保留两路 AND。公共模型若只在某一 context 或某一场景提供预测，只能
标注相应场景的 exploratory evidence，不能外推为双路径或治疗普遍性。

## 5. 无真实本地 KO/KD 时的证据分层

### 5.1 分层总表

| 层级 | 方法/来源 | 最低输入 | 主要输出 | 适用条件 | 默认状态 |
|---|---|---|---|---|---|
| A | 同 context 公共真实扰动 transfer | target-specific KO/KD、control、cell/state/dose/time | 可量化 delta 和方向 | 目标/上下文/机制高度可比 | 可进入 calibrated exploratory |
| B | GEARS | Perturb-seq 训练集、gene graph/GO graph、未知 target query | post-perturbation expression | 目标可由知识图谱外推；需 target-held-out 验证 | exploratory，当前资产 anchor fail |
| C | CPA/latent compositional model | 单细胞 perturbation、control、time/dose/covariate | latent shift、重建表达、组合/剂量反事实 | 有相同 perturbation 语义和剂量/时间轴 | target/context 匹配时 exploratory |
| D | AD-specific GRN/CellOracle | AD scRNA、TF-target prior，最好有 scATAC/motif | GRN 传播方向、状态转移/TF 反事实 | 目标是 TF/调控节点且网络可审计 | network_counterfactual |
| E | Geneformer/scGPT zero-shot | 目标 context、模型词表和冻结权重 | influence/embedding/排序 | 只能解释模型内部影响，方向需另行校准 | direction_only/ranking |
| F | PTM signed network、文献、自然扰动 | signed network、已发表结果、变异/药理 proxy | 机制方向假设、证据链接 | 作为上游 proposal 或独立先验 | direction_only |

选择规则：先用高层级、context 匹配且干预语义相同的证据；低层级证据只能
补充，不能把不匹配证据“加权相加”成高置信度。

### 5.2 公共 Perturb-seq transfer：首选

#### 输入

至少需要：

1. 原始或可追溯的 control/perturbed single-cell expression；
2. canonical Ensembl 或可审计的 symbol→Ensembl 映射；
3. target、perturbation mechanism、guide/剂量、时间点；
4. cell type、state、donor/batch 和处理条件；
5. 目标候选在测试数据中确实有真实 perturbation 标签；
6. 数据许可、版本、下载地址和完整 manifest。

#### 处理

先按 context 相似度分层，再训练或迁移：

1. 同物种、同细胞类型、同状态、同机制；
2. 同物种、相近细胞类型、同机制；
3. 同物种、不同细胞类型但存在可解释的 pathway/GRN 共享；
4. 仅有其他物种或其他机制时，不作为主预测证据。

必须使用 target-held-out split：同一 target 的全部 perturbation cells、guide、
donor 和重复不能同时出现在 train 与 test。若测试目标在训练数据中被任何 proxy
标签直接暴露，不能称为 unseen-target evaluation。

#### 优点与限制

- 优点：直接学习真实干预后的表达响应，输出最接近所需的 delta。
- 限制：context shift、CRISPR modality、时间/剂量和细胞组成会显著改变响应；
  公共细胞系结果不能直接作为 AD neuron/glia 的 ground truth。
- 进入 exploratory 的最低条件：target-held-out 测试结果、与 no-op/mean/linear
  基线比较、context 差异清单齐全。
- 进入 formal DAVF：仍不够，必须重新生成当前项目 route 的 scVI z0/z1 pair，
  并完成 LatentDAVF held-out 验证。

### 5.3 GEARS：知识图谱外推

GEARS 使用基因关系图和 GO-derived perturbation graph，在未直接观测目标的情况下
预测单基因或组合扰动的表达结果。原始论文在多个 Perturb-seq 数据集上采用
未见 target 的评估，并把 no-perturbation、GRN/linear 等作为基线。

#### 输入

- 训练 Perturb-seq 的 control 与已标注 perturbation；
- gene relationship graph 与 perturbation knowledge graph；
- target list 和 canonical mapping；
- cell context、训练配置、seed 和 checkpoint；
- 预先注册的 target-held-out 测试集。

#### 输出

- readout gene × target 的 predicted post-perturbation expression；
- 相对 control 的 predicted delta；
- top-k readout、方向和模型 seed/ensemble 统计。

#### 使用条件

GEARS 适用于候选与训练集存在可迁移的基因关系、GO 关系和 context。对 APP、
PSEN1、BACE1、MAPT，应先确认其 query 是否真的进入模型支持图，而不是只确认
它们在 PerturbGen vocabulary 中。

当前本地 GEARS 资产是 K562/Norman 的 <code>go_extrapolation</code>。现有
APOE anchor 的 top-100 一致率 0.25、Spearman -0.0428，整体 verdict=fail；
因此这批资产不能进入主线 lineage。重新训练或迁移后，必须在与目标 context
尽量接近的公共真实扰动数据上重新评估，不能沿用当前 fail 的结论作为通过证据。

GEARS 输出即使在标准 benchmark 通过，也只是外部模型 evidence；它不是当前
LatentDAVF 的训练轴，也不能生成正式所需的 scVI latent pair。

### 5.4 CPA：组合/剂量/时间的 latent response

CPA 学习基线细胞状态与 perturbation、dose、time、covariate 的组合表征，适合
在具有剂量/时间设计的单细胞扰动数据上估计 compositional response。它的输入可
包括 control/perturbed expression、perturbation identity、dose、time 和 batch；
输出通常是 latent response 与重建表达。

使用规则：

1. 若 target 在训练数据中已有真实扰动，CPA 可用于 context、dose 或 time 的
   迁移预测；
2. 若 target 从未出现，普通 CPA 不能凭 token vocabulary 自动获得该 target 的
   干预效应，必须有明确的 target embedding/knowledge transfer 机制并做
   target-held-out 验证；
3. KO 与 KD 不能只靠剂量值互换，必须确认原始数据的 perturbation semantics；
4. CPA 的 latent shift 不得直接命名为项目的 scVI z0/z1，除非重新按当前 scVI
   adapter 生成并验证 pair。

CPA 的优势是显式处理 dose/time/covariate；缺点是对 AD context 外推很敏感，
并且未见目标的泛化依赖训练设计，不能从方法名称推断可用性。

### 5.5 AD-specific GRN/CellOracle

CellOracle 类方法从 scATAC/motif 或 promoter prior 建立 base GRN，再结合
scRNA 推断 context-specific GRN，并进行调控因子扰动传播。它适合：

- 目标本身是 TF 或明确的调控节点；
- 有 AD 细胞类型对应的 scATAC/motif 或高质量 TF-target prior；
- 可以将网络边、符号、置信度、细胞类型和版本逐条追溯。

输出是网络传播的方向、向量场或状态转移得分，不是未经校准的 KO/KD expression
ground truth。APP、PSEN1、BACE1、MAPT 不是默认的 TF 候选；若目标没有直接调控
网络路径，CellOracle 只能作为 target downstream network 的补充，不能强行运行
并生成伪方向。

最低验证：

1. 与已知 AD TF perturbation 或公共 target-held-out 结果比较；
2. 对网络边做 signed consistency 和敏感性分析；
3. 与 no-op、线性传播基线比较；
4. 对 input target 无有效 network support 时输出 blocked/insufficient，
   不做 zero-fill。

### 5.6 Geneformer/scGPT zero-shot

foundation model 可以提供 token 删除、embedding influence 或状态相似度，但
这些值通常没有稳定的表达 up/down 语义。使用时：

- 只报告 influence/embedding score、top-k 影响基因或状态距离；
- 若要导出方向，必须用同 context 的真实 perturbation calibration；
- 不能把内部 cosine/influence 谱填入 <code>predicted_delta</code>；
- 不能因为候选在词表中就声称模型理解了其 KO/KD 反应。

当前 Geneformer 资产的 manifest 已明确：它是
<code>network_counterfactual</code>，对 250 个 GSE174367 EX disease cells
生成 influence spectrum，没有 up/down expression direction。因此当前只能用于
ranking，不能直接产生 DAVF direction evidence。

### 5.7 PTM signed network、文献和自然扰动

这一级用于保持项目上游主线和机制解释：

<code>PTM site → activity → signed propagation → global gene score → AD observed intersection</code>

可以产出：

- source activity 到 target gene 的 signed path；
- 文献或自然变异支持；
- candidate proposal 的方向假设和机制理由。

不能产出：

- 真实 KO/KD 后的 expression delta；
- scVI latent pair；
- 独立 observed gate 的显著性；
- PerturbGen 两路径的效用 PASS。

PTM 侧在没有独立 null 前保持 <code>prediction_status=direction_only</code>，不得
事后写 PTM q 值或用 AD 结果反调 PTM 阈值。

## 6. 推荐主路线、备选路线与候选分流

### 6.1 推荐主路线

推荐采用“匹配程度优先、验证结果决定保留”的单主路线：

~~~mermaid
flowchart TD
    A[冻结 candidate / context / intervention] --> B[公共数据 inventory]
    B --> C{有同机制 target-specific 数据?}
    C -->|有| D[transfer: baseline + GEARS/CPA]
    C -->|无| E[GEARS unseen + GRN/CellOracle 条件分支]
    D --> F[target-held-out/context-held-out/time-dose 评估]
    E --> F
    F --> G{超过预注册基线且语义可比?}
    G -->|是| H[exploratory_supported sidecar]
    G -->|否| I[direction_only 或 blocked]
    H --> J[人工审阅，不进入 formal gate]
    J --> K{真实 target-specific KO/KD pair 到位?}
    K -->|是| L[scVI z0/z1 + LatentDAVF + held-out]
    K -->|否| M[保持 exploratory，等待数据]
    L --> N[三方 gate → PerturbGen]
~~~

主路线的关键决策：

1. **优先公共真实扰动 transfer**，因为它能直接提供 post-perturbation readout；
2. **GEARS/CPA 必须和基线一起评估**，不能只看模型输出；
3. **GRN/foundation 只做补充**，不能填补缺失的真实 latent pair；
4. **任何公共模型结果都不自动接入 E2E**，必须先过 context/anchor/语义审阅；
5. **observed gate 与预测路径并行解决**，其中任一侧未满足，formal 结果都停止。

### 6.2 备选路线

| 路线 | 适用情况 | 产物 | 代价与边界 |
|---|---|---|---|
| B1：同机制 transfer | 有相同 target 的 CRISPRi/KO 数据，但 context 不完全相同 | calibrated exploratory delta | 需 domain shift 报告，不能直接当 AD ground truth |
| B2：GEARS unseen | 无 target-specific 数据但有足够同 context 图谱 | GO/graph 外推 delta | 需 target-held-out；当前 K562 APOE anchor fail |
| B3：CPA context/dose transfer | 有剂量/时间和 covariate 设计 | latent response + reconstructed delta | 未见 target 的泛化必须单独证明 |
| B4：AD GRN | 有 AD-specific ATAC/TF network | network counterfactual | 对非 TF 候选只作补充 |
| B5：foundation ranking | 没有可用训练扰动，但有目标 context | influence/embedding ranking | 默认无方向语义 |
| B6：候选收缩 | 正式时间/数据受限 | 只保留 APOE KO exploratory/formal 前置 | 不再声称覆盖五候选或 KD |

### 6.3 五个候选的分流

| 候选 | 当前可用路线 | 立即动作 | formal 前置 |
|---|---|---|---|
| APOE | 当前候选 KO 轴；外部模型只做交叉检查 | 使用当前 route-specific KO asset 做工程方向验证；不宣称 biology PASS | observed gate 通过；KO held-out/独立 donor 证据；KD 若要使用必须补真实 KD |
| APP | 公共 target-held-out transfer；GEARS/GRN 条件探索 | 先查同物种、相近 cell context 的真实 KO/KD 数据 | 真实 APP KO/KD target pair + 对应 route 重训/验证 |
| PSEN1 | 同上；若有 AD neuron/CRISPRi 数据优先 | 禁止用当前 K562 unseen 输出直接代替 AD effect | 真实 PSEN1 target pair |
| BACE1 | 同上；若 GRN 无直接 support，输出 blocked 而非补零 | 先确认同机制公共数据和网络 support | 真实 BACE1 target pair |
| MAPT | 同上；优先 neuron context、剂量/时间可比数据 | 评估 cell type/state shift，不能用 K562 结果替代 | 真实 MAPT target pair |
| 五者 KD | 仅接受 CRISPRi/partial knockdown/dose data | 当前全部标记 KD unavailable；不从 KO 取反 | 每个 target 的真实 KD/CRISPRi pair |

### 6.4 当前仓库外部证据的处理

当前已有 GEARS/Geneformer 资产保留为可审计外部结果，但按以下状态处理：

- <code>gears_predictions.manifest.json</code>：保留为 K562/Norman
  <code>go_extrapolation</code>；
- <code>geneformer_isp_EX.manifest.json</code>：保留为 GSE174367 EX
  <code>network_counterfactual</code>；
- <code>apoe_anchor_backtest.json</code>：<code>verdict=fail</code>，两源不进入 lineage；
- 不为这批资产新增 formal consumer，不把失败 anchor 改成 warning；
- 若重新获得 context-matched public data，应使用新 run_id、独立 manifest 和
  预注册评估，不覆盖当前失败资产。

## 7. 数据、manifest 与 schema 契约

### 7.1 ID 与索引契约

所有方法都必须遵守：

1. **canonical Ensembl 是跨数据源合并键**；symbol 仅作显示或输入解析；
2. 外部模型 readout matrix 的列/行必须显式声明 gene namespace；
3. PerturbGen token index 只用于 token/context 输入；
4. scVI decoder index 只从 route adapter 的有序 <code>gene_names</code> 解析；
5. 不得用 token index 直接当 decoder index；
6. mapping 冲突、未映射 target、重复 canonical ID 或命名空间混合时硬失败。

### 7.2 真实 DAVF pair 契约

正式 LatentDAVF 训练的 NPZ/metadata 必须包含：

~~~json
{
  "schema_version": "ptm2cellnet.latent-davf-pairs.v1",
  "scvi": {
    "model_path": "/absolute/path/to/scvi",
    "latent_dim": 64,
    "num_genes": 4018,
    "gene_names": ["full ordered scVI decoder gene order"]
  },
  "embedding_asset": {
    "path": "/absolute/path/to/embedding_asset",
    "manifest": "exact registered embedding manifest"
  },
  "intervention": {
    "type": "KO",
    "targets": ["ENSG..."],
    "directions": [0]
  },
  "provenance": {
    "input_h5ad": "/absolute/path/to/real_perturbation.h5ad",
    "control_rows": ["row-id"],
    "perturbed_rows": ["row-id"],
    "target_column": "target",
    "donor_column": "donor",
    "train_donors": ["D1"],
    "held_out_donors": ["D2"]
  }
}
~~~

数组必须同时存在：

- <code>z_0[N,64]</code>：同 route、同 scVI model 的 control latent；
- <code>z_1[N,64]</code>：真实干预细胞 latent；
- <code>gene_ids[N,K]</code>：PerturbGen token rows；
- <code>directions[N,K]</code>：KO/KD/OE 编码；
- <code>attention_mask[N,K]</code>；
- donor、target、dataset 和 cell row provenance。

没有真实 target-specific <code>z_0/z_1</code>，不能生成正式 DAVF checkpoint。外部
模型的 predicted expression delta 不能重命名为 <code>z_1-z_0</code>。

### 7.3 Exploratory external evidence 契约

当前仓库已经有：

- input manifest：<code>ptm2cellnet.external-perturbation-prediction/v1</code>；
- payload：<code>ptm2cellnet.external-perturbation-evidence/v1</code>；
- 主环境校验：<code>src/integration/perturbgen/external_perturbation_evidence.py</code>；
- 组装入口：<code>scripts/assemble_external_evidence.py</code>。

每个新 external evidence asset 至少写入：

~~~json
{
  "schema_version": "ptm2cellnet.external-perturbation-evidence/v1",
  "candidate": {
    "gene_symbol": "APP",
    "ensembl_id": "ENSG00000142192"
  },
  "semantic_context": {
    "context": "public_dataset/cell_type/state",
    "intervention": "KO",
    "comparison_baseline": "matched_control",
    "reference_axis": "intervention_minus_control",
    "research_objective": "replication",
    "evidence_source": "GEARS",
    "cohort": "public_dataset_version"
  },
  "evidence_kind": "go_extrapolation",
  "perturbation_semantics": "unseen_perturbation_extrapolation",
  "model": {
    "name": "GEARS",
    "version": "registered version",
    "commit": "registered commit",
    "seed": 42
  },
  "prediction": {
    "readout_namespace": "canonical_ensembl",
    "predicted_delta": "path/to/delta_matrix",
    "predicted_direction": "path/to/direction_table",
    "uncertainty": "path or unavailable"
  },
  "validation": {
    "split": "target_held_out",
    "baseline": ["no_op", "matched_mean", "linear"],
    "status": "heldout_pass_or_fail_or_inconclusive",
    "metrics": {}
  },
  "lineage_boundary": "supplementary_only"
}
~~~

该结构是执行所需的最小字段；不代表当前仓库已经实现新的 consumer。若要让
external evidence 被正式 lineage 消费，必须先有研究批准、通过 anchor、完成
context/semantic binding，并新增独立回归验证。当前 anchor fail 时保持
<code>lineage_boundary=supplementary_only</code>。

### 7.4 donor/context provenance

manifest 必须逐项记录：

- 原始数据集、版本、下载/许可来源；
- input h5ad 和其 gene namespace；
- cell type、state、cohort、donor、batch；
- intervention mechanism、guide、dose、time；
- train/validation/test 的 target、cell、donor 和 context 列表；
- observed 队列的 pairing（<code>within_donor</code> 或
  <code>between_donor</code>）；
- scVI checkpoint、embedding asset、decoder gene order；
- model package/version、代码 commit、seed、训练参数；
- 输出 matrix、方向表、评估表和错误/缺失记录；
- 当前使用的 config、manifest 和 lineage 路径。

同一 donor、同一 target、同一 guide 或同一 context 不能通过不同命名再次进入
train 与 test。无法解析来源时，不是低置信度，而是 blocked。

## 8. 评估设计与可校准性

### 8.1 必须同时执行的拆分

| 拆分 | 训练集/测试集关系 | 发现的泄漏或泛化问题 |
|---|---|---|
| target-held-out | 测试 target 的所有细胞、guide、donor 不进入训练 | 未见 target 外推能力 |
| cell-context-held-out | 留出 cell type/state/cohort，测试 target 可在别处出现 | context/domain shift |
| donor-held-out | 同 target 可见，但测试 donor 与训练 donor 不重叠 | donor memorization 和个体差异 |
| time-held-out | 训练时间点与测试时间点分离 | 时间外推，不把插值当 OOD |
| dose-held-out | 训练剂量与测试剂量分离 | 剂量响应外推 |
| target × context-held-out | target 与 context 同时留出 | 最接近当前 AD 候选场景，但通常最难 |

任何结果必须声明使用了哪些 split。只随机切 cell 不能证明 target 或 donor
泛化。

### 8.2 基线

所有模型必须与以下基线比较：

1. **no-op**：<code>predicted_delta = 0</code>；
2. **matched mean**：同 context、同 intervention class、训练集真实 delta
   的均值或中位数；
3. **linear**：冻结 signed GRN/共表达图上的正则线性传播；
4. 对有 time/dose 的数据，增加 context/time/dose 分层均值；
5. 对 unseen target，报告 target frequency 和 graph degree，避免把低覆盖目标
   与模型能力混淆。

若复杂模型不能稳定超过 no-op/mean/linear，结果只能为
<code>exploratory_unvalidated</code> 或 <code>blocked</code>，不能因为模型复杂
或图谱更大而保留。

### 8.3 指标

至少输出以下指标，并按 target、context、readout gene 和 top-k 分层：

- **direction accuracy**：预测方向与真实 delta 符号一致率；neutral 单独统计；
- **top-k overlap**：预测与真实 DE/readout top-k 的 precision、recall、Jaccard；
- **Spearman/Pearson**：预测 delta 与真实 delta 的 rank/linear correlation；
- **DE overlap**：以测试集预注册 DE 方法和阈值生成的集合重叠；
- **magnitude error**：MAE/RMSE 或相对 identity baseline 的误差；
- **calibration**：Brier/ECE、预测区间覆盖率、置信度分箱准确率；
- **OOD gap**：in-domain 与 target/context/time/dose-held-out 指标差；
- **baseline lift**：相对 no-op、matched mean、linear 的预注册提升。

阈值必须在看到候选测试结果前冻结。当前 APOE anchor 的 0.59 top-100 阈值只
属于该次预注册回测，不能未经重新注册直接移植到新的 context、模型或指标。

### 8.4 “相对准确”的定义

本方案不使用未经验证的准确率承诺。只有同时满足以下条件，才可写
<code>exploratory_supported</code>：

1. 测试 split 与目标问题相符；
2. 至少一个主要指标超过预先登记的 no-op/mean/linear 基线；
3. 方向和 top-k 结果不是由单个 donor/seed 驱动；
4. context/intervention semantics 没有混写；
5. 不确定性或失败范围已记录；
6. 目标候选没有被训练数据或映射表泄漏。

这仍然不等于 formal DAVF。没有 target-held-out 或可比 ground truth 时，只能为
<code>direction_only</code>。

### 8.5 当前 APOE anchor 的处理

当前 <code>outputs/external_evidence/apoe_anchor_backtest.json</code> 的结果：

- GEARS top-100 sign agreement = 0.25，预注册阈值 = 0.59，fail；
- all-shared Spearman = -0.0428，fail；
- 相对 mean-shift baseline 的比较为 pass，但不能挽救前两项；
- 总体 verdict = fail。

lessons 进一步显示 B2M、CD59、CTSD 锚点和 mean-shift baseline 也出现相似反向
模式，指向 Frangieh co-culture 与 Norman K562 单培养的 context/组成不可比。
因此后续必须使用 context-matched anchor 或重新定义可比性，不能只重新跑同一资产
并期待状态改变。

## 9. 端到端阶段计划

下表是实际执行顺序。每一阶段列出的“失败即停”是该阶段的局部停止，不允许用
下一阶段的模型输出覆盖前一阶段的失败。

### 阶段 0：研究语义和候选冻结

**预计**：0.5 工作日

**输入**

- 五个候选及 canonical Ensembl；
- PTM activity 主线 config；
- 研究目标（association/replication/reversal）；
- KO/KD route、context、reference axis。

**处理**

1. 为每个候选建立 candidate row；
2. 明确 proposal direction、observed direction、预测 direction 的来源；
3. 分别声明 source_intervention 和 within_state；
4. 将 KD 缺失写成 unavailable，不用 KO 推导；
5. 预注册 primary metric、baseline、split 和 anchor 判据。

**产物**

- <code>candidates.tsv</code>；
- <code>semantic_context.json</code>；
- <code>prediction_evaluation_plan.json</code>；
- 研究负责人批准记录。

**失败即停**

- candidate 无 canonical Ensembl；
- intervention、baseline、reference axis 或 objective 缺失；
- KO/KD 机制来源不明；
- 用 observed disease-minus-normal 充当干预标签。

**验收**

- 五个候选均能逐字段解释“谁在什么 context 下做什么干预、与什么 baseline 比”；
- 不同证据源没有共用一个方向字段。

### 阶段 1：observed gate 复核与研究决策分支

**预计**：1.5 工作日

**输入**

- d1/d2 DEG manifest 和 aggregate/donor-level 表；
- raw counts、donor provenance、pairing；
- 冻结的 <code>deg_max_fdr=0.05</code>。

**处理**

1. 复核 donor provenance 和 between_donor 不相交约束；
2. 复核 pseudobulk、Welch、BH 及 gene universe；
3. 分别报告 GSE174367、GSE157827、naive pooled、cohort-centered；
4. 计算 donor 数、文库深度范围、最小 FDR 和方向稳定性；
5. 由研究负责人选择唯一分支：
   - 增加独立 donor/队列；
   - 预注册独立小 gene panel；
   - 修订 formal gate 语义。

**产物**

- observed gate audit；
- <code>observed_gate_decision.json</code>；
- 若采用新队列，新的 cohort manifest；
- 若修订语义，修订批准记录。

**失败即停**

- donor 由 sample/batch/replicate 猜测；
- 把细胞当独立样本；
- 事后修改 FDR 或 universe；
- 发现 donor 标签不可靠但仍继续候选准入。

**验收**

- 当前两队列无显著行的事实可重现；
- 研究负责人明确下一分支；
- 在决策完成前，所有候选保持 observed gate inconclusive。

**2026-09-18 状态**：分支 3 已冻结为
`observed_admission_rule=signed_direction_without_fdr_cutoff`；报告 FDR 仍为
0.05。见 `src/analysis/ad_research_decision.py`。

### 阶段 2：公共扰动资产盘点

**2026-09-18 状态**：**取消**。不再检索公共 Perturb-seq；B1/B3 不启用。

**预计**：2.0 工作日，不含许可和大文件下载等待

**输入**

- 公共 Perturb-seq/CRISPRi 目录；
- GEARS/CPA 可用训练数据；
- 候选、Ensembl mapping、目标 context 要求。

**处理**

1. 逐数据集登记物种、cell type、state、donor、batch、time、dose；
2. 逐目标审计真实 perturbation label，而不是只看 vocabulary；
3. 逐记录 KO、KD、CRISPRi、OE 的原始语义；
4. 检查 control 与 perturbed 配对；
5. 生成 context compatibility matrix；
6. 剔除损坏、截断、缺 provenance 或不可区分机制的数据。

**产物**

- <code>public_perturbation_inventory.tsv</code>；
- 每数据集 manifest；
- <code>context_compatibility.json</code>；
- target × route coverage table。

**失败即停**

- 只有静态 token 没有真实扰动；
- target 机制未知；
- target、donor 或 context 无法追溯；
- mapping 产生冲突且没有人工解析。

**验收**

- 每个候选都有明确状态：同 context、有迁移数据、仅 unseen 图外推或无可用输入；
- KD 只在真实 CRISPRi/partial knockdown/dose 语义下记为 KD coverage。

### 阶段 3：canonical mapping 与 route 对齐

**预计**：1.0 工作日

**输入**

- inventory；
- <code>data/processed/gene_symbol_ensembl_map.tsv</code>；
- PerturbGen token dictionary；
- route scVI adapter gene order。

**处理**

1. symbol→canonical Ensembl 映射；
2. target、readout、token 和 decoder 四套空间分别审计；
3. 记录 unmapped、duplicate、ambiguous 和 dropped；
4. 对每个候选执行 route × target × intervention coverage audit；
5. 不生成任何 zero-fill perturbation target。

**产物**

- <code>canonical_target_map.tsv</code>；
- <code>route_coverage_audit.tsv</code>；
- mapping manifest。

**失败即停**

- target 只能通过别名猜测；
- token index 和 decoder index 被混用；
- candidate 在词表里但没有真实 perturbation 时被标为训练覆盖。

**验收**

- 所有 join key 为 canonical Ensembl；
- 每个模型输出都有明确 readout namespace；
- 轴审计可复现当前 KO={APOE}、KD=empty 的候选事实。

### 阶段 4：拆分、基线和评估冻结

**预计**：1.0 工作日

**输入**

- 阶段 2/3 manifest；
- 目标 context 与候选列表；
- 计算资源和 seed。

**处理**

1. 冻结 target-held-out；
2. 对 context transfer 冻结 cell-context-held-out；
3. 对 CPA/剂量数据冻结 time-held-out、dose-held-out；
4. 加入 donor-held-out；
5. 生成 no-op、matched mean、linear baseline；
6. 预注册指标、阈值、置信区间和失败语义。

**产物**

- split manifest；
- baseline prediction tables；
- evaluation config；
- seed list。

**失败即停**

- split 只按 cell 随机切；
- test target 在训练中以 guide、alias 或 proxy 出现；
- 阈值在查看 candidate test 结果后调整。

**验收**

- split manifest 可审计到 donor/target/context；
- 所有复杂模型都有同一批 baseline 和指标。

### 阶段 5：公共 transfer/GEARS/CPA 运行

**预计**：3.0 工作日，GPU wall time 单独记录

**输入**

- 通过阶段 4 的训练集和 split；
- GEARS/CPA 版本、配置、graph、seed；
- 外部独立环境和已登记权重。

**处理**

1. 先运行 no-op/mean/linear；
2. 对有真实 target 数据的候选训练 context-matched transfer；
3. 对 unseen target 运行 GEARS；
4. 对有 time/dose/covariate 的数据运行 CPA；
5. 保存每个 seed 的预测、delta、方向和不确定性；
6. 运行固定的 target/context/time/dose held-out 评估。

**产物**

- predictions h5ad/矩阵；
- <code>external-perturbation-prediction/v1</code> manifest；
- evidence payload；
- metrics、baseline comparison 和失败记录。

**失败即停**

- 模型实际 context 与 manifest 不一致；
- unseen query 未进入模型支持图；
- 输出只有 embedding 却写成表达方向；
- 复杂模型不超过预注册 baseline 且没有解释；
- 运行结果缺 seed、版本或输入 manifest 绑定。

**验收**

- 每个保留的结果都有 target-held-out 或明确的 not-evaluated 标记；
- 未通过的模型不进入融合为“支持证据”。

### 阶段 6：AD-specific GRN/CellOracle 条件分支

**预计**：2.0 工作日

**输入**

- AD cell type/state scRNA；
- scATAC/motif 或已登记 TF-target prior；
- signed network release；
- 阶段 4 的 baseline/evaluation config。

**处理**

1. 建立并冻结 base GRN；
2. 在每个 cell type/state 做 context-specific inference；
3. 只对有 TF/regulatory support 的 target 做传播；
4. 与 no-op/linear baseline 比较；
5. 输出网络边、路径、符号、置信度和 sensitivity。

**产物**

- GRN manifest；
- network counterfactual matrix；
- path/effect explanation；
- target applicability report。

**失败即停**

- 非 TF target 没有可解释调控路径；
- network release、sign 或 cell context 不明；
- 用 network delta 伪造 scVI latent pair。

**验收**

- 输出明确标记为 <code>network_counterfactual</code>；
- 对不适用候选返回 blocked/insufficient，而不是填零。

### 阶段 7：foundation model 影响排序

**预计**：1.5 工作日

**输入**

- 已冻结的 AD context；
- Geneformer/scGPT 版本、词表和权重；
- 目标候选。

**处理**

1. 运行 deletion/inhibition/overexpression 仅限模型支持的语义；
2. 保存 influence/embedding score；
3. 不强制转换为 up/down；
4. 如需方向，使用独立真实 perturbation calibration；
5. 与 GEARS/GRN 只做 field-separated comparison。

**产物**

- influence matrix；
- model manifest；
- ranking report；
- direction availability report。

**失败即停**

- 模型词表命中但输出语义仍是 embedding；
- context 不是 manifest 声明的 AD context；
- 试图把内部 score 直接作为 DAVF direction。

**验收**

- Geneformer/scGPT 结果最多是 ranking/direction_only；
- 不改变当前 formal candidate 或 invocation。

### 阶段 8：证据融合与 exploratory ranking

**预计**：1.0 工作日

**输入**

- GEARS/CPA、GRN、foundation、PTM、observed 结果；
- 各自 validation status 和 context compatibility。

**处理**

1. 每种来源保留独立字段；
2. 先按验证状态过滤，再按 context 相似度排序；
3. 生成可解释的 exploratory confidence；
4. 记录一致、冲突、缺失和无法比较；
5. 不把模型一致性当作三方独立证据。

**建议的探索性分数**

可使用非正式排序分数，但必须在 payload 中声明它不是 formal gate：

~~~text
exploratory_score =
    validation_calibration
  × context_similarity
  × model_agreement
  × network_support
~~~

任何因子 unavailable 时不填默认高值；输出 unavailable 并降低到
<code>direction_only</code>。不同模型共享训练数据或共享 graph 时，不能把它们
计数为独立证据。

**产物**

- <code>exploratory_candidate_ranking.tsv</code>；
- field-separated evidence report；
- conflict/uncertainty report；
- supplementary lineage（若当前入口允许；否则只保留 sidecar）。

**失败即停**

- observed、DAVF、外部模型方向被合并为一个 ground truth；
- anchor fail 的资产被纳入 formal lineage；
- 缺失证据被默认成 0、neutral 或高置信度。

**验收**

- 每个候选能回答：方向来自哪里、在哪个 context、什么干预、验证到什么程度；
- ranking 不改变三方 gate 的 pass/fail。

### 阶段 9：候选路由与报告

**预计**：1.0 工作日

**输入**

- exploratory ranking；
- observed gate decision；
- route coverage audit；
- 当前 E2E 和 bridge contract。

**处理**

1. APOE 分流到已有 KO route 或等待 observed gate；
2. APP/PSEN1/BACE1/MAPT 保留探索性证据，不创建伪 DAVF axis；
3. 五候选 KD 均标记 unavailable，除非新增真实 KD 输入；
4. 只有 formal gate pass 的候选才生成 PerturbGen invocation；
5. 报告中分开列出 predicted/observed/DAVF/target-set 字段。

**产物**

- candidate routing report；
- formal/exploratory 分离清单；
- blocked items；
- 下一轮数据需求表。

**失败即停**

- 外部模型结果被直接送入 E2E gate；
- candidate spec 的 intervention 与 route 不一致；
- 无 observed FDR 仍创建 formal invocation。

**验收**

- formal 和 exploratory 目录、manifest、报告相互隔离；
- 当前无显著 observed 行时 formal candidate 数保持 0。

### 阶段 10：真实 target 数据到位后的正式 DAVF/E2E

**预计**：人工准备 2.0 工作日；P40/GPU 执行约 3.0–5.0 GPU 工作日

**输入**

- 每个 target 的真实 KO/KD perturbation；
- 同 route scVI model；
- real <code>z0/z1</code> pair；
- train/held-out donor 和 target split；
- frozen embedding asset；
- observed gate pass 的 candidate spec。

**处理**

1. 用真实 perturbation H5AD 生成 pair；
2. 运行 pair schema、gene axis、token/decoder、donor provenance 校验；
3. 训练 route-specific LatentDAVF；
4. 运行 target-held-out/donor-held-out endpoint；
5. 生成 DAVF direction evidence；
6. 运行现有 E2E：三方 gate → shared prepare →
   <code>perturb → export_gene_embeddings → report</code>；
7. 组装 matched-null、质量、candidate p/q、BH-FDR 和 dual-path AND。

**失败即停**

- pair 没有真实 z1；
- train/held-out donor 泄漏；
- KO/KD route、checkpoint、scVI 或 embedding manifest 不一致；
- observed gate 未通过；
- 任一 formal statistical input 缺失。

**验收**

- 所有 formal 资产可从 manifest 追溯到真实输入；
- formal biology acceptance checklist 全部满足。

## 10. 当前仓库的最短落地路径

### 10.1 现在立即做的最小工作集

工程分流已由 `scripts/route_ad_candidates.py` 编码；不修改现有 checkpoint，
也不覆盖当前 fail 的 GEARS/Geneformer 资产。执行顺序：

1. 复用
   outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv
   和 d1/d2 DEG 表，不重新发明覆盖审计。
2. 运行 `python scripts/route_ad_candidates.py --axis-audit-tsv ... --deg-table ...
   --anchor-backtest ... --output-dir ...`，将当前五候选冻结为 canonical Ensembl
   表，明确 APOE=KO engineering、五者 KD=unavailable、formal invocation=0。
3. 对新的公共数据源先做 inventory 和 target/context/route 覆盖审计，再把 TSV
   交给 `--public-inventory-tsv`；无 inventory 时不得把词表命中写成训练覆盖。
4. 继续使用已有
   scripts/run_gears_predictions.py、
   scripts/run_geneformer_isp.py 和
   scripts/apoe_anchor_backtest.py
   作为外部环境 producer/diagnostic，但不得覆盖当前 fail 资产或跳过 anchor。
5. GEARS 只在有明确 target-held-out benchmark 和 context compatibility 记录后
   进入 exploratory ranking；Geneformer 当前只进入 influence ranking。
6. 需要主环境消费时，沿用
   src/integration/perturbgen/external_perturbation_evidence.py
   和现有 external evidence schema；payload 现强制
   `lineage_boundary=supplementary_only`。
7. 不运行正式 `--run-perturbgen` 来“试试看”。observed gate 或
   DAVF axis 未通过时，正确动作是 blocked/inconclusive，不是生成 formal report。

### 10.2 当前五候选的最短决策

| 候选 | 最短路线 | 当前结论 |
|---|---|---|
| APOE | 复核当前真实 KO/DAVF 方向；单独等待 observed gate；若需要 KD，补真实 KD | 可做 route 工程验证，不能绕过 observed gate |
| APP | 查找 AD neuron/相关细胞公共真实 KO/KD；无则 GEARS + GRN exploratory | 不创建本地 DAVF 轴 |
| PSEN1 | 优先 AD/neuron context 的真实 perturbation；再做 target-held-out transfer | 不使用当前 K562 输出作正式方向 |
| BACE1 | 同上；无 GRN support 就保持 insufficient | 不 zero-fill |
| MAPT | 优先 neuron context 和 time/dose 相近的 CRISPRi/KD | 不把模型内部 influence 当方向 |

### 10.3 当前不应投入的工作

- 不重跑当前 K562 GEARS/Geneformer 资产以期待 APOE anchor 自动通过；
- 不为五候选创建虚拟 KD target；
- 不把现有 KD checkpoint 的其他 target 改名成候选；
- 不把当前 d1/d2 的 exploratory DEG 行变成 formal candidate；
- 不为 external evidence 先造 lineage consumer，再寻找通过证据。

## 11. 失败、阻塞与禁止降级语义

### 11.1 状态定义

| 状态 | 触发条件 | 下一步 |
|---|---|---|
| <code>blocked</code> | 必需真实数据、manifest、mapping、route 或权限不存在 | 补输入或由负责人改变研究设计；不继续下游 |
| <code>failed</code> | 已执行检查且预注册判据失败，例如 anchor fail | 保留失败产物，重新选择可比数据/方法；不改判据 |
| <code>inconclusive</code> | 证据存在但 observed FDR、方向或校准不足 | 只能 exploratory/等待更强证据 |
| <code>direction_only</code> | 只有 PTM、文献、网络或无校准模型方向 | 只进入假设和排序 |
| <code>exploratory_supported</code> | 通过预注册 held-out/baseline/context 检查 | 可用于候选优先级，不进入 formal gate |
| <code>formal_pass</code> | 完成所有项目 biology acceptance 条件 | 才能进入正式报告/效用解释 |

缺失数据不是“模型预测为 0”；不适用不是 neutral；不可比不是 low confidence。
这些情况必须原样记录并停止依赖它们的下游阶段。

### 11.2 硬禁止项

以下任一项发生时，必须 hard fail 或 blocked：

- 把 <code>observed_direction=disease_minus_normal</code> 当作
  <code>KO/KD intervention effect</code>；
- 将 token coverage 写成 training-axis coverage；
- 用 KO checkpoint 推断 KD，或把 KO 方向全局取反；
- 用 zero-fill、随机 token、随机 latent 或 gene-space delta 生成 z1；
- observed gate 无 FDR≤0.05 行仍创建 formal invocation；
- 事后放宽 FDR、替换 raw p、缩小 gene universe；
- anchor fail 的 GEARS/Geneformer 资产进入 formal lineage；
- Geneformer/scGPT influence score 没有 calibration 却被写成 expression direction；
- 单路 source 或 within-state 结果外推为双路径治疗效用；
- 用 synthetic/mock/smoke/bridge 结果代替真实 biology acceptance。

## 12. 资源、工期与风险

### 12.1 工作日估计

| 阶段 | 人工工期 | 主要资源 |
|---|---:|---|
| 0 语义和候选冻结 | 0.5 天 | 研究负责人、数据负责人 |
| 1 observed gate 复核 | 1.5 天 | 统计/AD 分析人员 |
| 2 公共资产盘点 | 2.0 天 | 数据检索、许可和生物信息人员 |
| 3 mapping/route 对齐 | 1.0 天 | 项目主环境、Ensembl/token/scVI 维护者 |
| 4 split/baseline 冻结 | 1.0 天 | 统计人员、外部模型负责人 |
| 5 GEARS/CPA transfer | 3.0 天 | 独立 GPU 环境、模型维护者 |
| 6 GRN/CellOracle | 2.0 天 | scATAC/TF-GRN 专长 |
| 7 foundation ranking | 1.5 天 | 外部模型环境、GPU |
| 8 融合与 exploratory report | 1.0 天 | 主线维护者、研究负责人 |
| 9 路由与报告收口 | 1.0 天 | 主线维护者 |
| **探索性路线合计** | **13.5 天** | 不含许可等待和大文件下载 |
| 10 正式 DAVF/E2E 人工准备 | 2.0 天 | 数据、统计、GPU 维护者 |
| 正式 DAVF/E2E GPU 执行 | 3.0–5.0 GPU 天 | P40/eager、独立 PerturbGen 环境 |

如果公共数据缺失或许可未到位，阶段 2 直接 blocked；不能把“等待数据”折算成
模型成功。若研究负责人选择只收缩到 APOE KO，可跳过 APP/PSEN1/BACE1/MAPT
的 exploratory training，但不能跳过 observed gate 和正式验收条件。

### 12.2 资源要求

- 统计/AD：donor-level 设计、DEG、BH、队列和混杂审计；
- 单细胞扰动：Perturb-seq、CRISPR KO/CRISPRi、dose/time 语义；
- GRN/多组学：scATAC、motif、TF-target 和 signed network；
- 工程：canonical Ensembl、scVI adapter、PerturbGen token/decoder 区分；
- 外部环境：GEARS/CPA/Geneformer/scGPT 的独立 Python 环境；
- GPU：当前 P40 按 bridge guide 使用 eager 路径，formal wall time 单独记录；
- 研究负责人：批准 observed gate 语义、探索性阈值和 formal/exploratory 边界。

### 12.3 主要风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 公共 context 与 AD 不可比 | 方向反转或组成效应 | context matrix、anchor、matched baseline；不通过则 supplementary |
| target-held-out 无法通过 | unseen 预测不可靠 | 收缩到可验证 target 或获取真实扰动 |
| KO/KD 机制混淆 | 方向语义错误 | manifest 硬记录 intervention；未知机制 blocked |
| donor 数不足 | observed gate 不可达 | 增加 donor/预注册 panel/批准语义修订 |
| foundation model 无方向 | 误读 embedding | 只做 influence ranking |
| P40/外部 API 适配失败 | 训练中断 | 外部环境隔离、记录版本；不修改主线契约 |
| 多模型共享数据 | 虚假“独立证据” | evidence source 分字段，不做简单票数相加 |
| manifest 漂移 | 结果不可复现 | 新 run_id、输入/模型版本绑定、失败即停 |

## 13. Formal biology acceptance checklist

以下项目全部满足，才允许写正式 biology acceptance；任何一项缺失，状态必须是
blocked、failed 或 inconclusive。

### 13.1 上游 PTM/activity

- [ ] 有真实 PTM 全局数据，来源、版本、许可和 manifest 可追溯；
- [ ] KSTAR/PhosR activity 输入已冻结，方向字段与 PTM site 分离；
- [ ] signed network release、符号、confidence、path cap 已冻结；
- [ ] PTM q 值没有在看到 AD 结果后反调；
- [ ] source 与 target gene 没有混写。

### 13.2 observed gate

- [ ] 有真实 normal/disease raw counts；
- [ ] canonical Ensembl gene namespace；
- [ ] 显式 donor provenance；
- [ ] pairing 明确为 within_donor 或 between_donor；
- [ ] 每个目标 cell type 满足项目规定的 donor 数；
- [ ] donor-level estimand、gene universe、Welch/BH 和 FDR 预注册；
- [ ] formal candidate 的 observed row 满足冻结的 FDR≤0.05；
- [ ] 结果不是由细胞伪重复或事后阈值产生。

### 13.3 DAVF

- [ ] 每个 candidate route 有真实 target-specific KO 或 KD perturbation；
- [ ] KO/KD checkpoint、scVI、embedding asset 和 route 语义一致；
- [ ] 有同 scVI model 生成的真实 z0/z1 pair；
- [ ] latent_dim、num_genes、scVI gene order 完全匹配；
- [ ] token index 与 decoder index 分离并逐值核对；
- [ ] train/held-out donor 和 target split 无泄漏；
- [ ] held-out direction、top-k、Spearman/Pearson 和 baseline 结果已报告；
- [ ] 没有用 disease-minus-normal 代替 intervention effect。

### 13.4 三方 gate 和 E2E

- [ ] PTM proposal、DAVF direction、observed direction 三方字段独立；
- [ ] source 三方 gate pass 后才生成 invocation；
- [ ] source_intervention 和 within_state 均按同一 formal candidate 执行；
- [ ] PerturbGen 六阶段顺序正确，shared prepare artifact 可追溯；
- [ ] matched-null、未扰动质量、candidate empirical p/q 和 BH-FDR 已组装；
- [ ] dual-path AND 满足项目规定；
- [ ] downstream target concordance 与 source gate 分字段；
- [ ] external evidence 若存在，已过 context-matched anchor，且没有冒充独立
  三方证据。

### 13.5 发布边界

- [ ] 所有输入、模型、代码版本、seed、manifest 和输出可复现；
- [ ] synthetic/mock/smoke/bridge 结果未被写作 biology PASS；
- [ ] 单路效用未外推为普遍治疗结论；
- [ ] 失败和 blocked 项目仍保留在报告中，没有静默删除；
- [ ] formal biology PASS 由完整 checklist 支持，而不是由模型一致性分数支持。

## 14. 参考资料与本地证据

### 14.1 方法原始论文

- GEARS：Roohani, Huang, Leskovec. *Predicting transcriptional outcomes of
  novel multigene perturbations with GEARS*. Nature Biotechnology.
  [原始论文](https://www.nature.com/articles/s41587-023-01905-6)
- CPA：Lotfollahi et al. *Predicting cellular responses to complex perturbations
  in high-throughput screens*. Molecular Systems Biology.
  [原始论文与 DOI](https://doi.org/10.15252/msb.202211517)
- CellOracle：Kamimoto et al. *Dissecting cell identity via network inference
  and in silico gene perturbation*. Nature.
  [原始论文](https://www.nature.com/articles/s41586-022-05688-9)
- 扰动预测基准参考：*Systema: a framework for evaluating genetic perturbation
  response prediction beyond systematic variation*.
  [方法评估论文](https://doi.org/10.1038/s41587-025-02777-8)

### 14.2 当前项目事实源

- 当前状态与正式 PASS 边界：
  [docs/CURRENT_STATUS.md](../CURRENT_STATUS.md)
- 2026-09-17 综合审计：
  [project_analysis_20260917.md](../../project_analysis_20260917.md)
- 研究决策与 lessons：
  [lessons.md](../../lessons.md)，重点为 L-2026-0915-02、
  L-2026-0916-01、L-2026-0916-02、L-2026-0917-04
- KO/KD DAVF 训练与 pair 契约：
  [davf_ko_kd_training.md](davf_ko_kd_training.md)
- DAVF/PerturbGen bridge、external evidence 和 Gate-0/5：
  [perturbgen_bridge.md](perturbgen_bridge.md)
- PTM activity → AD → candidate 管线：
  [ptm_activity_pipeline.md](ptm_activity_pipeline.md)
- E2E gate、shared prepare、统计组装和双路径：
  [davf_perturbgen_e2e.md](davf_perturbgen_e2e.md)

### 14.3 当前运行产物

- observed d1：
  [ad_deg_manifest.json](../../outputs/ptm_activity/20260916_d1/ad_deg_manifest.json)
- observed d2 centered：
  [ad_deg_centered_manifest.json](../../outputs/ptm_activity/20260916_d2/ad_deg_centered_manifest.json)
- 候选轴审计：
  [davf_axis_coverage_audit.tsv](../../outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv)
- 轴审计 manifest：
  [davf_axis_coverage_audit.manifest.json](../../outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.manifest.json)
- 当前 GEARS 证据：
  [gears_evidence.json](../../outputs/external_evidence/gears_evidence.json)
- 当前 Geneformer 证据：
  [geneformer_evidence.json](../../outputs/external_evidence/geneformer_evidence.json)
- APOE 预注册回测：
  [apoe_anchor_backtest.json](../../outputs/external_evidence/apoe_anchor_backtest.json)

### 14.4 当前代码入口

- donor-level DEG：
  [src/analysis/ad_deg_table.py](../../src/analysis/ad_deg_table.py)
- DAVF/scPerturb pair：
  [src/data/davf_scperturb.py](../../src/data/davf_scperturb.py)
- LatentDAVF 训练：
  [scripts/train_latent_davf.py](../../scripts/train_latent_davf.py)
- 三方 direction gate：
  [src/integration/perturbgen/direction_gate.py](../../src/integration/perturbgen/direction_gate.py)
- 外部 evidence 校验：
  [src/integration/perturbgen/external_perturbation_evidence.py](../../src/integration/perturbgen/external_perturbation_evidence.py)
- 五候选分流：
  [src/analysis/ad_candidate_routing.py](../../src/analysis/ad_candidate_routing.py)、
  [scripts/route_ad_candidates.py](../../scripts/route_ad_candidates.py)

本方案规定在真实数据补齐前，外部模型结果如何被审计、评估、隔离和报告；真正的
formal route 仍以现有项目契约为准。分流 CLI 只编码第 6 章的路线选择，不替代
三方 gate 或 LatentDAVF。
