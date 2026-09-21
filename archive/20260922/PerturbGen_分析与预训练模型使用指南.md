# PerturbGen 分析与预训练模型使用指南

> 对话整理日期：2026年8月21日；当前契约复审：2026年9月13日
> 主题：疾病相关基因的双场景 PerturbGen 分析、官方资源及预训练模型使用方法

## 2026-09-13 当前项目执行契约

项目主线是外部 `PTM proposal/candidate_spec` → scVI/PTM 映射 → DAVF
方向与置信证据 → 独立 donor-level 表达方向三方 gate → evidence/invocation。
classifier 只预测 site presence；候选方向来自外部假设或逐 site override，不能把
rawsite 分类器写成自动因果表达方向推断。

每个候选先写清 `context`、`intervention`、比较基准和研究目标
（`association`、`replication` 或 `reversal`），再解释方向。观测 donor-level
disease−normal、DAVF 的干预后 decode−当前 context decode、以及 PerturbGen 的
rescue/效用属于不同证据来源；不使用全局同号或取反规则，不把病程签名或
normal/disease 对比当作 KO/KD 干预效应 ground truth，也不把同号写成三个统计独立
证据或治疗/临床因果结论。

`source_intervention=[src]` 是状态转移前场景，`within_state=[tgt]+pert_tps` 是
目标状态内场景。正式双路径判定保留 AND；单场景结果只能称对应场景的探索或计算
效用。六阶段为 `tokenise → train_mask → train_decoder → perturb →
export_gene_embeddings → report`。当前编排器仍按候选重复准备和两场景运行；固定
cohort、词表、训练配置和资产版本后公共 prepare 一次、候选只做两场景 perturb/效用
是待落实目标。

扰动模式遵循当前 route-specific `dual_path` 规则：对 up 方向，KO route 可用
`mask`、`pad`、`delete`，KD route 只用 `mask`；对 down 方向使用 `overexpress`。这些是各路径的
允许模式与敏感性分析，不把所有模式一律合并为 AND；正式 AND 只表示 source 与
within-state 两个场景均按各自规则通过。

Workflow A 是 gate 后的 PerturbGen 效用评估；Workflow B 是固定基础 encoder → 冻结
embedding asset → LatentDAVF 重训 → Gate-E 的独立资产生命周期。基础 export 不消费
候选结果、不训练本次 candidate checkpoint、不回灌本次 DAVF。正式 CLI 选择
`perturb`/`--path` 时需要绑定通过的 E2E gate report；invocation 拒绝 non-pass，
而低层 runner 当前只执行 StagePlan。正式资产合并使用 canonical Ensembl ID，
PerturbGen token index 与 scVI decoder index 不得混用。

本节执行依据见 [CURRENT_STATUS](docs/CURRENT_STATUS.md)、[中央双路径方案](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md) 和 [task_plan](task_plan.md)。

---

## 一、疾病状态下过表达基因的两套 KO 分析

### 用户问题

对于在疾病状态下过表达的基因，同时做两套分析：

1. `normal → disease，source KO`：探索状态转移前的模型干预效用；
2. `disease within-state KO`：探索已形成疾病状态内的模型干预效用。

是否也可以说明该基因比较值得进入实验验证？

### 回答

可以把两套结果作为候选排序的计算证据。**在明确参考轴、排除目标基因且两个场景都达到正式门槛时，该候选可进入实验验证清单。**只有一个场景有效时仍可作为相应场景的探索信号，但不能进入正式双路径清单；任何结果都不能单独证明致病驱动、治疗因果或临床疗效。

PerturbGen 官方设计中，`src` 扰动用于评估早期干预如何影响后续状态，`tgt` 扰动用于状态内扰动分析；KO 通常推荐使用 `mask` 模式，也可用 `pad` 和 `delete` 做敏感性分析。

预印本：

https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1

---

### 1. normal → disease，source KO

比较：

\[
\widehat X_{\text{disease}}^{\text{source-KO}}
\quad \text{vs.} \quad
\widehat X_{\text{disease}}^{\text{control}}
\]

如果 source KO 后，预测的疾病终点表现为：

- 疾病上调程序减弱；
- 正常功能程序恢复；
- 整体表达状态更接近正常；
- 目标细胞的病理亚状态比例下降；

在 `reversal` 研究目标下，这支持以下模型解释：

> 在当前 source/context/target 定义和模型假设下，该基因的早期抑制预测可能减弱状态转移中的疾病程序；这不是实测预防效果。

PerturbGen 的核心用途之一正是模拟 source 状态的扰动如何传播到后续 target 状态。

---

### 2. disease within-state KO

比较：

\[
\widehat X_{\text{disease+KO}}
\quad \text{vs.} \quad
\widehat X_{\text{disease control}}
\]

如果在疾病状态形成后再 KO，仍然可以：

- 降低疾病 signature；
- 恢复正常功能 signature；
- 逆转疾病 DEG；
- 使疾病细胞向正常参考状态移动；

在 `reversal` 研究目标下，这支持以下模型解释：

> 在当前目标状态和比较基准下，该基因的状态内抑制预测可能减弱已形成的疾病程序；这不是治疗因果或临床疗效证据。

官方文档将 `tgt` 扰动定义为直接编辑目标状态，适合 within-state perturbation 分析：

https://perturbgen.cog.sanger.ac.uk/docs/examples/06_perturbation.html

---

### 3. 两套结果如何组合解释？

| Source 场景 | Disease within-state 场景 | 研究解释边界 |
|---|---|---|
| 有效 | 有效 | 满足全部正式门槛时进入双场景候选清单；仍只是模型效用/方向筛选 |
| 有效 | 无效 | 仅支持 source 场景探索；不能外推到疾病状态内治疗效用 |
| 无效 | 有效 | 仅支持 within-state 场景探索；不能外推到早期状态转移效用 |
| 无效 | 无效 | 当前输入和模型下未得到场景化效用证据 |
| 方向不一致或参考轴不明 | 任一（诊断项，非准入条件） | 先检查 context、intervention、比较基准、来源和模型稳定性，不用全局同号/取反修正 |

因此，探索性排序可以保留单场景信号；项目正式实验验证清单要求两套场景同时有效并满足 AND。结果仍需独立实验验证，不能直接称为治疗靶点或普遍疗效。

---

### 4. normal source 中可能没有足够的目标基因表达

如果该基因表现为：

```text
正常细胞中几乎不表达
疾病形成后才强烈诱导
```

那么在正常 source 中做 KO 可能不够合理。

PerturbGen 基于排序后的表达基因 token 实施扰动。如果该基因根本不在正常细胞的有效 token 序列中，KO 可能无法模拟真实的早期干预，或者只能得到非常弱、难以解释的结果。

官方实现中的 `mask`、`pad` 和 `delete` 均依赖于对相应基因 token 的编辑。

这种情况下，更理想的设计是：

```text
早期疾病状态或刺激后早期状态 → 晚期疾病
source KO
```

而不是：

```text
完全健康的正常状态 → 晚期疾病
source KO
```

如果只有 normal 和 disease 两个横断面状态，那么 normal-source KO 仍然可以作为探索，但应解释为：

> 在模型假设的 normal→disease 映射中，早期移除该基因信息如何改变预测终点。

不能直接解释成真实的疾病预防效果。

---

### 5. 判断“向正常恢复”时必须排除目标基因本身

这是最容易产生假阳性的地方。

假设目标基因在疾病中高表达，进行 KO 后，它的表达自然降低。如果计算疾病细胞与正常细胞之间的距离时包含目标基因，那么：

\[
d(\text{disease+KO},\text{normal})
\]

几乎必然会变小一部分。

这并不能证明其他细胞程序得到恢复。

因此，至少要计算两种结果：

#### 5.1 包含目标基因

用于展示完整预测效应。

#### 5.2 排除目标基因

\[
d_{-G}\left(
\widehat X_{\text{disease+KO}},
X_{\text{normal}}
\right)
<
d_{-G}\left(
\widehat X_{\text{disease control}},
X_{\text{normal}}
\right)
\]

其中，\(-G\) 表示排除被 KO 的目标基因。

同时还应该：

- 从疾病 signature 中删除目标基因；
- 不把“目标基因表达降低”本身计入 rescue score；
- 重点观察其他疾病 DEG 和通路是否发生方向一致的逆转；
- 检查目标基因已知下游基因，而不只是它的共表达基因。

只有排除目标基因后仍然明显接近正常，才能称为**系统性状态恢复**。

---

### 6. 推荐定义 rescue score

若研究目标是 `reversal`，使用与 DAVF 训练/效用评估可追溯且按 donor 隔离的真实正常和疾病观测，在 donor 层面确定：

- \(D_{\uparrow}\)：疾病中上调的基因；
- \(D_{\downarrow}\)：疾病中下调的基因。

排除目标基因后，定义疾病分数：

\[
S_{\text{disease}}(X)
=
\operatorname{mean}_{g\in D_{\uparrow}}Z_g(X)
-
\operatorname{mean}_{g\in D_{\downarrow}}Z_g(X)
\]

KO 的恢复效果可以定义为：

\[
R_{\text{KO}}
=
S_{\text{disease}}(X_{\text{control}})
-
S_{\text{disease}}(X_{\text{KO}})
\]

如果 \(R_{\text{KO}}>0\)，表示在该参考轴下 KO 预测降低了疾病程序；它不等于已观测的干预效应。

分别得到：

\[
R_{\text{source-KO}}
\]

和：

\[
R_{\text{within-KO}}
\]

正式双场景候选应该表现为：

```text
两个 rescue score 均为正
+ 多数 donor 方向一致
+ 不依赖特定随机种子或配对方式
+ 排除目标基因后仍然成立
+ 真实 matched-null、未扰动质量和候选层 empirical-p/q 达到门槛
+ 至少 3 个共享 donor，且 observed/DAVF 训练/效用来源与 held-out 划分可追溯
```

---

### 7. 不能只看 UMAP 是否接近正常

UMAP 上的位置移动只能作为展示，不应作为主要证据。

建议至少同时评估：

1. 疾病 signature 的逆转程度；
2. 正常功能 signature 的恢复程度；
3. 疾病 DEG 的方向逆转比例；
4. pathway/GSEA 方向是否合理；
5. 目标基因已知下游程序是否改变；
6. 多个 donor 之间是否一致；
7. 是否降低异常状态，而不诱导毒性、应激或细胞死亡程序。

PerturbGen 的后处理流程提供了预测扰动前后表达以及嵌入相似性等结果，但最终仍需要对扰动后的表达矩阵进行标准化、差异分析和通路分析。

后处理文档：

https://perturbgen-docs.cog.sanger.ac.uk/examples/07_PostPerturbation_Analyses.html

---

### 8. 建议增加的稳健性检验

#### 8.1 比较不同 KO 模式

主要运行：

```yaml
perturbation_mode:
  - "mask"
```

再使用：

```text
pad
delete
```

进行敏感性分析。

`mask` 是官方推荐的主要 KO 模式；`pad` 和 `delete` 代表更强的移除假设。

如果三种模式方向一致，可信度更高；如果只有 `delete` 产生巨大效应，可能是过强 token 操作导致的模型外推。

#### 8.2 与匹配的随机基因比较

选择与目标基因匹配的背景基因：

- 疾病表达量相近；
- normal–disease fold change 相近；
- 表达细胞比例相近；
- token rank 相近；
- 变异程度相近。

对这些基因也进行 KO，建立 null distribution。

目标基因的 rescue score 应显著高于随机背景，而不能只是“KO 任何高表达基因都会使模型发生较大变化”。正式分析至少使用 99 个表达量/FC/检测率/token rank 匹配的 null，并在候选层计算 empirical-p 与 BH-FDR；20 个随机对照只用于 smoke。

#### 8.3 按 donor 做统计

不要把数千个细胞视为数千个独立重复。

建议对每个 donor 分别计算：

\[
R_{\text{source-KO}}^{(d)}
\]

和：

\[
R_{\text{within-KO}}^{(d)}
\]

再检查：

- 多数 donor 是否方向一致；
- 效应是否由单个 donor 驱动；
- 疾病严重程度、亚型或治疗状态是否影响结果。

#### 8.4 Held-out donor 验证未扰动预测

在解释 KO 前，首先确认模型能在未见过的 donor 上重建：

```text
normal → disease
```

尤其需要检查：

- 疾病 DEG 方向；
- 主要疾病通路；
- 目标细胞亚状态；
- 疾病状态的 donor 间异质性。

如果未扰动疾病状态本身预测不好，那么任一场景的结果都不应进入正式强解释。现有代码已有 null 生成、候选 p 聚合、formal 输入隔离、质量提取、donor split 和 AND 接口；E2E report 尚未自动接续这些统计，因此接口存在不等于正式证据完成。

---

### 9. 哪种结果足以让候选基因进入实验验证？

以下组合可以视为较强的计算候选证据；正式项目清单还要求真实 cohort 和完整统计 evidence：

- 目标基因在**同一细胞类型内部**疾病性上调，而不是仅由细胞比例改变造成；
- source KO 和 within-disease KO 都降低疾病 signature；
- 排除目标基因本身后，rescue 仍然成立；
- 多数 donor 中方向一致；
- 不同配对方案、随机种子和 KO 模式结果一致；
- 效应强于表达量匹配的随机基因；
- 没有明显诱导细胞应激、毒性或非特异性转录崩溃；
- 存在独立证据支持，例如遗传学、蛋白表达、调控网络、药物或 CRISPR 筛选证据；
- 该靶点具有现实可干预性。

正式 `PASS` 还必须同时满足：真实 `normal/disease` raw counts、canonical Ensembl、显式 donor、至少 3 个共享 donor、冻结 scVI gene order/embedding manifest、train-only/held-out donor 可追溯、真实 matched-null、未扰动质量、候选 empirical-p/q 以及 source/within-state 两个场景的统计门槛。四队列 IBD 规划、smoke、synthetic、mock 或 bridge 不能替代这些条件。

如果满足这些条件，可以将其列为：

> **高优先级、需要实验验证的计算候选。**

但建议表述为：

> 在明确参考轴和研究目标下，PerturbGen 预测 KO 可能逆转疾病相关转录程序。

而不是：

> PerturbGen 证明该基因致病。

---

### 10. 本部分结论

**两个场景在同一明确参考轴下都达到正式门槛，可以提高该候选进入实验验证的优先级；单场景结果只保留相应场景的探索解释。**

尤其是：

```text
source KO 阻止疾病程序形成
+
disease within-state KO 逆转已形成的疾病程序
```

这提示该基因在当前模型和参考轴下可能同时影响状态转移与状态内程序；不能据此声称疾病因果或治疗疗效。

不过，对于“疾病中特异性诱导、正常中几乎不表达”的基因，**disease within-state KO 通常比 normal-source KO 更直接、更可信**。

此时：

- normal-source KO 阴性不应直接否定靶点；
- 如果 normal-source KO 阳性，也必须确认目标基因在 source 中确实存在有效表达 token；
- 最终应以排除目标基因后的全局 rescue、跨 donor 稳定性、随机基因对照和实际 CRISPRi/KO 实验作为进入验证阶段的主要依据。

---

## 二、PerturbGen 官方资源

### 用户问题

PerturbGen 的官方网页、GitHub 仓库和预印本论文的链接分别是什么？

### 回答

#### 1. 官方网页或交互式数据门户

https://cellatlas.io/perturbgen

论文将其列为 PerturbGen web portal。

#### 2. 官方文档与教程

https://perturbgen.cog.sanger.ac.uk/

包含数据预处理、训练、基因扰动和扰动后分析教程。

#### 3. GitHub 主仓库

https://github.com/Lotfollahi-lab/Perturbgen

包含模型代码、配置文件和示例分析。

#### 4. bioRxiv 预印本

论文标题：

**Predicting how perturbations reshape cellular trajectories with PerturbGen**

预印本页面：

https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1

DOI：

https://doi.org/10.64898/2026.03.04.709254

预印本于 **2026年3月5日**发布。

#### 5. Hugging Face 预训练模型权重

https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main

---

## 三、如何使用 PerturbGen 预训练模型权重

### 用户问题

应如何使用 PerturbGen 的预训练模型权重？

是直接将 PerturbGen 的预训练模型权重用于自己的 scRNA 测序数据进行推理，还是需要在自己的 scRNA 测序数据上基于预训练模型权重进行重训练或微调？

### 回答

## 1. 核心结论

对于自己的 normal/disease scRNA-seq 数据，**通常不应只下载 Hugging Face 上的预训练权重后直接做最终扰动推理**。

更准确的官方工作流是：

> **加载预训练权重作为冻结的 source encoder → 在自己的 normal/disease 配对数据上训练 PerturbGen masking/transition model → 再训练 count decoder → 使用在自己数据上训练得到的 checkpoint 做 KO 推理。**

因此：

- **不需要**从头重新预训练超过一亿细胞规模的基础编码器；
- **需要**在自己的数据上做下游适配训练；
- 但这并不是传统意义上的“将整个基础模型全部解冻并微调”；
- 官方实现通常会冻结预训练的 scMaskGIT encoder；
- 主要训练数据集特异的状态转换部分和表达量解码器。

训练文档：

https://perturbgen-docs.cog.sanger.ac.uk/examples/03_train_perturbgen.html

---

## 2. Hugging Face 权重实际上是什么？

官方 Hugging Face 仓库主要提供一个预训练 checkpoint，例如：

```text
20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_
batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt
```

在官方教程中，它被指定为：

```python
ENCODER_PATH = "...pretrained checkpoint..."
```

也就是说，它主要作为 PerturbGen 的**预训练 encoder 初始化权重**，而不是一个已经适配任意疾病数据、可以直接输出 normal→disease 扰动表达矩阵的通用完整模型。

Hugging Face 仓库：

https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main

PerturbGen 的下游模型还包括：

1. source-state 预训练编码器；
2. target-state Transformer decoder；
3. 状态或时间条件编码；
4. 将隐空间结果映射回基因表达量的 count decoder。

其中后面的部分需要根据自己的状态设计、基因集合、条件变量和数据分布进行训练。

---

## 3. 为什么不能直接零样本推理？

### 3.1 疾病状态转换是数据集特异的

预训练 encoder 可以提供通用的单细胞基因表示，但它并不知道研究中的：

```text
normal → disease
```

具体对应什么表达变化，也不知道：

- normal 和 disease 如何定义；
- 哪些细胞类型相互对应；
- disease 是哪个 target state；
- donor、批次和疾病亚型如何处理；
- normal 细胞应该映射到什么样的疾病细胞。

官方训练流程要求输入：

```text
src_dataset
tgt_dataset_folder
src_adata
tgt_adata_folder
mapping_dict_path
```

这说明状态转换部分需要从自己的 source/target 数据中学习。

---

### 3.2 Target vocabulary 由自己的数据决定

官方代码会根据 tokenized target datasets 自动计算：

```text
target vocabulary size
maximum sequence length
```

自己的基因筛选方案、HVG 数量、token 映射和条件 token 都会影响 target 词表。

因此，一个在其他数据集上训练的完整 decoder 通常不能不经适配直接套用到新数据上。

---

### 3.3 表达量预测需要数据集特异的 count decoder

PerturbGen 先预测 target token 或 embedding，再通过 count decoder 重建可解释的基因表达量。

官方教程要求先获得在自己数据上训练的 masking checkpoint，然后以它为输入训练 count decoder：

```text
预训练 encoder
        ↓
在自己的数据上训练 masking model
        ↓
选择最佳 masking checkpoint
        ↓
在自己的数据上训练 count decoder
        ↓
获得 count-decoder checkpoint
        ↓
执行 in silico perturbation
```

扰动教程要求使用训练流程中得到的 count-decoder checkpoint，而不是只使用 Hugging Face 上的基础预训练 checkpoint。

---

## 4. 官方流程是否属于“微调”？

可以称为**下游适配训练**，但不宜简单理解为“全模型微调”。

### 4.1 第一阶段：训练 masking/transition model

官方代码通过以下参数载入预训练 scMaskGIT encoder：

```python
encoder_path=ENCODER_PATH
```

实现中会冻结预训练模型参数，逻辑类似：

```python
for param in self.model.parameters():
    param.requires_grad = False
```

并在 forward 中使用类似以下方式运行 encoder：

```python
with torch.no_grad():
    ...
```

因此，预训练 source encoder 默认是冻结的。

真正针对自己数据学习的主要是：

- target token embedding；
- cross-attention decoder；
- 状态转换模块；
- 输出层。

可以概括为：

```text
冻结的通用 source encoder
+
在自己的 normal/disease 数据上训练状态转换 decoder
```

---

### 4.2 第二阶段：训练 count decoder

官方实现会加载第一阶段获得的 masking checkpoint，然后再次冻结该模型的大部分参数：

```python
for param in self.pretrained_model.parameters():
    param.requires_grad = False
```

之后训练 count decoder，将学到的表示映射为基因表达量或计数分布。

因此，整体更像：

```text
预训练 encoder：冻结
状态转换 decoder：在自己的数据上训练
完整 masking model：随后冻结
count decoder：在自己的数据上训练
```

---

## 5. 推荐的 normal/disease 数据分析流程

### 第一步：准备 AnnData

按照官方预处理教程生成 `.h5ad` 文件。

建议在 `adata.obs` 中至少保留：

```text
disease_state: normal / disease
cell_type
donor
batch
```

如果还有严重程度或疾病阶段，也可以保留：

```text
disease_stage
severity
subtype
```

建议保留原始 count 信息用于后续 count decoder，而不是只提供批次整合后的表达矩阵。

---

### 第二步：Tokenization 和 source-target 配对

对于最简单的设计，可以设置：

```python
TIME_OBS = "disease_state"
TIME_POINT_ORDER = ["normal", "disease"]
REFERENCE_TIME = "normal"
```

并选择合理的配对方式：

```python
PAIRING_MODE = "stratified"
MAIN_PAIRING_OBS = "cell_type"
```

如果希望进一步约束配对，可以将 donor、疾病亚型等作为附加配对变量。

但如果 normal 和 disease 来自不同 donor，就不能要求同一 donor 一一匹配。

官方 tokenization 步骤还要求使用与预训练模型一致的：

```text
gene median dictionary
token dictionary
gene mapping dictionary
```

这些文件由 PerturbGen 仓库提供。

Tokenization 和配对教程：

https://perturbgen-docs.cog.sanger.ac.uk/examples/02_tokenization_pairing.html

---

### 第三步：训练 masking model

使用官方预训练 checkpoint 作为：

```python
ENCODER_PATH = "path/to/pretrained_encoder.ckpt"
```

同时输入自己的：

```python
SRC_DATASET
TGT_DATASET_FOLDER
SRC_ADATA
TGT_ADATA_FOLDER
MAPPING_DICT_PATH
```

然后运行类似：

```bash
python -m perturbgen train-mask ...
```

这里学习的是：

\[
P(X_{\text{disease}}\mid X_{\text{normal}},
\text{cell type},\text{context})
\]

而不是重新训练通用单细胞基础模型。

官方教程将预训练 checkpoint 作为 `--encoder_path` 传给 `train-mask`。

---

### 第四步：选择最佳 masking checkpoint

不要简单使用最后一个 epoch。

建议根据 held-out validation donor 上的指标选择 checkpoint，例如：

- validation perplexity；
- disease DEG 方向恢复；
- disease signature 相关性；
- predicted disease 与真实 disease 的表达距离；
- 各 donor、各细胞类型上的稳定性。

尤其建议采用 **donor-level 划分**，避免同一个 donor 的相似细胞同时进入训练集和测试集。

---

### 第五步：训练 count decoder

将上一步的最佳 checkpoint 作为：

```python
CKPT_MASKING_PATH = "path/to/best_masking_checkpoint.ckpt"
```

然后运行类似：

```bash
python -m perturbgen train-decoder ...
```

官方支持：

```text
mse
nb
zinb
```

教程示例使用 `zinb`。

训练完成后，可以得到适合自己基因集合和表达分布的 count-decoder checkpoint。

---

### 第六步：先验证未扰动预测

在进行 KO 之前，应先运行未扰动预测，检查模型能否完成：

```text
normal → disease
```

至少要确认：

1. 真实疾病 DEG 是否被正确预测；
2. 疾病上调和下调通路的方向是否正确；
3. 预测疾病状态是否比 normal 更接近真实 disease；
4. 结果是否在 held-out donor 上成立；
5. 是否由单个 donor 或单个细胞亚群驱动。

如果模型连未扰动的 normal→disease 都不能重建，后面的 KO 结果就不适合进行强生物学解释。

---

### 第七步：使用数据集特异的 checkpoint 做 KO

扰动配置中通常同时需要：

```yaml
trainer:
  encoder_path: "pretrained_encoder.ckpt"
  mapping_dict_path: "your_dataset_mapping.pkl"
  tokenid_to_rowid_path: "your_dataset_tokenid_to_rowid.pkl"
  genes_to_perturb:
    - "target_ensembl_id"

model:
  ckpt_masking_path: "your_count_decoder_checkpoint.ckpt"
```

最后运行：

```bash
python perturbgen/Perturb/val.py \
  --config docs/examples/configs/perturbation.yaml
```

官方扰动教程使用的是训练流程中得到的 count-decoder checkpoint，而不是仅使用 Hugging Face 的基础预训练 checkpoint。

扰动教程：

https://perturbgen-docs.cog.sanger.ac.uk/examples/06_perturbation.html

---

## 6. 什么时候可以直接使用预训练权重？

### 情形 A：只提取基础 encoder 表示

如果目标只是探索性地提取 source 细胞的通用表示，而不要求预测：

```text
normal → disease
KO → disease response
```

理论上可以冻结预训练 encoder，对兼容 tokenization 后的细胞进行 feature extraction。

但这只能得到通用表示，不能等同于已经完成疾病轨迹建模或 KO 效应预测。

---

### 情形 B：复现论文中的同一数据集

如果使用：

- 与论文完全一致的数据；
- 相同的基因筛选；
- 相同的词表和 token 映射；
- 相同的 source/target 定义；
- 官方提供的完整下游 checkpoint；

才可能直接运行相应推理。

但官方 Hugging Face 页面提供的主要是预训练 encoder checkpoint，并不是适用于任意疾病数据的完整 masking model 加 count decoder。

---

## 7. 针对疾病 KO 分析的具体建议

推荐采用：

```text
官方预训练 encoder
        ↓ 冻结
在自己的 normal/disease 数据上训练 transition/masking decoder
        ↓
在自己的数据上训练 count decoder
        ↓
验证未扰动 normal→disease 预测
        ↓
执行 source KO 和 within-state KO
```

而不是：

```text
官方预训练 checkpoint
        ↓
直接输入自己的 scRNA-seq 数据
        ↓
直接解释 KO 结果
```

后者缺少对自己的疾病状态转换和表达分布的学习。

另外，进行 normal-source KO 时，目标基因必须实际出现在 normal source 细胞的 token 序列中。

如果目标基因在 normal 中几乎不表达，相关细胞可能因为 source 序列中缺少该 token 而无法被有效扰动，甚至相应 batch 被跳过。

此时可以考虑：

- 使用早期疾病状态作为 source；
- 使用刺激后早期状态作为 source；
- 主要依赖 disease-state within-state KO 分析；
- 不将 normal-source KO 阴性直接解释为靶点无效。

---

## 8. 本部分一句话总结

**对于自己的 scRNA-seq 数据，需要的不是从头重训练基础模型，也不是直接进行零样本扰动推理，而是冻结官方预训练 encoder，在自己的 normal/disease 配对数据上训练疾病特异的 transition decoder 和 count decoder，然后使用得到的 checkpoint 进行 KO 推理。**

---

## 四、推荐的完整分析框架

综合以上内容，可以采用以下工作流：

```text
1. 准备原始 scRNA-seq count 数据和完整 metadata
   ↓
2. 进行细胞类型注释和样本级质量控制
   ↓
3. 在同一细胞类型内定义 normal 与 disease 状态
   ↓
4. 按 donor 划分训练集、验证集和测试集
   ↓
5. 使用 PerturbGen 官方词典完成 tokenization
   ↓
6. 使用官方预训练 encoder 作为冻结初始化
   ↓
7. 在自己的数据上训练 normal→disease transition/masking model
   ↓
8. 在自己的数据上训练 count decoder
   ↓
9. 在 held-out donor 上验证未扰动的 normal→disease 预测
   ↓
10. 用外部 candidate_spec 的目标和方向通过 DAVF/独立表达三方 gate
   ↓
11. 对通过 gate 的疾病高表达候选执行 source KO
   ↓
12. 对相同基因执行 disease within-state KO
   ↓
13. 排除目标基因本身后计算 disease rescue score
   ↓
14. 检查跨 donor 稳定性和不同 KO 模式的一致性
   ↓
15. 与表达量和 token rank 匹配的随机基因比较
   ↓
16. 接续真实 null、未扰动质量、候选 empirical-p/q 和双场景 AND
   ↓
17. 结合遗传学、蛋白表达和调控网络证据排序，再交实验验证
```

---

## 五、主要资源汇总

| 资源 | 地址 |
|---|---|
| PerturbGen Web Portal | https://cellatlas.io/perturbgen |
| 官方文档 | https://perturbgen.cog.sanger.ac.uk/ |
| GitHub 仓库 | https://github.com/Lotfollahi-lab/Perturbgen |
| Hugging Face 权重 | https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main |
| bioRxiv 预印本 | https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1 |
| DOI | https://doi.org/10.64898/2026.03.04.709254 |
| Tokenization 与配对教程 | https://perturbgen-docs.cog.sanger.ac.uk/examples/02_tokenization_pairing.html |
| 模型训练教程 | https://perturbgen-docs.cog.sanger.ac.uk/examples/03_train_perturbgen.html |
| 扰动分析教程 | https://perturbgen-docs.cog.sanger.ac.uk/examples/06_perturbation.html |
| 扰动后分析教程 | https://perturbgen-docs.cog.sanger.ac.uk/examples/07_PostPerturbation_Analyses.html |

---

## 六、最终结论

对于疾病状态下过表达的候选基因：

1. `normal → disease source KO` 是状态转移前场景，可用于探索模型中的早期干预效用；
2. `disease within-state KO` 是目标状态内场景，可用于探索模型中的状态内效用；
3. 两个场景均能在排除目标基因本身后逆转疾病程序且满足正式门槛，才可提高候选进入实验验证的优先级；
4. PerturbGen 的预训练权重主要作为通用 encoder 初始化，而不是针对任意疾病数据的零样本完整模型；
5. 应在自己的 normal/disease 数据上训练状态转换模块和 count decoder；
6. 在执行 KO 前，必须先验证模型能否在 held-out donor 上正确预测未扰动的 normal→disease 转换；
7. 计算结果只能用于方向筛选和候选效用排序，不能作为 KO/KD 因果或临床疗效结论，最终仍需 CRISPR、药理学或其他功能实验验证。
