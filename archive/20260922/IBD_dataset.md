# 人类 IBD 10x 3′ scRNA-seq 数据下载与整合分析方案

> **当前项目边界（2026-09-13）**：当前正式 PerturbGen cohort=0；下列四个 GEO 队列是数据来源与分析角色规划，
> 不是已经通过 Gate-0 的正式 PerturbGen cohort。正式效用验收仍需真实
> `normal/disease` raw counts、canonical Ensembl、明确 cell type/state、显式 donor
> 和至少 3 个跨状态共享 donor，并绑定 scVI gene order、冻结 embedding asset 与
> manifest；PerturbGen token index 与 scVI decoder index 必须通过 canonical Ensembl
> 和冻结 manifest 对齐且不得混用。下载、合并或完成 atlas 不能替代该 preflight。

执行依据见 [CURRENT_STATUS](docs/CURRENT_STATUS.md)、[中央双路径方案](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md) 和 [task_plan](task_plan.md)。

本方案的 normal/disease 比较只能作为 donor-level 观测方向证据；它不能作为
KO/KD 干预效应 ground truth。正式分析必须另外记录 `context`、`intervention`、
比较基准、研究目标、observed/DAVF/效用来源及 train-only/held-out donor 划分，
不得用全局同号或取反规则补齐方向，也不得把结果写成治疗因果或临床疗效。

## 1. 项目目标

本项目拟整合以下四个人类 IBD 10x Genomics 3′ scRNA-seq 数据集：

| 数据集       | 定位                                                    | 建议角色                          |
| --------- | ----------------------------------------------------- | ----------------------------- |
| GSE214695 | UC + CD + HC，colonic mucosa                           | 计划核心发现来源 / anchor（需 Gate-0 preflight） |
| GSE231993 | UC + HC，inflamed + patient-matched non-inflamed colon | 计划核心来源 / paired UC（需 Gate-0 preflight） |
| GSE282122 | UC + CD + HC，大型 longitudinal gut atlas                | 计划取 pretreatment + colon（需 Gate-0 preflight） |
| GSE266616 | CD + non-IBD，TI + ascending colon                     | 仅优先取 ascending colon，扩展/验证集   |

主要科学目标建议定义为：

**Primary objective**

> 鉴定成人 colonic IBD 中跨 UC/CD 共享的细胞组成、细胞状态和转录改变，即：
>
> `Inflamed IBD vs Healthy Control`

同时保留三个次级问题：

1. UC-specific 与 CD-specific changes；
2. non-inflamed IBD mucosa 与 HC 的 field effect；
3. inflammation gradient：HC → non-inflamed → adjacent → inflamed。

因此 metadata 中必须同时保留：

`IBD/HC`

和：

`UC/CD/HC`

不能在数据预处理阶段丢弃疾病亚型。

---

# 2. 四个数据集的最终使用策略

## 2.1 GSE214695：核心 anchor

当前 GEO 明确包含：

* 6 HC
* 6 UC
* 6 CD
* 共18名受试者
* 全部为 colonic mucosa

具体取材涉及 sigmoid、rectum、transverse colon、ascending colon、descending colon 等部位。GEO同时提供原始 count-level MTX/TSV 和作者 cell annotation。

### 纳入策略

全部18个样本均纳入 atlas。

主疾病比较：

```text
HC
vs
UC active
vs
CD active
```

同时建立：

```text
disease_group:
HC
IBD

disease_subtype:
HC
UC
CD
```

### 定位

**计划核心发现来源，建议作为 reference/anchor；是否进入正式效用 cohort 需 Gate-0 preflight。**

---

# 2.2 GSE231993：核心 paired UC 数据

包含：

* 4 HC；
* 4 UC患者；
* 每位UC患者各有：

  * inflamed colon biopsy；
  * patient-matched normal-appearing ascending-colon biopsy；
* 因此共12个scRNA样本。

建议定义三个状态：

```text
HC
UC_noninflamed
UC_inflamed
```

其中：

**UC self-control 绝对不能改标成 HC。**

### 主 atlas

12个样本全部保留。

### Primary IBD vs HC

只使用：

```text
UC_inflamed
vs
HC
```

### Secondary paired analysis

利用同一个患者的：

```text
UC_inflamed
vs
UC_noninflamed
```

做真正的 patient-paired differential analysis。

### Field-effect analysis

比较：

```text
UC_noninflamed
vs
HC
```

寻找尚未形成明显炎症病灶时已经存在的分子改变。

### 定位

**计划核心来源；是否进入正式效用 cohort 需 Gate-0 preflight。**

---

# 2.3 GSE282122：必须筛选后使用

这是整个项目中细胞数量最大的队列。

原研究包括：

* 38 biologic-naïve IBD patients；
* 3 healthy controls；
* CD + UC；
* 216 gut biopsies；
* 987,743 high-quality cells；
* 五个肠段：

  * terminal ileum；
  * ascending colon；
  * descending colon；
  * sigmoid；
  * rectum；
* adalimumab 治疗前后纵向采样。

作者明确发现肠段本身造成非常强的表达变化，尤其 epithelial compartment；因此 ileum 不能简单作为一个普通 batch 与 colon 混在一起。

## 主分析只纳入：

```text
timepoint = pretreatment / baseline
AND
site ∈ {
    ascending_colon,
    descending_colon,
    sigmoid,
    rectum
}
```

排除：

```text
terminal_ileum
post-adalimumab
```

### 如果 metadata 有 inflammation 状态

建议进一步建立：

Primary：

```text
baseline + colon + inflamed IBD
vs
HC colon
```

Secondary：

```text
baseline + colon + all IBD
vs
HC
```

以及：

```text
baseline + colon + non-inflamed IBD
vs
HC
```

### 特别注意

这里的患者是：

**biologic-naïve**

并不能自动解释为：

**完全 treatment-naïve**。

因此建议 metadata：

```text
treatment_status = biologic_naive
```

而不是：

```text
treatment_status = untreated
```

### 定位

筛选完成后的：

**pretreatment-colon subset = 计划核心来源；是否满足正式效用 cohort 需 Gate-0 preflight。**

post-treatment 部分全部保留在硬盘，但单独建立 future longitudinal analysis，不进入当前 disease atlas。

---

# 2.4 GSE266616：扩展验证集

GEO描述为：

* 17 non-IBD individuals；
* 32 CD patients；
* 79 gut biopsies；
* terminal ileum + ascending colon；
* 包含：

  * control；
  * non-inflamed；
  * adjacent-to-inflamed；
  * inflamed；
* 同时包含接受不同 biologic treatment 的患者。

每个 GSM 中具有相当详细的 metadata，例如：

```text
patient
condition
age
tissue
history
treatment (biological)
sample region
```

而且可以明确区分：

```text
ascending_colon
terminal_ileum
```

以及：

```text
inf
adjacent
non
```

具体样本确认使用 10x 3′ v3.1、Cell Ranger 7.0.1，并提供 barcodes/features/matrix 文件。

## 第一阶段只选择：

```text
tissue == ascending_colon
```

terminal ileum 暂时排除。

### Primary validation

```text
CD_inflamed_AC
vs
nonIBD_AC
```

### Secondary

```text
CD_noninflamed_AC
vs
nonIBD_AC
```

以及：

```text
CD_adjacent_AC
vs
nonIBD_AC
```

### biologic treatment

必须完整记录：

```text
biologic_status
```

例如：

```text
none
anti_TNF
anti_IL12_IL23
other
unknown
```

Primary sensitivity analysis 可以首先选择：

```text
biologic_status == none
```

但必须注意：

```text
no biologic
≠
treatment-naïve
```

### 定位

不建议与前三个数据集等权作为 discovery cohort。

更合理：

**external extension / replication cohort。**

---

# 3. 建议的数据目录结构

建议建立：

```text
IBD_scRNA/
│
├── 00_download/
│   ├── GSE214695/
│   ├── GSE231993/
│   ├── GSE282122/
│   └── GSE266616/
│
├── 01_raw_count/
│   ├── GSE214695/
│   ├── GSE231993/
│   ├── GSE282122/
│   └── GSE266616/
│
├── 02_metadata/
│   ├── metadata_GSE214695.tsv
│   ├── metadata_GSE231993.tsv
│   ├── metadata_GSE282122.tsv
│   ├── metadata_GSE266616.tsv
│   └── metadata_master.tsv
│
├── 03_objects_raw/
│
├── 04_objects_QC/
│
├── 05_integrated/
│
├── 06_celltype/
│
├── 07_pseudobulk/
│
├── 08_DEG/
│
├── 09_abundance/
│
├── 10_pathway/
│
├── figures/
│
└── scripts/
```

---

# 4. 第一阶段下载什么？

## 原则

目前不要下载 FASTQ。

目标是重新进行：

* QC；
* normalization；
* integration；
* clustering；
* DEG；
* cell abundance；
* pathway；

只需要作者提供的 **raw/filtered gene × cell count matrices**。

FASTQ 只有在准备：

* 统一 Cell Ranger version；
* 重新比对 genome；
* 重新 empty-droplet calling；
* 统一 ambient RNA correction；

时才真正需要。

目前完全没有必要。

---

# 5. 推荐下载文件

## GSE214695

下载：

```text
GSE214695_RAW.tar
GSE214695_cell_annotation.csv.gz
```

GEO当前标记约：

```text
RAW.tar             ~846 MB
cell_annotation     ~479 KB
```

RAW archive 中是各 GSM 的：

```text
barcodes.tsv.gz
features.tsv.gz
matrix.mtx.gz
```

---

## GSE231993

下载：

```text
GSE231993_RAW.tar
```

约：

```text
227 MB
```

其中包含12个样本对应的 MTX/TSV count matrices。

---

## GSE282122

第一阶段只下载：

```text
GSE282122_filtered_processed_data.tar.gz
```

约：

```text
2.8 GB
```

暂时不要下载：

```text
GSE282122_raw_processed_data.tar.gz
```

后者约：

```text
8.4 GB
```

除非以后准备重新做 ambient-RNA / droplet-level 分析。

此外必须下载论文：

```text
Supplementary Table 1
```

它是：

> Study cohort summary and metadata

是建立：

```text
donor
disease
gut site
baseline/post-treatment
remission
```

等变量的主要依据。

---

## GSE266616

下载：

```text
GSE266616_RAW.tar
```

约：

```text
3.7 GB
```

archive 中进一步包含逐样本 tar 文件，每个样本最终提供：

```text
barcodes.tsv.gz
features.tsv.gz
matrix.mtx.gz
```

注意一个需要人工核验的地方：

GEO Series 描述目前仍写：

```text
79 gut biopsies
```

但页面显示：

```text
Samples (85)
```

因此绝对不要把85个GSM直接解释为85个独立患者或85个独立 biological replicates。

必须通过：

```text
patient
tissue
sample_region
GSM
```

重新构建样本表。

---

# 6. 推荐使用 GEOquery 自动下载

R 中：

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("GEOquery")

library(GEOquery)

dir.create("00_download", showWarnings = FALSE)

getGEOSuppFiles(
    "GSE214695",
    baseDir = "00_download",
    filter_regex = "RAW|cell_annotation"
)

getGEOSuppFiles(
    "GSE231993",
    baseDir = "00_download",
    filter_regex = "RAW"
)

getGEOSuppFiles(
    "GSE282122",
    baseDir = "00_download",
    filter_regex = "filtered_processed_data"
)

getGEOSuppFiles(
    "GSE266616",
    baseDir = "00_download",
    filter_regex = "RAW"
)
```

这样不会下载 SRA FASTQ。

---

# 7. 下载后的完整性检查

首先记录：

```text
filename
file_size
md5
download_date
```

Linux：

```bash
find 00_download -type f -exec md5sum {} \; \
    > 00_download/md5_local.txt
```

然后解压。

GSE214695：

```bash
mkdir -p 01_raw_count/GSE214695

tar -xf \
00_download/GSE214695/GSE214695_RAW.tar \
-C 01_raw_count/GSE214695
```

GSE231993：

```bash
mkdir -p 01_raw_count/GSE231993

tar -xf \
00_download/GSE231993/GSE231993_RAW.tar \
-C 01_raw_count/GSE231993
```

GSE282122：

```bash
mkdir -p 01_raw_count/GSE282122

tar -xzf \
00_download/GSE282122/GSE282122_filtered_processed_data.tar.gz \
-C 01_raw_count/GSE282122
```

GSE266616：

```bash
mkdir -p 01_raw_count/GSE266616

tar -xf \
00_download/GSE266616/GSE266616_RAW.tar \
-C 01_raw_count/GSE266616
```

GSE266616 是嵌套 archive，之后：

```bash
find 01_raw_count/GSE266616 \
    -name "*.tar.gz" \
    -print
```

逐样本解压到独立目录。

---

# 8. 不要马上 merge：先建立 master metadata

这是整个项目最重要的一步。

建议 `metadata_master.tsv` 至少具有：

```text
dataset
GSM
library_id
sample_id
donor_id

disease_group
disease_subtype

inflammation_status
tissue
intestinal_region

treatment_status
biologic_status
timepoint
response_status

age
sex

chemistry
cellranger_version

analysis_role
include_atlas
include_primary_DE
include_field_effect
exclude_reason
```

---

# 9. 推荐的标准化 metadata 编码

## disease_group

```text
HC
IBD
```

## disease_subtype

```text
HC
UC
CD
```

## inflammation_status

统一为：

```text
healthy
noninflamed
adjacent
inflamed
unknown
```

不要保留各论文自己完全不同的缩写。

例如：

```text
inf → inflamed
ad  → adjacent
non → noninflamed
UCSC → noninflamed
```

---

# 10. intestinal_region 标准化

建议：

```text
terminal_ileum
ascending_colon
transverse_colon
descending_colon
sigmoid
rectum
colon_unspecified
```

额外建立：

```text
organ_level
```

仅两个水平：

```text
ileum
colon
```

Primary discovery：

```text
organ_level == colon
```

---

# 11. treatment/timepoint 标准化

例如：

```text
baseline
post_treatment
not_applicable
unknown
```

GSE282122：

```text
baseline → include
post_adalimumab → exclude from primary atlas
```

不要把两者一起当 IBD。

---

# 12. 四个数据集建议的 analysis_role

## GSE214695

```text
analysis_role = core_discovery
```

全部18 donor纳入。

---

## GSE231993

```text
HC:
core_discovery

UC_inflamed:
core_discovery

UC_noninflamed:
paired_secondary
```

---

## GSE282122

```text
baseline + colon:
core_discovery

post-treatment:
longitudinal_only

terminal_ileum:
ileal_secondary
```

---

## GSE266616

```text
ascending_colon:
external_validation

terminal_ileum:
ileal_secondary
```

---

# 13. 推荐建立两个不同的“纳入变量”

不要只创建：

```text
include = TRUE/FALSE
```

而应该至少有：

```text
include_atlas
include_primary_DE
```

因为一个样本完全可能：

```text
include_atlas = TRUE
include_primary_DE = FALSE
```

典型例子：

GSE231993 的：

```text
UC_noninflamed
```

它非常值得显示在 UMAP 和细胞状态 atlas 中，但不应该进入：

```text
inflamed IBD vs HC
```

Primary DEG。

---

# 14. Primary discovery cohort 的计划定义（不等于 Gate-0 已通过）

我建议第一版严格定义为：

## GSE214695

```text
HC
UC
CD
```

全部 colon。

## GSE231993

```text
HC
UC_inflamed
```

UCSC 不进入 Primary DEG。

## GSE282122

```text
baseline
AND colon
```

如果能够获得可靠 inflammation metadata：

优先：

```text
baseline
AND colon
AND inflamed
```

对比 HC。

---

# 15. GSE266616 不进入第一版 discovery（规划角色，仍需 preflight）

它作为 replication cohort：

```text
ascending_colon
AND
(
    nonIBD
    OR
    CD
)
```

然后分别：

```text
CD_inflamed vs HC
CD_noninflamed vs HC
CD_adjacent vs HC
```

做独立分析。

这一设计比四个数据集混在一个 differential model 中更加可靠。

---

# 16. 数据读取后统一处理什么？

四个数据集均保留：

> 原始 UMI counts

不要先对各数据集做：

```text
LogNormalize
```

后再把 normalized matrices 合在一起。

正确方法：

```text
raw counts
↓
统一 gene identity
↓
建立单个对象
↓
QC
↓
concatenate
↓
统一 normalization / integration
```

---

# 17. Gene ID harmonization

首先检查：

```text
gene symbol
Ensembl ID
feature_type
```

全部转换到同一 gene vocabulary。

推荐使用：

```text
HGNC gene symbol
```

作为主 display identifier，同时保留：

```text
Ensembl_gene_id
```

如果某些数据只有 symbol：

不要通过简单：

```text
make.unique()
```

长期掩盖问题。

先检查：

* duplicated symbols；
* outdated symbols；
* mitochondrial genes；
* ENSG版本后缀。

建议最终构建：

```text
gene_symbol
ensembl_id
```

映射表。

---

# 18. 不要强求所有数据集完全相同的 gene list

整合对象可以使用 union gene set。

用于：

* HVG；
* PCA；
* scVI；
* DEG；

再根据具体方法选择共有的有效表达基因。

不要为了得到完全相同矩阵而过早删除大量基因。

---

# 19. 第二轮 QC

这些都是作者已经处理过的 filtered matrices，因此：

**不要重新用非常激进的统一阈值过滤。**

推荐 per-sample robust QC。

至少计算：

```text
n_genes
n_UMI
percent_mito
percent_ribo
```

建议以：

```text
sample × MAD
```

识别异常细胞。

例如：

```text
n_genes < median - 3 MAD
n_genes > median + 3 MAD
n_UMI   > median + 3 MAD
mito    > median + 3 MAD
```

然后结合实际分布人工检查。

---

# 20. 不建议直接统一规定 percent.mt < 10%

IBD inflamed biopsy 中：

* epithelial stress；
* tissue digestion；
* neutrophils；
* inflammatory macrophages；

均可能导致 mitochondrial fraction 上升。

如果机械使用：

```text
percent.mt < 10%
```

可能 preferentially 删除真正的疾病细胞。

更合理的是：

> dataset/sample-specific QC。

---

# 21. Doublet detection

建议使用：

```text
scDblFinder
```

或：

```text
Scrublet
```

在每个 library/sample 内单独运行。

不要把所有样本合并以后再寻找 doublets。

---

# 22. Ambient RNA

第一版建议不做跨数据集 SoupX。

原因：

SoupX 最理想需要 unfiltered droplet matrix。

目前四个数据集可获得的公开处理层级并不完全一致。

如果：

GSE282122 使用 raw droplets 做 SoupX，

而其他数据不做，

反而可能引入新的 study-specific preprocessing bias。

因此第一版：

```text
ambient correction = none
```

后续有需要再统一重处理 FASTQ/raw droplet matrices。

---

# 23. 第一轮 broad cell annotation

建议统一到：

```text
Epithelial

T
NK/ILC

B
Plasma

Myeloid

Mast

Fibroblast
Pericyte

Endothelial

Other
```

首先建立 broad lineage，不要立即追求几十个亚群。

---

# 24. 原论文 annotation 如何使用？

原则：

> 使用作者 annotation 作为 reference，而不是绝对 ground truth。

GSE214695已经直接提供：

```text
cell_annotation.csv.gz
```

GSE282122原研究定义了9个主要 compartment 和109个cell states，也非常适合作为 cell-state reference。

建议最终保存：

```text
original_annotation
harmonized_annotation
```

两个字段。

绝对不要覆盖作者原始 annotation。

---

# 25. 最重要的原则：annotation 和 integration 分层进行

推荐：

```text
all cells
    │
    ▼
broad lineage annotation
    │
    ├── epithelial
    ├── myeloid
    ├── T/NK
    ├── B/plasma
    └── stromal/endothelial
    │
    ▼
lineage-specific integration
    │
    ▼
fine cell-state annotation
```

比一次性：

```text
~1 million cells
→ one global integration
→ clustering
```

可靠得多。

---

# 26. 推荐 integration 工具

对于这个项目规模，我更推荐：

> **Scanpy + scVI**

而不是把全部数据进行传统 Seurat SCT anchor integration。

原因是：

* GSE282122 接近100万细胞；
* sparse count matrix很大；
* scVI mini-batch 模式更适合大规模数据；
* 非线性 batch effect 处理较好。

---

# 27. scVI 中什么作为 batch？

推荐：

```text
batch_key = dataset
```

即：

```text
GSE214695
GSE231993
GSE282122
```

对于扩展 atlas 再加入：

```text
GSE266616
```

可以考虑同时记录 chemistry：

```text
v3
v3.1
...
```

---

# 28. 哪些变量绝对不能作为“需要消除的 batch”？

不要把：

```text
disease
inflammation_status
intestinal_region
treatment
```

作为普通 batch 强行消掉。

尤其：

```text
colon vs ileum
```

是真实生物学。

TAURUS/GSE282122 已经显示，仅 epithelial compartment，ileum 与 colon 就存在极强的表达差异。

因此：

**不要 Harmony(region)**

也不要：

**scVI batch_key = intestinal_region**

否则可能把真正的肠段生物学删除。

---

# 29. 推荐的 scVI 框架

概念上：

```python
adata.layers["counts"] = adata.X.copy()

scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="dataset"
)

model = scvi.model.SCVI(
    adata,
    n_latent=30,
    gene_likelihood="nb"
)

model.train()

adata.obsm["X_scVI"] = model.get_latent_representation()
```

然后：

```text
X_scVI
↓
neighbors
↓
UMAP
↓
Leiden
```

但：

> integrated latent embedding 只用于 clustering / visualization / annotation。

---

# 30. 绝对不要使用 integrated expression 做正式 DEG

正式 DEG 必须回到：

> raw UMI counts。

即：

```text
scVI/Harmony embedding
    ↓
cell-type definition
    ↓
raw counts
    ↓
pseudobulk
    ↓
DEG
```

---

# 31. 为什么不能 cell-level FindMarkers？

错误方法：

```text
500,000 IBD cells
vs
200,000 HC cells
```

因为这些细胞不是：

```text
n = 700,000
```

真正 biological replicate 是：

```text
patient/donor
```

否则会产生严重 pseudoreplication。

---

# 32. 推荐 pseudobulk 单位

最基础的 aggregation key：

```text
dataset
donor_id
intestinal_region
inflammation_status
cell_type
```

GSE282122还必须加入：

```text
timepoint
```

例如：

```text
GSE282122
patient_01
ascending_colon
inflamed
baseline
macrophage
```

对应一个 pseudobulk sample。

---

# 33. 多个 biopsy 属于同一个患者怎么办？

特别是 GSE282122，一个患者可能：

* 多个 colon sites；
* baseline/post；
* inflamed/non-inflamed。

不能把它们都作为独立 donor。

推荐 pseudobulk 后用：

> donor 作为 random effect。

适合：

```text
dreamlet / variancePartition
```

模型例如：

```text
expression ~
    disease_group +
    inflammation_status +
    intestinal_region +
    dataset +
    age +
    sex +
    (1 | donor_id)
```

实际可根据 metadata 完整程度简化。

---

# 34. 我更推荐“study-level DEG + meta-analysis”

这是本项目统计上更稳健的方案。

不要把所有 pseudobulk 样本简单扔进一个巨大模型。

分别在：

```text
GSE214695
GSE231993
GSE282122
```

内部做：

```text
IBD_inflamed vs HC
```

然后获得每个 cell type 的：

```text
log2FC
SE
P
FDR
```

再做：

> cross-study meta-analysis。

---

# 35. IBD-shared gene 推荐定义

例如一个 macrophage gene：

至少满足：

```text
≥2 independent discovery datasets
方向一致
```

并且 meta-analysis：

```text
FDR < 0.05
```

进一步用：

```text
GSE266616 AC
```

验证方向。

最终可以定义：

```text
robust IBD-shared macrophage genes
```

比 pooled FindMarkers 得到的结果可靠得多。

---

# 36. UC/CD 合并策略

Primary：

```text
UC + CD → IBD
```

但是所有统计结果必须继续进行：

```text
UC vs HC
CD vs HC
```

敏感性分析。

例如某基因：

```text
IBD vs HC ↑
UC vs HC ↑
CD vs HC ↑
```

可以定义为：

> IBD-shared。

但如果：

```text
UC ↑
CD ↓
```

即使 pooled IBD 显著，也不能称为 shared IBD signature。

---

# 37. Cell abundance analysis

对于：

```text
cell-type proportion
```

也不要简单把四个数据集的所有细胞合并后计算。

不同研究：

* biopsy size；
* digestion；
* cell recovery；
* sequencing depth；

均不同。

因此推荐：

### study内部

每个 donor：

```text
cell type cells /
total recovered cells
```

计算 abundance。

再比较：

```text
IBD vs HC
```

最后 meta-analysis effect direction。

---

# 38. GSE266616 对 abundance 特别需要谨慎

其组织处理涉及 epithelial fraction 和 lamina propria 的独立消化步骤。具体GSM记录显示 epithelium 使用 EGTA/TrypLE，而 lamina propria 使用 Liberase TM。

因此不能把：

```text
GSE266616 macrophage %
```

直接与：

```text
GSE214695 macrophage %
```

作为绝对比例比较。

更合理：

> 比较各 study 内 IBD/HC fold-change，然后比较效应方向。

---

# 39. 推荐最终建立四类分析对象

## Object A：core_colon_atlas

包括：

```text
GSE214695
+
GSE231993
+
GSE282122 baseline-colon
```

用途：

* UMAP；
* broad/fine annotation；
* IBD shared cell states；
* discovery。

---

## Object B：core_inflamed_colon

包括：

```text
GSE214695 active IBD + HC

GSE231993 inflamed UC + HC

GSE282122 baseline inflamed colon + HC
```

用途：

> Primary IBD vs HC statistical analysis。

这是最重要的 inferential dataset。

---

## Object C：noninflamed_field_effect

包括：

```text
GSE231993 UC-self-control

GSE282122 baseline noninflamed colon

GSE266616 AC noninflamed
```

与各自 HC 比较。

用途：

> IBD field effect / pre-inflammatory state。

---

## Object D：GSE266616_AC_validation

包括：

```text
nonIBD AC
CD_noninflamed AC
CD_adjacent AC
CD_inflamed AC
```

用途：

> 独立 CD validation。

---

# 40. 第一阶段明确排除什么？

当前主分析暂时排除：

```text
GSE282122 terminal ileum
GSE282122 post-adalimumab
GSE266616 terminal ileum
```

不是删除。

全部保留在硬盘并进入 metadata：

```text
include_primary = FALSE
```

未来可以建立：

```text
ileal_CD_atlas
```

和：

```text
adalimumab_longitudinal_atlas
```

---

# 41. Primary analysis 的逻辑结构

```text
                     ┌─ GSE214695 ─────────────┐
                     │ HC / UC / CD            │
                     │ colon                    │
                     │                          │
                     ├─ GSE231993 ─────────────┤
Download counts ────►│ HC / UC-inf / UC-non   │
                     │ colon                    │
                     │                          │
                     ├─ GSE282122 ─────────────┤
                     │ baseline + colon only    │
                     │                          │
                     └─ GSE266616 ─────────────┘
                       AC validation only
                               │
                               ▼
                      metadata harmonization
                               │
                               ▼
                        sample-level QC
                               │
                               ▼
                       concatenate counts
                               │
                               ▼
                        broad annotation
                               │
                               ▼
                       lineage-specific
                          integration
                               │
                               ▼
                         cell states
                               │
              ┌────────────────┴───────────────┐
              ▼                                ▼
       donor pseudobulk                 cell abundance
              │                                │
              ▼                                ▼
       study-level DEG                   study-level test
              │                                │
              └───────────┬────────────────────┘
                          ▼
                     meta-analysis
                          │
                          ▼
                robust shared IBD signals
                          │
                          ▼
                GSE266616 validation
```

---

# 42. 第一轮最值得得到的结果

完成上述流程后，应首先获得：

### Figure 1

整合 atlas：

```text
UMAP by cell type
UMAP by dataset
UMAP by disease
UMAP by intestinal region
```

### Figure 2

各数据集：

```text
cell composition
```

但不直接比较绝对跨-study比例。

### Figure 3

每个 major lineage：

```text
IBD vs HC pseudobulk DEG
```

### Figure 4

跨数据集：

```text
effect-size forest plot
```

### Figure 5

定义：

```text
IBD-shared signatures
UC-specific signatures
CD-specific signatures
```

### Figure 6

GSE266616：

```text
external validation
```

---

# 43. 软件环境建议

如果采用 Python 主线：

```text
Python 3.11
scanpy
anndata
scvi-tools
scrublet
decoupler
pandas
numpy
scipy
```

统计分析：

```text
R
edgeR
limma
dreamlet / variancePartition
muscat
fgsea
clusterProfiler
```

这是一个比较理想的：

> Python负责百万细胞 atlas + R负责 donor-level statistics

的组合。

---

# 44. 如果坚持全R

可以使用：

```text
Seurat v5
+
BPCells
+
Harmony
+
scDblFinder
+
edgeR/dreamlet
```

对于百万细胞规模：

> 不建议简单使用传统全数据 SCTransform + anchor integration。

BPCells on-disk matrix 会明显降低内存压力。

---

# 45. 硬件需求估算

仅下载目前推荐 processed matrices：

大约：

```text
GSE214695       0.85 GB
GSE231993       0.23 GB
GSE282122       2.8 GB
GSE266616       3.7 GB
------------------------
合计            ~7.6 GB compressed
```

建议项目目录至少：

```text
100–200 GB free disk
```

因为还会产生：

* 解压数据；
* h5ad；
* Seurat object；
* intermediate matrices；
* pseudobulk；
* figures。

推荐 RAM：

```text
≥64 GB
```

更稳妥：

```text
128 GB+
```

scVI 本身可以 mini-batch GPU training，因此显存需求远低于一次把百万细胞 dense matrix 放进 GPU。

---

# 46. 第一阶段不建议做的事情

暂时不要：

```text
下载全部 FASTQ
```

不要：

```text
把四个数据集直接 merge 后 FindMarkers
```

不要：

```text
把 intestinal_region 当 batch 消除
```

不要：

```text
把 UC-self-control 当 HC
```

不要：

```text
把 GSE282122 post-treatment 与 baseline 混合
```

不要：

```text
把 GSE266616 TI 与 AC 无差别混合
```

不要：

```text
把 cell 当 biological replicate
```

---

# 47. 第一阶段实施顺序

建议实际执行顺序：

1. 下载四个 GEO processed count data，并为每个来源登记 manifest；
2. 下载 GSE282122 Supplementary Table 1；
3. 解压但不合并；
4. 建立4张 dataset-specific metadata；
5. 建立 `metadata_master.tsv`；
6. 明确每一个 GSM/library 对应 donor；
7. 给所有样本标记：

   * disease；
   * subtype；
   * site；
   * inflammation；
   * treatment；
   * timepoint；
8. 生成 inclusion/exclusion table；
9. 逐数据集创建 AnnData/Seurat；
10. sample-level QC；
11. doublet filtering；
12. broad lineage annotation；
13. 构建 core colon atlas；
14. lineage-specific integration；
15. harmonized cell-type annotation；
16. donor-level pseudobulk；
17. 先执行 Gate-0 preflight，确认真实 raw counts、canonical Ensembl、显式 donor 和至少 3 个共享 donor，再进入 PerturbGen 正式 utility；
18. study-level differential analysis；
19. meta-analysis；
20. GSE266616 replication；
21. 最后再决定是否加入 TI 和 longitudinal 数据。

---

# 48. 我建议第一阶段下载后先停止在什么位置？

最理想的停止节点不是：

> 四个 Seurat object 已经 merge。

而是：

> **四个数据集已经下载、解压，并完成准确的 sample/GSM/donor metadata 和 inclusion table。**

这仍只是数据准备完成条件，不是 PerturbGen Gate-0 或生物学 PASS；还需通过项目
的 raw-count、normal/disease、canonical Ensembl、≥3 共享 donor、scVI/embedding
manifest、train-only/held-out 划分及真实统计 evidence 要求。

也就是先得到：

```text
metadata_master.tsv
```

和：

```text
sample_inclusion.tsv
```

例如：

| dataset   | GSM | donor | disease | region    | inflammation | timepoint | atlas | primary_DE | role         |
| --------- | --- | ----- | ------- | --------- | ------------ | --------- | ----- | ---------- | ------------ |
| GSE214695 | ... | HC1   | HC      | sigmoid   | healthy      | baseline  | Y     | Y          | discovery    |
| GSE231993 | ... | UC1   | UC      | colon     | inflamed     | baseline  | Y     | Y          | discovery    |
| GSE231993 | ... | UC1   | UC      | ascending | noninflamed  | baseline  | Y     | N          | paired       |
| GSE282122 | ... | CD... | CD      | colon     | inflamed     | baseline  | Y     | Y          | discovery    |
| GSE282122 | ... | CD... | CD      | ileum     | inflamed     | baseline  | N     | N          | ileal        |
| GSE282122 | ... | UC... | UC      | colon     | ...          | post      | N     | N          | longitudinal |
| GSE266616 | ... | HA... | CD      | AC        | inflamed     | ...       | Y     | validation | validation   |
| GSE266616 | ... | HA... | CD      | TI        | inflamed     | ...       | N     | N          | ileal        |

只有这张表完全正确以后，才建议正式启动百万细胞级 integration。

---

# 49. 最终推荐方案

本项目第一阶段正式定义：

## Core discovery

```text
GSE214695
+
GSE231993
+
GSE282122 pretreatment-colon
```

用于建立：

> Human colonic IBD scRNA atlas。

## Primary contrast

```text
inflamed IBD
vs
healthy colon
```

UC 和 CD 合并为：

```text
IBD
```

但保留 subtype。

## External validation

```text
GSE266616 ascending colon
```

重点验证 CD 中：

```text
cell states
DEGs
pathways
abundance trends
```

## Secondary analyses

```text
UC vs HC
CD vs HC
UC vs CD
noninflamed IBD vs HC
inflamed vs noninflamed
```

## 暂不纳入

```text
ileum
post-adalimumab
```

二者后续分别建立：

```text
ileal CD atlas
```

和：

```text
longitudinal anti-TNF atlas
```

这一方案的核心思想不是尽可能把细胞数做大，而是：

> **先控制 sampling frame、解剖部位和 treatment/timepoint，再利用多个独立 cohort 对 IBD 共同信号进行重复验证。**

对于这四个数据集，这是比简单 pooled integration 更可靠的分析框架。
