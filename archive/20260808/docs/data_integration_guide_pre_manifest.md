# 数据整合指南

## 概述

本指南介绍如何使用PTM2CellNet数据整合脚本，将多个公开PTM数据库整合为统一的训练数据格式。

## 数据源

| 数据库 | 内容 | 下载方式 |
|--------|------|---------|
| [EPSD 2.0](http://epsd.biocuckoo.cn/) | 磷酸化位点 | 直接下载 |
| [CPLM 4.0](http://cplm.biocuckoo.cn/) | 赖氨酸修饰 | 直接下载 |
| [dbPTM 2025](https://biomics.lab.nycu.edu.tw/dbPTM/) | 综合PTM | 直接下载 |
| [UniProt](https://www.uniprot.org/) | 蛋白序列 | REST API |

## 快速开始

### 1. 运行数据整合

```bash
# 基本整合 (使用已有序列)
python scripts/integrate_data_v2.py --max-seq-len 1000

# 从UniProt API获取缺失序列
python scripts/integrate_data_v2.py --max-seq-len 1000 --fetch-api
```

### 2. 输出文件

整合完成后，会在 `data/processed/` 目录生成：

- `ptm_integrated_human.csv` — 24,195条整合数据
- `ptm_integrated_human_labeled.csv` — 112,012条已标记数据

### 3. 数据格式

```csv
id,sequence,ptm_sites,uniprot_id,seq_length,num_ptm_sites,cell_state
PTM_O00115,MVAMAAGPSG...,"[{""position"":20,""type"":""phosphorylation""}]",O00115,360,10,unknown
```

### 输入数据格式规范（权威）

无论你是用上面的整合脚本生成数据，还是自带真实数据，**训练/推理消费的 CSV 必须满足以下列契约**。`DataPreprocessor` 与 `PTMDataset` 据此读取：

| 列名 | 是否必填 | 类型 | 说明 |
|---|---|---|---|
| `sequence` | ✅ 必填 | string | 蛋白质氨基酸序列（单字母大写）。非标准字符（U/X/J/B/Z/O）会被自动标准化。 |
| `cell_state` | 训练必填，推理可选 | string | 细胞状态标签（如 `proliferation` / `apoptosis`）。所有出现过的取值构成类别集合。 |
| `ptm_sites` | 推荐 | JSON 字符串 | PTM 位点列表，形如 `[{"position": 20, "type": "phosphorylation"}]`。缺失或非法位点的行会被规整为 `[]`。 |
| `id` | 可选 | string | 样本唯一标识，便于结果对齐。 |
| `uniprot_id` | 可选 | string | UniProt 入库号，便于追溯。 |

**最小可用训练 CSV 示例**（仅必填列）：

```csv
sequence,cell_state
ACDEFGHIKLMNPQRSTVWY,proliferation
ACDEFGHIKLMNPQRSTVWYG,apoptosis
MKTAYIAKQRQ,quiescence
```

**带 PTM 位点的训练 CSV 示例**：

```csv
id,sequence,ptm_sites,cell_state
S1,MVAMAAGPSG,"[{""position"":20,""type"":""phosphorylation""}]",proliferation
S2,MKTA...,"[{""position"":5,""type"":""ubiquitination""},{""position"":12,""type"":""acetylation""}]",apoptosis
S3,ACDEFGHIK,[],quiescence
```

`ptm_sites.position` 为从 1 起的残基索引（1-based），`type` 取值见 `src/data/features.py` 的 `DEFAULT_PTM_TYPES`（磷酸化、泛素化、乙酰化、甲基化、SUMO 化等）。

#### 字段约束与质量校验

`DataPreprocessor` / `helpers.py` 会在预处理阶段做以下校验，非法样本默认被丢弃（`handle_missing="drop"`）：

- **序列**：非空、长度 ≤ `data.max_sequence_length`（默认 1000）、字符 ∈ `valid_amino_acids`。
- **PTM 位点**：`position` 为正整数且 ≤ 序列长度；`type` 在已知 PTM 类型集合内。
- **标签**：非空字符串；类别数 ≥ 2（单类无法训练）。

训练入口 `scripts/train.py` 还会在启动时执行 `DataPreprocessor.validate_data_quality()`（见下文“数据质量预检”），对标签分布、序列长度分布、PTM 位点合法性做聚合报告，**不达标即 fail-fast**，避免在低质量数据上浪费时间。

## 氨基酸字符标准化

整合脚本会自动将非标准氨基酸字符映射为标准字符：

| 字符 | 含义 | 映射为 |
|------|------|--------|
| U | 硒代半胱氨酸 | C |
| X | 未知 | A |
| J | Leu/Ile模糊 | L |
| B | Asx | D |
| Z | Glx | E |
| O | 吡咯赖氨酸 | K |

## 常见问题

### Q: 整合后只有24K条记录，是否太少了？

A: 这是因为只有同时有**完整序列**和**PTM位点**的蛋白才会被纳入。如果需要更多数据：

1. 从UniProt API获取更多序列 (`--fetch-api`)
2. 下载更多公开数据 (PhosphoSitePlus等)
3. 使用数据增强 (见`src/data/augmentation.py`)

### Q: cell_state只有2种标签怎么办？

A: 当前数据来自PMADS，只有Quiescent/Activated两种状态。可以：

1. 使用二分类模式训练
2. 从CPTAC/CCLE获取更多标签
3. 使用生物学知识做标签映射
