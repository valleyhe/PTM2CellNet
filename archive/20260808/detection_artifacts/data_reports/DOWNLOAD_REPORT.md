# 公开数据集下载报告 — 最终版

**日期**: 2026-04-11  
**执行方式**: 全自动化脚本下载（无需注册/登录/授权）

---

## ✅ 已成功下载的数据集（4个，全部开放获取）

### 1. EPSD 2.0 — 真核生物磷酸化位点数据库

| 项目 | 详情 |
|------|------|
| **来源** | http://epsd.biocuckoo.cn/Download.php |
| **下载文件** | `Homo sapiens.txt` + `Homo sapiens.fasta` |
| **文件大小** | 243MB (位点) + 47MB (参考序列) |
| **磷酸化位点数** | **1,071,726** 条（100% 实验验证） |
| **覆盖蛋白** | 75,248 个唯一 UniProt ID |
| **字段** | EPSD ID, UniProt ID, 氨基酸, 位置, 来源, 参考文献 |
| **认证要求** | ❌ 无需任何注册或登录 |

### 2. CPLM 4.0 — 蛋白质赖氨酸修饰数据库

| 项目 | 详情 |
|------|------|
| **来源** | http://cplm.biocuckoo.cn/Download.php |
| **下载文件** | `Homo sapiens.txt` (完整) + `Acetylation.txt` (单独) |
| **文件大小** | 292MB + 172MB |
| **修饰位点总数** | **298,634** 条 |
| **覆盖蛋白** | 24,327 个唯一 UniProt ID |
| **修饰类型** | 泛素化(139K), 乙酰化(59K), SUMO化(53K), 巴豆酰化(13K) 等 15+ 种 |
| **字段** | CPLM ID, UniProt ID, 位置, 修饰类型, 基因名, 物种, **完整蛋白序列**, 证据, PMID |
| **认证要求** | ❌ 无需任何注册或登录 |

### 3. dbPTM 2025 — 综合PTM数据库

| 项目 | 详情 |
|------|------|
| **来源** | https://biomics.lab.nycu.edu.tw/dbPTM/download.php |
| **下载文件** | 5种PTM类型的实验验证数据 |
| **文件大小** | ~200MB（解压后） |
| **总位点数** | **2,123,533** 条 |
| **分布** | 磷酸化(1,615K), 泛素化(348K), 乙酰化(138K), 甲基化(16K), SUMO化(6K) |
| **字段** | 条目名, UniProt ID, 位置, PTM类型, 参考文献, 序列窗口(21aa) |
| **认证要求** | ❌ 无需任何注册或登录 |

### 4. UniProt 人类参考蛋白质组

| 项目 | 详情 |
|------|------|
| **来源** | UniProt FTP (`ftp.uniprot.org`) |
| **下载文件** | `human_proteome.fasta.gz` + `human_idmapping.gz` |
| **文件大小** | 1.9MB + 678KB |
| **蛋白序列数** | 5,523 条参考蛋白质 |
| **ID映射** | UniProt ID ↔ Gene Name ↔ RefSeq ↔ Ensembl 等 |
| **认证要求** | ❌ 无需注册，开放获取 (`Access-Control-Allow-Origin: *`) |

---

## ❌ 无法自动下载的数据集

| 数据库 | 原因 | 数据量 | 获取建议 |
|--------|------|--------|---------|
| **PhosphoSitePlus** | Cloudflare 人机验证防护 | 35万+ PTM | 手动通过浏览器下载 |
| **CPTAC 泛癌蛋白质组** | 需通过 PDC 门户交互式查询 | ~1000 样本 | 浏览器访问 pdc.cancer.gov |
| **CCLE/DepMap** | 需同意使用条款（免费注册） | ~1000 细胞系 | 注册后下载 |

---

## 📊 数据总量汇总

| 数据库 | 位点/记录数 | 磁盘占用 | PTM 类型 | 数据来源 |
|--------|-----------|---------|---------|---------|
| **EPSD** | 1,071,726 | 290MB | 磷酸化 | 实验验证 |
| **CPLM** | 298,634 | 464MB | 15+ 种赖氨酸修饰 | 实验/数据库策展 |
| **dbPTM** | 2,123,533 | 200MB | 5 种核心 PTM | 41个数据库整合 |
| **UniProt** | 5,523 蛋白 | 2.5MB | 参考序列 + 映射 | 参考蛋白质组 |
| **已有本地** | ~4,200,000 | ~328MB | 6 种 PTM | 前期处理 |
| **合计** | **~7,700,000+** | **~1.3GB** | — | — |

---

## 📁 本地文件目录结构

```
data/raw/
├── uniprot/
│   ├── human_proteome.fasta.gz       # 人类参考蛋白质组 (1.9MB)
│   ├── human_proteome.fasta          # 解压后 (3.8MB)
│   ├── human_idmapping.gz            # ID 映射 (678KB)
│   └── human_reviewed_batch1.fasta   # 部分 reviewed 蛋白 (68KB)
├── epsd/
│   ├── Homo sapiens.txt              # 磷酸化位点 (243MB, 1.07M条)
│   └── Homo sapiens.fasta            # 参考序列 (47MB)
├── cplm/
│   ├── Homo sapiens.txt              # 全部赖氨酸修饰 (292MB, 299K条)
│   └── Acetylation.txt               # 仅乙酰化 (172MB, 208K条)
├── dbptm/
│   ├── Phosphorylation               # 磷酸化 (162MB, 1.62M条)
│   ├── Ubiquitination                # 泛素化 (26MB, 348K条)
│   ├── Acetylation                   # 乙酰化 (9MB, 138K条)
│   ├── Methylation                   # 甲基化 (1.1MB, 16K条)
│   └── Sumoylation                   # SUMO化 (393KB, 6K条)
└── DOWNLOAD_REPORT.md                # 本报告
```

---

## 🔗 数据字段对照表

| 数据库 | UniProt ID 列 | 位置列 | 序列列 | PTM类型列 |
|--------|-------------|--------|--------|----------|
| EPSD | Column 2 | Column 4 | 需从FASTA获取 | 仅磷酸化 |
| CPLM | Column 2 | Column 3 | Column 7（完整序列） | Column 4 |
| dbPTM | Column 2 | Column 3 | Column 6（21aa窗口） | Column 4 |
| UniProt | FASTA header `|XXXXX|` | N/A | 完整序列 | N/A |

---

## 💡 下一步：数据整合建议

```
步骤 1: 从 UniProt FASTA 构建 UniProt ID → 完整序列 的映射字典
步骤 2: 从 EPSD 提取磷酸化位点 (UniProt ID + Position)
步骤 3: 从 CPLM 提取赖氨酸修饰位点 (UniProt ID + Position + Type)
步骤 4: 从 dbPTM 提取5种PTM位点 (UniProt ID + Position + Type)
步骤 5: 通过 UniProt ID 合并 → 每个蛋白的多PTM组合标注
步骤 6: 与现有 train/val/test.csv (351条) 的 cell_state 标签关联
步骤 7: 输出最终训练集: id, sequence, ptm_sites(JSON), cell_state
```

---

**报告生成时间**: 2026-04-11  
**下载工具**: wget (5个并行), curl, Python urllib  
**总耗时**: ~10 分钟
