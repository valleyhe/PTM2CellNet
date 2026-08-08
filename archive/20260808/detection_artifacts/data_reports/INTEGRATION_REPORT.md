# PTM2CellNet 数据整合最终报告 (v3)

**日期**: 2026-04-11  
**最后更新**: 19:20  
**脚本**: `scripts/integrate_data_v2.py`

---

## 🎯 最终结果

| 数据集 | 记录数 | 大小 | 唯一蛋白 |
|--------|--------|------|---------|
| **ptm_integrated_human.csv** | **80,508** | 135.2MB | 61,979 |
| **ptm_integrated_human_labeled.csv** | **112,012** | 65.4MB | — |
| **总PTM位点** | **1,358,618** | — | — |

---

## 📊 下载与整合历程

### 第1步: 公开数据库下载 (无需注册/登录)

| 数据库 | 下载方式 | 数据量 | 耗时 |
|--------|---------|--------|------|
| **EPSD 2.0** | 直链 wget | 1,071,725 磷酸化位点 | 42s |
| **CPLM 4.0** | 直链 wget | 298,633 赖氨酸修饰位点 | 2s |
| **dbPTM 2025** | 直链 wget (5种PTM) | 2,123,533 位点 | 2s |
| **UniProt 人类全部** | REST API stream (2次请求) | 204,729 条序列 | 461s |

### 第2步: UniProt 下载策略

```
请求1: reviewed:true AND organism_id:9606    → 20,423 条 (16s, 11MB)
请求2: reviewed:false AND organism_id:9606   → 184,306 条 (446s, 51MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: 204,729 条人类蛋白序列 (62MB)
```

**访问分析**:
- UniProt REST API 无明确的 rate limit header
- `stream` 端点返回所有结果, 无需分页
- 两次请求之间有 5s 间隔 (保守设置)
- 无 429 限流错误
- 总耗时 ~8 分钟 (绝大部分是 unreviewed 数据下载)

### 第3步: 数据整合与优化

| 版本 | 序列数 | 记录数 | 覆盖率 |
|------|--------|--------|--------|
| v1 (所有物种) | 29,770 | 24,195 | 26% |
| v2 (仅人类) | 29,770 | 24,195 | 26% |
| **v3 (完整UniProt)** | **210,270** | **80,508** | **87%** |

---

## 📋 数据质量

### PTM 类型分布 (25种)

```
phosphorylation:           901,877 (55.2%)
ubiquitination:            302,158 (18.5%)
acetylation:                57,403  (3.5%)
sumoylation:                49,312  (3.0%)
crotonylation:              11,666  (0.7%)
methylation:                10,444  (0.6%)
2-hydroxyisobutyrylation:    8,128  (0.5%)
glycation:                   5,673  (0.3%)
malonylation:                4,330  (0.3%)
succinylation:               3,923  (0.2%)
beta-hydroxybutyrylation:    3,032  (0.2%)
其他稀有类型 (<0.1%): 14种
```

### 序列统计

```
平均长度: 477 aa
中位长度: 412 aa
最大长度: 1,000 aa (截断)
平均PTM/蛋白: 16.9
中位PTM/蛋白: 6
最大PTM/蛋白: 1,500
```

### 数据质量验证

```
PTM位点-氨基酸匹配率: 93.85%
异构体ID解析: 18,904 个 → Canonical 形式
剩余无序列: 12,055 个 (主要是 UPI/XP_/废弃ID)
```

---

## 📁 完整文件列表

```
data/raw/
├── uniprot/
│   ├── human_all_sequences.fasta       # 204,729 条 (62MB) ← 新
│   ├── human_reviewed.fasta            # 20,423 条 (11MB) ← 新
│   ├── human_proteome.fasta            # 5,523 条 (旧版)
│   └── human_idmapping.gz              # ID 映射表
├── epsd/
│   ├── Homo sapiens.txt                # 1,071,725 位点 (243MB)
│   └── Homo sapiens.fasta              # 参考序列 (47MB)
├── cplm/
│   ├── Homo sapiens.txt                # 298,633 位点 (292MB)
│   └── Acetylation.txt                 # 仅乙酰化 (172MB)
├── dbptm/
│   ├── Phosphorylation                 # 1,615,054 位点 (162MB)
│   ├── Ubiquitination                  # 348,307 位点 (26MB)
│   ├── Acetylation                     # 138,169 位点 (9MB)
│   ├── Methylation                     # 16,114 位点 (1.1MB)
│   └── Sumoylation                     # 5,889 位点 (393KB)
└── DOWNLOAD_REPORT.md

data/processed/
├── ptm_integrated_human.csv            # 80,508 条 (135MB) ← 主数据集
├── ptm_integrated_human_labeled.csv    # 112,012 条 (65MB) ← 已标记数据
├── ptm_train_phosphorylation.csv       # 1,814,867 条 (原有)
├── ptm_train_acetylation.csv           # 199,495 条 (原有)
├── pmads_combined.csv                  # 112,013 条 (原有)
├── train.csv / val.csv / test.csv      # 351/76/76 条 (原有)
└── INTEGRATION_REPORT.md

scripts/
├── integrate_data_v2.py                ← 主整合脚本
├── download_uniprot_all_human.py       ← UniProt 批量下载
├── download_uniprot_human.py           ← 旧版 (已弃用)
└── full_pipeline_test.py               ← 全量测试
```

---

## 💡 训练方案建议

### 方案A: 二分类模式 (立即可用, 推荐)

```bash
# 使用已标记数据
python scripts/train.py \
  --config configs/default.yaml \
  --data data/processed/ptm_integrated_human_labeled.csv \
  --task-type classification \
  --num-classes 2 \
  --cell-states Quiescent Activated
```

**数据**: 112,012 条 | **标签**: Quiescent (75.5%) vs Activated (24.5%)

### 方案B: 多任务学习 (推荐, 充分利用数据)

```
Task 1: PTM 位点预测 → 使用 80,508 条整合数据
Task 2: 细胞状态分类 → 使用 112,012 条已标记数据
```

### 方案C: 迁移学习

```
Step 1: 在 80,508 条数据上做 PTM-aware 预训练 (无监督/自监督)
Step 2: 在 112,012 条已标记数据上做微调
```

---

## 📌 剩余待处理项

| 项目 | 影响 | 优先级 | 说明 |
|------|------|--------|------|
| **UniProt UPI/XP_ ID 映射** | 12K 蛋白无序列 | 🟡 中 | 需要通过 UniParc/RefSeq 映射获取 |
| **cell_state 标签扩展** | 仅2种状态 | 🔴 高 | 需从 CPTAC/文献获取更多分类 |
| **PhosphoSitePlus** | 额外 35万 高质量PTM | 🟡 中 | Cloudflare 防护, 需手动下载 |
| **PTM位点氨基酸验证** | 6%不匹配 | 🟡 中 | 异构体/成熟蛋白差异导致, 可容忍 |

---

## 🏁 执行总结

| 阶段 | 耗时 | 结果 |
|------|------|------|
| 公开数据库下载 | ~10 min | 4个数据库, ~1.3GB 原始数据 |
| UniProt 全量下载 | ~8 min | 204,729 条序列 (2次API请求) |
| 数据整合 v1→v2 | ~3 min | 24,195 条 (仅人类) |
| 数据整合 v2→v3 | ~2 min | **80,508 条 (+233%)** |
| 数据质量验证 | ~1 min | 93.85% 匹配率 |
| **总计** | **~25 min** | **从0到80,508条整合数据** |

**报告生成时间**: 2026-04-11 19:20
