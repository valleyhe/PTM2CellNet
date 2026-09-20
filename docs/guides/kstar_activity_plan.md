# KSTAR activity 方案（独立环境，主环境只收标准表）

对应主线：`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §5.2 / §7 阶段 2，
命令契约见 [`ptm_activity_pipeline.md`](ptm_activity_pipeline.md) §3。

**状态（2026-09-18）**：方案冻结，**未执行**。当前只有 PTM smoke 与 `method_version=smoke-stub-20260918`
的 activity 占位表。这不是 KSTAR，也不是 biology PASS。

## 1. 目标与边界

KSTAR 只做一件事：从**磷酸化位点证据**推断 **kinase activity**（`activity_direction` /
`activity_score`），交给已有阶段 3 做 signed-network 传播。

不做：

- 不把 `kstar` / PhosphoSitePlus 客户端 / R 的 PhosR 装进 `requirements-core.txt`
- 不在主环境 `import kstar`
- 不把 KSTAR 分数当成 log2FC、DAVF 方向或治疗结论
- 不把 smoke stub（`method=KSTAR` 但 `method_version=smoke-stub-*`）回标成真 KSTAR
- 不在 AD DEG 上反调 KSTAR 阈值（方案 §8.7）
- 不把当前 `omnipath-2026-09-16`（**只有 `tf_regulation`**）当成 kinase→gene 传播网

主环境交接物必须是 `ptm_activity.tsv`（方案 §4.2 十三列）+ activity manifest。

## 2. 为什么现在还不能跑正式 KSTAR

| 缺口 | 事实 | 正式运行前必须 |
|---|---|---|
| 真实 PTM | `ptm_cohort: PENDING_EXTERNAL_PTM_COHORT` | 真实 phosphosite 定量，donor/condition 可审计 |
| 工具环境 | 主环境没有 kstar | 独立 conda/env，版本钉扎 |
| 先验网络 | KSTAR 需要 kinase–substrate 注释 | 冻结 PhosphoSitePlus（或 KSTAR 自带资源）release + 许可 |
| 传播网络 | 2026-09-16 OmniPath 导出是 DoRothEA/tf_target，**没有 kinase 边** | 另做 signed enzyme–substrate/signaling release，**不得**与 TF-only freeze 混用 |
| Benchmark | 方案 §9.1 要求已知 kinase perturbation 评估 | 预注册 1 个公开 kinase inhibitor/KO phosphoproteome，过线前进 lineage |

PTM smoke（`scripts/generate_ptm_smoke.py`）只证明阶段 1/3/4/5 的表契约。
它同时写出 `kstar_evidence_from_sites.tsv`，作为未来 adapter 的**输入形状**，不是 KSTAR 输出。

## 3. 独立环境

建议环境名：`kstar`（或 `ptm-activity`，与 PerturbGen 环境分离）。

```text
python >= 3.10
kstar          # Naegle lab, Nat Commun 2022 https://doi.org/10.1038/s41467-022-32017-5
pandas, numpy
# PhosphoSitePlus kinase-substrate 资源按 kstar 文档下载/缓存，写入 env 内路径
```

安装、资源路径、包版本写入该环境自己的 `environment.yml` / pin 文件，**不**进入
`requirements-core.txt` / `setup.py`。主仓库只增加 **adapter CLI**（见 §5），用
`subprocess` 或约定目录消费产物，失败则硬失败，不静默跳过。

PhosR（sensitivity）走 **另一个 R 环境**，同样只交 `ptm_activity.tsv`；primary 与
sensitivity **分两次阶段 3，禁止平均**（方案 §4.2）。

## 4. 输入适配：标准化 PTM → KSTAR evidence

上游：阶段 1 `standardized_ptm.tsv`（已做 replicate 策略、site/total 归一化、Ensembl map）。

KSTAR 要的是 **二元位点证据**（某个 contrast 下该位点是否“增加/减少”），不是连续 intensity
直接当 activity。

建议冻结的 binarize 规则（写进 activity manifest，禁止事后改）：

1. 只用 `has_donor=True` 的行；无 donor 行保持 exploratory，不进 KSTAR。
2. 位点键：`{UniProt}_{residue}`（例 `P49841_S9`），与 PhosphoSitePlus 对齐；对不上的位点计数后丢弃，不猜。
3. 每个位点分别在 normal / disease donor 上取归一化值的 donor 均值（或预注册的 Welch，二选一，冻结后不得换）。
4. `disease_mean / normal_mean >= τ_up` → increased 证据；`<= 1/τ_up` → decreased 证据；中间 unchanged。
   第一版建议 `τ_up=1.2`，与 smoke evidence 表同一口径，便于对照；正式数据可以改 τ，但必须换 manifest id。
5. 输出两张 0/1 表（increased / decreased），列 = 一次 contrast（本项目第一版只有
   `disease-minus-normal`），行 = site_id。

Smoke 已按此口径生成：

`outputs/ptm_activity/20260918_ptm_smoke/inputs/kstar_evidence_from_sites.tsv`

（若尚未生成，运行 `python scripts/generate_ptm_smoke.py --output-dir outputs/ptm_activity/20260918_ptm_smoke`）。

**禁止**：把连续 `ptm_value` 直接叫 kinase activity；用 AD 表达 DEG 当 KSTAR 输入。

## 5. 输出适配：KSTAR → `ptm_activity.tsv`

KSTAR 激酶分数映射到方案 §4.2：

| 我方字段 | 来源 |
|---|---|
| `activity_unit` | KSTAR 内部量纲名（如 enrichment z / median statistic），原样记录 |
| `regulator_id` | **必须与阶段 3 网络 `source_id` 同一 identifier 空间**（见 §6） |
| `regulator_type` | `kinase` |
| `condition_or_contrast` | 冻结 `config.contrast`（`disease-minus-normal`） |
| `activity_score` | 有符号分数；0 分禁止传播（现有契约） |
| `activity_direction` | 与 score 符号一致的 `up`/`down` |
| `activity_pvalue` / `activity_qvalue` | KSTAR 自带检验；无 q 则留空并标 exploratory，不得填假 q |
| `n_substrates` | 该激酶在本次 evidence 中命中的注释底物数 |
| `network_coverage` | 命中底物 / 该激酶注释底物总数，∈[0,1] |
| `method` | `KSTAR` |
| `method_version` | **真实包版本 + 资源 hash**，禁止 `smoke-stub-*` |
| `input_manifest` | 阶段 1 `ptm_input_manifest.json` 路径 |

每行唯一键：`(regulator_id, condition_or_contrast, method)`。

Adapter 写出 `ptm_activity_manifest.json`：kstar 版本、资源 sha256、τ、site 丢弃计数、
对比轴、`biology_pass=false`（直到 §7 benchmark 过线）。

## 6. 传播网络缺口（阶段 3 的硬前提）

KSTAR 输出的是 **kinase**。当前正式 signed network
`data/processed/signed_network/signed_network_omnipath_20260916.tsv` 的 `edge_type`
全是 `tf_regulation`，source 是 TF Ensembl。把 GSK3B 的 activity 丢进这张网，
`seeds_without_node` 会吃掉激酶，**不会**得到 tau/APP 一类下游 gene score。

因此正式 KSTAR → gene score 之前必须另冻一张网，至少包含：

```text
kinase --(signed enzyme-substrate / signaling)--> substrate 或 TF
TF     -- tf_regulation --> gene          # 可复用 2026-09-16 边，但要新 release id
```

`propagation.gene_edge_types` 仍以基因终止边为准（现为 `tf_regulation`）。
kinase 边的 `edge_type` 必须是中间 hop，不能冒充 gene 终止边。

候选边来源（均外部、需许可与 freeze）：

- OmniPath `enzsub` / SIGNOR（仓库已有 `scripts/import_kinase_substrate.py`，**非正式 PTM 传播网**）
- 不得使用 `src/models/signaling_network.py` 硬编码路径（方案 §5.3）

`regulator_id` 与 `source_id` 对齐规则（冻结一条，不要混用）：

- **推荐**：双方都用 canonical Ensembl（与 scVI/DEG 一致）
- 若 KSTAR 只出 symbol：经 `data/processed/gene_symbol_ensembl_map.tsv` 显式 map，冲突硬失败

新网的 `release` 必须是新 id（例如 `omnipath-enzsub+tf-YYYYMMDD`），config
`network_release` 跟着改。禁止改 2026-09-16 文件原地加点边。

## 7. 验收（仍不是 biology PASS）

工程验收（adapter + 独立 env）：

- 用 smoke `kstar_evidence_from_sites.tsv` 能在 kstar env 跑通并写出合法 `ptm_activity.tsv`
- `method_version` 不含 `smoke-stub`
- 主环境 `load_ptm_activity_table` + `activities_for_propagation(method="KSTAR")` 通过
- 在 **kinase-containing** 测试网上阶段 3 不把全部 seed 标成 `seeds_without_node`

科学验收（方案 §9.1，另开 run）：

- 预注册的 kinase perturbation phosphoproteome：目标激酶 enrichment 方向与扰动符号一致
- 未过线的分数不得进 formal lineage（比照 GEARS 锚点 fail 的处理）

主线 biology PASS 仍要求真实 PTM **和** 真实 AD 队列契约；KSTAR 单独跑通不等于 PASS。

## 8. 建议执行顺序

1. PTM smoke 契约（阶段 1/3/4/5，activity=stub）→ 已于 2026-09-18 跑通
   `outputs/ptm_activity/20260918_ptm_smoke/`（`kstar_ran=false`，`biology_pass=false`）
2. 建 `kstar` 环境，钉版本，下载并 hash PhosphoSitePlus/KSTAR 资源
3. 实现 adapter：`standardized_ptm` → evidence →（env 内 kstar）→ `ptm_activity.tsv`
4. 冻结 kinase+TF signed network 新 release
5. 真实 PTM 到位后跑 primary KSTAR；PhosR 作 sensitivity 另跑
6. 预注册 kinase benchmark；过线前 `may_enter_lineage=false`

步骤 1 只证明表契约。2–6 在真实 PTM 或至少 KSTAR 环境就绪前不要宣称完成。
