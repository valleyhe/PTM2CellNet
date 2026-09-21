# KSTAR activity 方案（独立环境，主环境只收标准表）

对应主线：`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §5.2 / §7 阶段 2，
命令契约见 [`ptm_activity_pipeline.md`](ptm_activity_pipeline.md) §3。

**状态（2026-09-21）**：独立 conda `kstar`（Python 3.12.14 + `kstar==1.2.0`）已钉包，
PhosphoSitePlus 衍生 companion 文件 hash 已冻结于
`environments/kstar/resource_hashes.json`（`unique_reference_id=a7dfa119…`，
`HumanPhosphoProteome.csv` sha256 `c2ebbcac…`）。Adapter 把阶段 1 标准化表打成
KSTAR 二元 evidence（τ=1.2），并把成对 increased/decreased 结果写成十三列
`ptm_activity.tsv`；`--mode analysis` 强制 `--ensembl-mapping`，kinase symbol 冲突硬失败。
kinase+TF 传播网已另冻为 `omnipath-kinase+tf-2026-09-21`（15,133 条
`kinase_substrate:signaling` + 12,878 条 2026-09-16 `tf_regulation`，合计 28,011 行；
2026-09-16 TF-only 文件 sha256 仍为 `ba500f61…`，未原地改边）。官方 KSTAR ST/Y
network 仍须匹配同一 `unique_reference_id`；缺 network 时 setup/analysis 硬失败，
不伪造资产。`biology_pass=false`，`may_enter_lineage=false`。

PTM smoke 与 `method_version=smoke-stub-20260918` 的 activity 占位表仍只是工程 fixture，
不是 KSTAR，也不是 biology PASS。

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

## 2. 为什么正式 KSTAR 仍未完成

| 缺口 | 事实 | 正式运行前必须 |
|---|---|---|
| 真实 PTM | `ptm_cohort: PENDING_EXTERNAL_PTM_COHORT` | 真实 phosphosite 定量，donor/condition 可审计 |
| 工具环境 | 独立 `kstar` 环境 Python 3.12.14 + 钉包 + RESOURCE_FILES hash 已冻结；ST/Y network 须匹配同一 `unique_reference_id` | 用显式 `--network-dir` 提供通过 hash/`unique_reference_id` 校验的官方网络 |
| 先验网络 | KSTAR 1.2.0 PhosphoSitePlus 衍生 companion 文件 hash 已写入 `environments/kstar/resource_hashes.json` | 保持该 pin；禁止用活下载覆盖而不改 manifest |
| 传播网络 | `omnipath-kinase+tf-2026-09-21` 已另冻（kinase signaling + 2026-09-16 TF）；TF-only 文件未改 | 正式传播须显式改 `network_release`；禁止把 TF-only freeze 当 kinase 网 |
| Benchmark | 方案 §9.1 要求已知 kinase perturbation 评估 | 预注册 1 个公开 kinase inhibitor/KO phosphoproteome，过线前进 lineage |

PTM smoke（`scripts/generate_ptm_smoke.py`）只证明阶段 1/3/4/5 的表契约。
它同时写出 `kstar_evidence_from_sites.tsv`，作为 adapter 的**输入形状**，不是 KSTAR 输出。

## 3. 独立环境与已落地入口

正式环境名为 `kstar`，与 PerturbGen 环境分离。当前环境声明为：

```text
python=3.12.14
kstar==1.2.0   # Naegle lab, Nat Commun 2022 https://doi.org/10.1038/s41467-022-32017-5
pandas==3.0.6 numpy==2.5.3 scipy==1.18.1
# PhosphoSitePlus 衍生资源：kstar 1.2.0 RESOURCE_FILES（ProteomeScout 2020-02-26 / KinPred）
# hash 见 environments/kstar/resource_hashes.json，setup 与 CLI 都校验
```

安装、资源路径、包版本写入 `environments/kstar/environment.yml` 与
`environments/kstar/requirements-lock.txt`，**不**进入 `requirements-core.txt` /
`setup.py`。`scripts/setup_kstar_env.sh` 使用
`/home/scu/anaconda3/envs/kstar/bin/python` 做版本、`sys.prefix`、解释器、钉包和
PhosphoSitePlus 衍生文件 hash 校验；缺少与 `unique_reference_id` 一致的 ST/Y
network 时硬失败，不静默跳过或伪造 network。HTTP 202 / HTML / 过小响应都不是
network 资产。setup 会在 `Path(kstar.__file__).parent` 下复用或校验
`NETWORKS/`（以及同目录的 `NETWORKS.tar.gz`）；不会把 `CONDA_PREFIX` 根目录当成
network 安装目录，也不会覆盖已有网络资产。

主环境不 import KSTAR。`scripts/run_kstar_activity.py` 只有在隔离环境校验通过后才
import KSTAR；`mapping` 只写 mapping handover，`analysis` 才分别运行 increased /
decreased 官方 analysis 并写出标准 `ptm_activity.tsv`。两个 CLI 模式都要求在 `kstar`
环境中运行并显式提供存在的 `--network-dir`。2026-09-21 修复后：analysis 模式不再要求
两方向 network metrics 全表相等（TD-01）——每个 kinase 按胜出方向（较小 p 值）绑定该方向
自己的 `n_substrates`/`network_coverage`，两方向 metrics 分别落盘
`<name>_increased_network_metrics.tsv` / `<name>_decreased_network_metrics.tsv` 并写入
manifest（schema `ptm2cellnet.kstar-adapter-manifest/v2`）。`--min-donors-per-state`
（默认 1）为正式运行提供 donor 下限硬门禁；输入 manifest 必须是结构化 stage-1 manifest
（schema/source.sha256/standardization），空 JSON 不再通过。

PhosR（sensitivity）走 **另一个 R 环境**，同样只交 `ptm_activity.tsv`；primary 与
sensitivity **分两次阶段 3，禁止平均**（方案 §4.2）。

### 3.1 KSTAR 1.2.0 输出交接的已验证修复

`scripts/run_kstar_activity.py` 在每次官方 `kstar.calculate.run_kstar_analysis` 返回后，严格检查
`activities_mann_whitney` 与 `fpr_mann_whitney`：非预期索引（含两张结果索引不一致）、缺少 directional
data 列、空表或重复 kinase 都硬失败。空索引名只规范为 `KSTAR_KINASE`，随后调用官方 `save_kstar` 重写 TSV，再由
`from_kstar` 重载。根因是 KSTAR 1.2.0 新 `run_kstar` 路径返回未命名索引；官方 `save_kstar` 因此写出
空首列表头，而 `from_kstar` 硬编码 `index_col=KSTAR_KINASE`。

post-fix 临时 synthetic smoke 已确认两路结果首行分别为 `KSTAR_KINASE` 加 directional data 列，CLI
内部两次 `from_kstar` 均成功；但 test-only synthetic 网络两路 p 值没有方向性激酶证据，CLI 按预期以
`paired KSTAR outputs contain no directional kinase evidence` 退出 1，未生成 `ptm_activity.tsv`。
这只验证了 handover 修复，不是完整 CLI 十三列表 smoke，也不是正式 KSTAR activity 或 biology PASS；
不添加 fallback。

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
| `activity_unit` | `signed_-log10_kstar_pvalue` |
| `regulator_id` | **必须与阶段 3 网络 `source_id` 同一 identifier 空间**（见 §6） |
| `regulator_type` | `kinase` |
| `condition_or_contrast` | 冻结 `config.contrast`（`disease-minus-normal`） |
| `activity_score` | 配对方向中较小 p-value 的 `-log10(p)`；increased 为正、decreased 为负；0 分禁止传播（现有契约） |
| `activity_direction` | 配对 evidence 的 `increased` → `up`、`decreased` → `down`，与 score 符号一致 |
| `activity_pvalue` | 选中的 increased/decreased 原始 p-value |
| `activity_qvalue` | 将两路 directional p-value 合并后统一做 Benjamini–Hochberg 的 q-value |
| `n_substrates` | 该激酶在本次 evidence 中命中的注释底物数 |
| `network_coverage` | 命中底物 / 该激酶注释底物总数，∈[0,1] |
| `method` | `KSTAR` |
| `method_version` | **真实 KSTAR 1.2.x 包版本**（当前为 `1.2.0`），禁止 `smoke-stub-*` |
| `input_manifest` | 阶段 1 `ptm_input_manifest.json` 路径 |

每行唯一键：`(regulator_id, condition_or_contrast, method)`。

Adapter 写出 `ptm_activity_manifest.json`：KSTAR 版本、τ、site 丢弃计数、对比轴和
方向分析语义；其 `lineage` 固定为 `biology_pass=false`、`may_enter_lineage=false`
（直到 §7 benchmark 过线且真实 PTM/AD 前置条件满足）。

adapter 的 signed score 来自成对的 increased/decreased KSTAR p-value 比较：较小的
p-value 决定方向，不能对单一路结果做全局取反。`activity_qvalue` 是两路 p-value
合并后再做 BH，不是全局取反或由 score 符号推导。

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

`scripts/build_kinase_signed_edges.py` 从 OmniPath `omnipath+kinaseextra` 导出有符号
中间边（`edge_type=kinase_substrate:signaling`），再用
`scripts/build_kinase_tf_network_release.py` 与 2026-09-16 TF-only 表组装新
release。2026-09-16 文件只读，禁止原地加点边。当前冻结：

- kinase 表：`data/processed/signed_network/signed_network_omnipath_kinase_20260921.tsv`
  （15,133 边，release=`omnipath-kinase-2026-09-21`）
- 组合表：`data/processed/signed_network/signed_network_omnipath_kinase_tf_20260921.tsv`
  （28,011 边，release=`omnipath-kinase+tf-2026-09-21`）
- 追踪摘要：`data/manifests/kstar_signed_network_release_20260921.json`

这是工程冻结，`biology_pass=false`。正式 PTM 传播仍须把研究 config 的
`network_release` 显式改到该新 id，不得把 TF-only freeze 当成 kinase 网。

`regulator_id` 与 `source_id` 对齐规则（冻结一条，不要混用）：

- **推荐**：双方都用 canonical Ensembl（与 scVI/DEG 一致）
- 若 KSTAR 只出 symbol：经 `data/processed/gene_symbol_ensembl_map.tsv` 显式 map，冲突硬失败

新网的 `release` 必须是新 id（例如 `omnipath-enzsub+tf-YYYYMMDD`），config
`network_release` 跟着改。禁止改 2026-09-16 文件原地加点边。

## 7. 验收（仍不是 biology PASS）

工程验收（adapter + 独立 env）：

- 独立解释器校验通过；PhosphoSitePlus 衍生资源 hash 与钉包版本校验通过
- adapter 的二元 evidence、paired directional output 与 Ensembl 映射单元契约通过
- 官方 `ExperimentMapper` mapping smoke 已通过，`mapped_rows=7`；匹配冻结
  `unique_reference_id` 的 ST/Y 官方网络（各 50 个非空网络文件，Unique Network ID 已 pin）
  已在本机装齐并通过严格 verifier（2026-09-21）
- `method_version` 不含 `smoke-stub`
- 主环境 `load_ptm_activity_table` + `select_activities_for_propagation(method="KSTAR",
  policy=<frozen activity_admission>)` 通过（2026-09-21 起旧 `activities_for_propagation`
  已删除，传播前必经 admission gate）
- analysis 模式两方向 metrics 允许合理不同（TD-01 修复）；真实 PTM 双方向 full analysis
  仍待真实 cohort
- `omnipath-kinase+tf-2026-09-21` 上 GSK3B（`ENSG00000082701`）不再是 `seeds_without_node`
  （工程连通，不是 biology PASS）

科学验收（方案 §9.1，另开 run）：

- 预注册的 kinase perturbation phosphoproteome：目标激酶 enrichment 方向与扰动符号一致
- 未过线的分数不得进 formal lineage（比照 GEARS 锚点 fail 的处理）

主线 biology PASS 仍要求真实 PTM **和** 真实 AD 队列契约；KSTAR 单独跑通不等于 PASS。

## 8. 建议执行顺序

1. PTM smoke 契约（阶段 1/3/4/5，activity=stub）→ 已于 2026-09-18 跑通
   `outputs/ptm_activity/20260918_ptm_smoke/`（`kstar_ran=false`，`biology_pass=false`）
2. 独立 `kstar` 环境钉包 + PhosphoSitePlus 衍生资源 hash → 2026-09-21 已冻结
3. Adapter：standardized_ptm → 二元 evidence → 成对 KSTAR → 十三列 `ptm_activity.tsv`
   （Ensembl `regulator_id`）；mapping smoke 已通过；官方 ST/Y network 已装齐并通过
   verifier（2026-09-21），runner 双方向 metrics 语义已修复，待真实 PTM 执行 full analysis
4. kinase+TF signed network 新 release `omnipath-kinase+tf-2026-09-21` → 2026-09-21 已冻结
5. 真实 PTM 到位后跑 primary KSTAR；PhosR 作 sensitivity 另跑
6. 预注册 kinase benchmark；过线前 `may_enter_lineage=false`

步骤 1–4 只证明工程契约和资产冻结。正式 KSTAR analysis、kinase→TF 生物学结论和
biology PASS 仍需真实 PTM、匹配 `unique_reference_id` 的 KSTAR network、benchmark
及真实 AD 队列前置条件；不得把 smoke、mapping、hash pin 或 HTTP 202 当成正式分析。
