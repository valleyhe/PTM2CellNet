# PTM2CellNet 修复执行报告（2026-09-16）：U2–U7 收口批次

> 执行依据：[`project_analysis_20260916.md`](project_analysis_20260916.md) §7「技术债与解决策略」（§7.2 债务清单 U1–U8、§7.3 严重债务策略、§7.4 高等级策略、§7.5 中等级策略）。
>
> 事实边界：代码、测试、配置与运行产物是实现事实；smoke/synthetic/工程 context 不等于正式生物学 PASS，正式 biology PASS 维持 **0**。本报告区分 U2–U7/T1–T6 历史批次结果与本轮收口命令结果。

## 目录

- [1. 修复汇总（按严重程度分类）](#1-修复汇总按严重程度分类)
- [2. 每轮迭代执行情况与结果](#2-每轮迭代执行情况与结果)
- [3. 系统性复核：E2E 训练与推理技术要求](#3-系统性复核e2e-训练与推理技术要求)
- [4. 未解决问题与后续解决策略](#4-未解决问题与后续解决策略)
- [5. 验证矩阵](#5-验证矩阵)
- [6. 本轮产出清单](#6-本轮产出清单)
- [7. 引用索引](#7-引用索引)

## 1. 修复汇总（按严重程度分类）

严重程度沿用 [`project_analysis_20260916.md` §7.1](project_analysis_20260916.md) 的判定标准。

### 1.1 严重级（阻塞性）

| 编号 | 债务（§7.2） | 本轮处置 | 结果 | 证据 |
|---|---|---|---|---|
| U2 | GSE → scVI context、缺基因 policy、`davf_batch` binding 未冻结（§7.3-U2 方案 A） | 新增唯一准备入口 `src/data/scvi_context.py` + `scripts/prepare_scvi_context.py`：`missing_gene_policy`（zero_fill/fail）、`davf_batch`（硬校验 ∈ route scVI batch registry）、`batch_binding_rationale`（必填非空，逐字进 manifest）三项显式合同化；真实生成 KO route 正式 context 并经 E2E 同款加载路径（`_load_davf_config → DAVFInferenceModule → load_scvi_adapter → encode`）验证 | **已解决（KO 路径）**：61,472 cells × 4018 基因、33 缺失轴基因显式零填、54,691 队列外基因丢弃计数、encode 输出 (256,64) 全 finite。KD route 无候选消费者（见 U3），不强行绑定 55-batch 决策 | [`gse174367_ko_context.h5ad`](outputs/ptm_activity/20260916_d1/scvi_context/gse174367_ko_context.h5ad) + [manifest](outputs/ptm_activity/20260916_d1/scvi_context/gse174367_ko_context.manifest.json)；单测 9 项 |
| U3 | DAVF gene axis 不覆盖 4/5 AD 候选（§7.3-U3 方案 A/B） | 数据可及性审计 `scripts/audit_davf_axis_coverage.py` 扫描本地 30 个 scPerturb 数据集 + PerturbGen vocab + 双 route 轴 | **决策支撑落地，方案 A 判定为本地不可行**：APP/PSEN1/BACE1/MAPT 在 PerturbGen vocab 全覆盖（ENSG 直查）但在本地 30 个数据集**从未被扰动**（obs target 全空）——Workflow B 重训无训练输入；per-route 收缩结论 KO={APOE}、KD=∅。另发现 4 个 h5ad 下载截断损坏（GasperiniShendure2019_atscale、LaraAstiasoHuntly2023_invivo、NadigOConner2024_hepg2、SunshineHein2023），显式记入 manifest，现有 KO/KD 资产（Dixit jurkat 完好）不受影响 | [audit TSV](outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv) + [manifest](outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.manifest.json)（`axis_covered_per_route: {ko: [APOE], kd: []}`） |
| U4 | donor DEG 统计口径无显著行（§7.4-U4，高级） | `build_ad_deg_tables` 增加显式 `donor_aggregation` estimand（`per_cell_log2_mean` 旧 / `pseudobulk_counts` 新，config 字段 `deg_donor_aggregation` + manifest 记录）；冻结新口径并重算真实表 | **口径已实现并重算，但"产出可过阈值输入"的目标被证伪**：pseudobulk 重算 117,352 行 min FDR=0.9154、FDR≤0.05 行数 0；进一步功效诊断（logCPM pseudobulk min p=1.56e-5；raw-counts t=1.9e-4；表达过滤 universe 16k–19k 后 min BH FDR 0.28–0.94）确认 **GSE174367（EX 7 vs 11 between_donor，donor 文库 15.6 万–2374 万 reads 异质）在任何无偏口径下 FDR≤0.05 不可达**。上轮 lessons L-2026-0915-02 "pseudobulk min p≈1e-13"不可复现，已在 lessons L-2026-0916-01 修正——observed gate 阻塞是队列统计功效限制，不是代码缺陷，不得调阈值掩盖 | [pseudobulk DEG 表](outputs/ptm_activity/20260916_d1/ad_deg_aggregate.tsv) + [manifest](outputs/ptm_activity/20260916_d1/ad_deg_manifest.json)（`donor_aggregation: pseudobulk_counts`）；单测 +6 |
| U5 | source proposal 外部输入（§7.3-U5） | 边界核实：消费方 `load_source_proposals`（`scripts/build_celltype_candidate_specs.py:80`）校验 `proposed_direction ∈ {up,down}`，集成测试覆盖 | **维持外部依赖（按设计）**：方向假设是不可自动生成的研究输入（方案 §5.5；bridge §3），本轮不发明方向、不捏造 proposal 表 | [`tests/integration/test_ptm_activity_pipeline.py`](tests/integration/test_ptm_activity_pipeline.py) |
| U6 | 正式六阶段/null/统计闭环未执行（§7.3-U6） | 工程链 preflight：PerturbGen 独立环境验证（`/home/scu/anaconda3/envs/perturbgen`，torch 2.5.1+cu124 CUDA 可用）；750-cell 真实三阶段探针 | **工程执行路径打通，formal 仍被候选 gate 阻塞**：探针 tokenise/train_mask/train_decoder 8.7s/43.0s/37.4s 全部成功（eager 模式，VRAM 峰值 4,359 MiB @99% util）；正式执行的前置仍是 pass invocation（被 U4 observed gate 与 U5 外部 proposal 阻塞） | [d3_probe stage manifests](outputs/perturbgen/d3_probe_20260916/) |

### 1.2 高等级

| 编号 | 债务 | 本轮处置 | 结果 |
|---|---|---|---|
| U7 | Workflow B Gate-E ≥200 条真实 benchmark 缺失（§7.4-U7） | `src/analysis/gate_e_benchmark.py` + CLI：从**真实数据库注释**组装——CPLM `Homo sapiens.txt`（自带 gene symbol 列，human 过滤）覆盖非磷酸化类型，dbptm `Phosphorylation`（UniProt acc→symbol 经 CPLM 自带 24,327 个零冲突配对映射，映射率 90.4%），symbol→ENSG 经 PerturbGen ensembl mapping（173,697 条）；唯一 (gene, ptm_type) 去重、不抽样、全量保留 | **已解决（benchmark 数据部分）**：60,030 行 / 25 种 PTM 类型（phosphorylation 14,286、ubiquitination 11,975、acetylation 9,443、sumoylation 7,505，覆盖 DAVF 全部三个方向 code）；Gate-E evaluator 端到端消费验证通过（validate ≥200 ✓；identity 对照 coverage=0.9874、collisions=0、action_agreement=1.0）。LatentDAVF 重训与 held-out 验收仍受 U3 无训练输入限制。附带发现：本机 `data/raw/uniprot/human_proteome.fasta` 等实为酵母数据、`human_idmapping.gz` 无 Ensembl 行——均不可用，已在 lessons 记录 |

### 1.3 中等级（D1–D4）

| 编号 | 债务 | 本轮处置 | 结果 |
|---|---|---|---|
| D1 | 文档指针与 FDR 事实漂移（§7.5-D1） | `docs/CURRENT_STATUS.md` 权威指针 → `project_analysis_20260916.md`（20260915 作为上一份证据、注明 §2.3 FDR 文字校正）；Quick Reference 首条写入本批次事实；文档入口链接更新；`docs/guides/ptm_activity_pipeline.md` 增补 §7.1 context 准备命令与 `deg_donor_aggregation` 字段 | 已完成 |
| D2 | runner 与 formal invocation 边界未下沉（§7.5-D2 方案 A/B） | 调用者审计：src 内 runner 调用方为 `run_perturbgen_pipeline.py`（CLI wrapper，选 perturb/--path 时强制 gate report）、`run_davf_perturbgen_e2e.py`（经 orchestrator gate）、`null_generation.py`（合法内部 null 用途）；runner docstring F-09 已把"runner=工程执行层、formal 唯一入口=invocation wrapper"写成显式契约 | **采纳方案 B 维持现设计**：所有直接调用均为合法工程用途，无绕过 wrapper 的真实用例；在 runner 内复制 gate 校验属重复防御（违反仓库 scope 约束），不采纳 |
| D3 | VRAM 40GB 估计未实测（§7.5-D3 方案 A） | 真实 750-cell 三阶段探针 + 5s 间隔 VRAM 采样 | **实测替换估计并发现真根因**：第一阻塞不是显存而是 `torch.compile`（trainer `compile_model=True`，triton 需 CC≥7.0，P40 为 6.1）——**`TORCHDYNAMO_DISABLE=1` eager 是 P40 唯一执行路径**；smoke 载荷峰值 4,359 MiB（≪24GB），formal 载荷峰值待正式输入复测 |
| D4 | 正式资产验收未进入可重复验证层（§7.5-D4 方案 B） | 环境证据已固化（`outputs/perturbgen/env_evidence_20260823.json`：P40/24576MiB/torch 2.5.1+cu124/全部离线 import OK）；本报告与 CURRENT_STATUS 均维持"契约具备 ≠ 生物学具备"措辞 | 已按方案 B 固化；真实 asset lane（方案 A）仍依赖外部数据与 GPU 排期 |

### 1.4 低等级

U8 已在 T5 完成：bridge guide 的 §2.1 登记了 22 个 formal `PTM2CELLNET_PERTURBGEN_*` 变量、9 个 local adaptation 变量及 P40 的 `TORCHDYNAMO_DISABLE=1` eager 要求；`/tmp/d3_env.sh` 仍是未入库的临时文件，不作为交付资产。剩余低等级问题不阻塞当前 U2–U7 结论。

## 2. 每轮迭代执行情况与结果

| 轮次 | 动作 | 结果 |
|---|---|---|
| 1 侦察 | 通读 `project_analysis_20260916.md` §7 债务清单；核对 ad_deg_table/scvi_adapter/runner/gate_e/davf_scperturb/config 现状；确认执行依据为今日 02:22 生成的 20260916 审计报告 | 明确 U2–U7 + D1–D4 的落地方案与边界 |
| 2 U4 实现 | `_donor_pseudobulk_log2`（gse_normal_disease.py）+ `donor_aggregation` 参数（ad_deg_table.py）+ config 字段 + CLI 传递 + 冻结 `pseudobulk_counts` + 单测 | 定向 12 passed；config 21 passed |
| 3 U4 重算 | 真实 GSE174367 重算 DEG 表 | **结果与文档预期相反**：min FDR 0.9154、0 显著行 |
| 4 U4 根因诊断 | 对照实验：raw-counts t / logCPM pseudobulk t / per-cell 口径 / 表达过滤 universe（K=3/6/11）×（EX/INH） | 定位：任何无偏口径不可达 FDR≤0.05；上轮 1e-13 记录作废（lessons 修正） |
| 5 U2 实现 | `scvi_context.py` 模块 + CLI + 9 项单测（fake adapter）；读真实 checkpoint registry 验证 batch 校验可行（KO 2 batch、KD 55 batch） | 单测全绿 |
| 6 U2 真实生成 | KO context 真实生成（后台 gzip）+ 契约验证 + E2E 同款路径 encode 复验 | 61,472×4018 全契约通过；encode (256,64) finite；KD 判定为无消费者不生成 |
| 7 U3 审计 | 审计脚本 + 全 30 数据集扫描 | 首跑撞上 4 个截断 h5ad（OSError）→ 改为显式记录不可读文件后重跑；per-route 结论 KO={APOE}、KD=∅、4 候选零扰动数据 |
| 8 U7 benchmark | 映射链勘察（uniprot 辅助文件为酵母、idmapping 无 Ensembl → 弃用；CPLM 自带 symbol；dbptm phosph 90.4% 可映射）→ 模块 + CLI + 5 单测 → 真实组装 → evaluator 端到端验证 | 60,030 行；identity 对照 coverage 0.9874/collisions 0/agreement 1.0 |
| 9 U6/D3 探针 | 重建环境变量 → 三跑：首跑缺 `PYTHON` 变量 → 二跑 train_mask 被 triton CC 硬阻塞 → 三跑（清 failed stage + `TORCHDYNAMO_DISABLE=1` + `--resume`） | 三阶段全部成功；VRAM 峰值 4,359 MiB；resume 契约对 failed stage 的拒绝行为验证正确 |
| 10 D1/U5/指南 | CURRENT_STATUS 三处指针 + 本批次 Quick Reference；管线指南 §7.1 新命令；U5 接口核实 | 完成 |
| 11 验证 | U2–U7/T1–T6 全量结果沿用各批次历史记录；本轮按收口要求重新执行 compileall、50 项定向 pytest、ruff、format、mypy 和 requirements consistency | 历史：U2–U7 为 2818 passed，T1–T6 为 2819 passed；本轮定向结果见 §5，为 50 passed / 26 warnings |
| 12 收口 | 当前代码批次已提交；本报告与综合报告待单独提交并在提交后复核 SHA/状态 | 归档不扩张；biology PASS 仍为 0 |

## 3. 系统性复核：E2E 训练与推理技术要求

按主线链路逐层复核（要求出处：[E2E guide](docs/guides/davf_perturbgen_e2e.md)、[bridge guide §4–§5](docs/guides/perturbgen_bridge.md)、[方案 §4/§7](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md)）。

| 层 | 技术要求 | 本轮后状态 | 满足？ |
|---|---|---|---|
| 上游 PTM 输入（U1，范围外） | 真实 PTM 定量 / KSTAR/PhosR activity / signed network release | signed network 已绑定 `omnipath-2026-09-16`；PTM 定量与 KSTAR/PhosR activity 仍显式 PENDING | ✗ 外部依赖 |
| AD DEG 表（U4） | donor-level estimand + Welch/BH + between_donor ≥3 donor | pseudobulk estimand 冻结并重算；**0 显著行（队列功效限制）** | 代码 ✓ / 数据 ✗ |
| source proposal（U5） | 外部审阅的方向假设 TSV | 消费接口与校验完备 | ✗ 外部输入 |
| scVI context（U2） | 4018 轴一致 + `davf_batch` + counts 层 + ensembl_id var | 唯一入口 + KO 真实 context + encode 验证 | ✓（KO；KD 待候选） |
| DAVF 方向解码 | route checkpoint + alias 轴 + token/decoder 分离 | APOE KO 链上轮 dry-run 已验证 finite decode；本轮 context encode 复验通过 | ✓（仅 APOE） |
| 三方方向 gate | proposal + DAVF + observed 三方同号且 observed FDR≤0.05 | 工程链完整；**observed FDR 在本队列数学上不可达** | 工程 ✓ / 科学 ✗ |
| Gate-0/invocation | 真实队列 raw counts/donor/pairing + pass-only invocation | 已具备（上轮 dry-run 正确拦截） | ✓ |
| 六阶段执行（U6） | 外部 PerturbGen env + gate-bound CLI + shared prepare | 环境 + 三阶段真实探针成功（eager）；formal 被候选 gate 阻塞 | 工程 ✓ / formal ✗ |
| 统计验收链 | matched-null ≥99 + quality + empirical p + BH + dual-path AND + frozen replay | 接口全在（F-01），等 formal 资产 | ✗ 等资产 |
| Workflow B / Gate-E（U7/U3） | ≥200 benchmark + 重训 + held-out | **benchmark 60,030 行就绪**；重训本地无训练输入（4 候选零扰动数据） | benchmark ✓ / 重训 ✗ |
| 资源（D3） | GPU 可执行性 | P40 需 `TORCHDYNAMO_DISABLE=1`；smoke 峰值 4.4GB；formal 峰值待复测 | ✓（eager 路径） |

**结论**：E2E 的**工程链**（context 准备 → DAVF decode → 三方 gate → invocation → 六阶段 → 统计组装）在 KO/APOE 路径上全链可执行且契约完备；**正式生物学链**存在两处数据层硬阻塞——(a) observed 三方 gate 的 FDR≤0.05 在 GSE174367 任何无偏统计口径下不可达（U4 实测），(b) 除 APOE 外 4 个 AD 候选既无 DAVF 轴覆盖也无本地扰动训练数据（U3 审计）。二者均不是代码缺陷，属研究设计/数据供给决策，本轮已把决策所需的全部事实证据（审计表、功效诊断、benchmark）产出。

## 4. 未解决问题与后续解决策略

| 编号 | 问题 | 严重度 | 解决策略与实施步骤 | 时间节点（估） |
|---|---|---|---|---|
| U1 | 真实 PTM/activity 输入（signed network release 已就绪） | 严重（范围外） | 按 §7.3-U1 方案 A：研究负责人冻结 PTM/activity 标准表 → 登记 manifest → 仓库 contract 校验并复用 `omnipath-2026-09-16` | 外部依赖 5–10 天 |
| U4-残余 | observed gate 在本队列不可达 | 高 | 三选一（需研究负责人决策，不得调阈值）：① 扩大/更换 donor 队列；② 预注册独立于 DEG 结果的小基因面板（如 PTM 网络 target set，依赖 U1）缩 BH universe；③ 修订 gate 语义为方向一致性+标称 p（需方案 §4.5 修订与审批） | 决策 0.5 天 + 实施 1–2 天 |
| U5 | source proposals | 严重（外部） | 按 §7.3-U5 方案 A：文献/实验审阅后提供 TSV（接口已就绪） | 外部 1.5 天 |
| U3-残余 | 4 候选无 DAVF 轴覆盖 | 严重 | 本地重训已证不可行；需外部 AD 相关 perturb 数据（如 iPSC-neuron CRISPR 队列）或正式收缩研究问题至 APOE；4 个 scPerturb h5ad 已重下并通过完整性验证，但不增加候选扰动覆盖 | 数据获取外部；收缩决策 0.5 天 |
| U6-残余 | formal 六阶段 + null + 统计闭环 | 严重 | U4/U5 解锁后：KO/APOE pass invocation → eager 六阶段（`TORCHDYNAMO_DISABLE=1`，formal 载荷先做 VRAM 复测）→ matched-null 批跑 → `--assemble-statistical-evidence` | 3–5 GPU 天 |
| U7-残余 | LatentDAVF 重训 + held-out Gate-E | 高 | 受 U3-残余数据限制；benchmark 资产已就绪，数据到位即可执行 | 数据依赖 5–7 GPU 天 |
| U8 | 环境变量集中 runbook | 低 | bridge guide §2.1 已增补 22 个 formal 与 9 个 local 变量清单，含 P40 eager 注记 | **已完成（T5）** |

## 5. 验证矩阵（区分历史批次与本轮收口）

| 检查 | 命令/方式 | 结果 |
|---|---|---|
| 本轮定向 pytest | `python -m pytest tests/unit/analysis/test_ad_deg_table.py tests/unit/analysis/test_gate_e_benchmark.py tests/unit/data/test_scvi_context.py tests/unit/analysis/test_ptm_research_config.py tests/integration/test_scvi_davf_connection.py --timeout=300 -q` | **50 passed / 26 warnings / 4.36s / exit 0** |
| U2–U7 批次全量回归（历史记录） | `pytest -m "not slow and not gpu" --timeout=600 -q` | **2818 passed / 0 failed / 22 skipped / 771.67s / exit 0**；非本轮重新运行 |
| T1–T6 批次全量回归（历史记录） | `python -m pytest -m "not slow and not gpu" --timeout=600 -q` | **2819 passed / 0 failed / 22 skipped / 859.27s / exit 0**；非本轮重新运行，含 Frangieh route 测试 |
| 本轮静态检查 | `ruff check src scripts tests` | **All checks passed / exit 0** |
| 本轮格式检查 | `python -m ruff format --check`（18 个 Python 触碰文件） | **18 files already formatted / exit 0** |
| 本轮类型检查 | `python -m mypy src/ --ignore-missing-imports` | **174 source files / 0 errors / exit 0** |
| 本轮编译检查 | `python -m compileall -q src scripts tests` | **exit 0** |
| 本轮依赖一致性 | `python scripts/check_requirements_consistency.py` | **274 lock pins 一致 / exit 0** |
| 本轮 VCS | `git fetch origin main`；`git ls-remote origin refs/heads/main`；`git rev-list --left-right --count HEAD...origin/main` | fetch exit 0；远端 `219b81fe9d77d1f49cd916989e227ca88a2417d7`；代码提交 `40f115e0985a6cdbf9406cc94051ab84334a570b`；当前 `9 0`；无 merge、无 push |
| 本轮空白检查 | `git diff --check` | exit 0 |
| U2 真实验收 | context 契约字段 + E2E 同款 encode | 61,472×4018 ✓ / (256,64) finite ✓ |
| U4 真实验收 | pseudobulk 重算 + 4 组功效对照 | 表产出 ✓；min FDR 0.9154（如实记录） |
| U3 真实验收 | 30 数据集扫描 + vocab 双查 | per-route 结论 + 4 损坏文件显式记录 |
| U7 真实验收 | 组装 + `load_benchmark.validate` + identity migration | 60,030 行 ✓ / evaluator 消费 ✓ |
| D3 真实验收 | 三阶段 GPU 探针 + VRAM 采样 | 全成功 / 峰值 4,359 MiB |

## 6. 本轮产出清单

**代码**：`src/data/scvi_context.py`（新）、`src/analysis/gate_e_benchmark.py`（新）、`scripts/prepare_scvi_context.py`（新）、`scripts/audit_davf_axis_coverage.py`（新）、`scripts/build_gate_e_benchmark.py`（新）；修改 `src/analysis/ad_deg_table.py`、`src/analysis/ptm_research_config.py`、`src/data/gse_normal_disease.py`、`scripts/build_ad_deg_table.py`、`configs/research/ptm_research_config.yaml`。
**测试**：`tests/unit/data/test_scvi_context.py`（新，9 项）、`tests/unit/analysis/test_gate_e_benchmark.py`（新，5 项）、`test_ad_deg_table.py`（+6）、`test_ptm_research_config.py`（+1）。
**真实产物**（`outputs/ptm_activity/20260916_d1/`）：pseudobulk DEG 三件套、KO scVI context + manifest、轴覆盖审计 TSV + manifest、Gate-E benchmark CSV + manifest；`outputs/perturbgen/d3_probe_20260916/`（三阶段 stage manifests）。
**文档**：`docs/CURRENT_STATUS.md`、`docs/guides/ptm_activity_pipeline.md`、`lessons.md`（L-2026-0916-01）、本报告。

## 7. 引用索引

| 引用 | 章节/位置 | 用途 |
|---|---|---|
| [`project_analysis_20260916.md`](project_analysis_20260916.md) | §7.1 严重度标准 / §7.2 债务清单 U1–U8 / §7.3-U2、U3、U5、U6 / §7.4-U4、U7 / §7.5-D1–D4 / §2.3 FDR 校正 / §9 后续顺序 | 本轮执行依据 |
| [`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md) | §4.5 AD DEG 输入 / §5.5 proposal 边界 / §6.4 Gate-E 触发条件 / §8.7 阈值隔离 | 研究契约 |
| [`docs/guides/ptm_activity_pipeline.md`](docs/guides/ptm_activity_pipeline.md) | §1 阶段 0（新增 `deg_donor_aggregation`）/ §7.1（新增 context 准备命令） | 命令指南 |
| [`docs/guides/perturbgen_bridge.md`](docs/guides/perturbgen_bridge.md) | §3 步骤 1（proposal 不可自动生成）/ §4 六阶段 / §5 Workflow B/Gate-E | 执行边界 |
| [`project_analysis_20260915.md`](project_analysis_20260915.md) | §2.3 所引 FDR 校正 / §3 三项发现 | 上轮证据基线 |
| `lessons.md` | L-2026-0915-02（三项发现，其中发现三的 1e-13 已由 L-2026-0916-01 修正）/ L-2026-0916-01 | 决策记录 |

---

# 追加批次：T1–T6（PerturbGen 之外主线推进，2026-09-16 下午）

按用户指令"依次完成任务 1–6"执行（任务定义见本轮对话；1=扩充 AD 队列、2=signed network、3=Frangieh KO 链、4=重下损坏 h5ad、5=U8 runbook、6=R-01~03 核实）。

## 8. T1–T6 执行结果

| 任务 | 结果 | 关键证据 |
|---|---|---|
| T1 扩充 AD 队列 | **标准化与合并落地；observed gate 在本地两队列确认不可达（结论固化）**：GSE157827 标准化（177,654 nuclei × 33,538 ENSG；12 AD + 9 control donors；7 cell type 数据驱动注释全记 provenance）；与 GSE174367 合并（33,427 基因 × 239,126 cells）。三种统计口径实测：157827 单队列 min FDR **0.1365**（HES4）、朴素池化 min FDR **1.0**（队列基线偏移稀释效应）、per-cohort centering（16N+23D=39 donors）EX **0.51** / INH **0.75**——全部无 FDR≤0.05 行。出路（研究决策）：第三/更大队列，或 gate 语义修订 | [`GSE157827_ad_cohort.h5ad`](data/AD/standardized/GSE157827_ad_cohort.h5ad) + provenance；[合并队列](data/AD/standardized/GSE174367_GSE157827_combined.h5ad)；[157827 DEG](outputs/ptm_activity/20260916_t1/gse157827_only/)；[合并 DEG](outputs/ptm_activity/20260916_t1/) |
| T2 signed network release | **完成**：OmniPath REST（`datasets=dorothea,tf_target`、`dorothea_levels=A..E`——默认查询只回 1,258 行，必须显式全等级）导出 13,565 行 → 组装 **12,878 条 signed TF→gene 边**（+10,124/−2,754；761 TF → 3,727 target；符号 consensus 优先；confidence=min(1, 0.5+0.1×(n资源−1)) 冻结公式）；`load_signed_network(expected_release='omnipath-2026-09-16')` 验证通过；研究 config `network_release` 已从 PENDING 更新为真实 release——阶段 3 传播的网络输入就绪，PTM 上游只剩 KSTAR/activity 外部依赖 | [`signed_network_omnipath_20260916.tsv`](data/processed/signed_network/signed_network_omnipath_20260916.tsv) + [manifest](data/processed/signed_network/signed_network_omnipath_20260916.manifest.json) |
| T3 Frangieh KO 链重建 | **完成（GPU 全链）**：Frangieh var ensembl 列 5,649 行符号污染修复（492 remap / 5,157 drop）→ Dixit×2+Frangieh prepare（240,646 × 4,018，**216 扰动 target 含 APOE**，原 Dixit-only 链仅 10 个 TF）→ scVI（P40，40 epochs）→ latent pairs（train 43,063 / val 5,306 / test 5,387）→ LatentDAVF（best_epoch=4，early stop 19）。新 4018 轴被 GSE174367 **零缺失**覆盖（原链缺 33）；EX context 绑定最大训练份额 batch（`frangieh_remapped:batch_0`，82%）；**APOE ubiquitination（KO code 0）真实解码 finite 非零**；新正式测试 `test_real_ko_frangieh_route_serves_apoe_ko_direction` 通过（token 12707 ≠ decoder 206，方向 code 0 契约验证）。注意：新链轴与旧 KO 链不同，资产不可混用 | [`davf_ko_frangieh`](checkpoints/davf/davf_ko_frangieh/) + [prepared manifest](data/processed/davf_scperturb/ko_frangieh/prepared.manifest.json) + [context](outputs/ptm_activity/20260916_t1/ex_context_ko_frangieh.h5ad) + [测试](tests/integration/test_scvi_davf_connection.py) |
| T4 重下损坏 h5ad | **完成（4/4，三重验证通过）**。Lara invivo（760,312,556 B）、Nadig hepg2（850,590,740 B）、Sunshine 2023（743,672,579 B）、Gasperini at-scale（1,866,293,566 B）size 与 Zenodo record 13350497 元数据精确一致 + anndata **全量（非 backed）读取**通过（Gasperini 207,324 × 13,135）；四者实际 SHA256 均与本地 `SHA256SUMS.txt`（figshare 基准）不一致但 size 一致——Zenodo 修订版重打包的已知版本差异，实际 hash 与差异逐文件显式记入 verify 文件，不篡改基准。**事故与处置**：多波次后台续传曾对同一 `.new` 并发 append，Gasperini 首次补齐副本交错损坏（size 超期望 + X gzip filter 读失败 + B-tree signature 错，backed 元数据读不可信，必须全量读）；弃损坏副本后单写者全新下载，size 匹配 + 全量读取通过后才原子替换 | [`redownload_verify.json`](data/raw/scperturb/redownload_verify.json)；`data/raw/scperturb/*.h5ad` |
| T5 U8 环境变量 runbook | **完成**：bridge guide 新增 §2.1——22 个 formal `PTM2CELLNET_PERTURBGEN_*` 变量语义 + 本地验证值、9 个 local adaptation 变量、P40 `TORCHDYNAMO_DISABLE=1` eager 注记 | [`perturbgen_bridge.md`](docs/guides/perturbgen_bridge.md) §2.1 |
| T6 R-01～03 核实 | **blocked 确认（如实报告）**：replogle/scgenescope 原始数据本地不存在（raw/ 与 ref/ 全量检索无结果），真实图/扰动训练维持数据供给依赖 | 检索记录见本轮会话 |

## 9. T1–T6 验证

| 检查 | 结果 |
|---|---|
| 全量回归（非 slow/gpu） | **2819 passed / 0 failed / 22 skipped / 859.27s**（+1 新链 DAVF 测试） |
| mypy src/ | 174 files / 0 errors |
| ruff check / format（触碰文件） | 全部通过 |
| 新链 DAVF 端到端 | APOE KO 解码 8/8 evidence `model_source='davf'`、finite 非零 |
| signed network loader | `expected_release` 校验通过 |

## 10. 对主线的影响

1. **observed gate 阻塞的性质最终确认**：不是代码、不是口径、不是单一队列——是本地可用 AD 队列（2 个）的统计功效结构。任何"调参让它显著"的路径都违反冻结语义。
2. **PTM 上游从 2 个外部依赖减为 1 个**：signed network 已就绪，只剩 KSTAR/PhosR activity + PTM 定量队列。
3. **KO route 从 10 个 TF target 扩到 216 个（含 APOE 真实扰动信号）**，为 APOE 候选提供了有训练基础的 DAVF 证据来源（科学验收仍需三方 gate + formal 六阶段）。
