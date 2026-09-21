# PTM2CellNet 代码修复与系统性复核报告（2026-09-15）

> 执行依据：[`project_repair_report_20260915.md`](/home/scu/PTM2CellNet/project_repair_report_20260915.md) §5「D0–D5 收口计划」。
> 本报告为当前权威状态文件；`project_analysis_20260914.md` §7 的口径由本文接续。
> 结论边界：本文所有"通过"均指代码契约与工程验证层，**正式 biology PASS = 0** 维持不变（修复报告 §6 发布口径）。

## 0. 结论先行

1. **两项验证技术债已收口**：全仓 format 债（324 files）一次性修复；全量非 slow/gpu pytest 实际可在 818 秒完成（此前 420s timeout 是外层截断而非挂起），暴露的 3 个失败测试已全部修复，当前全量绿。
2. **D0 可执行部分已完成**：冻结 `ptm_research_config.yaml`（AD 侧真实值、PTM 侧资产绑定字段显式 PENDING）；用真实 GSE174367 生成 donor-level DEG 表（aggregate 117,352 行 + donor-level 1,290,872 行）。
3. **E2E 链路完成真实数据 dry-run 复核**，并得到三项修复报告未明确列出的硬发现：
   - **发现一（阻塞性）**：缺 GSE174367 → scVI-aligned context h5ad 的准备工具与跨数据集 batch 绑定语义（§3.1）。
   - **发现二（阻塞性）**：AD 冻结候选 5 基因中 4 个（APP/PSEN1/BACE1/MAPT）不在当前任何 DAVF route 的 scVI 基因轴中，仅 APOE 在 KO 链（§3.2）。
   - **发现三（高）**：当前 DEG 统计口径下全表 FDR 恒为 1.0（0/117,352 显著），三方 gate 的 observed 方在数学上不可通过（§3.3）。
4. D1–D5 formal GPU 不具备启动条件：除修复报告 §3 已列的外部研究输入缺失外，上述发现一/二/三均须先解决。

## 1. 问题修复汇总（按严重程度）

| # | 问题 | 严重度 | 来源 | 修复方式 | 验证 |
|---|---|---|---|---|---|
| F1 | 全仓 format check 失败（324 files reformat、168 formatted，exit 1） | 中（验证债） | 修复报告 §3「验证债务」、§4 验证矩阵 | 执行 `python -m ruff format src scripts tests`（324 files reformatted） | `ruff format --check`：492 files already formatted、exit 0；`compileall` 通过 |
| F2 | 全量非 slow/gpu pytest 420s timeout 截止、无最终摘要 | 中（验证债） | 修复报告 §3、§4（collected 2820，停在 test_davf_pipeline） | 确认非挂起（该文件单独 87.8s 通过），以充分外层超时完整跑全量 | **817.99s：3 failed / 2795 passed / 22 skipped**，随后 3 个失败逐项修复（F2a–F2c） |
| F2a | `test_formal_cli_rejects_hand_filled_candidate_pvalue` 失败 | 高（测试与新契约脱节） | 本轮全量暴露 | fixture 补齐 formal 必填声明 `evidence_class=empirical_null`、`pvalue_source=empirical_matched_null` 与匹配的 `contract` mapping（[`replay_evaluation.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/replay_evaluation.py:446) 的新校验） | 该文件 8 passed |
| F2b | `test_formal_cli_aggregates_empirical_p_for_kd_mask` 失败 | 高（同上） | 同上 | 同上 | 同上 |
| F2c | `test_real_current_davf_direction_keeps_token_and_decoder_indices_separate` 失败（lessons.md 已知失败） | 高（资产链过时） | 本轮全量暴露；`lessons.md` L-2026-0914-03 记录旧 checkpoint 被正确拒绝 | 测试从 Norman 旧链迁移到 KD 正式链 `davf_kd_nadig`（canonical ENSG）：PTM 类型 phosphorylation→sumoylation（KD code 1）、`resolve_target_gene_indices` 传 ENSG、断言 token 328 / decoder 2260、obs 用 `davf_batch`（[`test_scvi_davf_connection.py`](/home/scu/PTM2CellNet/tests/integration/test_scvi_davf_connection.py:21)） | 该文件 2 passed（含真实 GPU decode） |
| F3 | 无冻结 `ptm_research_config.yaml`（D0 前置） | 高（主线阻塞） | 修复报告 §5 D0；方案 §7 阶段 0 | 冻结 [`configs/research/ptm_research_config.yaml`](/home/scu/PTM2CellNet/configs/research/ptm_research_config.yaml)：AD 侧全部真实值（EX/INH、between_donor、GSE174367 绝对路径、7/11 donor 覆盖实测）；PTM 侧 `network_release`/`ptm_cohort` 显式 `PENDING_*` 标记，注释声明 PENDING 状态下运行阶段 1–3 视为契约违规 | `load_ptm_research_config` 校验通过 |
| F4 | 缺 donor-level AD DEG 表（阶段 4/5 输入） | 高（主线阻塞） | 修复报告 §3「研究输入尚未齐备」 | 用冻结 config + 真实 cohort 运行 `build_ad_deg_table.py`，产出 `outputs/ptm_activity/20260915_d0/`（aggregate/donor-level/manifest） | 表结构与方案 §4.5 八列契约一致；manifest 绑定 h5ad 绝对路径与 donor 计数 |

未修复项及其理由见 §4；三项新发现（不属"修复"而属"研究设计决策"）见 §3。

## 2. 每轮迭代执行情况

| 轮次 | 动作 | 结果 |
|---|---|---|
| R1 环境与事实核对 | `zmemory resume`/claim；读取修复报告、指南、执行方案；盘点 data/、outputs/、checkpoints/、conda 环境 | 确认真实资产在位：GSE174367 h5ad、冻结 cohort manifest（EX、90 runs、donor split 12+6）、embedding asset/vocab、DAVF KO/KD checkpoint、scVI、PerturbGen 独立环境（perturbgen 0.1.0 + torch 2.5.1）；`data/external` 为空、无 KSTAR/PhosR 环境 |
| R2 format 收口 | `ruff format`（F1）+ `ruff check` + `compileall` | 全绿 |
| R3 全量测试 | 后台完整跑全量（F2），复现并修复 3 个失败（F2a–F2c） | 全量 2798 passed（2795+3 修复）/ 22 skipped；改动文件单独复跑通过 |
| R4 D0 执行 | 冻结 config（F3）；生成 DEG 表（F4）；核对 cell type donor 覆盖（每型 11 disease + 7 normal） | D0 的 AD 侧闭环；PTM 侧确认缺口（§4） |
| R5 E2E dry-run 复核 | 构造显式工程 context + APOE 单候选 spec，逐层通过 Gate-0 → DAVF 推理 → 三方 gate → invocation 决策 | 全链路跑通；得到 §3 三项硬发现 |
| R6 复核根因分析 | 重现 DEG p 值分布（两种聚合口径对比）；核对候选基因在 KO/KD alias 的覆盖 | 定位 §3.2/§3.3 根因 |
| R7 收尾 | mypy（172 files 0 errors）、requirements（274 pins OK）、`lessons.md` 追加 L-2026-0915-02、ZMemory 事件 | 全部通过 |

## 3. 系统性复核分析：E2E 训练与推理技术要求

### 3.0 已确认满足的部分

| 技术要求 | 证据 |
|---|---|
| 真实 cohort + provenance + Gate-0 证据 | `data/AD/standardized/GSE174367_ad_cohort.h5ad`（61,472×58,676，int32 counts，`layers['counts']`）；`outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json` |
| 冻结 cohort manifest + donor split + acceptance plan（5 候选 × 2 path × 3 seed × 3 mode = 90 runs） | `outputs/perturbgen/frozen/20260914_gse174367_ex/`（train 12 + held-out 6，两性均衡） |
| DAVF KO/KD checkpoint + scVI + alias + embedding asset | `checkpoints/davf/davf_ko_dixit`、`davf_kd_nadig`（72MB，canonical ENSG 4018）；`checkpoints/scvi/*`；`embedding_asset_20260822`（18967 vocab） |
| PerturbGen 外部环境与资产 | conda env `perturbgen`（0.1.0/torch 2.5.1/scanpy 1.11.2）；`ref/Perturbgen-src`（commit a9a9375）；encoder ckpt 574MB；token/gene-mapping/median pkl 齐 |
| GPU | Tesla P40 24GB，driver 580.178.04，CUDA 13，torch cuda available（本轮沙箱内实测可见） |
| E2E 工程链路（spec schema → Gate-0 → DAVF decode → 三方 gate → invocation 决策 → donor split 绑定） | 本轮 dry-run 真实跑通：`outputs/ptm_activity/20260915_d0/e2e_dryrun/e2e_report.json`，DAVF 真实 decode delta finite（APOE predicted_delta 0.00114），gate 正确因 `observed_expression_not_significant` 返回 inconclusive 并拒绝创建 invocation |
| donor-level DEG 表 | 本轮生成（F4） |

### 3.1 发现一（阻塞性）：scVI-aligned context 准备缺口

E2E 的 DAVF 推理把 `context_h5ad` 直接行子集后传给 scVI adapter（[`run_davf_perturbgen_e2e.py:829-833`](/home/scu/PTM2CellNet/scripts/run_davf_perturbgen_e2e.py:829)），契约要求其基因轴与 scVI 4018 基因**完全一致**并含 `davf_batch` 列（[`davf_perturbgen_e2e.md` §候选清单](/home/scu/PTM2CellNet/docs/guides/davf_perturbgen_e2e.md:63)、[`davf_ko_kd_training.md`](/home/scu/PTM2CellNet/docs/guides/davf_ko_kd_training.md:303)）。原始 GSE174367（58,676 基因、无 `davf_batch`）直接传入会硬失败（本轮实测 `Expected: 4018 Received: 58676`）。

具体缺口：
- 仓库**没有**任何工具/脚本/文档定义"GSE174367 → scVI-aligned context"这一步；
- 基因缺失按 route 不同：KO 链 scVI 4018 基因中 **33 个不在 GSE174367**（3985/4018 交集），KD 链仅缺 **1 个**（4017/4018）；缺失基因的处理（填零/剔除/重训）未定义；
- `davf_batch` 跨数据集绑定语义未冻结：KO scVI 只有 2 个 batch（DixitRegev2016 K562），KD scVI 是 55 个 jurkat batch；把 AD EX cells 绑定到哪个 batch 是影响 z_0 的研究决策。

本轮为链路验证构造的 `engineering_context_ex.h5ad`（247 cells、donor 轮转采样、33 基因显式填零、全部绑定第一个 KO batch）**仅是 dry-run 工程资产，不是正式 context**。

### 3.2 发现二（阻塞性）：DAVF 资产基因轴不覆盖 AD 候选

冻结 cohort 的 5 个 AD 候选基因在两条 DAVF route 的 scVI/alias 资产中的覆盖（本轮实测）：

| 基因 | KO 链（Dixit K562） | KD 链（Nadig jurkat） | PerturbGen vocab |
|---|---|---|---|
| APOE | ✅ ENSG00000130203 | ❌ | ✅ |
| APP / PSEN1 / BACE1 / MAPT | ❌ | ❌ | ✅（token 13453/9618/8218/11423） |

DAVF 方向 gate 需要在 scVI decoder 轴 resolve 候选基因（[`orchestrator.py`](/home/scu/PTM2CellNet/src/integration/perturbgen/orchestrator.py:441) 硬校验 alias）。当前 DAVF 训练资产来自血液细胞系 perturbation 数据，其 4018 HVG 轴天然不含多数 AD 神经元基因。**即使方向假设、context、统计全部齐备，4/5 冻结候选也无法产生 DAVF 方向证据。**

出路（研究决策，二选一或组合）：
- 走 Workflow B：以 AD 队列（或含 AD 基因的 perturbation 数据）重训 scVI + LatentDAVF（修复报告 §5 D5、[`perturbgen_bridge.md` §5](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:285)）；
- 或冻结"以候选轴为先"的新训练输入选择，使 scVI HVG 集覆盖候选基因。

### 3.3 发现三（高）：当前 DEG 口径下三方 gate observed 方不可通过

本轮生成的真实 DEG 表全表 **FDR 恒等于 1.0（唯一值）**，0/117,352 行显著。根因（本轮数值重现定位）：

- 管线聚合 [`_donor_log2_means`](/home/scu/PTM2CellNet/src/data/gse_normal_disease.py:596) 是"每细胞 log2(CPM+1) → donor 内平均"，该口径下 GSE174367 EX 的 Welch t（7 vs 11 donor）**min p ≈ 6.3e-5**；
- BH 校正 58,676 基因要求 rank-1 p < 1.37e-6，数学上不可达 → 全表 FDR=1.0；
- 对照口径"donor pseudobulk counts 先聚合再变换"：min p ≈ 8.9e-13，7 个基因可达 p<1e-4。

这不是代码 bug（实现与契约自洽），而是统计方法功效问题：**三方 gate 的 observed 方（`deg_max_fdr=0.05`）在当前口径下对整个队列不可通过**，正式候选链路（含 APOE）都会停在 `observed_expression_not_significant`（本轮 dry-run 已实证该拦截路径正确工作）。改口径（pseudobulk counts 聚合 + 检验选择，如 edgeR 风格负二项或保持 Welch-on-pseudobulk）是研究设计决策，须与 `deg_max_fdr` 语义一起冻结后重算 DEG 表，不得静默更换。

### 3.4 其他复核结论

- **VRAM 风险（中）**：正式配置 `configs/integration/perturbgen.yaml` 的 dry_run_estimates 标注 train_mask/train_decoder `peak_vram_gb: 40`，而 P40 仅 24GB；runner 另要求磁盘 `2×estimated+10GiB`。D1 启动前须实测或调 batch/精度后更新估计（历史 local_adaptation 运行在 P40 上完成过 train_mask，说明该估计偏保守，但未在本配置组合上验证）。
- **环境变量口径（低）**：正式 PerturbGen 配置依赖 18 个 `PTM2CELLNET_PERTURBGEN_*` 运行时变量，无文档集中登记其取值约定（本轮从 v10 历史 manifest 考古还原）。建议在 runbook 固化变量清单。
- **E2E runner 边界符合设计**：gate 未通过 → 不创建 invocation → 不构造 prepare plans（`perturbgen_runs=0`），与"runner 不自行重查 gate、由 E2E wrapper 保证"的口径一致（AGENTS.md 2026-09-13 约束）。

## 4. 未解决问题与后续解决策略（D0–D5 修订口径）

| # | 问题 | 严重度 | 解决策略 | 实施步骤 | 时间节点 |
|---|---|---|---|---|---|
| U1 | PTM 侧外部输入（真实 PTM 定量、KSTAR/PhosR 独立环境与 activity 表、signed network + id map、source proposals） | 阻塞性（外部） | 由研究负责人提供外部资产；仓库侧已具备消费契约（阶段 1/3/5 CLI） | ① 选定并冻结 PTM 队列与 OmniPath release（更新 config 的两个 PENDING 字段）② 建 KSTAR/PhosR 独立 conda 环境 ③ 按方案 §10 取标准表 ④ 文献锚点 source_proposals.tsv | D0 剩余，外部依赖（1–2 周） |
| U2 | GSE174367 → scVI-aligned context 准备工具 + 33 缺失基因处理 + batch 绑定语义（§3.1） | 阻塞性 | 新增一个显式 context 准备入口（脚本或标准流程文档），缺失基因与 batch 绑定作为研究设计冻结项写入 config/manifest | ① 冻结绑定决策（填零 vs 重训；batch 选择）② 实现 `prepare_scvi_context` 类工具 + 单测 ③ 生成正式 context 并登记 lineage | 与 U3 同批，1–2 天（决策依赖研究设计） |
| U3 | DAVF 基因轴不覆盖 AD 候选（§3.2） | 阻塞性 | 优先评估 Workflow B（AD 队列重训 scVI+LatentDAVF）；或调整候选集与资产对齐 | ① 研究决策（重训 vs 候选集收缩）② 若重训：按 [`davf_ko_kd_training.md`](/home/scu/PTM2CellNet/docs/guides/davf_ko_kd_training.md) 流程重训并过 Gate-E 边界 | 3–5 GPU 日（对应修复报告 §5 D5 扩展） |
| U4 | DEG 统计口径功效为零（§3.3） | 高 | 冻结 pseudobulk counts 聚合口径 + 检验选择，重算 DEG 表；同步复核 `deg_max_fdr` 语义 | ① 研究决策冻结口径 ② 修改 `_donor_log2_means` 消费路径或新增口径参数（显式、不静默）③ 重算并冻结新 DEG manifest ④ 回归测试 | 0.5–1 天（CPU） |
| U5 | 候选 spec 方向假设（proposed_direction/site 锚点/七字段语义确认） | 阻塞性（外部研究输入） | 不可自动生成（runbook 步骤 1 明示）；等 U1 的 source proposals 或研究负责人直接给定 | 文献锚点 → spec（可经 `build_celltype_candidate_specs.py` 或手写，均需 U4 先产出可 gate 的 DEG） | 随 U1/U4 |
| U6 | formal 六阶段 + matched-null + 统计组装 + verify（D1–D4） | 阻塞性（依赖 U1–U5） | 按修复报告 §5 D1–D4 顺序执行；VRAM 估计先行实测（§3.4） | 依 runbook（[`perturbgen_bridge.md` §3](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:139)）逐步执行，不跳步 | 3–4 GPU 日（U1–U5 齐备后） |
| U7 | Workflow B Gate-E（冻结 embedding → LatentDAVF 重训 → ≥200 真实 benchmark） | 高（独立生命周期） | 若 U3 选择重训路线则与本项合并规划 | 按 [`perturbgen_bridge.md` §5](/home/scu/PTM2CellNet/docs/guides/perturbgen_bridge.md:285) 边界执行，与 A 分开记账 | 与 U3 合并评估 |
| U8 | PerturbGen 运行时环境变量清单未文档化 | 低 | 在 bridge runbook 固化 18 个 `PTM2CELLNET_PERTURBGEN_*` 变量的名称/语义/示例值 | 文档补丁 | 0.5 天，可随任一后续 PR |

**修订后的推荐顺序**：U4（DEG 口径，快且解锁 gate 可行性评估）→ U3/U7（资产基因轴，决定主线走向的最大决策）→ U2（context 准备）→ U1/U5（外部输入）→ U6（D1–D4）。

## 5. 验证矩阵（本轮最终快照）

| 检查 | 结果 |
|---|---|
| `ruff check src scripts tests` | 通过 |
| `python -m ruff format --check src scripts tests` | **492 files already formatted，exit 0**（修复前 exit 1） |
| 全量 `pytest -m "not slow and not gpu" --timeout=300` | **2798 passed / 0 failed / 22 skipped，825.69s，exit 0**（修复后全量实测重跑；修复前同命令 3 failed/2795 passed/818s） |
| `python -m mypy src/ --ignore-missing-imports` | 172 files，0 errors |
| `python scripts/check_requirements_consistency.py` | 通过，274 pins |
| `python -m compileall src scripts tests` | 通过 |
| E2E dry-run（真实 GSE174367 工程子集 + APOE） | 全链路跑通；gate=inconclusive（observed 不显著，正确拦截）；无 invocation |
| DAVF 真实 decode（GPU） | 通过（finite delta，checkpoint/embedding provenance 完整） |
| formal GPU / biology PASS | **0**（维持；U1–U6 未齐备前不得启动） |

## 6. 本轮产出物清单

| 产出 | 路径 |
|---|---|
| 冻结研究配置 | `configs/research/ptm_research_config.yaml` |
| AD DEG 表 + manifest | `outputs/ptm_activity/20260915_d0/ad_deg_aggregate.tsv`、`ad_deg_donor_level.tsv`、`ad_deg_manifest.json` |
| E2E dry-run 工程 context（非正式资产） | `outputs/ptm_activity/20260915_d0/e2e_dryrun/engineering_context_ex.h5ad` |
| dry-run 候选 spec（工程占位方向） | `outputs/ptm_activity/20260915_d0/e2e_dryrun_candidate_spec.json` |
| E2E dry-run 报告 | `outputs/ptm_activity/20260915_d0/e2e_dryrun/e2e_report.json` |
| 测试修复 | `tests/unit/scripts/test_evaluate_perturbgen_dual_path.py`、`tests/integration/test_scvi_davf_connection.py` |
| 全仓格式化 | 324 files（`src/ tests/ scripts/`） |
| lessons 新条目 | `lessons.md` L-2026-0915-02 |

## 7. 引用索引

### 本地文档（章节号）

- `project_repair_report_20260915.md`：§1（已完成修复基线）、§3（未解决根因与边界，含两项验证债）、§4（验证矩阵与前口径）、§5（D0–D5 收口计划，本轮执行依据）、§6（发布口径）
- `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`：§3.1（方向字段分离）、§4.5（AD DEG 八列契约）、§5.3（传播）、§5.5（候选与三方 gate）、§7 阶段 0–6（含"正式 yaml 由真实研究运行时冻结"）、§8（泄漏与统计边界）、§10（外部标准表）
- `docs/guides/ptm_activity_pipeline.md`：§0（边界）、§1（config 契约）、§4（gene score）、§5（交集）、§6（候选 spec 与 source gate）、§7（target-set 评估）、§8（测试）
- `docs/guides/davf_perturbgen_e2e.md`：§候选清单（context_h5ad/scVI 一致性、七字段、方向语义、"候选 JSON 可以手工编写"）、§Gate-0、§统计接续
- `docs/guides/perturbgen_bridge.md`：§3（GPU runbook 步骤 1–4，方向假设不可自动生成；matched-null 骨架）、§5（Workflow B/Gate-E）、§6（统计闭环）
- `docs/guides/davf_ko_kd_training.md`（4018 基因轴、`adapter.gene_names` 契约、新进程 decode 要求）
- `lessons.md`：L-2026-0914-01…05、L-2026-0915-01、**L-2026-0915-02（本轮新增）**
- `docs/CURRENT_STATUS.md`（Quick Reference 口径）

### 代码与测试

- 修复：`tests/unit/scripts/test_evaluate_perturbgen_dual_path.py`（fixture 补 formal 声明）、`tests/integration/test_scvi_davf_connection.py`（KD 链迁移）
- 复核依据：`scripts/run_davf_perturbgen_e2e.py`（context 子集/Gate-0/donor split）、`src/models/scvi_adapter.py`（`resolve_target_gene_indices`/加载契约）、`src/integration/perturbgen/orchestrator.py`（alias 校验、shared prepare）、`src/integration/perturbgen/replay_evaluation.py`（formal 契约）、`src/analysis/ad_deg_table.py` 与 `src/data/gse_normal_disease.py`（`_donor_log2_means`/`_bh_adjust`）
- 新产出：见 §6
