# PTM2CellNet 修复执行报告（2026-09-16）：U1–U7 / D1–D4 全批次收口

> 执行依据：[`project_analysis_20260916.md`](project_analysis_20260916.md) §7「技术债与解决策略」（§7.1 严重度标准、§7.2 债务清单 U1–U8、§7.3 严重债务策略、§7.4 高等级策略、§7.5 中等级策略 D1–D4）、§5.2 缺失接口清单、§4.4 需求-实现差异表。
>
> 本报告覆盖 2026-09-16 当日两个修复批次：**批次一**（U2–U7 收口 + T1–T6 主线推进，见 §2.1/§8）、**批次二**（本轮：U6-残余 rescue_extractor 真实绑定 + D4 统一验收编排，见 §2.2）。
>
> 事实边界：代码、测试、配置与运行产物是实现事实；smoke/synthetic/工程 context 不等于正式生物学 PASS，正式 biology PASS 维持 **0**。

## 目录

- [1. 修复汇总（按严重程度分类）](#1-修复汇总按严重程度分类)
- [2. 每轮迭代执行情况与结果](#2-每轮迭代执行情况与结果)
- [3. 系统性复核：E2E 训练与推理技术要求](#3-系统性复核e2e-训练与推理技术要求)
- [4. 未解决问题与后续解决策略](#4-未解决问题与后续解决策略)
- [5. 验证矩阵](#5-验证矩阵)
- [6. 产出清单](#6-产出清单)
- [7. 引用索引](#7-引用索引)
- [8. 批次一执行记录（U2–U7 收口 + T1–T6）](#8-批次一执行记录u2u7-收口--t1t6)

## 1. 修复汇总（按严重程度分类）

严重程度沿用 [`project_analysis_20260916.md` §7.1](project_analysis_20260916.md) 的判定标准。每个条目标注修复批次：〔1〕=当日批次一，〔2〕=本轮批次二。

### 1.1 严重级（阻塞性）

| 编号 | 债务（§7.2） | 处置 | 结果 |
|---|---|---|---|
| U1 | 真实 PTM 定量与 KSTAR/PhosR activity 缺失（§7.3-U1） | 〔1〕T2 已把 OmniPath signed network 组装为 `omnipath-2026-09-16` release（12,878 条 signed TF→gene 边）并绑定研究 config；PTM 定量与 activity 按方案 §10 保持外部标准表输入，不在 core 重建 KSTAR/PhosR | **PTM 上游外部依赖从 2 个减为 1 个**；剩余为外部数据供给，代码侧无缺口 |
| U2 | KD 与 formal 全 route scVI context 未冻结（§7.3-U2 方案 A） | 〔1〕新增唯一准备入口 [`src/data/scvi_context.py`](src/data/scvi_context.py) + CLI：`missing_gene_policy`/`davf_batch`/`batch_binding_rationale` 三项显式合同化；KO route 真实 context 61,472×4,018、encode (256,64) finite；T3 Frangieh 新链 4018 轴被 GSE174367 **零缺失**覆盖 | **已解决（KO 路径）**；KD route 无候选消费者（见 U3），不强行绑定 55-batch 决策 |
| U3 | DAVF gene axis 不覆盖 4/5 AD 候选（§7.3-U3） | 〔1〕`scripts/audit_davf_axis_coverage.py` 扫描本地 30 个 scPerturb 数据集 + PerturbGen vocab：APP/PSEN1/BACE1/MAPT 在 vocab 全覆盖但本地零扰动数据——**Workflow B 重训方案 A 判定本地不可行**；T3 Frangieh 链把 KO route 扩到 216 targets（含 APOE 真实扰动信号） | **决策支撑落地**；per-route 结论 KO={APOE}、KD=∅。4 个 scPerturb 截断 h5ad 已由 T4 重下修复（不增加候选扰动覆盖） |
| U5 | source proposal 外部输入（§7.3-U5） | 〔1〕边界核实：消费方 `load_source_proposals` 校验 `proposed_direction ∈ {up,down}`，集成测试覆盖；方案 §5.5 与 bridge guide §3 明确方向假设不可自动生成 | **维持外部依赖（按设计）**；不发明方向、不捏造 proposal 表 |
| U6 | 正式六阶段、matched-null、quality、p/q、dual-path 未执行（§7.3-U6） | 〔1〕工程链打通：PerturbGen 独立环境验证 + P40 三阶段真实探针（eager，VRAM 峰值 4,359 MiB）；D3 根因确认 `TORCHDYNAMO_DISABLE=1` eager 是 P40 唯一路径。〔2〕**本轮补齐最后两处代码缺口**：`rescue_extractor` 真实实现（[`src/integration/perturbgen/null_rescue.py`](src/integration/perturbgen/null_rescue.py)，原为 bridge guide §3 的 `NotImplementedError` 骨架，分析报告 §4.4 标"实现但有缺陷"、§5.2 标"正式研究代码仍缺绑定"）+ `verify_formal_workflow_a` 统一验收编排（见 D4） | **工程执行层完整就绪**：GPU null 批跑所需的 rescue 提取器与验收编排均不再是缺口；formal 执行仍被 U4 observed gate 与 U5 外部 proposal 阻塞（数据层，见 §3/§4） |

### 1.2 高等级

| 编号 | 债务 | 处置 | 结果 |
|---|---|---|---|
| U4 | donor DEG 统计口径无显著行（§7.4-U4 方案 A） | 〔1〕`donor_aggregation` estimand 冻结（`pseudobulk_counts`）并重算真实表；功效对照实验（logCPM pseudobulk min p=1.56e-5；16k–19k universe 下 min BH FDR 0.28–0.94）；T1 扩充 GSE157827 队列（177,654 nuclei）+ 两队列三口径合并实测：157827 单队列 min FDR 0.1365、朴素池化 1.0、per-cohort centering 0.51/0.75——全部无 FDR≤0.05 行 | **estimand 已实现并穷尽本地口径**；结论固化为"本地两 AD 队列统计功效结构性不可达"（lessons L-2026-0916-01/L-2026-0916-02），不是代码缺陷，不得调阈值掩盖 |
| U7 | Workflow B LatentDAVF 重训与 held-out 缺失（§7.4-U7 方案 A） | 〔1〕`src/analysis/gate_e_benchmark.py` + CLI 从真实 CPLM/dbPTM 注释组装 **60,030 行 / 25 类 PTM** benchmark（≥200 要求满足 300 倍），evaluator identity 对照 coverage=0.9874/collisions=0/agreement=1.0 | **benchmark 数据部分已解决**；重训与 held-out 受 U3 无本地训练输入限制（外部数据依赖） |

### 1.3 中等级（D1–D4，§7.5）

| 编号 | 债务 | 处置 | 结果 |
|---|---|---|---|
| D1 | 文档指针与 FDR 事实漂移（§7.5-D1 方案 A） | 〔1〕`docs/CURRENT_STATUS.md` 权威指针更新；管线指南 §7.1 新增 context 准备命令与 `deg_donor_aggregation` 字段；20260915 报告 FDR 文字在分析报告 §2.3 校正 | 已完成 |
| D2 | runner 与 formal invocation 边界未下沉（§7.5-D2） | 〔1〕调用者审计：src 内直接调用方均为合法工程用途（CLI wrapper 强制 gate report、E2E 经 orchestrator、null_generation 合法内部用途）；采纳方案 B 维持"runner=工程执行层"设计，runner docstring F-09 已写明显式契约 | 已完成（方案 B） |
| D3 | VRAM 40GB 估计未实测（§7.5-D3 方案 A） | 〔1〕P40 真实三阶段探针 + 5s 间隔 VRAM 采样：smoke 载荷峰值 4,359 MiB（≪24GB）；第一阻塞实为 `torch.compile`（triton 需 CC≥7.0，P40=6.1） | 已完成（smoke 实测）；formal 载荷峰值待正式输入复测 |
| D4 | 正式资产验收未进入可重复验证层（§7.5-D4 方案 B） | 〔1〕环境证据固化 + "契约具备 ≠ 生物学具备"措辞维持。〔2〕**本轮落地分析报告 §5.2 缺失接口 `verify_formal_workflow_a`**：新增 [`src/integration/perturbgen/formal_verification.py`](src/integration/perturbgen/formal_verification.py) + [`scripts/verify_formal_workflow_a.py`](scripts/verify_formal_workflow_a.py)，一次消费 E2E gate report、frozen cohort manifest、matched-null manifests、未扰动质量与 dual-path eval input/report manifest，输出单一 verdict（pass/fail/blocked）+ `missing_inputs` 清单；纯编排绑定既有校验器（`load_frozen_manifest`+`audit_donor_leakage`、`_validate_distribution_manifest`、quality 来源校验、`verify_eval_input_against_manifest`+`replay_verdicts`），不重复实现任何校验逻辑；blocked 绝不降级为 pass | **验收编排就绪**：真实 formal 资产到位即可单命令复核 Gate-4/5；方案 A 的真实 asset lane 仍依赖外部数据与 GPU 排期 |

### 1.4 低等级

U8 已闭合（〔1〕T5：bridge guide §2.1 登记 22 个 formal + 9 个 local 环境变量及 P40 eager 注记）。其余低风险格式问题不做全仓重写。

### 1.5 本轮（批次二）明确不修复项及第一性原理依据

| 项 | 不修复理由 |
|---|---|
| `evaluate_driver_target_gate`（分析报告 §5.2、M10） | 方案 §5.5 明确要求另行批准后实现；target-set concordance 不能替代 source 三方 gate，擅自实现等于放宽研究契约 |
| U1 PTM 定量 / U5 source proposals | 研究输入，不可自动生成（方案 §5.5、bridge guide §3 步骤 1）；捏造即违反"不可捏造数据"约束 |
| U4 调低 `deg_max_fdr` / 回填 p 值 | 分析报告 §2.3 与 §7.4-U4 明确禁止；功效问题不是代码缺陷 |
| runner 内复制 gate 校验（D2 方案 A） | 所有直接调用均为合法工程用途，属重复防御（仓库 scope 约束） |

## 2. 每轮迭代执行情况与结果

### 2.1 批次一（当日早间-下午，12 轮）

见 [§8](#8-批次一执行记录u2u7-收口--t1t6) 完整记录：U4 estimand 实现→重算→功效诊断（结果与预期相反并固化结论）、U2 context 实现→真实生成、U3 审计（4 截断 h5ad 显式记录）、U7 benchmark 组装、U6/D3 GPU 探针（三次迭代定位 triton CC 阻塞）、D1/D2/D4 处置、T1–T6 批次。

### 2.2 批次二（本轮，7 轮）

| 轮次 | 动作 | 结果 |
|---|---|---|
| 1 侦察 | 通读分析报告 §7/§5.2/§4.4 与现有修复报告；核对 `null_generation.py`/`runner.py`/`results.py`/`frozen_cohort.py`/`eval_assembly.py` 现状；ZMemory 声明 7 个目标文件 | 确认当日仅剩两处代码层缺口：`rescue_extractor` 绑定（U6）与 `verify_formal_workflow_a`（D4）；`evaluate_driver_target_gate` 按方案 §5.5 不擅自实现 |
| 2 rescue_extractor 设计 | 确认 `results.py` 已有完整统计核心（`validate_perturbgen_h5ad_schema`、`_aggregate_donor_expression_from_h5ad`（baseline=pred_counts/perturbed=X）、`summarize_rescue_by_donor`）；确认 perturb stage 唯一 h5ad artifact 名固定为 `result_h5ad`（`configs/integration/perturbgen.yaml`） | 设计为纯绑定层：不重算统计、不猜命名空间，DEG 列名/var 列由调用方显式传入 |
| 3 实现 | 新增 [`null_rescue.py`](src/integration/perturbgen/null_rescue.py)：`build_stage_rescue_extractor(deg_table, donor_obs_column, var_gene_column, ...)` → extractor(request, StageExecutionResult) → NullStageRecord | 硬失败契约落地：failed stage、缺 `result_h5ad` artifact、<3 可评估 donor、基因命名空间不匹配、非 finite median、manifest 缺 sha256 全部 `NullGenerationError` |
| 4 验收编排实现 | 新增 [`formal_verification.py`](src/integration/perturbgen/formal_verification.py)（5 项独立检查 + verdict 规则）+ [`scripts/verify_formal_workflow_a.py`](scripts/verify_formal_workflow_a.py)（退出码 0/1/2）；发现 `load_frozen_manifest` 只接受路径且校验 cohort hash → frozen 参数改为纯路径契约 | 检查项：e2e_gate / frozen_cohort / matched_null / unperturbed_quality（显式 payload 或 eval_input 候选，拒绝 hand_filled）/ acceptance_replay |
| 5 测试 | 新增 [`test_null_rescue.py`](tests/unit/integration/perturbgen/test_null_rescue.py)（6 项）+ [`test_formal_verification.py`](tests/unit/integration/perturbgen/test_formal_verification.py)（8 项）；首跑 3 失败——fixture 用了非法 Ensembl ID（`ENSGN`，`normalize_ensembl_id` 硬校验 ENSG+11 位）与 donor 泄漏断言措辞（构造期即拒，报 `frozen_manifest_invalid: donor leakage`）→ 修正 fixture | **14/14 passed**；3.18s |
| 6 文档 | bridge guide §3 示例从 `NotImplementedError` 骨架替换为 `build_stage_rescue_extractor` 真实用法（保留"不得用 synthetic rescue 分数代替真实输出"边界）；§6 登记 `verify_formal_workflow_a.py` 命令 | 完成 |
| 7 验证 | 定向 81 passed（null_rescue/formal_verification/null_generation/results/frozen_cohort/eval_assembly）；CLI 冒烟（dry-run E2E report → verdict=fail、退出码 1、缺失项与失败原因如实列出）；ruff/format/mypy(176 files)/compileall/requirements 全部 exit 0；**全量回归 2833 passed / 0 failed / 22 skipped / 732.94s**（当日批次一历史 2819 + 本轮 14） | 全绿 |

## 3. 系统性复核：E2E 训练与推理技术要求

按主线链路逐层复核（要求出处：[E2E guide](docs/guides/davf_perturbgen_e2e.md)、[bridge guide §3–§6](docs/guides/perturbgen_bridge.md)、[方案 §4/§7](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md)）。状态为两批次累计后现状。

| 层 | 技术要求 | 当前状态 | 满足？ |
|---|---|---|---|
| 上游 PTM 输入（U1） | 真实 PTM 定量 / KSTAR/PhosR activity / signed network release | signed network 已绑定 `omnipath-2026-09-16`；PTM 定量与 activity 显式 PENDING | ✗ 外部依赖 |
| AD DEG 表（U4） | donor-level estimand + Welch/BH + between_donor ≥3 donor | pseudobulk estimand 冻结并重算，两队列三口径实测；**0 显著行（队列功效限制）** | 代码 ✓ / 数据 ✗ |
| source proposal（U5） | 外部审阅的方向假设 TSV | 消费接口与校验完备 | ✗ 外部输入 |
| scVI context（U2） | 4018 轴一致 + `davf_batch` + counts 层 + ensembl_id var | 唯一入口 + KO 真实 context（Frangieh 链零缺失）+ encode 验证 | ✓（KO；KD 待候选） |
| DAVF 方向解码 | route checkpoint + alias 轴 + token/decoder 分离 | APOE KO 双链（Dixit/Frangieh）decode finite 非零验证 | ✓（仅 APOE） |
| 三方方向 gate | proposal + DAVF + observed 三方同号且 observed FDR≤0.05 | 工程链完整；**observed FDR 在本地两队列任何无偏口径下不可达** | 工程 ✓ / 科学 ✗ |
| Gate-0/invocation | 真实队列 raw counts/donor/pairing + pass-only invocation | 已具备（dry-run 正确拦截 non-pass） | ✓ |
| 六阶段执行（U6） | 外部 PerturbGen env + gate-bound CLI + shared prepare | 环境 + 三阶段真实探针（eager）；shared `_prepare` 复用契约完整 | 工程 ✓ / formal ✗ |
| **matched-null 批跑（U6，本轮修复）** | GPU null 执行需 `rescue_extractor(request, stage_result)` 真实绑定 | **本轮落地**：`build_stage_rescue_extractor` 复用与 candidate path 提取完全相同的 pred_counts/X donor 聚合与 held-out signature 打分，排除被扰动 null 基因，sha256 从 stage manifest 登记，<3 donor/命名空间不匹配硬失败 | **接口 ✓ / 真实批跑等 formal 资产** |
| 统计验收链 | matched-null ≥99 + quality + empirical p + BH + dual-path AND + frozen replay | 接口全在（F-01）；`--assemble-statistical-evidence` 强制真实输入 | ✗ 等资产 |
| **统一验收编排（D4，本轮新增）** | Gate-4/5 单命令可复核 verdict | **本轮落地**：`verify_formal_workflow_a` 五项检查 + pass/fail/blocked verdict + missing_inputs；dry-run report 实测正确判 fail 并列出全部缺失项 | **编排 ✓ / 等真实资产** |
| Workflow B / Gate-E（U7/U3） | ≥200 benchmark + LatentDAVF 重训 + held-out | benchmark 60,030 行就绪 + evaluator identity 通过；重训无本地训练输入 | benchmark ✓ / 重训 ✗ |
| 资源（D3） | GPU 可执行性 | P40 需 `TORCHDYNAMO_DISABLE=1`；smoke 峰值 4.4GB；formal 峰值待复测 | ✓（eager 路径） |

**结论**：E2E 的**工程链**（context 准备 → DAVF decode → 三方 gate → invocation → 六阶段 → matched-null rescue 提取 → 统计组装 → 统一验收）在 KO/APOE 路径上**全链可执行且契约完备，已无代码层缺口**；本轮之后，剩余阻塞全部位于**数据层**——(a) observed 三方 gate 的 FDR≤0.05 在本地两个 AD 队列任何无偏统计口径下不可达（U4 实测，分析报告 §2.3），(b) 除 APOE 外 4 个 AD 候选既无 DAVF 轴覆盖也无本地扰动训练数据（U3 审计），(c) 真实 PTM 定量/KSTAR-PhosR activity/source proposal 未供给（U1/U5）。三者均不是代码缺陷，属研究决策/数据供给问题；formal 资产一旦到位，rescue 批跑与 Gate-4/5 验收均可在现有接口上直接执行。

## 4. 未解决问题与后续解决策略

| 编号 | 问题 | 严重度 | 解决策略与实施步骤 | 时间节点（估） |
|---|---|---|---|---|
| U1-残余 | 真实 PTM/activity 输入 | 严重（外部） | 按 §7.3-U1 方案 A：研究负责人冻结 PTM/activity 标准表 → 登记 manifest → 仓库 contract 校验并复用 `omnipath-2026-09-16` | 外部 5–10 天 |
| U4-残余 | observed gate 在本地两队列不可达 | 高 | 三选一（研究决策，不得调阈值）：① 扩大/更换 donor 队列；② 预注册独立于 DEG 结果的小基因面板缩 BH universe（依赖 U1）；③ 修订 gate 语义为方向一致性+标称 p（需方案 §4.5 修订与审批） | 决策 0.5 天 + 实施 1–2 天 |
| U5 | source proposals | 严重（外部） | 按 §7.3-U5 方案 A：文献/实验审阅后提供 TSV（消费接口已就绪） | 外部 1.5 天 |
| U3-残余 | 4 候选无 DAVF 轴覆盖/训练数据 | 严重 | 本地重训已证不可行；需外部 AD 相关 perturb 数据（如 iPSC-neuron CRISPR 队列）或正式收缩研究问题至 APOE | 数据获取外部；收缩决策 0.5 天 |
| U6-残余 | formal 六阶段 + 真实 null + 统计闭环执行 | 严重 | U4/U5 解锁后：KO/APOE pass invocation → eager 六阶段（`TORCHDYNAMO_DISABLE=1`，formal 载荷先做 VRAM 复测）→ matched-null 批跑（**本轮 rescue extractor 直接可用**：bridge guide §3 步骤 3）→ `--assemble-statistical-evidence` → **`verify_formal_workflow_a` 单命令验收** | 3–5 GPU 天 |
| U7-残余 | LatentDAVF 重训 + held-out Gate-E | 高 | 受 U3-残余数据限制；benchmark 资产已就绪（60,030 行），数据到位即可执行 §7.4-U7 方案 A | 数据依赖 5–7 GPU 天 |
| M10 | driver–target gate contract 写入 lineage | 中 | 方案 §5.5 要求另行批准后实现；sidecar 分字段评估已就绪，等待研究负责人批准 | 批准后 1–2 天 |

## 5. 验证矩阵

| 检查 | 命令/方式 | 结果 | 批次 |
|---|---|---|---|
| 全量回归（本轮实际执行） | `python -m pytest -m "not slow and not gpu" --timeout=300 -q` | **2833 passed / 0 failed / 22 skipped / 732.94s / exit 0** | 2 |
| 新增单测 | `test_null_rescue.py`（6 项）+ `test_formal_verification.py`（8 项） | **14 passed / 3.18s** | 2 |
| null/rescue/frozen 定向回归 | 6 个相关测试文件 | **81 passed / 14.19s** | 2 |
| CLI 冒烟（验收编排） | `verify_formal_workflow_a.py --e2e-report <20260915_d0 dry-run>` | verdict=**fail**、退出码 **1**；缺失项 4 项与 e2e 失败原因（no_passing_invocation/no_perturbgen_runs/statistical_evidence_not_assembled）如实列出 | 2 |
| 静态检查 | `ruff check src scripts tests` | **All checks passed / exit 0** | 2 |
| 格式检查 | `python -m ruff format --check`（本轮触碰文件） | 全部通过（format 后） | 2 |
| 类型检查 | `python -m mypy src/ --ignore-missing-imports` | **Success: no issues in 176 source files / exit 0**（批次一 174 + 本轮 2） | 2 |
| 编译检查 | `python -m compileall -q src scripts tests` | **exit 0** | 2 |
| 依赖一致性 | `python scripts/check_requirements_consistency.py` | **274 lock pins 一致 / exit 0** | 2 |
| U2 真实验收 | context 契约字段 + E2E 同款 encode | 61,472×4,018 ✓ / (256,64) finite ✓（Frangieh 链零缺失） | 1 |
| U4 真实验收 | pseudobulk 重算 + 4 组功效对照 + T1 两队列三口径 | 表产出 ✓；min FDR 0.9154/0.1365/1.0（如实记录，0 显著行） | 1 |
| U3 真实验收 | 30 数据集扫描 + vocab 双查 | per-route 结论 KO={APOE}/KD=∅；4 损坏文件重下修复（T4 三重验证） | 1 |
| U7 真实验收 | 组装 + `load_benchmark.validate` + identity migration | 60,030 行 ✓ / evaluator 消费 ✓ | 1 |
| D3 真实验收 | 三阶段 GPU 探针 + VRAM 采样 | 全成功 / 峰值 4,359 MiB（P40 eager） | 1 |
| 批次一全量回归（历史记录） | 同全量命令 | 2818 passed（U2–U7 批次）/ 2819 passed（T1–T6 批次）；非本轮重新运行 | 1（历史） |

## 6. 产出清单

**批次二（本轮）代码**：`src/integration/perturbgen/null_rescue.py`（新）、`src/integration/perturbgen/formal_verification.py`（新）、`scripts/verify_formal_workflow_a.py`（新）。
**批次二测试**：`tests/unit/integration/perturbgen/test_null_rescue.py`（新，6 项）、`tests/unit/integration/perturbgen/test_formal_verification.py`（新，8 项）。
**批次二文档**：`docs/guides/perturbgen_bridge.md`（§3 rescue extractor 真实示例、§6 验收命令）、本报告。
**批次一产出**（详见 §8）：`src/data/scvi_context.py`、`src/analysis/gate_e_benchmark.py`、`scripts/prepare_scvi_context.py`、`scripts/audit_davf_axis_coverage.py`、`scripts/build_gate_e_benchmark.py`、`scripts/build_signed_network_release.py` 等；真实产物在 `outputs/ptm_activity/20260916_d1/`、`20260916_t1/`、`data/processed/signed_network/`、`checkpoints/davf/davf_ko_frangieh/`。

## 7. 引用索引

| 引用 | 章节/位置 | 用途 |
|---|---|---|
| [`project_analysis_20260916.md`](project_analysis_20260916.md)（2026-09-16 第三批次起为批次三分析报告；下述章节号属同日早间版综合分析，已归档 `git show HEAD:project_analysis_20260916.md`） | §2.3 DEG estimand 与 FDR 复核 / §4.4 需求-实现差异（rescue_extractor"实现但有缺陷"） / §5.1 M1–M11 / §5.2 缺失接口（rescue_extractor、verify_formal_workflow_a、evaluate_driver_target_gate） / §7.1 严重度标准 / §7.2 债务清单 U1–U8 / §7.3-U1/U2/U3/U5/U6 严重债务策略 / §7.4-U4/U7 高等级策略 / §7.5-D1–D4 中等级策略 / §9 最小后续顺序 | 本日修复执行依据 |
| [`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md) | §2.3 正式验收边界 / §3.1 方向字段分字段契约 / §4.1 PTM 输入 / §4.5 AD DEG 输入 / §5.5 proposal 边界与 driver–target gate 另行批准 / §6.4 Gate-E / §7 阶段 6 / §10 外部标准表 | 研究契约 |
| [`docs/guides/perturbgen_bridge.md`](docs/guides/perturbgen_bridge.md) | §2.1 环境变量 runbook（U8/T5） / §3 步骤 1（proposal 不可自动生成）、步骤 3（matched-null，本轮改真实 rescue extractor） / §4 六阶段 / §5 Workflow B/Gate-E / §6 评估与发布证据（本轮新增 verify_formal_workflow_a 命令） | 执行边界 |
| [`docs/guides/davf_perturbgen_e2e.md`](docs/guides/davf_perturbgen_e2e.md) | 候选清单与三方 gate 合同 | E2E 运行边界 |
| [`docs/guides/ptm_activity_pipeline.md`](docs/guides/ptm_activity_pipeline.md) | §1 阶段 0（`deg_donor_aggregation`） / §7.1 context 准备命令 | 命令指南 |
| [`src/integration/perturbgen/runner.py`](src/integration/perturbgen/runner.py) | 模块 docstring F-09 执行边界契约（D2） | runner 契约 |
| [`src/integration/perturbgen/results.py`](src/integration/perturbgen/results.py) | `validate_perturbgen_h5ad_schema`、`_aggregate_donor_expression_from_h5ad`（pred_counts/X 基线契约）、`summarize_rescue_by_donor`（held-out signature） | 本轮 rescue extractor 复用的统计核心 |
| [`src/integration/perturbgen/frozen_cohort.py`](src/integration/perturbgen/frozen_cohort.py) | `load_frozen_manifest`+`audit_donor_leakage`、`verify_eval_input_against_manifest`、`replay_verdicts` | 本轮验收编排复用的 Gate-4/5 校验器 |
| [`project_analysis_20260915.md`](project_analysis_20260915.md) | §2.3 所引 FDR 校正 / §3 三项发现 | 上轮证据基线 |
| `lessons.md` | L-2026-0915-02（1e-13 已作废）/ L-2026-0916-01（observed gate 功效限制修正）/ L-2026-0916-02（T1–T3 队列与 Frangieh 链）/ L-2026-0916-03/04（T4 重下完整性口径） | 决策记录 |

## 8. 批次一执行记录（U2–U7 收口 + T1–T6）

### 8.1 U2–U7 收口批次（12 轮）

| 轮次 | 动作 | 结果 |
|---|---|---|
| 1 侦察 | 通读分析报告 §7 债务清单；核对 ad_deg_table/scvi_adapter/runner/gate_e/davf_scperturb/config 现状 | 明确 U2–U7 + D1–D4 落地方案与边界 |
| 2 U4 实现 | `_donor_pseudobulk_log2` + `donor_aggregation` 参数 + config 字段 + CLI + 冻结 `pseudobulk_counts` + 单测 | 定向 12 passed；config 21 passed |
| 3 U4 重算 | 真实 GSE174367 重算 DEG 表 | **结果与文档预期相反**：min FDR 0.9154、0 显著行 |
| 4 U4 根因诊断 | 对照实验：raw-counts t / logCPM pseudobulk t / per-cell 口径 / 表达过滤 universe ×（EX/INH） | 任何无偏口径不可达 FDR≤0.05；上轮 1e-13 记录作废（lessons 修正） |
| 5 U2 实现 | `scvi_context.py` 模块 + CLI + 9 项单测；真实 checkpoint registry 验证 batch 校验（KO 2 batch、KD 55 batch） | 单测全绿 |
| 6 U2 真实生成 | KO context 真实生成 + 契约验证 + E2E 同款路径 encode 复验 | 61,472×4,018 全契约通过；encode (256,64) finite；KD 无消费者不生成 |
| 7 U3 审计 | 审计脚本 + 全 30 数据集扫描 | 首跑撞上 4 个截断 h5ad → 显式记录不可读文件后重跑；KO={APOE}、KD=∅ |
| 8 U7 benchmark | 映射链勘察（uniprot 辅助文件为酵母、idmapping 无 Ensembl → 弃用；CPLM 自带 symbol；dbptm phosph 90.4% 可映射）→ 模块 + CLI + 5 单测 → 真实组装 → evaluator 端到端验证 | 60,030 行；identity 对照 coverage 0.9874/collisions 0/agreement 1.0 |
| 9 U6/D3 探针 | 三跑：缺 `PYTHON` 变量 → train_mask 被 triton CC 硬阻塞 → 清 failed stage + `TORCHDYNAMO_DISABLE=1` + `--resume` | 三阶段全部成功；VRAM 峰值 4,359 MiB；resume 契约验证正确 |
| 10 D1/U5/指南 | CURRENT_STATUS 指针 + Quick Reference；管线指南 §7.1 新命令；U5 接口核实 | 完成 |
| 11 验证 | compileall、50 项定向 pytest、ruff、format、mypy（174 files）、requirements | 全部 exit 0 |
| 12 收口 | 代码批次提交（`40f115e`）+ 报告批次（`c78ada1`） | 归档不扩张；biology PASS 0 |

### 8.2 T1–T6 扩展批次（下午）

| 任务 | 结果 |
|---|---|
| T1 AD 队列扩充 | GSE157827 标准化（177,654 nuclei × 33,538 ENSG；12 AD + 9 control）+ 两队列合并（33,427 × 239,126）；三口径实测 min FDR 0.1365 / 1.0 / 0.51–0.75——observed gate 本地不可达结论固化 |
| T2 signed network | OmniPath 全等级导出 13,565 行 → 12,878 signed TF→gene 边 release；研究 config `network_release` 绑定真实 release |
| T3 Frangieh KO 链 | GPU 全链：240,646×4,018、216 targets（含 APOE）→ scVI → latent pairs → LatentDAVF；APOE KO 真实解码 finite 非零；新正式测试通过；4018 轴被 GSE174367 零缺失覆盖 |
| T4 重下损坏 h5ad | 4/4 完成（size 精确匹配 Zenodo + 非 backed 全量读取通过）；版本 hash 差异显式登记 |
| T5 U8 runbook | bridge guide §2.1：22 formal + 9 local 变量 + P40 eager 注记 |
| T6 R-01–03 核实 | blocked 确认：replogle/scgenescope 原始数据本地不存在 |

### 8.3 对主线的影响

1. **observed gate 阻塞性质最终确认**：不是代码、不是口径、不是单一队列——是本地两个 AD 队列的统计功效结构（分析报告 §2.3、lessons L-2026-0916-01/02）。
2. **PTM 上游从 2 个外部依赖减为 1 个**：只剩 KSTAR/PhosR activity + PTM 定量队列。
3. **KO route 从 10 个 TF target 扩到 216 个（含 APOE 真实扰动信号）**，为 APOE 候选提供有训练基础的 DAVF 证据来源。
