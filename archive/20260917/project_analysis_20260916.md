# PTM2CellNet 分析报告（2026-09-16）：§4 未解决问题第三批次修复与系统性复核

> 执行依据：[`project_repair_report_20260916.md`](project_repair_report_20260916.md) §4「未解决问题与后续解决策略」（7 项：U1-残余、U3-残余、U4-残余、U5、U6-残余、U7-残余、M10）及其引用的分析报告 §5.2 缺失接口契约、[方案 §5.5](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md)。
>
> 本报告为当日**第三修复批次**：输入是同日的修复报告（其 §4 列出剩余未解决问题），输出本轮修复记录、复核与后续策略。当日批次一/二记录见修复报告 §2.1/§2.2/§8。同日早间版综合分析报告（546 行，含 §5.2/§7 债务清单）已归档于 git 历史，其契约要点由修复报告 §1/§7 转引。
>
> 事实边界：代码、测试、配置与运行产物是实现事实；smoke/synthetic/工程 context 不等于正式生物学 PASS，正式 biology PASS 维持 **0**。

## 目录

- [1. 问题修复汇总（按严重程度分类）](#1-问题修复汇总按严重程度分类)
- [2. 每轮迭代执行情况与结果](#2-每轮迭代执行情况与结果)
- [3. 系统性复核分析：E2E 训练与推理技术要求](#3-系统性复核分析e2e-训练与推理技术要求)
- [4. 未解决问题及后续解决策略](#4-未解决问题及后续解决策略)
- [5. 验证矩阵](#5-验证矩阵)
- [6. 产出清单](#6-产出清单)
- [7. 相关文档引用](#7-相关文档引用)

## 1. 问题修复汇总（按严重程度分类）

严重程度沿用修复报告 §4 与其所引分析报告 §7.1 的判定标准。§4 共 7 项；从第一性原理逐项判定后，**代码层可推进的为 2 项**（M10、U4-残余的工具支撑），其余 5 项为外部数据供给或研究决策（判定依据见 §1.2）。

### 1.1 本轮修复项

| 编号 | 严重度 | 问题（修复报告 §4） | 本轮处置 | 结果 |
|---|---|---|---|---|
| M10 | 中 | driver–target gate contract 未写入 lineage；`evaluate_driver_target_gate` 缺失（原分析报告 §5.2） | **已实现**：[`src/integration/perturbgen/downstream_target_evaluation.py`](src/integration/perturbgen/downstream_target_evaluation.py) 新增 `evaluate_driver_target_gate` + `driver_target_gate_to_payload`（schema `ptm2cellnet.driver-target-gate/v1`）；E2E `--downstream-target-sidecar` 路径自动为每个 gated 候选写入 `downstream_target_evaluation.driver_target_gate`（方案 §5.5 要求的 lineage 落地） | **契约层已解决**：source 三方 gate 证据与 target-set concordance 分字段记录，`driver_target_status` 为无阈值事实分类，永不产生 pass/fail——source 三方 gate 仍是 formal 候选唯一准入门，source 无自身 DEG 的候选仍只进 exploratory（不放宽研究契约） |
| U4-残余 | 高 | observed gate 在本地两队列不可达；策略①（扩大/更换队列）的多队列合并此前只有临时代码 | **工具固化**：`build_ad_deg_tables` 新增 `pseudobulk_counts_centered` 口径（每个 (cell_type, cohort) 减各自 normal donor pseudobulk 基线后再做 disease-vs-normal Welch t + BH）；CLI `--cohort-column`；config 值域同步；真实合并队列实测 | **策略①支撑就绪**：GSE174367+GSE157827 合并（`dataset` 列）centered 实测 min FDR EX 0.5254 / INH 0.7908（vs 朴素池化 1.0），仍无 FDR≤0.05 行——**工具固化不改变科学结论**，后续新队列可用单命令复评 |

### 1.2 判定为不可代码修复项及第一性原理依据

| 编号 | 严重度 | 不修复理由 |
|---|---|---|
| U1-残余 | 严重（外部） | 真实 PTM 定量与 KSTAR/PhosR activity 是研究输入，不可自动生成（方案 §5.5、[bridge guide §3 步骤 1](docs/guides/perturbgen_bridge.md)）；消费与 contract 校验接口已在位 |
| U3-残余 | 严重 | 4/5 AD 候选无本地扰动训练数据是审计事实（修复报告 §1.1-U3：KO={APOE}、KD=∅）；需外部数据获取或研究收缩决策，代码侧无缺口 |
| U5 | 严重（外部） | source proposals 必须来自文献/实验审阅；方向假设不可自动发明（方案 §5.5）；消费接口 `load_source_proposals` 已就绪 |
| U6-残余 | 严重 | formal 六阶段执行被 U4 决策与 U5 输入阻塞（三方 gate 需要 observed FDR≤0.05 与外部 proposal）；工程链（rescue extractor、统计组装、统一验收）在批次二已无缺口 |
| U7-残余 | 高 | LatentDAVF 重训受 U3-残余数据限制；benchmark（60,030 行）已就绪，数据到位即可执行（修复报告 §1.2-U7） |

### 1.3 明确不修复项

| 项 | 理由 |
|---|---|
| U4 策略②（预注册基因面板缩 BH universe）的接口 | 修复报告 §4 明确标注"依赖 U1"——面板内容必须来自独立于 DEG 结果的外部输入；在面板供给前实现该参数没有可验证的消费者，属超前建设 |
| U4 策略③（修订 gate 语义为方向一致性+标称 p） | 明确要求"方案 §4.5 修订与审批"；擅自实现等于放宽 observed 显著性契约 |
| driver–target gate 产生 pass/fail 决策 | 方案 §5.5 与仓库协作约定（2026-09-14 节）一致要求 target-set 一致性不能替代 source 三方 gate；本轮实现严格保持补充证据定位 |

## 2. 每轮迭代执行情况与结果

| 轮次 | 动作 | 结果 |
|---|---|---|
| 1 侦察 | 通读修复报告 §4 全部 7 项与所引原分析报告 §5.2、方案 §5.5/§6.2/§6.3；核对 `downstream_target_evaluation.py`（sidecar/`evaluate_target_set_deltas` 已在位但无 gate 合并接口）、`run_davf_perturbgen_e2e.py`（`_assemble_downstream_target_evaluation` 已接线但不含 gate evidence）、`ad_deg_table.py`（`VALID_DONOR_AGGREGATIONS` 仅两口径）、`ptm_research_config.py`、`build_ad_deg_table.py` CLI；ZMemory 注册并声明 10 个目标文件 | 判定：M10 与 U4 策略①工具为仅剩代码层可推进项；其余 5 项外部/决策依赖（§1.2） |
| 2 M10 设计 | 确认契约输入：source gate 的 plain dict（E2E report `candidates[i].direction_gate`，离线可从磁盘 report 复评）+ `SourceTargetSetEvaluation`（既有）+ semantic_context；verdict 定为无阈值事实四分类（参照 target 自身 `observed_direction`） | 不发明判据阈值；source/target 字段物理分离（dataclass 两段）；身份绑定硬校验（ensembl/gene 不匹配即失败） |
| 3 M10 实现 | `evaluate_driver_target_gate` + `driver_target_gate_to_payload`（`DRIVER_TARGET_GATE_SCHEMA_VERSION`）；E2E gated 循环保存 `direction_gate`（pass 候选缺失该段硬失败），主循环组装 per-source evidence，返回结构新增 `driver_target_gate` 键，lineage `gate_boundary` 注记更新 | 编译通过；E2E report 加载校验确认宽松（只查 schema_version/runs，嵌套新字段安全） |
| 4 U4 实现 | `pseudobulk_counts_centered`：`_direction_adata` 透传 cohort 列、`_clean_obs` 显式检查列存在、centering 主逻辑（per (cell_type, cohort) normal 基线；donor 跨 cohort / cohort 缺 normal donor 硬失败）、donor-level log2fc 分支（centered 下为相对各自 cohort 基线值）、audit 扩展（cohort_column/cohorts/donor_counts_by_cohort）；config 值域与 CLI `--cohort-column` 同步 | 编译通过；合成数据数值验证：cohort 组成性偏移下真实效应基因 FDR 0.412 → 4.9e-15（方差收缩），单 cohort 下与 pseudobulk 数值恒等 |
| 5 测试 | DEG 追加 7 项（单队列不变性、偏移移除、4 个硬失败契约）；target 评估追加 7 项（mixed/all/none/not_evaluable 分类、source-target 分离 payload、身份不匹配、非法 status/direction）；E2E fixture 补 `direction_gate` 并追加断言 + 1 项缺失硬失败负测试。首跑修 3 处：Ensembl ID 位数笔误（18→15 位）、字符串方向列误用 `np.allclose`、`not_evaluable` 预期错置（target 侧参照不被 source 观测缺失阻断——恰为分离语义的验证点） | 定向 **81 passed**（4 文件）；fixture 偏移构造修正为单基因组成性偏移（全基因乘性偏移会被 library 归一化吸收，测试会假阴） |
| 6 真实验收 | 从 T1 manifest 内嵌 config 生成 centered 研究 config（`outputs/ptm_activity/20260916_d2/ptm_research_config_centered.yaml`）；正式 CLI 跑合并队列（239,126 × 33,427，`--cohort-column dataset`） | CLI exit 0；min FDR EX 0.5254 / INH 0.7908、0 个 FDR≤0.05 行；audit 完整（两 cohort、per-cohort per-state donor 计数）；与上一轮临时实验结论一致 |
| 7 静态与回归 | ruff check / format（触碰文件）/ mypy（176 files，修一处 audit dict 显式注解）/ compileall / requirements 一致性 | 全部 exit 0；**全量回归 2848 passed / 0 failed / 22 skipped / 778.21s**（上轮 2833 + 本轮 15） |
| 8 文档 | [管线指南](docs/guides/ptm_activity_pipeline.md)：config 值域注记、新增 §4a DEG 生成命令段（含 centered 用法与实测数字）、§7.2 库接口示例与 driver-target gate 边界；[E2E guide](docs/guides/davf_perturbgen_e2e.md) sidecar 段补 `driver_target_gate` 说明；lessons 追加 L-2026-0916-05（事实分类非判据、单队列不变性、centered 实测） | 完成 |

## 3. 系统性复核分析：E2E 训练与推理技术要求

按主线链路逐层复核（要求出处：[E2E guide](docs/guides/davf_perturbgen_e2e.md)、[bridge guide §3–§6](docs/guides/perturbgen_bridge.md)、[方案 §4/§5.5/§7](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md)）。状态为当日三批次累计后现状。

| 层 | 技术要求 | 当前状态 | 满足？ |
|---|---|---|---|
| 上游 PTM 输入（U1） | 真实 PTM 定量 / KSTAR-PhosR activity / signed network release | network 绑定 `omnipath-2026-09-16`；PTM 定量与 activity 显式 PENDING | ✗ 外部依赖 |
| AD DEG 表（U4） | donor-level estimand + Welch/BH + between_donor ≥3 | 三口径（per_cell/pseudobulk/**centered**）全部工具化并真实实测；0 显著行（队列功效限制） | 代码 ✓ / 数据 ✗ |
| source proposal（U5） | 外部审阅的方向假设 TSV | 消费接口与校验完备 | ✗ 外部输入 |
| scVI context（U2） | 4018 轴一致 + `davf_batch` + counts 层 | 唯一入口 + KO 真实 context + encode 验证（修复报告 §1.1-U2） | ✓（KO） |
| DAVF 方向解码 | route checkpoint + alias 轴 + token/decoder 分离 | APOE KO 双链 decode finite 非零验证 | ✓（仅 APOE） |
| 三方方向 gate | proposal + DAVF + observed 三方同号且 observed FDR≤0.05 | 工程链完整；observed FDR 本地不可达（三口径实测） | 工程 ✓ / 科学 ✗ |
| **driver–target gate（M10，本轮）** | target-set concordance 写入 lineage 且不替换 source gate | **本轮落地**：per-source evidence（source 三方 gate 原样 + target 分字段 concordance + 事实 verdict），E2E sidecar 路径自动写入 | ✓（契约层；消费等 formal 资产） |
| Gate-0/invocation | 真实队列 raw counts/donor/pairing + pass-only invocation | 已具备（dry-run 正确拦截 non-pass） | ✓ |
| 六阶段执行（U6） | 外部 PerturbGen env + gate-bound CLI + shared prepare | 环境 + 三阶段真实探针（eager）；`_prepare` 复用契约完整 | 工程 ✓ / formal ✗ |
| matched-null 批跑 | `rescue_extractor` 真实绑定 | 批次二落地（`build_stage_rescue_extractor`） | 接口 ✓ / 批跑等资产 |
| 统计验收链 | matched-null ≥99 + quality + empirical p + BH + dual-path AND | 接口全在；`--assemble-statistical-evidence` 强制真实输入 | ✗ 等资产 |
| 统一验收编排 | Gate-4/5 单命令 verdict | 批次二落地（`verify_formal_workflow_a`） | 编排 ✓ / 等资产 |
| Workflow B / Gate-E（U7/U3） | ≥200 benchmark + LatentDAVF 重训 + held-out | benchmark 60,030 行就绪；重训无本地训练输入 | benchmark ✓ / 重训 ✗ |
| 资源 | GPU 可执行性 | P40 eager（`TORCHDYNAMO_DISABLE=1`）；smoke 峰值 4.4GB | ✓（eager 路径） |

**结论**：E2E 的**工程链**（context 准备 → DAVF decode → 三方 gate → invocation → 六阶段 → matched-null rescue 提取 → 统计组装 → 统一验收 → **driver–target 补充证据**）在 KO/APOE 路径上全链可执行且契约完备，**已无代码层缺口**。本轮后剩余阻塞全部为数据层/研究决策层，与修复报告 §3 结论一致且未回退：

1. **observed 三方 gate 的 FDR≤0.05 不可达**：本轮 centered 口径真实实测（EX 0.5254 / INH 0.7908）再次确认这是本地两个 AD 队列的统计功效结构，不是代码或口径缺陷（lessons L-2026-0916-01/05）。
2. **候选扰动训练数据缺失**：除 APOE 外 4 个 AD 候选既无 DAVF 轴覆盖也无本地扰动数据。
3. **上游外部输入未供给**：真实 PTM 定量、KSTAR-PhosR activity、source proposals 三项。

## 4. 未解决问题及后续解决策略

| 编号 | 问题 | 严重度 | 解决策略与实施步骤 | 时间节点（估） |
|---|---|---|---|---|
| U1-残余 | 真实 PTM/activity 输入 | 严重（外部） | 研究负责人冻结 PTM/activity 标准表 → 登记 manifest → 仓库 contract 校验并复用 `omnipath-2026-09-16`（修复报告 §4-U1 原策略不变） | 外部 5–10 天 |
| U4-残余 | observed gate 三选一研究决策 | 高 | 决策仍未做，但**策略①工具已就绪**：新队列到位后用 `build_ad_deg_table.py`（单队列 `pseudobulk_counts` 或多队列 `pseudobulk_counts_centered --cohort-column <列>`）单命令复评 FDR 分布；策略②等 U1 面板；策略③需方案 §4.5 修订审批。不得调 `deg_max_fdr` | 决策 0.5 天；新队列评估单命令数小时 |
| U5 | source proposals | 严重（外部） | 文献/实验审阅后提供 TSV（消费接口就绪） | 外部 1.5 天 |
| U3-残余 | 4 候选无 DAVF 轴覆盖/训练数据 | 严重 | 外部 AD 相关 perturb 数据获取，或正式收缩研究问题至 APOE（修复报告 §4-U3 原策略不变） | 数据获取外部；收缩决策 0.5 天 |
| U6-残余 | formal 六阶段 + 真实 null + 统计闭环执行 | 严重 | U4 决策/U5 输入解锁后：KO/APOE pass invocation → eager 六阶段 → matched-null 批跑 → `--assemble-statistical-evidence` → `verify_formal_workflow_a` 验收（链路全就绪）；本轮起可同时携带 `--downstream-target-sidecar` 把 driver–target 补充证据一并写入 report | 3–5 GPU 天 |
| U7-残余 | LatentDAVF 重训 + held-out Gate-E | 高 | 受 U3-残余数据限制；benchmark 就绪，数据到位即执行 | 数据依赖 5–7 GPU 天 |
| M10-残余 | driver-target gate 的研究消费 | 低 | 契约层已完成；对 exploratory 候选（source 无自身 DEG）的离线批量评估可用库接口直接消费磁盘 report + sidecar；是否将该证据纳入 exploratory 候选的正式排序属研究决策 | 随 U6-残余一起 |

## 5. 验证矩阵

| 检查 | 命令/方式 | 结果 |
|---|---|---|
| 全量回归（本轮实际执行） | `python -m pytest -m "not slow and not gpu" --timeout=300 -q` | **2848 passed / 0 failed / 22 skipped / 778.21s / exit 0**（上轮 2833 + 本轮 15） |
| 新增单测 | DEG 7 项 + driver-target gate 7 项 + E2E 负测试 1 项 | **15/15 passed**（含于定向 81） |
| 定向回归 | `test_ad_deg_table.py` + `test_downstream_target_evaluation.py` + `test_run_davf_perturbgen_e2e.py` + `test_ptm_research_config.py` | **81 passed** |
| 真实数据验收（U4） | `build_ad_deg_table.py --config <centered> --cohort-column dataset`（239,126 × 33,427 合并队列） | exit 0；min FDR EX 0.5254 / INH 0.7908、0 显著行；audit 含 cohort_column/cohorts/donor_counts_by_cohort |
| 合成数值性质（U4） | 单队列不变性（与 pseudobulk 恒等）+ 双 cohort 组成性偏移移除（效应基因 FDR 0.412 → 4.9e-15） | 通过（进单元测试） |
| 静态检查 | `ruff check src scripts tests` | **All checks passed / exit 0** |
| 格式检查 | `python -m ruff format --check src scripts tests` | **508 files 全部已格式化** |
| 类型检查 | `python -m mypy src/ --ignore-missing-imports` | **Success: no issues in 176 source files** |
| 编译检查 | `python -m compileall -q src scripts tests` | exit 0 |
| 依赖一致性 | `python scripts/check_requirements_consistency.py` | **274 lock pins 一致 / exit 0** |
| E2E report 兼容性 | `load_e2e_report` 对新增嵌套字段的处理 | 校验只查 schema_version/perturbgen_runs，`driver_target_gate` 字段安全（宽松契约，非破坏性新增） |

## 6. 产出清单

**代码**：[`src/integration/perturbgen/downstream_target_evaluation.py`](src/integration/perturbgen/downstream_target_evaluation.py)（新增 `evaluate_driver_target_gate`/`driver_target_gate_to_payload`/`DriverTargetGateEvidence`）、[`scripts/run_davf_perturbgen_e2e.py`](scripts/run_davf_perturbgen_e2e.py)（gate evidence 接线）、[`src/analysis/ad_deg_table.py`](src/analysis/ad_deg_table.py)（centered 口径）、[`src/analysis/ptm_research_config.py`](src/analysis/ptm_research_config.py)（值域同步）、[`scripts/build_ad_deg_table.py`](scripts/build_ad_deg_table.py)（`--cohort-column`）。

**测试**：[`tests/unit/integration/perturbgen/test_downstream_target_evaluation.py`](tests/unit/integration/perturbgen/test_downstream_target_evaluation.py)（+7）、[`tests/unit/analysis/test_ad_deg_table.py`](tests/unit/analysis/test_ad_deg_table.py)（+7）、[`tests/unit/scripts/test_run_davf_perturbgen_e2e.py`](tests/unit/scripts/test_run_davf_perturbgen_e2e.py)（+1、fixture 更新）。

**文档**：[`docs/guides/ptm_activity_pipeline.md`](docs/guides/ptm_activity_pipeline.md)（值域注记、§4a DEG 生成命令、§7.2 接口与边界）、[`docs/guides/davf_perturbgen_e2e.md`](docs/guides/davf_perturbgen_e2e.md)（sidecar 段）、`lessons.md`（L-2026-0916-05）、本报告。

**真实产物**：`outputs/ptm_activity/20260916_d2/`（centered 研究 config、aggregate/donor-level DEG 表、manifest、CLI log）。

## 7. 相关文档引用

| 引用 | 章节/位置 | 用途 |
|---|---|---|
| [`project_repair_report_20260916.md`](project_repair_report_20260916.md) | §4 未解决问题与后续解决策略（本轮执行依据）/ §1 修复汇总 / §3 系统性复核 / §5 验证矩阵 | 本轮输入 |
| 同日早间版综合分析报告（已归档 git 历史，契约要点经修复报告 §1/§7 转引） | §5.2 缺失接口（`evaluate_driver_target_gate` 建议参数与返回契约）/ §5.1 M10 / §7.4-U4 策略 | 契约建议 |
| [`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`](docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md) | §3.1（方向字段分字段契约）/ §3.3（source/target 角色分离）/ §5.5（driver–target gate 另行批准边界；target-set 一致性不能替代 source 三方 gate）/ §6.2（downstream_target_evaluation 模块边界）/ §6.3（网络下游一致性是补充 evidence）/ §8.6（非因果验证措辞） | 研究契约 |
| [`docs/guides/ptm_activity_pipeline.md`](docs/guides/ptm_activity_pipeline.md) | §1 config 值域 / §4a（本轮新增）DEG 生成命令 / §7.2 库接口与 driver-target gate 边界 | 命令指南 |
| [`docs/guides/davf_perturbgen_e2e.md`](docs/guides/davf_perturbgen_e2e.md) | sidecar/downstream 段（本轮更新） | E2E 运行边界 |
| [`docs/guides/perturbgen_bridge.md`](docs/guides/perturbgen_bridge.md) | §3 步骤 3（rescue extractor）/ §6（验收命令） | 批次二产出引用 |
| [`src/integration/perturbgen/eval_assembly.py`](src/integration/perturbgen/eval_assembly.py) | `load_e2e_report`（宽松校验，新字段兼容证据） | 兼容性复核 |
| `lessons.md` | L-2026-0916-01（observed gate 功效限制）/ L-2026-0916-02（队列与 Frangieh 链）/ L-2026-0916-05（本轮：事实分类非判据、单队列不变性、centered 实测） | 决策记录 |
