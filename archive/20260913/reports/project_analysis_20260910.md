# PTM2CellNet 最终代码修复与 E2E 收口分析报告

日期：2026-09-10
范围：当前工作树中的全部既有未提交修改；本报告不以提交、回滚或清理工作树为前提。

## 1. 执行摘要

本次按 project_repair_report_20260910.md 的未完成项重新核对了代码、测试和文档，并直接完成了仍影响主线目标的修复。结论如下：

- G-1 / N-05 的代码缺口已补齐：显式 cohort 的 deterministic matched-null 选择、selection manifest、每条 null record 的 candidate/path/mode/seed/rescue_excl_target 绑定、严格 99 条下限，以及从成功 stage manifest 到 dual-path eval input 的组装链已经存在并有回归测试。
- G-2 已补齐：DAVF 方向推理输出有限的 confidence，明确标注为 bounded relative effect-size proxy，不冒充 calibrated probability；direction gate 将同一数值写入 candidate.davf_score。
- G-3 已收口：PerturbGenInvocation 和 run_perturbgen_pipeline 的 perturb/path 执行都要求显式、结构完整且通过的 E2E gate report；JSON 边界和库级对象均严格校验 candidate、invocation、DAVF evidence 的身份及 score/confidence 一致性。
- TD-NEW-05 的目标入口已统一到 dependency_check.py 的可选依赖检查，并保留核心调用边界的硬 ImportError 行为；TD-NEW-10 的 PMADS 目标路径不再使用 iterrows，并有固定 golden feature-matrix 回归。未拆 TD-NEW-07/15 的大文件或长函数。
- E2E 的工程链已经可以从候选、DAVF gate、双路径 stage、manifest、null distribution 到重放评估形成可审计闭环；但本机没有合规的 normal/disease raw-count cohort，因此没有执行正式生物学验收，也没有把 offline、synthetic、mock 或本地 smoke 结果写成 biology PASS。

最终离线回归是在最后一次代码变更之后执行的：

    python -m pytest -m 'not slow and not gpu and not real_assets' --timeout=600 -q
    2573 passed, 15 skipped, 7 deselected, 55 warnings in 805.68s (0:13:25), exit 0

静态结果是 Ruff check、mypy、compileall、requirements consistency 和 git diff --check 通过；全仓 Ruff format check 未通过，报告在第 6 节保留了完整事实，不把它写成通过。

本结论与主线决策一致：lessons.md 的 L-2026-0901-01 要求 DAVF 负责方向筛选、PerturbGen 负责下游效用；L-2026-0902-02 要求串联 gate 不可绕过；L-2026-0902-03 要求真实桥接或 smoke 不能替代生物学验收。

## 2. 依据、范围和证据口径

本轮实际读取并以其当前内容为依据的入口：

1. project_repair_report_20260910.md：§2.1 的 N-01～N-05、§2.2 的判定链补腿、§2.4 的 TD-NEW-05/10/07/15、§5.2 的 G-1～G-6。
2. docs/CURRENT_STATUS.md:3-26：当前测试基线、PerturbGen smoke、Gate-0 donor 审计和真实资产边界。
3. lessons.md:L-2026-0821-02、L-2026-0822-04～06、L-2026-0901-01、L-2026-0902-01～03：底座资产、manifest、donor cohort、DAVF/PerturbGen 分工和 gate 口径。
4. docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md：§4.1/§4.3/§4.6/§4.7 的工作流和判定契约，§5.1/§5.4 的 T4、T5、Gate-E，§7.2 的 M3/M4/M6。
5. docs/guides/perturbgen_bridge.md：v1.5（2026-09-10）的实际入口、gate hard-fail、null manifest、Gate-0/Gate-E/Gate-5 状态。

工程测试证据和科学证据分开记账。默认 pytest 使用 fixture、synthetic 或 mock；real_assets 标记和外部 PerturbGen 环境没有在本轮打开。不能由本报告的 2573 个离线通过用例推导 DAVF 方向准确率、双路径 rescue 生物学效用、Gate-E 非劣或正式 release PASS。

## 3. 本轮实际修复清单

### 3.1 G-1 matched null 与 null distribution 契约

文件和符号：

- src/integration/perturbgen/null_selection.py:155-264 的 select_matched_nulls：
  - 只接受显式 cohort.var['ensembl_id']，统一去掉 ENSG version suffix，并拒绝重复或非 ENSG ID。
  - 从 cohort.X 计算 mean_expression 和 detection_rate；从 feature_table 读取 log2fc/fc/fold_change；从 token vocabulary 记录 token_rank。
  - 按固定 feature_order 做归一化距离匹配，最后按 distance、canonical ENSG 排序，保证同一输入确定性选择。
  - excluded_ids 明确等于 target 加全部 candidate；target 和 candidates 不进入 eligible null；required_count 及可用数量至少为 99。
  - 返回 target features、feature sources、excluded_candidate_ids、selected_nulls 和距离，供 selection manifest 审计。
- src/integration/perturbgen/null_selection.py:274-350 的 _validate_selection_manifest 和 write_null_selection_manifest：
  - 要求 target、全部 candidate、excluded_ids 三者一致；
  - 拒绝 target/candidate 出现在 selected null、重复 selected ID、非有限 feature/distance；
  - 对 schema、feature order、matching vector 和至少 99 个 null 做硬校验。
- src/integration/perturbgen/null_selection.py:392-464 的 summarize_null_distribution：
  - 每条记录必须含 candidate_ensembl_id、null_ensembl_id、path、mode、seed、rescue_excl_target；
  - candidate/path/mode/seed 必须与汇总参数相同，null ID 不得等于 candidate；
  - 重复 null ID、非有限 rescue 值、缺失字段和少于 99 条都直接失败；
  - 输出 schema 为 perturbgen_null_distribution/v1，并保留 null_ensembl_ids 和 selection_manifest_path。
- src/integration/perturbgen/null_selection.py:467-625 的 loader：
  - 默认严格要求带 schema、candidate/path/mode/seed/null IDs 的 manifest 和至少 99 个值；
  - indexed manifest 必须对每个 path+mode+seed+candidate 选择唯一记录；
  - 旧的无绑定 list/mapping 只有在显式 legacy 调用中才允许，且只接受非空、有限值；一旦要求绑定就硬失败。

对应测试：tests/unit/integration/perturbgen/test_null_selection.py:50-207，覆盖 canonical ENSG、四种匹配特征、确定性、target+candidate 排除、99 下限、重复/NaN/binding 错误和旧 payload 边界。

### 3.2 N-05 E2E 输出组装与 inline null 绑定

- src/integration/perturbgen/eval_assembly.py:54-149 的 load_e2e_report 和 resolve_run_artifacts 只接受 schema 正确、status=success、含 result_h5ad 的 perturb stage manifest；文件名的 gene、src/tgt sequence、mode 与 stage path 交叉校验，seed 只能从 fingerprint_material.random_seed 读取。
- src/integration/perturbgen/eval_assembly.py:188-321 的 build_eval_input_payload 把 DEG、candidate p-value、质量状态和 null distribution 作为显式输入，不从 E2E 报告编造证据。严格 manifest 逐条按 candidate/path/mode/seed 加载。
- src/integration/perturbgen/eval_assembly.py:351-395 的 _passing_invocation 只允许 preparation status=pass、candidate.direction_gate_status=pass 和 invocation candidate gate=pass 的候选进入评估；失败 gate 或缺 invocation 不能被 assembly 隐式升级。
- scripts/build_dual_path_eval_input.py:31-70 同时提供显式 legacy --null-distribution 和严格 --null-distribution-manifest 两条入口。
- scripts/evaluate_perturbgen_dual_path.py:213-250 对 assembler 写入的 inline null_distribution 再从 manifest 按四元组加载，并要求 inline 数值逐项完全相等；:329-360 只保留旧 null_distribution_path 的明确兼容校验，不能用它满足严格绑定。

对应测试：tests/unit/integration/perturbgen/test_eval_assembly.py:193-339、tests/unit/scripts/test_evaluate_perturbgen_dual_path.py；覆盖成功/失败 stage、path-sequence mismatch、seed 缺失、每个 path+mode+seed 的 indexed manifest、inline 值篡改和短 legacy 分布。

### 3.3 G-2 DAVF confidence 与 direction gate

- src/models/davf_inference.py:970-1008 的 predict_expression_direction 仍严格使用 scVI decoder gene order 和显式资产。
- src/models/davf_inference.py:1135-1138 将 delta 转为 abs(delta)/(abs(perturbed)+abs(baseline))，零分母定义为 0，之后检查 finite 并限制在 [0,1]。
- src/integration/perturbgen/contracts.py:65-121 明确写出 confidence 是 bounded relative effect-size proxy，不是 calibrated probability，并拒绝非有限或越界值。
- src/integration/perturbgen/direction_gate.py:130-170 在三方 gate 通过后写入 davf_score=davf_evidence.confidence；不通过时不创建 candidate。

对应测试：tests/unit/integration/perturbgen/test_direction_gate.py:153-202、tests/unit/test_davf_direction_integration.py、tests/unit/test_davf_inference.py。

### 3.4 G-3 gate hard-fail、报告边界和模式/seed 产物

- src/integration/perturbgen/orchestrator.py:64-149 的 PerturbGenInvocation 校验 gene/Ensembl、gate status、方向、provenance、finite score/confidence，并要求 candidate.davf_score 与 davf_evidence.confidence 精确相等；测试回归位于 tests/unit/integration/perturbgen/test_orchestrator.py:100-125。
- scripts/run_perturbgen_pipeline.py:56-121 的 _validate_e2e_gate_report 现在要求：
  - candidate、invocation、invocation.candidate、invocation.davf_evidence 都是完整 mapping；
  - record 状态不是 fail，两个 candidate gate 都是 pass；
  - gene_symbol/ensembl_id 在 candidate、invocation、nested candidate、DAVF evidence 之间完全绑定；
  - 三份 score/confidence 都是有限 [0,1] 数值且精确相等。
- scripts/run_perturbgen_pipeline.py:184-189 对 perturb stage 或 --path 缺少 --e2e-gate-report 直接 parser.error；training-only stage 不被不必要地阻断。
- src/integration/perturbgen/orchestrator.py:427-488、:572-740 将 materialized expected output glob 随 mask/pad/delete/overexpress 模式重写，避免 sensitivity 或 overexpress 仍搜索 tmask 文件；src/integration/perturbgen/runner.py:259-275 按 driver=perturb_script 解析 auto dimensions，修复 path 名称为 source_intervention/within_state 后的真实 E2E 断点。

对应测试：tests/unit/scripts/test_run_perturbgen_pipeline.py:56-178、tests/unit/integration/perturbgen/test_orchestrator.py:100-210、tests/integration/test_perturbgen_pipeline_mocked.py:330-420。

### 3.5 TD-NEW-05、TD-NEW-10 和明确跳过项

- src/utils/dependency_check.py:56-210 提供统一 DependencyStatus、check_dependency、check_extras、require_extras 及安装提示。
- src/analysis/ibd_qc.py:27-42、src/data/davf_scperturb.py:44-50、src/models/scvi_adapter.py:79-92 使用同一可选依赖边界；scVI/anndata/scanpy 缺失或 broken install 在真正调用边界报告 ImportError，不把缺包静默当成正常结果。回归见 tests/unit/test_dependency_check.py、tests/unit/analysis/test_ibd_qc.py、tests/unit/data/test_davf_scperturb.py、tests/unit/test_scvi_adapter.py。
- src/baselines/pmads_ridge.py:343-377 的 _feature_matrix 使用 to_dict(orient="records") 和数值列缓存，不再调用 iterrows；tests/unit/baselines/test_pmads_ridge.py:52-146 固定了 feature matrix golden 值和空输入行为。序列/PTM 解析仍是逐行语义计算，本轮没有把它虚报为全量向量化。
- 未拆分 TD-NEW-07/15 的大文件/长函数，符合本次明确范围；它们在剩余项中保留为未完成，不宣称修复。

### 3.6 文档同步

docs/guides/perturbgen_bridge.md 已更新到 v1.5（2026-09-10），同步了 gate hard-fail、strict null manifest、N-05 入口、Gate-0 0 合规 cohort 和 Gate-5 未执行状态，并去掉与当前代码不一致的旧百分比和旧日期。文档中的工程 smoke、真实资产和科学验收仍分开描述。

## 4. 从训练到推理的 E2E 链路审计

| 链路 | 源码事实 | 当前判定 |
|---|---|---|
| 数据与 DAVF 训练输入 | src/integration/perturbgen/data_prep.py、src/data/gse_normal_disease.py、scripts/train_latent_davf.py；方案 §4.6 要求 raw counts、无版本 ENSG、state/cell_type/donor、训练 donor 与 held-out donor 分离 | 工具契约有测试；本机没有满足正式 normal/disease donor 的新 cohort，未执行正式科学训练验收 |
| PTM proposal 到 candidate spec | src/integration/perturbgen/candidate_spec.py、scripts/build_candidate_spec.py；方向值必须显式提供或来自独立 direction map，不能从 observed 证据反推 | N-01/N-02 代码和 tests/unit/integration/perturbgen/test_candidate_spec.py 通过 |
| scVI latent 与 DAVF direction | src/models/scvi_adapter.py、src/models/davf_inference.py、src/integration/perturbgen/direction_gate.py；decoder index 取 scVI gene order，不能使用 PerturbGen token index | 工程接口和离线回归通过；confidence 是 proxy，未做 calibration 或 held-out biology claim |
| 三方 gate 与 invocation | src/integration/perturbgen/orchestrator.py、scripts/run_davf_perturbgen_e2e.py；PTM proposal、DAVF 方向、独立 normal/disease expression direction 必须同时通过 | G-2/G-3 已硬化；失败 candidate 不生成 invocation，报告边界也再次校验 |
| PerturbGen 六阶段训练/推理 | src/integration/perturbgen/config_builder.py、runner.py、orchestrator.py、scripts/run_perturbgen_pipeline.py；source_intervention 与 within_state 独立计划，多 seed 和 KO pad/delete sensitivity 独立目录 | mocked/integration 证明计划、seed、mode、manifest 接线；本轮没有下载或调用外部权重，也没有把 smoke 当正式 donor 结果 |
| 输出 manifest 与 h5ad | src/integration/perturbgen/runner.py、src/integration/perturbgen/eval_assembly.py；stage status、result_h5ad、文件名 path/mode、fingerprint seed 都需要真实存在且一致 | N-05 组装器和失败/错绑回归通过 |
| matched null 与统计重放 | null_selection.py、build_dual_path_eval_input.py、evaluate_perturbgen_dual_path.py、src/integration/perturbgen/results.py、dual_path.py | 代码闭环和 ≥99/schema/binding 硬门通过；真实 cohort 上的 distribution 尚未生成 |
| 冻结验收与 Gate-E | src/integration/perturbgen/frozen_cohort.py、gate_e.py、scripts/run_frozen_acceptance.py、scripts/evaluate_gate_e.py；方案 §5.1/§5.4、§7.2 M4/M6 | 编排和独立重算工具、单测已在；真实 T4/Gate-5 和 ≥200 样本 Gate-E 未运行 |

因此，当前可以确认的是“工程链可执行、输入输出契约可审计、错误会硬中断”；不能确认“候选在真实疾病队列上具有生物学效用”。

## 5. 剩余高/中/低项、根因和收口条件

### 5.1 高等级或科学阻塞项

| 项目 | 当前根因 | 负责人/输入 | 完成条件 | 预计时间 |
|---|---|---|---|---|
| G-4 / Gate-0 合规 cohort | docs/CURRENT_STATUS.md:21-24 和 outputs/perturbgen/spike/20260903_donor_audit/evidence.json 表明本机审计的 30 个 scPerturb 文件没有满足正式 normal/disease + raw counts + explicit donor + 至少 3 个 shared donor + Ensembl 的队列；部分样本缺 state/donor，通用 scPerturb 不能替代冻结 cohort | 用户或数据提供方；提供真实 normal/disease H5AD、layers['counts']、canonical ENSG、cell_type/state/donor、至少 3 个共享 donor | data_prep/preflight 逐项 PASS；产出可追溯 cohort manifest 和 donor audit；train/held-out donor 无交集且 held-out ≥3 | 外部输入到位后约 0.5–1 天预检；数据到位时间不由本机决定 |
| G-1 的正式运行证据 | null_selection 和 distribution loader 已完成，但本轮没有合规 cohort 和外部 PerturbGen 结果，因此没有生成可用于正式候选的 selection/distribution manifests | 实验执行者；合规 cohort、candidate spec、独立 PerturbGen 环境和真实 h5ad | 每个 candidate/path/mode/seed 有 selection manifest、至少 99 个排除目标和全部候选的 null、99 条有限 rescue 记录、独立可重放 manifest；再运行 N-05 和 BH-FDR | cohort 与环境具备后约 1–2 天，取决于 stage 运行时间 |

### 5.2 中等级项

| 项目 | 当前状态和根因 | 完成条件与负责人 | 预计时间 |
|---|---|---|---|
| G-5 / Gate-E | gate_e.py 和 evaluate_gate_e.py 已实现，但没有用户提供的至少 200 个可追溯 PTM→gene 样本及旧冻结基线；本轮没有伪造 benchmark | 研究/数据负责人提供固定划分、GeneMap/CPTAC 来源、旧基线和新底座资产；满足 coverage ≥99%、token collision=0、action code 100%、下游指标相对基线不下降超过约 1 个百分点且 CI 不劣化 | 资产齐后约 0.5 天准备、2–6 小时计算；以实际 GPU/数据规模为准 |
| G-6 / T4 或 Gate-5 release gate | frozen_cohort.py 和 run_frozen_acceptance.py 已有 manifest、donor 泄漏检查、验收矩阵和独立重算；但没有真实冻结 cohort、held-out donor、3 seeds、双路径和真实 ≥99 null evidence | release/实验负责人提供冻结 H5AD、donor 划分、真实 PerturbGen 产物和可留存 evidence；每候选通过双路径、seed/mode/null/FDR，并由独立 replay 复核 | G-4 后约 0.5–1 天执行；外部 GPU/队列时间另计 |
| TD-NEW-05 的全仓风格收束 | 本轮已统一目标入口和核心 ImportError 行为；仓库仍有历史上其他可选库的 lazy/availability 代码，本轮没有在用户跳过的范围外做全仓重写 | 若要求全仓单一风格，另做按模块清单迁移并保持每个 feature 的 hard import boundary；当前不构成主线 gate 缺口 | 约 1 天专项审计，不纳入本轮完成 |

### 5.3 低等级或明确延期项

- TD-NEW-07/15：大文件、长函数拆分按用户要求未做；若后续需要，先按现有测试拆段、每段单独回归，预计 2–3 天，不应与本次功能收口混做。
- TD-NEW-10：目标 iterrows 已移除，PMADS golden 已有；如果要继续优化，应另行做向量化与数值对拍，不能仅凭性能猜测改写。
- 全仓格式债：python -m ruff format --check src scripts tests 真实结果为 329 files would be reformatted, 124 files already formatted，涉及大量既有文件。本轮没有用批量格式化制造无关 diff；这是低级维护项，不是功能通过证据。
- legacy null_distribution_path：只作为明确兼容路径保留；正式结果必须使用带 candidate/path/mode/seed 的 manifest。不得把兼容路径的短分布当成正式 PASS。

## 6. 实际验证命令与结果

以下结果均来自当前工作树，不包含猜测结果：

| 检查 | 实际命令 | 实际结果 |
|---|---|---|
| 相关 broad focus | python -m pytest -q（G-1/G-2/G-3/optional dependency/PMADS/PerturbGen 相关套件，命令包含 test_null_selection.py、test_eval_assembly.py、test_direction_gate.py、test_orchestrator.py、test_run_perturbgen_pipeline.py、test_evaluate_perturbgen_dual_path.py、dependency、IBD、scPerturb、scVI、PMADS、mocked pipeline） --timeout=600 | 129 passed, 2 skipped, 14 warnings in 5.56s |
| 最后 gate/assembly/orchestrator 回归 | python -m pytest -q tests/unit/scripts/test_run_perturbgen_pipeline.py tests/unit/integration/perturbgen/test_orchestrator.py tests/unit/integration/perturbgen/test_eval_assembly.py --timeout=600 | 32 passed, 8 warnings in 2.90s |
| 离线全量回归 | python -m pytest -m 'not slow and not gpu and not real_assets' --timeout=600 -q | 2573 passed, 15 skipped, 7 deselected, 55 warnings in 805.68s (0:13:25), exit 0 |
| Ruff lint | ruff check src scripts tests | All checks passed, exit 0 |
| Ruff format | python -m ruff format --check src scripts tests | 失败，329 files would be reformatted，124 files already formatted，exit 1；未把它写成通过 |
| 类型 | python -m mypy src/ --ignore-missing-imports | Success: no issues found in 161 source files，exit 0 |
| 编译 | python -m compileall -q src scripts tests | exit 0 |
| 依赖一致性 | python scripts/check_requirements_consistency.py | requirements/lock consistency OK: 274 lock pins satisfy all core constraints (optional tracks checked where present)，exit 0 |
| diff 空白 | git diff --check | exit 0 |

全量命令按要求排除了 slow、gpu、real_assets；没有运行真实资产测试、没有下载权重或外部数据、没有刷新 coverage。测试中的 warnings 主要是 Pydantic v1 API、torch/mamba CUDA API、Lightning dataloader worker、scVI 小 category、anndata index、历史 roadmap deprecation；它们没有导致本轮失败，但也没有被隐藏。

## 7. 工作树变更范围

所有已有未提交修改均保留，没有执行 git reset、git checkout 或 git clean。与本次收口直接相关的文件包括：

- N-01/N-02/N-03/N-04/N-05 新增模块和入口：src/integration/perturbgen/candidate_spec.py、eval_assembly.py、frozen_cohort.py、gate_e.py、null_selection.py，以及 scripts/build_candidate_spec.py、build_dual_path_eval_input.py、evaluate_gate_e.py、run_frozen_acceptance.py。
- G-1/G-2/G-3 和 E2E：src/integration/perturbgen/orchestrator.py、runner.py、direction_gate.py、contracts.py、reports.py、results.py、scripts/run_davf_perturbgen_e2e.py、scripts/run_perturbgen_pipeline.py、scripts/evaluate_perturbgen_dual_path.py、docs/guides/perturbgen_bridge.md。
- 可选依赖、PMADS、类型和数据契约：src/utils/dependency_check.py、src/analysis/ibd_qc.py、src/data/davf_scperturb.py、src/models/scvi_adapter.py、src/baselines/pmads_ridge.py、scripts/predict.py、scripts/predict_ptm_sites.py，以及对应 tests。
- 回归测试：tests/unit/integration/perturbgen/test_candidate_spec.py、test_eval_assembly.py、test_frozen_cohort.py、test_gate_e.py、test_null_selection.py、test_orchestrator.py、tests/unit/scripts/test_run_perturbgen_pipeline.py、tests/integration/test_perturbgen_pipeline_mocked.py，以及 optional dependency、DAVF、PMADS 相关测试。

工作树当前还有其余同一任务批次的修改，最终以 git status --short 为准；本报告没有覆盖或重置它们。

## 8. 引用索引

### 需求与修复输入

- project_repair_report_20260910.md：§2.1 N-01～N-05、§2.2 判定链补腿、§2.4 TD-NEW-05/10/07/15、§5.2 G-1～G-6、§6 后续策略。
- docs/CURRENT_STATUS.md:3-26、:36-42：2026-09-10 当前基线、Gate-0 donor 审计、真实资产与 mock 证据边界。
- docs/guides/perturbgen_bridge.md:1-33、:51-81、:196-222：当前命令、data contract、gate hard-fail、N-05 和 Gate 状态。

### 方案和项目决策

- docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md §4.1：两条工作流；§4.3：CandidateEvidence 和方向契约；§4.6：raw counts/ENSG/donor/held-out；§4.7：rescue、null、donor、FDR、双路径 AND；§5.1：T4/T5；§5.4：Gate-E 阈值；§7.2：M3/M4/M6。
- lessons.md:L-2026-0822-04：动态产物必须从 manifest 绑定；L-2026-0822-05：正式证据必须可留存；L-2026-0822-06：通用 scPerturb 不能冒充冻结 donor cohort。
- lessons.md:L-2026-0901-01：DAVF 方向筛选与 PerturbGen 效用分工；L-2026-0902-01：LatentDAVF/scVI/asset schema；L-2026-0902-02：串联 gate 不可绕过；L-2026-0902-03：真实桥接不等于生物学验收。

### 源码与测试

- src/integration/perturbgen/null_selection.py:155-264、:274-350、:392-625；tests/unit/integration/perturbgen/test_null_selection.py:50-207。
- src/integration/perturbgen/eval_assembly.py:54-149、:188-395；scripts/build_dual_path_eval_input.py:31-70；scripts/evaluate_perturbgen_dual_path.py:213-250、:329-360；tests/unit/integration/perturbgen/test_eval_assembly.py:193-339。
- src/models/davf_inference.py:970-1008、:1135-1138；src/integration/perturbgen/contracts.py:65-121；src/integration/perturbgen/direction_gate.py:130-170；tests/unit/integration/perturbgen/test_direction_gate.py:153-202。
- src/integration/perturbgen/orchestrator.py:64-149、:160-204、:427-740；src/integration/perturbgen/runner.py:259-275；scripts/run_perturbgen_pipeline.py:56-189；tests/unit/scripts/test_run_perturbgen_pipeline.py:56-178；tests/unit/integration/perturbgen/test_orchestrator.py:100-210。
- src/utils/dependency_check.py:56-210；src/analysis/ibd_qc.py:27-42；src/data/davf_scperturb.py:44-50；src/models/scvi_adapter.py:79-92；tests/unit/test_dependency_check.py、tests/unit/analysis/test_ibd_qc.py、tests/unit/data/test_davf_scperturb.py、tests/unit/test_scvi_adapter.py。
- src/baselines/pmads_ridge.py:343-377；tests/unit/baselines/test_pmads_ridge.py:52-146。

## 9. 最终结论

代码侧要求 G-1/N-05、G-2、G-3、TD-NEW-05 目标入口和 TD-NEW-10 iterrows 已按当前契约完成并通过离线验证；TD-NEW-07/15 明确未动。主线可以在获得合规真实 cohort 后继续执行，不需要再用手写 null 数组、未绑定的旧路径或无 gate 的 runner 旁路掩盖缺口。

当前唯一决定正式科学结论的阻塞仍是外部数据和真实验收：合规 normal/disease raw counts、显式 donor、至少 3 个 shared donor、Gate-E 至少 200 benchmark、以及 T4/Gate-5 的真实多 seed 双路径 evidence。它们的负责人、输入、完成条件和预计时间已在第 5 节逐项列出。在这些输入到位并实际运行前，本项目的正确结论是“工程闭环通过，生物学验收未完成”。
