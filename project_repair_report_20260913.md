# PTM2CellNet 工程契约修复与科学验收最终报告（2026-09-13）

## 结论先行

本轮必须把两件事分开：

1. **工程契约修复**：当前代码已经补齐或收紧若干 formal 输入边界，包括语义上下文、候选与 invocation 绑定、null 身份、Gate-E 必要证据、canonical Ensembl、冻结 cohort 的 route/mode，以及训练 donor metadata 的 fail-fast。
2. **科学验收**：仍未完成。当前没有生物学 PASS，也没有正式的候选 p/q、未扰动质量、双路径统计或 Gate-E 科学证据结论。

E2E report 当前只会明确写出 statistical_evidence 为 inconclusive、scientific_acceptance 为 false。它不会把现有统计接口自动接续成 null、quality、p/q 或双路径科学结论。当前工作树保留 21 个源码/测试文件的既有改动，统计为 **21 files / 576 insertions / 34 deletions**；本轮只同步本文档，不提交、不回滚、不修改这些源码/测试改动。

正式科学 PASS 仍受真实 cohort、合规 checkpoint、held-out donor、Gate-E benchmark、匹配 null、GPU 和统计 lineage 阻断。smoke、synthetic、mock、bridge 或单元测试通过都不等于 biology PASS。

## 事实来源与复核范围

本报告以代码、测试和当前验证记录为事实来源，使用以下章节作为需求与边界依据：

| 来源 | 复核内容 |
|---|---|
| project_analysis_20260913.md §4.2、§4.3、§4.4、§4.5、§6.1、§6.2、§7 | U-01～U-07 的现有接口状态、A-01～A-05 资产缺口、主线流程、F-01～F-09 技术债与后续顺序 |
| docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md §4.6、§4.7、§5.4、§7.2 | cohort、donor split、rescue/null、正式 PASS、科学质量门、M0～M7 停止条件 |
| lessons.md：L-2026-0913-01、L-2026-0902-01、L-2026-0902-02、L-2026-0902-03 | 方向语义、资产与索引契约、串联 gate、真实 bridge 不等于生物学验收 |
| docs/CURRENT_STATUS.md、docs/guides/davf_perturbgen_e2e.md | 当前公开状态和运行边界 |

源码复核范围包括：

- src/integration/perturbgen/contracts.py、orchestrator.py、null_generation.py、gate_e.py、frozen_cohort.py、direction_gate.py；
- src/models/davf_checkpoint_contract.py；
- scripts/run_davf_perturbgen_e2e.py、scripts/run_perturbgen_pipeline.py、scripts/train_latent_davf.py；
- src/data/davf_scperturb.py。

测试复核范围包括：

- tests/unit/integration/perturbgen/test_contracts.py、test_null_generation.py、test_gate_e.py、test_frozen_cohort.py、test_orchestrator.py；
- tests/unit/models/test_davf_checkpoint_contract.py；
- tests/unit/scripts/test_run_davf_perturbgen_e2e.py、test_run_perturbgen_pipeline.py、test_train_latent_davf.py；
- tests/integration/test_perturbgen_pipeline_mocked.py。

## 1. 本轮实际落地与严重程度汇总

### 高：F-04 语义合同进入 formal invocation

src/integration/perturbgen/contracts.py 新增 SemanticContext，固定七个字段：

- context
- intervention
- comparison_baseline
- reference_axis
- research_objective
- evidence_source
- cohort

research_objective 只接受 association、replication、reversal。缺字段、空值、非法 objective 会硬失败；formal PerturbGenInvocation 不接受缺失 semantic_context。CandidateEvidence 与 invocation 必须携带相同的七字段语义上下文，invocation 的 semantic_context.intervention 必须与 KO/KD route 相同，候选 gene、Ensembl、DAVF evidence 和 gate 状态也必须一致。

这解决的是“观测 donor-level disease−normal、DAVF decode delta 和 PerturbGen utility 被误写成同一个因果效应”的工程风险，不是参考轴已经由代码自动确定。参考轴仍需要研究定义。相关实现和测试见 src/integration/perturbgen/contracts.py、orchestrator.py、direction_gate.py，以及 tests/unit/integration/perturbgen/test_contracts.py、test_orchestrator.py、tests/unit/scripts/test_run_perturbgen_pipeline.py、tests/unit/scripts/test_run_davf_perturbgen_e2e.py。

KO/KD route 仍由显式 DAVF 配置决定。PTM site classifier 只表示 site presence；candidate direction 来自外部 proposal 或逐 site override，不能从 classifier 自动推断因果表达方向。

### 高：F-05/U-01 null stage record 身份绑定

src/integration/perturbgen/null_generation.py 的 NullStageRecord 绑定：

- candidate Ensembl ID；
- null Ensembl ID 与 null gene symbol；
- path；
- mode；
- seed；
- rescue_excl_target；
- stage manifest、result h5ad 及 result h5ad.sha256。

收集时会重新按 candidate/path/mode/seed 校验记录身份；selected nulls 排除 candidate，null 只执行 perturb-only stage，不创建伪造的 PerturbGenInvocation，也不把 null 当候选。GPU 路径必须提供显式 rescue_extractor，模块不会猜 DEG 列或 h5ad 来源。tests/unit/integration/perturbgen/test_null_generation.py 覆盖 99 个 null 的 dry-run、收集、candidate mismatch 和缺少 rescue extractor 的拒绝。

这只是已有 null 生成与收集接口的严格边界复核；当前没有执行正式的 >=99 null GPU 矩阵，也没有形成科学 p/q。

### 高：F-06 Gate-E 必要证据、canonical coverage 与 benchmark 硬失败

src/integration/perturbgen/gate_e.py 的 formal Gate-E report 必须同时有：

- benchmark section；
- vocabulary_migration section；
- davf_noninferiority section。

缺任一 required section 时 gate_e_passed 为 false；benchmark 缺必要列、ID 非法或样本数低于 200 时拒绝 formal 证据。vocabulary coverage 按 canonical Ensembl ID 计算，symbol-only vocabulary 不计覆盖；token collision 必须为 0，PTM action code agreement 必须为 100%。tests/unit/integration/perturbgen/test_gate_e.py 覆盖缺 section、至少 200、非法 Ensembl、symbol-only coverage、collision 和 paired DAVF 输入。

Gate-E 工具和硬失败边界存在，不等于当前已有 >=200 个可追溯 benchmark，也不等于 Gate-E 已通过。当前 A-04 仍未完成。

### 高：F-07 formal checkpoint.scVI gene_names 严格 canonical ENSG

src/models/davf_checkpoint_contract.py 现在要求 formal checkpoint.scvi.gene_names：

- 数量与 formal 4018 gene contract 一致；
- 每个值都是 canonical ENSG；
- 不带版本后缀；
- 不重复；
- 顺序与 scVI adapter 和 decoder vocabulary 一致。

这与冻结 embedding manifest、PerturbGen token vocabulary 和 scVI decoder index 的身份校验共同构成 formal asset contract；PerturbGen token index 不能冒充 scVI decoder index。tests/unit/models/test_davf_checkpoint_contract.py 覆盖 symbol、带版本后缀和 legacy checkpoint 的拒绝。

该硬校验暴露了本地旧资产 checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt 不合规：其 checkpoint.scvi.gene_names 不是 canonical ENSG。因此旧资产不能作为本轮 formal DAVF 结果，必须重训或重导出。

### 高：F-03 donor split metadata/row donor fail-fast

scripts/train_latent_davf.py 在显式 donor split 或要求 frozen manifest 时，要求 train/val NPZ metadata 同时包含：

- donor_split 且其 SHA 与 CLI split 一致；
- dataset.donor_rows 非空；
- 每个训练行的 donor 只能属于 train_donors。

缺 metadata、缺 donor_rows、SHA 不一致或某行 donor 越界都会直接失败，不再允许仅凭 donor 名单继续声称 held-out 无泄漏。tests/unit/scripts/test_train_latent_davf.py 覆盖缺 metadata 和训练行混入 held-out donor 的失败路径。

当前 src/data/davf_scperturb.py 生成的既有 pair metadata 仍需核对并重导出为含 row-level donor provenance 的资产；所以 F-03 的代码 fail-fast 已在，正式 held-out DAVF 结果仍未形成。

### 中高：F-08 frozen cohort 每行 modes 与 KO/KD route 校验

src/integration/perturbgen/frozen_cohort.py 将 intervention_type、modes、seeds 和 matched_nulls 记录到每个 FrozenCandidate 行，而不是用一个不分 route 的全局 mode 掩盖候选差异：

- KO formal sensitivity plan 要求 mask、pad、delete；
- KD 只能使用 mask，拒绝 pad/delete；
- overexpress 是独立的单 mode，不能与 KO sensitivity modes 混用；
- 每个候选至少 3 个 seed、至少 99 个 matched null。

train/held-out donor 仍要求至少 2/3 且互斥。tests/unit/integration/perturbgen/test_frozen_cohort.py 覆盖 manifest、acceptance plan、route/mode 和 donor leakage。这里是现有冻结 cohort 接口的 route 边界复核，不是已有真实 M6 PASS。

### 中高：F-01 E2E 只增加明确的统计状态，不虚报自动接续

scripts/run_davf_perturbgen_e2e.py 当前会明确写：

    statistical_evidence.status = "inconclusive"
    statistical_evidence.scientific_acceptance = false

未请求 PerturbGen 时 reason 为 perturbgen_not_requested；请求六阶段但未组装统计时 reason 为 statistical_evidence_not_assembled，并仅列出现有 null、report、empirical-p 和 dual-path 接口。

这一步只提供状态边界，绝不能写成统计自动接续已经实现。E2E 仍只汇总 stage results/stage_manifest；它没有自动调用 matched-null、未扰动质量提取、candidate p/q 聚合和 dual-path formal evaluator，也没有生成正式科学结论。

### 未闭合：F-02 跨候选公共 prepare/reuse

当前编排器能在单个 invocation 内规划六阶段，E2E 仍按候选建立输出目录并重复执行 tokenise、train_mask、train_decoder、两路 perturb、export_gene_embeddings 和 report。固定 cohort、词表、训练配置和资产版本后公共 prepare 一次、候选循环只运行两路 perturb/效用仍是目标，不是当前实现。

### 未闭合：F-09 outer invocation gate 与 runner 边界

正式 scripts/run_perturbgen_pipeline.py 在选择 perturb 或 --path 时要求绑定通过的 E2E gate report，并校验 gene、mode、canonical Ensembl、route、seed、path 和 config 绑定；orchestrator 只为 direction gate 通过的候选创建 invocation。

但低层 src/integration/perturbgen/runner.py 仍只执行 StagePlan、GPU lock、manifest 和 resume，不自行重查 gate。当前外层已有 gate，低层 runner 仍只跑 StagePlan；正式公开入口边界尚未统一到 runner。不能把 runner 缺口写成已解决，也不能因此放宽 outer invocation 要求。

### 既有接口/边界复核：U-02、U-03、U-04、U-06、U-07

U-02 candidate empirical-p 聚合、U-03 formal 输入隔离、U-04 未扰动质量提取、U-06 pathway 次级接口、U-07 Gate-E 前保留 Geneformer，及同类 U 条目在本报告只记为**现有接口/边界复核**：

- 它们不是本轮新增；
- 接口存在不等于真实资产已经运行；
- engineering/synthetic/mock 结果不等于 formal scientific result；
- U-06 pathway 不改变双路径硬判定；
- Gate-E 未完成前不能删除 Geneformer。

## 2. 系统性训练/E2E 推理复核

下表把代码现状、资产现状和正式 PASS 分开记录。

| 项目 | 代码现状 | 资产现状 | 能否正式 PASS |
|---|---|---|---|
| donor split / held-out | CLI、frozen manifest 绑定和训练 metadata/row donor fail-fast 已有；缺绑定时不能声明无泄漏 | 现有训练资产没有完成可追溯的 formal train-only/held-out row provenance | 否；需 A-02/A-03 |
| scVI gene order / embedding manifest | 当前 checkpoint contract 校验 64/4018、canonical ENSG、无版本后缀、无重复，并校验冻结 embedding manifest | 本地旧 checkpoint 的 gene_names 不合规 | 否；必须重训/重导出合规 checkpoint |
| DAVF gate | route-specific DAVF → decode delta → PTM proposal/independent expression 三方 gate；gate 不通过不生成 invocation；zero_fallback 证据被拒绝 | 本次 real_assets 失败暴露旧 checkpoint，不能转成真实方向证据 | 只能 PASS 工程准入，不能 PASS 生物学方向 |
| PerturbGen 六阶段 | 固定为 tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report；runner 执行 StagePlan | 只有本地小规模 smoke/模拟边界，未形成 formal cohort 运行 | 否 |
| 双路径 AND | source_intervention=[src] 与 within_state=[tgt]+pert_tps 分开执行；正式结果保留双路径 AND；KO/KD mode 规则已约束 | 没有合规 donor、三 seed 和可审计两路结果 | 否 |
| null / p/q / quality | null 生成、身份记录、candidate empirical-p、未扰动质量等接口存在；formal 缺输入会失败 | 没有正式 >=99 matched null × dual path × >=3 seeds 和真实 quality evidence | 否 |
| Gate-E | required sections、benchmark IDs、canonical Ensembl coverage、collision 和 paired non-inferiority 有硬边界 | 没有 >=200 个固定且可追溯 benchmark 的正式结果 | 否 |
| 统计自动接续 | E2E 明确写 inconclusive/false，不自动串接 null、quality、p/q、dual-path | 没有统计 lineage report | 否；F-01 未闭合 |
| 跨候选 prepare/reuse | 当前每候选重复准备和训练；没有公共 prepare/reuse 生命周期 | 无可复用的正式公共准备 manifest | 否；F-02 未闭合 |
| outer gate / runner | outer CLI 和 invocation 有 gate；runner 只执行 StagePlan | 其他低层 caller 仍可能绕过 outer formal wrapper | 当前只能按 outer 入口运行，F-09 未统一 |

上述判断对应 project_analysis_20260913.md §4.3、§4.4、§4.5、§6.2、§7，以及方案 §4.6、§4.7、§5.4、§7.2。L-2026-0902-01 要求当前 LatentDAVF/scVI/embedding asset 契约；L-2026-0902-02 要求串联 gate；L-2026-0902-03 明确真实 bridge 不等于 PerturbGen 生物学验收；L-2026-0913-01 又明确了方向语义、证据独立性和双路径执行边界。

## 3. 迭代验证记录

### 3.1 针对性 pytest

针对性 pytest 原始结果（原样记录）：

    100 passed / 8 warnings / 13.17s

这只证明本轮合同、身份绑定和边界测试通过，不证明真实 cohort、GPU null、Gate-E 或 biology PASS。

### 3.2 全量回归

原始命令：

    python -m pytest -m "not slow and not gpu" --timeout=300

原始结果：

    exit 1
    2623 passed / 1 failed / 21 skipped / 67 warnings / 800.73s

唯一失败：

    tests/integration/test_scvi_davf_connection.py::test_real_current_davf_direction_keeps_token_and_decoder_indices_separate

原因是本地 checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt 的 checkpoint.scvi.gene_names 不是 canonical ENSG，新的 contract hard-fail；随后 zero_fallback 证据被拒绝。这里没有代码回退，也不是通过 skip 或 fallback 掩盖失败。

### 3.3 静态与构建检查

| 命令 | 原始结果 |
|---|---|
| ruff check src scripts tests | 通过 |
| python -m mypy src/ --ignore-missing-imports | 165 个源文件，0 errors |
| python scripts/check_requirements_consistency.py | 通过，274 lock pins |
| python -m compileall -q src scripts | 通过 |
| python -m ruff format --check src scripts tests | exit 1；339 文件需格式化，124 文件已格式化；未批量改写 |

slow、gpu 本轮未执行。real_assets 没有形成正式验收。默认测试中的 mock、synthetic、smoke 和 bridge 只能证明工程接口或执行链，不等于 biology PASS。本次失败的 real_assets 标记测试恰好暴露旧 checkpoint，不得用 skip 或 fallback 掩盖。

### 3.4 当前工作树

当前源码/测试既有改动统计仍为：

    21 files / 576 insertions / 34 deletions

本轮没有提交，也没有执行 reset、checkout、回滚或长时间测试。

## 4. 未解决根因、策略与时间窗

时间以取得合规外部资产并完成负责人确认的工作日为起点；GPU 墙钟不折算成编码工作日。

| 项目 | 未解决根因与最低完成条件 | 依赖与时间窗 |
|---|---|---|
| A-01 合规 cohort | 提供真实 normal/disease raw counts、显式 donor、至少 3 个共享 donor、canonical Ensembl、冻结 scVI gene order 和 embedding/manifest；通过 Gate-0 | 外部数据 owner、anndata/scVI 处理；T0–T10 工作日；没有数据就停止科学声明 |
| 旧 checkpoint | 隔离 checkpoints/davf/latent_davf_perturbgen_4018/best_model.pt；按当前 LatentDAVF schema、canonical ENSG、冻结 embedding asset 和 manifest 重训/重导出 | 外部合规 scVI 与 GPU；A-01 后约 T+2–T+8 工作日，资产就绪后 1–3 个 GPU 日 |
| A-02/A-03 held-out DAVF | 冻结至少 2 个 train donor 与至少 3 个 held-out donor，写入 pair row metadata 和 checkpoint provenance，报告 held-out 方向指标 | A-01、真实 donor split、GPU；T+2–T+8 工作日 |
| A-04 Gate-E >=200 | 提供至少 200 个固定、可追溯 PTM→gene benchmark，old/new vocabulary 和 paired DAVF 结果全部齐全，满足 coverage、collision、action-code 和 non-inferiority 门 | benchmark owner、旧基线、DAVF 资产；可与 A-02/A-03 并行，T+5–T+12 工作日 |
| A-05 formal null | 对每个候选执行 >=99 matched null，覆盖两路径、route/mode 和 >=3 seeds，保留 stage/result hash、rescue、candidate p/q 和双路径 AND | A-01、旧 checkpoint、PerturbGen 独立环境、GPU/磁盘；A-01 后 T+3–T+10 GPU 日，墙钟另计 |
| F-01 统计自动接续 | 将已有 null、quality、empirical-p/q 和 dual-path 接口串入 E2E report，所有 provenance 写回同一 lineage；未定义的 estimand 先由统计负责人确认 | A-01/A-05、统计负责人、PerturbGen 维护者；接口实现约 3–5 工作日，不包含 GPU |
| F-02 公共 prepare/reuse | 固定 cohort、词表、训练配置和资产版本后公共执行 tokenise/train_mask/train_decoder；候选循环只执行两路 perturb/效用 | runner/orchestrator、外部环境、GPU 和磁盘；约 2–4 工作日，必须先设计可审计生命周期 |
| F-09 invocation/runner 边界 | 明确 formal invocation wrapper 是唯一公开正式入口，或让 runner 接受并验证 gate context；不得影响内部 StagePlan/null/engineering caller | API/runner owner；约 0.5–1.5 工作日；当前 outer gate 已有、低层 runner 仍未统一 |
| 语义参考轴 | 由研究负责人冻结 context、intervention、comparison_baseline、reference_axis、research_objective 和各证据来源；不使用全局同号/取反 | 统计与生物学负责人；约 1–3 工作日；必须先于正式候选验收 |

推荐顺序是：先冻结语义与 cohort/train-only/held-out 划分；取得 A-01 并重导出合规 checkpoint；再完成 F-09/F-02 的执行边界和 A-02/A-03；随后接续 F-01，执行 A-05；A-04 Gate-E 和 Gate-4/Gate-5 证据齐全后才讨论发布。不要新增 hash、调度框架或兼容开关来假设公共 reuse 已实现。

## 5. 公开入口与审计锚点

正式候选入口仍是：

    PTM site presence
    → external proposal / candidate_spec
    → scVI context + route-specific DAVF
    → independent donor-level expression direction
    → three-way direction gate
    → CandidateEvidence / PerturbGenInvocation
    → six stages
    → source_intervention + within_state
    → statistical evidence

source_intervention=[src] 表示状态转移前场景；within_state=[tgt]+pert_tps 表示目标状态内场景。单路结果只属于对应场景，不能外推成普遍治疗疗效；两路都通过才允许 formal dual-path AND。

关键源码与测试锚点：

- semantic_context、CandidateEvidence、PerturbGenInvocation：src/integration/perturbgen/contracts.py、orchestrator.py；tests/unit/integration/perturbgen/test_contracts.py、test_orchestrator.py；
- 三方 gate 与 route：src/integration/perturbgen/direction_gate.py、orchestrator.py；tests/unit/integration/perturbgen/test_orchestrator.py；
- null 计划、记录和非候选边界：src/integration/perturbgen/null_generation.py；tests/unit/integration/perturbgen/test_null_generation.py；
- Gate-E：src/integration/perturbgen/gate_e.py；tests/unit/integration/perturbgen/test_gate_e.py；
- frozen cohort、modes、donor split：src/integration/perturbgen/frozen_cohort.py、scripts/train_latent_davf.py、src/data/davf_scperturb.py；tests/unit/integration/perturbgen/test_frozen_cohort.py、tests/unit/scripts/test_train_latent_davf.py；
- checkpoint/scVI gene order：src/models/davf_checkpoint_contract.py；tests/unit/models/test_davf_checkpoint_contract.py；
- E2E 状态和 Gate-0：scripts/run_davf_perturbgen_e2e.py；tests/unit/scripts/test_run_davf_perturbgen_e2e.py；
- outer gate binding 和 runner 边界：scripts/run_perturbgen_pipeline.py、tests/unit/scripts/test_run_perturbgen_pipeline.py、tests/integration/test_perturbgen_pipeline_mocked.py。

最终判定：本轮完成的是工程契约收紧与真实失败暴露；正式科学验收仍为 inconclusive，**没有生物学 PASS**。
