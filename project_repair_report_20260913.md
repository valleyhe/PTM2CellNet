# PTM2CellNet E2E 代码断点修复报告（2026-09-13）

> 本报告先以初稿落盘，再完成源码修复和检查；不覆盖 2026-09-10 的历史检查结果。
>
> 范围：保留当前工作树全部既有改动；本轮没有创建 worktree、commit 或启动外部 Codex CLI。`frozen_cohort.py`、`scripts/run_frozen_acceptance.py` 及其专属 fixture 由 Volta worker 负责，本 worker 没有修改；Volta 已提供真实工程检查结果，已在本报告中与本轮成果分开记录。

## 1. 结论先行

本轮已落地并验证以下工程修复：

- `scripts/run_perturbgen_pipeline.py` 先解析真实 base YAML，再将通过 invocation 的 gene、mode、声明的 Ensembl ID/route、`pipeline.random_seed` 与请求的 path 子集绑定；report 若显式带 `perturbgen_config_path`，还必须指向当前 base config。没有 legacy helper、异常 fallback 或空 config 免校验分支。
- `dual_path.py` 现在显式接收 KO/KD route：KO/up 要求 `mask,pad,delete` 并执行二取三；KD/up 只要求 `mask`；down 的 `overexpress` 与 route 独立。evaluator、`src/integration/perturbgen/mainline.py` 和 Volta 完成的 frozen replay 已显式传 route。
- N-05 assembler 以成功 perturb manifest 的精确 `src_dataset/src_h5ad/tgt_dataset_folder/tgt_h5ad_folder` 消费链寻找唯一 tokenise stage manifest，写入精确 `tokenise_stage_manifest`；不按目录 latest、同名文件或 hash 猜身份。
- N-05 assembler 现在还严格匹配 runner `stage_manifest.outputs` 中与 `artifacts.result_h5ad` 相同的实际输出记录，把 runner 已登记的 `sha256` 原值写入 `h5ad_provenance`；不在 assembler 新算 hash。
- evaluator 每个 run 记录实际传给 extractor 的 `target_gene`、donor/DEG 列名、阈值、donor 下限、`top_k`、bootstrap iterations 和 seed；缺少 bootstrap seed 直接硬失败。顶层 report manifest entry 复用已生成的 `manifest["replay"]`，不再让 frozen replay 只读不存在的字段。
- `run_davf_perturbgen_e2e.py` Gate-0 继续在准备副本上做校验，但已登记的外部 tokeniser 实际读取原始 `tokenise.args.h5ad_path`，所以本轮对原始输入的 Ensembl 列做 canonical 硬检查；登记源码 `GF_tokenisation.py:223-226,238,301-305` 已静态核对原始 counts→`X`→恢复 counts→重算 `n_counts` 的接线，prepared copy 的新增 `n_counts` 不需 materialize；本轮未运行真实 tokenisation，也没有新增 X/layer 转换。

Volta 已完成冻结 verifier/OE/replay 的工程接线和其针对性检查；其真实 M6 cohort 尚未运行。首次旧 frozen fixture 检查的 4 个失败是 fake H5AD、缺 schema 或旧调用缺 manifest，随后仅更新既有 fixture 到当前契约并验证 14 passed，未通过弱化生产契约放行。CSV 类型根因也已修正，明确 synthetic AnnData/runner metadata 的 assembler→evaluator→frozen 生成链已独立重算并验证篡改失败。上述结果只证明工程边界和错误传播，不证明生物学结果。真实合规 normal/disease raw-count cohort、训练-only/held-out donor lineage、Gate-E ≥200 benchmark、正式多 seed/null、冻结 M6/OE 均未完成正式资产验收。不能把 `null_selection.py` 的选择/汇总/loader、G-1、BH 或规则重放称为完整自动化科学闭环。

## 2. 事实来源与章节引用

- 本修复完成时的根目录分析是 [`project_analysis_20260910.md`](archive/20260913/reports/project_analysis_20260910.md)（现已归档）。本报告具体更正其 §4“从训练到推理的 E2E 链路审计”和 §9“最终结论”中过强的 G-3/完整闭环/唯一外部阻塞表述；不改写其 9/10 检查日期和历史命令结果。当前权威分析是 [`project_analysis_20260913.md`](project_analysis_20260913.md)。
- [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) 的 `Quick Reference`、`解释测试结果的边界` 保留 9/10 基线；新增 `2026-09-13 本轮修复状态` 提供本报告入口及本轮边界。该文件没有按不存在的 §1–§4 章节引用。
- [`lessons.md`](lessons.md) 的 L-2026-0821-01、L-2026-0822-04～06、L-2026-0901-01、L-2026-0902-01～03 分别约束双路径主线、manifest 绑定、单次自洽证据、真实 donor cohort、DAVF/PerturbGen 分工、LatentDAVF/scVI/asset schema、串联 gate 和“真实桥接不等于生物学验收”。
- [`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md) §4.1/§4.3/§4.6/§4.7、§5.1/§5.4、§7.2/§7.3 是本轮 E2E、donor、rescue/null、Gate-E 和 M3/M4/M6 判断依据。
- [`docs/guides/davf_perturbgen_e2e.md`](docs/guides/davf_perturbgen_e2e.md) 的 `核心原则`、`候选清单`、`只执行真实 DAVF 与方向 gate`、`触发六阶段 PerturbGen`、`当前真实边界`，以及 [`docs/guides/perturbgen_bridge.md`](docs/guides/perturbgen_bridge.md) §1、§3、§4、§5.1、§6、§7 已同步当前入口和限制。
- 已查阅登记外部源码 `ref/Perturbgen-src`（commit `a9a9375`）：`perturbgen/pp/GF_tokenisation.py:223-226` 在原始 counts 存在时明确把 `adata.X` 置为 `layers['counts'].copy()`，`:238` 在 before-HVG 后恢复 `X=counts`，`:301-305` 重新计算 `n_counts`；同一文件直接读取原始 `args.h5ad_path` 并从 `var['ensembl_id']` 建 token gene name，`perturbgen/pp/tokenizer.py` 再读取生成 h5ad 的 `data.X` 和 `n_counts`。这是已登记源码契约；本轮未运行真实 tokenisation。
- `scripts/train_latent_davf.py:285-328` 可核对正式 asset/scVI/schema/route/方向契约，但没有直接 donor split 输入；训练输入契约与正式冻结 donor 验证在本报告 §6 分开记账。

## 3. 既有成果与本轮成果

### 3.1 既有成果（截至 2026-09-10，保留历史归属）

9/10 分析及当前工作树此前已有的 N-01～N-05、G-1 null selection/summary/loader、G-2 confidence/direction gate、orchestrator/stage manifest、Gate-E 工具、scVI/LatentDAVF schema、可选依赖边界和 PMADS 改动，本轮不重新归功，也不撤销。9/10 的离线回归数字继续作为历史结果保存。

其中必须更正两点：

1. `null_selection.py` 没有按候选批量生成和执行 null stage、收集 null rescue 的生成端；它只有选择、汇总和 loader。因此 G-1/null/BH 链不能称完全自动化工程闭环。
2. 9/10 所称 G-3 report binding 已闭合只覆盖结构/对象层校验，不能替代 report 与当前 YAML 的 gene/mode/path/config 身份绑定；candidate p 也仍可由外部输入改变正式 verdict。

### 3.2 本轮直接改动

- `scripts/run_perturbgen_pipeline.py`：删除 legacy report helper、异常 fallback 和 empty-config 免校验；复用 `PerturbGenInvocation` 完整 `__post_init__` 契约，并新增 base YAML 的 gene/mode/route/ENSG/seed/path/config binding。`paths` 按 invocation 的授权双路径集合解释；`perturbgen_config_path` 按原契约作为 base config 路径，不要求 materialized stage YAML。
- `src/integration/perturbgen/dual_path.py`、`src/integration/perturbgen/mainline.py`、`scripts/evaluate_perturbgen_dual_path.py`：route 显式贯穿共享评价和生产 evaluator/mainline 调用；不由 observed corrective mode、`davf_action` 或输入中偶然出现的 mode 推断 route。
- `scripts/run_davf_perturbgen_e2e.py`、`src/integration/perturbgen/orchestrator.py`：Gate-0 原始 tokenise 输入边界硬化；E2E invocation seed 从 base config 的 `pipeline.random_seed` 读取，避免模板 seed 42 与旧默认 0 静默错配。
- `src/integration/perturbgen/eval_assembly.py`：增加精确 tokenise manifest lineage，并从 runner `outputs` 对 `result_h5ad` 的既有记录复制 `sha256`；N-05 默认将每个 artifact 的真实运行 seed 写为 bootstrap seed，保留原有显式全局 bootstrap override 并记录实际值，不强制所有多 seed 等于同一 override。
- `scripts/evaluate_perturbgen_dual_path.py`、`src/integration/perturbgen/reports.py`：记录 extractor 实际解释参数和显式 bootstrap seed；顶层 candidate manifest entry 复用 `manifest["replay"]`。
- `tests/unit/scripts/test_run_perturbgen_pipeline.py`、`tests/unit/integration/perturbgen/test_dual_path.py`、`test_eval_assembly.py`、`test_reports.py`、`tests/unit/scripts/test_evaluate_perturbgen_dual_path.py`、`tests/unit/test_davf_direction_integration.py`、`tests/unit/integration/perturbgen/test_direction_gate.py`、`tests/integration/test_perturbgen_dual_path_mocked.py`：仅将既有调用/fixture 补到真实显式 route、完整 binding、lineage 和 bootstrap schema；没有新增测试套件。
- `docs/CURRENT_STATUS.md`、`docs/guides/davf_perturbgen_e2e.md`、`docs/guides/perturbgen_bridge.md`：只追加/更新 9/13 状态、真实 tokeniser/seed/binding/replay 边界和报告入口，保留 9/10 历史检查日期。

分工记录：本 worker 未编辑 `frozen_cohort.py`、`scripts/run_frozen_acceptance.py` 及其专属 fixture；Volta 已实际修改这两个生产文件并完成 M6 verifier/OE、cohort lineage、frozen replay 调用和失败传播。其结果见 §5、§6、§8；真实 cohort/M6 仍未跑。

## 4. 修复前后高/中断点

| 严重度 | 断点、证据与真实影响 | 当前处置 |
|---|---|---|
| 高 | direct runner 只找任意 pass report，未绑定当前 YAML 的 gene/mode/path/config；合法报告可能放行另一份 YAML。 | 已修复。实际 config 缺失、gene/mode/seed/path 不一致均硬失败；声明的 ENSG/route 也比较。未用 filename 相同推断内容身份，未加 hash。 |
| 高 | KD/up 只有 mask 是合法路线，但旧 helper 对所有 observed up 强制 mask/pad/delete，KD/up 永远 inconclusive。 | 已修复全部生产调用：KO/up 才要求三种模式且二取三，KD/up 只要求 mask，down 只要求 overexpress；Volta frozen replay 已同步 route。 |
| 高 | 候选 p 值链（`eval_assembly.py:333-426`、`evaluate_perturbgen_dual_path.py:45-189,193-294`）从外部 p 先 BH；每-run empirical p 没有候选级聚合，外部 p 可改变 q 及正式 verdict。`uniform_candidate_pvalue` 可构造人为通过输入。 | 本轮不硬编码未定义的多 path/mode/seed 聚合；报告明确为正式科学阻塞。uniform 只可作为工程/合成演示输入，不能标正式验收。 |
| 高 | `null_selection.py`（选择/汇总/loader）没有 batch null stage 生成/运行/收集端。 | 未解决。不能把 G-1/null/BH 链称完全自动化闭环；需要真实 cohort 后补生成、运行、收集和绑定流程。 |
| 高 | Gate-0 ≥3 shared donor 只证明输入 cohort 的表面条件，当前 train 入口无 `train_donors`/`held_out_donors` 传给 tokenise/训练；不能推出 DAVF 训练未用 held-out donor。 | 未解决。需要正式训练 split/input manifest 与冻结 donor manifest 的显式绑定；Volta 固定 DEG train-only/H5AD heldout-only 只能防 signature leakage，不能证明模型训练无 held-out 泄漏。 |
| 高 | 本机没有正式 normal/disease raw-count cohort；2026-09-13 evidence 的 30 个文件中 26 可读、0 个合规候选。 | 外部阻塞仍在，但不再是唯一阻塞；candidate p、null generator、training split、M6/OE 也独立阻断正式结论。 |
| 中 | Gate-0 检查 `prepare_perturbgen_anndata` 返回的 copy，而外部 tokeniser 消费原始 h5ad；若误以为 copy 中新增字段自动进入 tokeniser，会造成边界误判。 | 已按真实消费者修复边界：原始 tokenise 输入 ENSG 非 canonical 直接失败；登记源码已静态核对同一原始文件的 counts→`X`→恢复 counts→`n_counts` 接线，prepared copy 的新增 `n_counts` 不需 materialize；本轮未运行真实 tokenisation。该项不再作为独立工程阻塞。 |
| 中 | N-05 原先只由路径/文件名寻找结果；报告无法证明 perturb 实际消费了哪一次 tokenise，且 evaluator 结果缺少真实重算参数/`result_h5ad` hash。 | 已修复 assembler 精确 lineage、复制 runner `outputs` 的既有 `result_h5ad.sha256` 与 result 参数记录；Volta 已接独立重读 H5AD/重算 rescue/donor/CI/verdict，不能只重放 summary。 |
| 中 | frozen 独立 replay 的 DEG CSV 经 `csv.DictReader` 读取后 `fdr` 保留字符串，而 `results` 的签名筛选要求数值；因此 evaluator 的 `pd.read_csv` 正常不代表 replay runtime 正常。 | 首次 runtime 复核已定位根因；Volta 改为 `pd.read_csv`/JSON DataFrame 后，生成链复跑无 mismatch、`independent=True`，正常 CLI verify exit 0，篡改生成 verdict 后 exit 1。 |
| 中 | `scripts/train_latent_davf.py:285-328` 的正式 asset/scVI/schema/route 校验未接收 donor split；不能由该入口的 schema 通过推导 held-out DAVF biology。 | 不新增训练功能；在 E2E 矩阵和未解决项中明确区分。 |
| 中 | Gate-E ≥200、正式多 seed/null、冻结 M6/OE 没有真实资产验收。 | 未验证；不以 synthetic/mock/smoke 结果替代。 |

## 5. 分轮执行结果

| 轮次 | 内容 | 结果 |
|---|---|---|
| 0 | 读取全局/项目 `AGENTS.md`、最新 `project_analysis_20260910.md`、`CURRENT_STATUS`、lessons 最新相关条目、方案、指南、源码调用链和登记外部 tokeniser；先写报告。 | 完成。报告初稿先于源码修复落盘；未创建 worktree/commit/外部 Codex。早期一次 shell 引号未闭合，命令失败；之后读取和写入均成功。 |
| 1 | 审计 direct binding、Gate-0 copy/原始消费者、candidate p/null、KD 路线和 production helper 调用。 | 确认高断点及调用点：evaluator、`src/integration/perturbgen/mainline.py:91` 和 frozen replay；确认外部 tokeniser 契约和 9/13 donor evidence。 |
| 2 | 落地 config binding、显式 route、Gate-0 原始 ENSG 硬检查、config seed 同步、N-05 lineage、replay/extraction 参数与 bootstrap seed。 | 完成。无 legacy/fallback/empty-config 免校验；无新增生产测试或门禁；冻结 verifier 的 route/manifest/replay 由 Volta 单独完成。 |
| 3 | 将既有 route 调用/fixture 补到新显式契约，并更新状态/指南。 | 完成。未新增测试套件；仅更新既有测试的 route、完整 binding、lineage、candidate route 和 provenance seed 字段。CURRENT_STATUS 保留 9/10 历史数字和日期，新增 9/13 状态入口。 |
| 4 | 运行目标静态检查、现有相关回归和一次性契约探针。 | SHA 接线前代码版本的静态检查和 89 项相关回归结果见 §8；探针确认 binding match、seed mismatch hard fail、KD/up mask-only pass、KO/up 缺 pad/delete inconclusive。 |
| 5 | Volta 完成 M6 verifier/OE、cohort lineage、frozen replay 接线；显式 KO/KD、KD mask-only、KO mask-only 不完整、OE-only 完整、显式 bootstrap seed、manifest 必需、真实 extraction 独立重算和失败 exit1。 | 首次 Volta targeted 检查为 `test_results` 15 passed、`test_eval_assembly` 17 passed、`test_evaluate` 6 passed；旧 frozen fixture 当时为 10 passed/4 failed，失败原因为 fake H5AD、缺 schema、旧调用缺 manifest，未弱化生产契约。 |
| 6 | 修复 assembler 从 runner `outputs` 精确复制 `result_h5ad.sha256`，再复跑本轮既有相关检查。 | `test_eval_assembly.py` 17 passed；完整相关集合收集 89，`89 passed, 19 warnings`，exit 0，用时 4.92s。该结果是 SHA 接线后的最终本轮相关回归结果；仍不是 biology/M6 验收。 |
| 7 | Volta 修正 frozen replay 的 DEG CSV 数值解析、更新 4 个旧 fixture，并完成实际生成链复核；本 worker 补跑此前未纳入 89 项集合的三个既有入口。 | Volta 最终 4 文件现有回归 `test_results`、`test_eval_assembly`、`test_evaluate`、`test_frozen` 合计 `52 passed`、exit 0；旧 frozen fixture 更新后 `14 passed`、exit 0。明确 synthetic AnnData/runner metadata 的 assembler→evaluator 实际生成 JSON 经 frozen 独立重算无 mismatch、`independent=True`、CLI verify exit 0，篡改生成 verdict 后 CLI exit 1。本 worker 三组入口首轮为 24 passed/1 failed，唯一失败是既有 fixture 缺显式 `pipeline.random_seed`；补字段后最终 `25 passed, 8 warnings`、exit 0。上述均为工程/synthetic 结果，不是正式 M6 biology 验收。 |

## 6. E2E 全链系统复核矩阵

| 环节 | 当前源码事实 | 当前判定 |
|---|---|---|
| PTM site → proposal/candidate | `candidate_spec.py`/`build_candidate_spec.py` 可把显式方向和独立表达证据接入 candidate；方向不能从 observed 证据反推 | 既有工程成果；不等于真实候选科学有效 |
| 训练输入、scVI schema、PerturbGen asset | `scripts/train_latent_davf.py:285-328` 能核对正式 asset/scVI/schema/route/direction；方案 §4.6、§7.2 还要求 donor split 和冻结验收。 | 只证明输入 schema/asset 接口可查。训练入口没有直接 donor split 输入，当前 config/orchestrator/E2E 也未见 `train_donors`/`held_out_donors` 进入 tokenise/训练；held-out DAVF biology 未验证。 |
| 正式冻结 donor 验证 | 冻结 M6 manifest 的现有口径是至少 2 个 training donor、至少 3 个 held-out donor 且两者 disjoint；因此完整 M6 至少需要 5 个共享 donor（normal/disease 两状态和目标 celltype 范围都要满足）。Volta 已在 verifier 中保留该 manifest 约束，并接入 lineage/replay。 | 与训练输入分开记账，不能用 Gate-0 ≥3 shared donor 代替完整 M6 的 ≥5 donor split。真实 split/input manifest 到位后，先做 0.5–1 天 lineage/preflight，再做约 0.5–1 天 held-out DAVF 方向验证。 |
| Gate-0 cohort | `data_prep.py` 检查 raw counts、canonical/unique Ensembl、normal/disease、显式 donor、≥3 shared donors；`outputs/perturbgen/spike/20260913_donor_audit/evidence.json` 记录本机 30 文件、26 可读、0 合规候选。 | Gate-0最低口径是目标 celltype/normal-disease 配对中 ≥3 个 shared donor；完整 M6 口径是至少 2 training + ≥3 held-out、disjoint，即至少 5 个 shared donor，并且这些 donor 在两状态都可评估。当前两种口径都没有真实合规资产；不能用 Datlinger/scPerturb smoke 替代。 |
| DAVF direction → 三方 gate | `orchestrator.py`/`direction_gate.py` 要求 PTM proposal、DAVF 方向、独立表达方向一致后才生成 invocation | 工程 gate 存在；正式 held-out direction accuracy 未验证 |
| gate report → direct runner | `_validated_report_invocation` 构造真实 `PerturbGenInvocation`，再由 `_config_gate_binding` 比较 gene/mode/声明 ENSG/route/seed/path/config。 | 已修复并有正/负契约探针；模板 `pipeline.random_seed=42` 与 report seed 0 不会 fallback。 |
| PerturbGen 六阶段 | `config_builder.py`/`runner.py` 有 stage 顺序、artifact、hash、resume、双路径计划；`run_davf...` 可触发外部 stage | 工程接线存在；无合规 cohort/正式多 seed 结果 |
| tokenise lineage → eval input | `eval_assembly.py` 只接受成功 perturb manifest，并按其 resolved data artifacts 精确找到唯一成功 tokenise manifest；从 runner `outputs` 中与 `artifacts.result_h5ad` 精确相同的记录复制既有 `sha256`，写入 `h5ad_provenance`。 | N-05 来源链和 hash 传递已硬化；旧 fixture 缺 lineage/runner output record 会失败，不能用目录 latest 猜测，也不由 assembler 新算 hash。 |
| Null generation → empirical p | `null_selection.py` 只有选择/汇总/loader；N-05 仍要求外部 null manifest；candidate p 仍可由 table/uniform 外部提供。 | 统计消费/绑定组件存在，但 batch null 生成端和 candidate-level p 聚合未闭合；外部 p 可改变正式 verdict。 |
| 双路径/mode/seed verdict | `dual_path.py` route-aware：KO/up 三模式二取三，KD/up mask-only，down overexpress-only；每路径仍要求既定 seed/donor/null 条件。 | KD 高断点已修复；candidate 多 path/mode/seed empirical-p 聚合规则仍未定义，不能宣称 G-1/null/BH 全闭环。 |
| report replay / raw-data replay | evaluator result 保存实际列名、阈值、donor/top-k/bootstrap 参数、null manifest 和含 runner hash 的 h5ad/tokenise provenance；top manifest entry 复用 `manifest.replay`。 | Volta 已接 extraction helper 独立重算 rescue/donor/CI/verdict，且任一 verify 失败返回 exit1。CSV `fdr` 字符串根因已修正为 `pd.read_csv`/JSON DataFrame；明确 synthetic 生成 JSON→frozen 独立重算无 mismatch、`independent=True`、正常 verify 0，篡改 verdict verify 1。仍不是真实 cohort 的 biology/M6 验收。 |
| 冻结 M6/OE | Volta 已修改 `frozen_cohort.py`、`run_frozen_acceptance.py`：显式 KO/KD；KD mask 完整、KO mask-only 不完整、OE-only 完整；bootstrap seed 显式但不强制等 artifact；manifest 必需且删除默认 mask fallback。 | Volta 最终 4 文件现有回归合计 `52 passed`、exit 0；4 个旧 fixture 更新后 `14 passed`、exit 0，未弱化生产契约。synthetic 生成链正常 verify 0、篡改 verdict verify 1。真实 M6/OE/cohort 仍未跑。 |
| Gate-E benchmark | `gate_e.py` 工具存在；缺少真实、可追溯、至少 200 个 PTM→gene benchmark 与固定旧基线 | 未验证；不能用 synthetic/mock PASS 替代 |

训练输入与正式冻结 donor 验证必须分开：Gate-0 只要求目标 celltype/normal-disease 配对至少 3 个 shared donor；完整 M6 还要求至少 2 training + 至少 3 held-out 且 disjoint，即至少 5 个 shared donor（两状态均可评估）。依赖到位后，先用 0.5–1 天完成 cohort/preflight 和 split lineage，再用约 0.5–1 天完成 held-out DAVF latent pair/方向验证；随后约 1–2 天执行真实 PerturbGen 双路径、多 seed、null 与 replay，Gate-E 资产齐后约 0.5 天准备、2–6 小时计算。以上是依赖到位后的时间计划，不是已运行结果。

## 7. 未解决问题、根因、步骤与时间节点

| 问题 | 根因 | 具体步骤 | 时间节点 |
|---|---|---|---|
| config-bound gate report | 报告验证器与 config 解析分离，只检查任意 pass 结构 | 已先 `load_pipeline_config`；从真实 trainer/pipeline 读取 gene/mode/声明 route/ENSG/seed，并由 `--path`/sequence 解析期望 path；report 不匹配硬失败 | 已完成；后续只需在真实 E2E 资产上复核 |
| Gate-0 实际消费数据漂移 | `prepare_perturbgen_anndata` 返回规范化副本，外部 tokenise 读取原始 path | 不覆盖输入；Gate-0 对原始 context 的 Ensembl 值执行 canonical 检查；若需规范化，必须由上游显式产出新文件并让 config 指向该文件 | 代码已完成；真实 cohort 到位后 T0+0.5–1 天重新 preflight |
| batch null stage 生成端 | 现有模块只消费 null 结果，没有批量 stage 运行器/证据 manifest | 合规 cohort 后，为每 candidate/path/mode/seed 固定 selection manifest，调用现有 stage plan 生成 null，收集成功 manifest/rescue，输出绑定 distribution，再接 N-05/evaluator | T0=资产齐备；T0+1–2 天，GPU 运行另计 |
| candidate p provenance/聚合 | candidate p 外部提供；per-run empirical p 未定义如何跨 path/mode/seed 聚合 | 统计负责人先定义 candidate estimand/聚合规则，再实现 empirical-null 计算、provenance 绑定和固定 BH 输入；规则确定前禁止 formal PASS | 规则确认后约 1 天实现/重放，确认时间外部决定 |
| uniform candidate p | `eval_assembly.py` 提供 uniform 输入入口，缺少 formal/smoke 语义隔离 | uniform 仅能作为工程/合成演示并明确标注；正式验收必须由真实 empirical null 生成 candidate p，不把 uniform 输入用于 release | 当前仍未解决；与 candidate p 统计契约一并收口 |
| Gate-0 与完整 M6 cohort | 本机 9/13 audit 0 合规候选；Gate-0最低是 ≥3 shared donor，而完整 M6 需至少 2 training + ≥3 held-out、disjoint，即至少 5 shared donor，且各 normal/disease 状态和目标 celltype 范围可评估 | 提供真实 normal/disease H5AD、raw counts、canonical ENSG、cell_type/state/donor、train/held-out manifest；分别完成 Gate-0 与 M6 lineage audit | 外部输入时间未知；到位后 T0+0.5–1 天完成两种口径预检 |
| held-out DAVF biology | 训练入口可查 asset/schema，但不接 donor split；无真实 held-out 方向指标 | 固定 donor split，生成携带 scVI gene order/asset manifest 的真实 latent pair，独立 donor 验证方向；不由 schema PASS 代替 | cohort/split/latent pair 到位后 T0+1–2 天 |
| Gate-E ≥200 | 缺少固定、可追溯 benchmark 与旧基线 | 提供 ≥200 PTM→gene 样本、GeneMap/CPTAC 来源、旧底座基线，运行 coverage/token/action/non-inferiority | 资产齐后约 0.5 天准备、2–6 小时计算 |
| M6 verifier/OE 与真实 replay | 工程 verifier 已由 Volta 接入显式 KO/KD、manifest、exact lineage、实际 extractor 参数、bootstrap seed 和独立 H5AD/CI 重算；验证失败任一即 exit1。首次生成链暴露 `DictReader` 的 `fdr` 字符串类型根因，已改为 `pd.read_csv`/JSON DataFrame。 | synthetic 真实生成链工程复核已完成：生成 JSON→frozen 独立重算无 mismatch、`independent=True`、正常 CLI verify 0，篡改 verdict CLI 1；旧 fixture 已更新为 14 passed。真实 cohort/M6 尚未运行，仍需真实 normal/disease、raw-count、split lineage 和正式候选执行。 | 真实依赖到位后约 0.5 天做 verifier/replay preflight；biology 时间按 §6 |

## 8. 本轮真实检查结果

以下是本轮实际命令或实际读取结果，没有把历史结果冒充本轮结果：

| 检查 | 实际结果 |
|---|---|
| `git diff --check -- scripts/run_perturbgen_pipeline.py scripts/run_davf_perturbgen_e2e.py scripts/evaluate_perturbgen_dual_path.py src/integration/perturbgen/dual_path.py src/integration/perturbgen/mainline.py src/integration/perturbgen/orchestrator.py src/integration/perturbgen/reports.py src/integration/perturbgen/eval_assembly.py tests/unit/scripts/test_run_perturbgen_pipeline.py tests/unit/integration/perturbgen/test_dual_path.py tests/unit/integration/perturbgen/test_eval_assembly.py tests/unit/scripts/test_evaluate_perturbgen_dual_path.py tests/unit/integration/perturbgen/test_reports.py tests/unit/test_davf_direction_integration.py tests/unit/integration/perturbgen/test_direction_gate.py tests/integration/test_perturbgen_dual_path_mocked.py docs/CURRENT_STATUS.md docs/guides/davf_perturbgen_e2e.md docs/guides/perturbgen_bridge.md project_repair_report_20260913.md` | exit 0（SHA 接线后最终检查）。 |
| `python -m compileall -q scripts/run_perturbgen_pipeline.py scripts/run_davf_perturbgen_e2e.py scripts/evaluate_perturbgen_dual_path.py src/integration/perturbgen/dual_path.py src/integration/perturbgen/mainline.py src/integration/perturbgen/orchestrator.py src/integration/perturbgen/reports.py src/integration/perturbgen/eval_assembly.py` | exit 0。 |
| `ruff check` 上述 8 个源码/脚本文件 | `All checks passed!`，exit 0。 |
| `python -m mypy src/integration/perturbgen/dual_path.py src/integration/perturbgen/mainline.py src/integration/perturbgen/orchestrator.py src/integration/perturbgen/reports.py src/integration/perturbgen/eval_assembly.py --ignore-missing-imports` | `Success: no issues found in 5 source files`，exit 0。 |
| `python -m ruff format --check` 本轮相关源码/既有测试文件 | 6 files would be reformatted、10 files already formatted，exit 1；未运行批量 formatter，避免引入无关格式 diff。Volta worker 对其冻结范围的 format 检查通过。 |
| 既有相关回归（SHA 接线前代码版本）：`python -m pytest -q tests/unit/scripts/test_run_perturbgen_pipeline.py tests/unit/integration/perturbgen/test_dual_path.py tests/unit/integration/perturbgen/test_eval_assembly.py tests/unit/scripts/test_evaluate_perturbgen_dual_path.py tests/unit/integration/perturbgen/test_reports.py tests/unit/test_davf_direction_integration.py tests/unit/integration/perturbgen/test_direction_gate.py tests/integration/test_perturbgen_dual_path_mocked.py --timeout=300` | 收集 89；`89 passed, 19 warnings`，exit 0，pytest 用时 4.89s。该结果不冒充 SHA 接线后的最终集成结果。 |
| SHA 接线后相同既有相关回归：`python -m pytest -q tests/unit/scripts/test_run_perturbgen_pipeline.py tests/unit/integration/perturbgen/test_dual_path.py tests/unit/integration/perturbgen/test_eval_assembly.py tests/unit/scripts/test_evaluate_perturbgen_dual_path.py tests/unit/integration/perturbgen/test_reports.py tests/unit/test_davf_direction_integration.py tests/unit/integration/perturbgen/test_direction_gate.py tests/integration/test_perturbgen_dual_path_mocked.py --timeout=300` | 收集 89；`89 passed, 19 warnings`，exit 0，pytest 用时 4.92s；这是 SHA 接线后的最终本轮相关回归。 |
| direct binding 一次性契约探针 | `binding_match=PASS; route=KO; mode=overexpress; seed=42; paths=both`；expected seed 改为 0 得 `binding_seed_mismatch=HARD_FAIL`，错误为 `E2E gate seed does not match config pipeline.random_seed: 42 != 0`。这是工程契约检查，不是生物学验收。 |
| route 一次性契约探针 | `KD/up mask-only=pass; required=['mask']`；`KO/up mask-only=inconclusive; reasons=missing_mode_pad,missing_mode_delete,ko_modes_not_two_of_three_positive`；`KD/down overexpress=pass; required=['overexpress']`。只验证评价分支，不代表真实数据 PASS。 |
| 9/13 donor evidence | 实际读取 `outputs/perturbgen/spike/20260913_donor_audit/evidence.json`：30 files audited、26 readable、0 compliant candidates。没有用旧 2026-09-03 evidence 改写本轮日期。 |
| 外部 tokeniser | 实际读取登记 repo `ref/Perturbgen-src` commit `a9a9375`：`GF_tokenisation.py:223-226,238,301-305` 静态确认原始 counts→`X`→恢复 counts→重算 `n_counts`，并确认原始 `h5ad_path`/`var['ensembl_id']` 接线；`tokenizer.py` 消费生成文件的 `X`/`n_counts`。本轮未运行真实 tokenisation。 |
| Volta 冻结/回放静态检查 | Volta worker 实际报告 compileall、Ruff、format、target mypy 均通过；冻结 verifier 显式 KO/KD、manifest 必需、删除默认 mask fallback、bootstrap seed 显式且不要求等 artifact；verify 任一失败 exit1。 |
| Volta 既有 targeted checks | 首轮为 `test_results` 15 passed、`test_eval_assembly` 17 passed、`test_evaluate` 6 passed（合计 38 passed）；随后 4 文件现有回归 `test_results`、`test_eval_assembly`、`test_evaluate`、`test_frozen` 合计 `52 passed`、exit 0。旧 frozen fixture 更新后 `14 passed`、exit 0；首次 10/4 失败保留为迭代记录，未为旧 fixture 弱化生产契约。 |
| Volta 真实生成链迭代 | 首次 runtime 复核发现 frozen replay 的 DEG CSV 经 `DictReader` 得到字符串 `fdr`，与 `results` 数值比较契约不符；修正为 `pd.read_csv`/JSON DataFrame 后，明确 synthetic AnnData/runner metadata 的 assembler→evaluator 实际生成 JSON 经 frozen 独立重算无 mismatch、`independent=True`、CLI verify exit 0；篡改生成 verdict 后 CLI exit 1。该工程结果不等于 biology PASS。 |
| 本 worker 最终补跑三个既有入口 | 同一命令首轮收集 25、24 passed/1 failed，失败为 `run_davf_perturbgen_e2e.py` 既有 fixture 缺显式 `pipeline.random_seed`，生产代码直接硬失败；仅更新该 fixture 后再次收集 25，`25 passed, 8 warnings`，exit 0，用时 3.21s。未改生产 fallback。 |
| legacy/fallback/empty-config 静态查错：`rg -n "_validate_legacy_gate_record|validated_report_invocation.*fallback|if not config:return None" scripts/run_perturbgen_pipeline.py` | exit 1、无匹配；direct runner 没有该 legacy/fallback/空 config 免校验代码。 |

本轮没有运行全量 pytest、真实资产 biology 验收、正式多 seed/null、Gate-E ≥200 或真实 cohort 的冻结 M6/OE；Volta 的冻结工程 fixture/targeted checks 已单独如实列出。9/10 全量回归和静态结果仍只作为 `CURRENT_STATUS`/历史分析中的 9/10 记录。没有下载新权重、没有伪造 cohort、没有把 mock/synthetic 当 biology PASS。

## 9. 最终状态

本轮已消除 direct runner 任意 report 放行风险，修复共享评价的 KD/up route 语义，补齐 E2E seed 绑定、N-05 tokenise lineage/runner hash 传递、evaluator 实际解释参数、确定性 bootstrap seed 和顶层 replay entry；Volta 已完成冻结 verifier/OE/独立 raw-data replay 的主体接线，并修正 DEG CSV `fdr` 字符串类型根因。89 项相关回归先有 SHA 接线前版本结果，随后已在 SHA 接线后复跑并得到 `89 passed, 19 warnings`；本 worker 最终补跑的三个既有入口为 `25 passed, 8 warnings`。Volta 最终 4 文件回归为 `52 passed`，旧 fixture 为 `14 passed`，synthetic 生成链正常 verify 0、篡改 verify 1；没有用兼容 fallback 掩盖缺陷。

仍不能写成“G-1/null/BH/报告重放全部闭环”或“只有外部 cohort 一个阻塞”：candidate p 可以由外部输入改变正式 verdict，候选级多 path/mode/seed empirical-p 聚合未定义，null batch 生成端缺失，正式训练 donor split/input manifest 未绑定，真实 cohort/Gate-E/M6/OE/多 seed/null 尚未验证。Gate-0 与已登记 tokeniser 的原始 counts→`X`→`n_counts` 接线已完成静态核对，但本轮未运行真实 tokenisation；prepared copy 不需要被冒充为消费者输入。

因此截至 2026-09-13 的准确结论是：工程入口、冻结 verifier 和错误边界已按全任务范围硬化，Volta 的工程 replay 检查已完成但真实生物学效用与冻结 M6 验收仍未完成；后续必须以 Gate-0 ≥3 与完整 M6 ≥5 shared donor 的双口径 cohort、显式训练/held-out lineage、统计负责人确认的 candidate-p 聚合规则、真实 null 生成证据和正式真实数据运行逐项收口。
