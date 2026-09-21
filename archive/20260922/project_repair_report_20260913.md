# PTM2CellNet 第四轮代码修复报告：F-01/F-02/F-03 剩余缺口闭合与系统性复核（2026-09-13）

## 结论先行

本轮按 `archive/20260914/project_analysis_20260913.md` §4.5a/§6.2 的剩余缺口完成三项高优先级修复：
**F-01（E2E 统计接续）**、**F-02（跨候选公共 prepare/reuse）**、**F-03 剩余（生成端
tokenise/latent-pair 行级 donor 绑定）**，并落地 F-09 边界文档化与两项中等技术债
（TD-13-13 文档同步、TD-13-14 损失函数数值单测）。全量验证 **2639 passed / 1 failed
（旧 checkpoint 真实资产失败，与第三轮基线相同，非本轮引入）/ 21 skipped**；ruff
check 通过、mypy 166 文件 0 errors、requirements 274 lock pins 通过。

工程侧的"未实现接口与业务流程"缺口（分析文档 §4.5 的 F-01～F-09）至此全部代码
闭合或显式文档化。**正式生物学 PASS 仍未成立**：真实合规 cohort（A-01）为 0、旧
checkpoint 不合规待重训、GPU matched-null 矩阵（A-05）与 Gate-E ≥200 benchmark
（A-04）未执行；这些是资产/数据缺口，不是接口缺口。E2E 统计链路已可用显式输入
（DEG 表 + null manifest）驱动，但用真实资产跑出的结果才可作为 formal 证据。

---

## 事实来源与需求依据

| 来源 | 引用章节 | 本轮对应动作 |
|---|---|---|
| `archive/20260914/project_analysis_20260913.md` | §4.5（F-01～F-09 审计）、§4.5a（第三轮状态）、§6.2（方案/资源/风险）、§6.3a（TD-13-13/14） | F-01 方案 A、F-02 方案 A、F-03 方案 A（生成端）、F-09 方案 A 的实现依据 |
| `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | §4.7（rescue 与正式判定 7 条件）、§5.4（质量门）、§7.2 M3/M6 | 统计接续 estimand 与质量门语义 |
| `lessons.md` | L-2026-0913-01（研究契约）、L-2026-0913-02（第三轮落地）、L-2026-0913-03（本轮新增） | 主线边界与新增决策记录 |
| `docs/CURRENT_STATUS.md` | Quick Reference、研究边界（第 2/3 条） | 本轮已同步更新 |
| `docs/guides/davf_perturbgen_e2e.md`、`docs/guides/perturbgen_bridge.md` | statistical_evidence 段、六阶段段、runner 边界段 | 公开行为变更的指南同步（AGENTS 修改规则） |
| `API_DOCUMENTATION.md` | PerturbGen 集成合同小节（本轮新增） | TD-13-13 |

---

## 1. 问题修复汇总（按严重程度）

### 高：F-01 E2E 没有自动接续 null、质量、p/q 与 dual-path —— 已闭合（代码）

- **原缺口**（分析 §4.5 F-01、§6.2 F-01）：`run_davf_perturbgen_e2e.py` 只序列化
  stages，`null_generation/eval_assembly/results/empirical_pvalue/dual_path` 五组接口
  无 E2E 调用，报告无统计证据。
- **本轮实现**（方案 A：单条 lineage、正式入口清晰）：
  - 重放评估核心从 `scripts/evaluate_perturbgen_dual_path.py` 抽离为
    `src/integration/perturbgen/replay_evaluation.py::replay_dual_path_evaluation`
    （`replay_evaluation.py:53`），CLI 变薄封装，两处共用同一实现，不再有两份
    重放逻辑。
  - E2E 新增 `--assemble-statistical-evidence --deg-table --null-distribution-manifest
    [--statistical-output-dir]`（`scripts/run_davf_perturbgen_e2e.py:689-706`），在六阶段
    完成后执行 `_assemble_statistical_evidence`（`:312`）：按候选从 primary-mode
    `within_state` h5ad 提取未扰动质量（`extract_unperturbed_quality_from_h5ad`）→
    `build_eval_input_payload`（formal）→ `replay_dual_path_evaluation`（候选
    empirical-p 按 `conservative_max_required_runs` 聚合、BH-FDR q、双路径严格
    AND），全部 lineage 写回报告 `statistical_evidence` 节。
  - `eval_assembly.build_eval_input_payload` 新增 per-candidate 质量入口
    `candidate_unperturbed_quality`（`eval_assembly.py:350`），formal 模式要求每个
    候选都有 h5ad 提取的质量 payload；与全局 `unperturbed_quality` 互斥（`:377`）。
  - 边界：GPU matched-null 批跑仍由 `run_matched_null_stages.py` 独立执行并产出
    `perturbgen_null_distribution/v1` 索引；E2E 只消费显式提供的 manifest，缺
    null/质量/seed 覆盖硬失败或保持 INCONCLUSIVE，绝不生成伪造 p/q。`--dry-run`
    与统计组装互斥（`run_davf_perturbgen_e2e.py:622`）。
- **测试**：`tests/unit/scripts/test_run_davf_perturbgen_e2e.py::
  test_assemble_statistical_evidence_chains_null_quality_pq_dual_path`（真实 h5ad +
  6 组 null 分布条目 + 3 seeds 全链路）、`test_assemble_statistical_evidence_requires_completed_runs`。

### 高：F-02 多候选没有公共 prepare/reuse —— 已闭合（代码）

- **原缺口**（分析 §4.5 F-02、§6.2 F-02）：每候选重复 `tokenise/train_mask/
  train_decoder`（重复 GPU），固定 tokenise 外部输出在第二个候选处冲突
  （`runner._ensure_external_output_targets_clean` 拒绝已存在输出）。
- **本轮实现**（方案 A：公共准备一次，候选只跑两路 perturb/效用）：
  - `orchestrator.build_shared_prepare_plans`（`orchestrator.py:534`）：对固定
    cohort/词表/训练配置的 config 重写 `pipeline.output_root`（`_rewrite_pipeline_output_root`
    逻辑自 `materialize_candidate_config` 提取复用，`:503`），产出
    `tokenise → train_mask → train_decoder` 三个共享计划。
  - `build_candidate_stage_plans(..., skip_prepare_stages=True,
    prepare_artifact_paths=...)`（`orchestrator.py:601`）：候选计划只含两路径
    perturb + export + report；`resolve_prepare_artifact_references`（`:565`）把
    `@artifact:tokenise/train_*` 引用解析为共享准备产物绝对路径，缺失引用硬失败
    （"shared prepare artifact was never produced"）；dry-run 预览保留 `@artifact`
    字符串（与 runner 行为一致）。
  - E2E 统一走共享准备（单候选同样受益）：`<root>/<KO|KD>/_prepare/` 执行一次，
    报告新增 `perturbgen_prepare` 节与每个 run 的 `prepare_root`
    （`run_davf_perturbgen_e2e.py:582-592`）。共享阶段沿用 runner manifest/resume
    指纹（重跑传 `--resume` 复用，不改 runner）。
  - 未新增 hash、调度框架或兼容开关（遵守 lessons L-2026-0913-01 第 4 条）。
- **测试**：`tests/integration/test_perturbgen_pipeline_mocked.py::
  test_shared_prepare_plans_run_once_and_candidates_reuse_artifacts`（真实 runner：
  共享三阶段执行一次→候选四阶段执行→第二次 prepare 全 reused）、
  `test_candidate_stage_plans_fail_when_shared_prepare_artifact_is_missing`、
  `test_candidate_stage_plans_keep_artifact_references_for_dry_run_preview`。

### 高：F-03 donor split 未绑定 tokenise 行与下游资产 —— 生成端已闭合（代码）

- **原缺口**（分析 §4.5 F-03、§4.5a"tokenise 行绑定仍开放"）：上轮已实现训练侧
  `_validate_donor_split_metadata` fail-fast（NPZ metadata 须含 `donor_split` 与
  `dataset.donor_rows`），但生成端 `build_scperturb_latent_pairs` 从不产出这些
  字段，行级 provenance 无法形成闭环。
- **本轮实现**（方案 A：row→donor membership 写入资产并由训练端比对）：
  - `build_scperturb_latent_pairs` 新增 `donor_obs_column` + `donor_split`
    （canonical `ptm2cellnet.donor_split/v1` payload，`davf_scperturb.py:543-544`）：
    train/val 行只允许来自 `train_donors`、test 行只允许来自 `held_out_donors`
    （target 策略在 train donor 池内二分、cell 策略按层二分、held 池全部入
    test；control 池同样按 donor 池分配）；未列入任一池的 donor 硬失败，两池
    必须都有细胞。
  - NPZ 新增 `target_donors`/`control_donors` 数组（`:930`），metadata 写入
    `donor_split` 原始 payload、`dataset.donor_rows`（行级 donor 列表，`:909`）、
    `donor_pool` 与 `donor_obs_column`；`pair_manifest.json` 记录 donor split。
    训练侧 `_validate_donor_split_metadata` 消费同一契约，闭环成立。
  - CLI `build_davf_scperturb_pairs.py` 新增 `--donor-obs-column/--train-donors/
    --held-out-donors/--donor-split-json`（`build_davf_scperturb_pairs.py:47-67`），
    canonical split 由 `build_donor_split` 构造（SHA 绑定）。
  - 边界：既有 NPZ/checkpoint 无行级 donor provenance，正式 held-out 声明仍需
    用该入口重建资产后由训练端校验（方案 §7.2 M6 真实运行未执行）。
- **测试**：`tests/unit/data/test_davf_scperturb.py::
  test_donor_bound_pairs_keep_train_and_held_out_donors_in_separate_npz`（train/val
  与 test 的 donor 池分离、metadata/NPZ 字段齐全）及三个失败路径测试
  （池外 donor、缺 donor 列、两池泄漏）。

### 高（上轮已闭合，本轮复核维持）：F-04/F-05/F-06/F-07

F-04（SemanticContext 七字段）、F-05（NullStageRecord 身份绑定）、F-06（Gate-E
必需证据/canonical coverage）、F-07（checkpoint gene_names 强制 canonical ENSG）
在提交 `96ee544` 已代码闭合（分析 §4.5a），本轮全量回归 2639 passed 维持成立，
无回归。

### 中：F-09 gate 边界未统一 —— 已按方案 A 文档化（边界显式，接口层未下沉）

- **实现**：`src/integration/perturbgen/runner.py` 模块契约（docstring）明确
  runner 是工程执行层、只执行 `StagePlan`、不重查 DAVF gate；formal invocation
  wrapper（`run_perturbgen_pipeline.py --e2e-gate-report` / orchestrator
  `PerturbGenInvocation`）是唯一公开正式入口。`docs/guides/davf_perturbgen_e2e.md`
  （gate 边界段）与 `docs/guides/perturbgen_bridge.md`（runner 边界段）同步。
- **未做**（有意）：方案 B（runner 接收并验证 gate context）会波及内部
  null/engineering StagePlan 调用方；当前按分析 §6.2 F-09 方案 A 落地，接口层
  统一仍开放（见 §4 后续策略）。

### 中：TD-13-13 新公开合同未同步 API 文档与 bridge 指南 —— 已闭合

`API_DOCUMENTATION.md` 新增"PerturbGen 集成合同"小节：SemanticContext 七字段
（`research_objective` 限 association/replication/reversal）、PerturbGenInvocation
校验、共享准备/候选计划入口、runner 执行边界、统计证据组装 CLI。
`docs/guides/perturbgen_bridge.md` 评估入口段补七字段语义与 E2E 内置统计组装
说明。

### 中：TD-13-14 `davf_losses.py` 无测试引用 —— 已闭合

新增 `tests/unit/models/test_davf_losses.py`（7 个数值断言测试）：零误差锚点、
已知输入的 MSE/幅度/方向项精确值、相对幅度误差缩放、自适应权重 10000 步饱和值
（mse 1.0→0.7、mag 0.3→0.45）、DirectionConsistencyLoss 的符号奖惩（BCE(logit)
精确值）、padding mask、KD 与 KO 同号语义。

### 中：TD-13-05 全仓 ruff format 债 —— 部分处理（维持上轮取舍）

本轮触碰的 12 个源码/测试文件已 `ruff format` 通过；全仓 339 文件一次性
format 沿用分析 §6.3 TD-13-05 与第三轮的取舍（独立机械 PR，避免淹没功能
diff），未在本轮批量改写。真实 `format --check` 结果见 §3。

### 其余中等技术债处置

| 条目 | 处置 |
|---|---|
| TD-13-06 pip 元数据冲突 | 不改（分析 §6.3：不为核心 pin 放宽；CI full-test 已知取舍，维持文档化现状） |
| TD-13-07 覆盖率未刷新 | 本轮仍无 `--cov` 运行；`.coveragerc fail_under=74` 不变，不用历史 75.27% 冒充（刷新排入 §4） |
| TD-13-08/10/11/12 | 上轮已闭合/文档化，本轮无动作 |
| TD-13-15/16（低） | 不动（分析 §6.3a 推荐方案 A：触碰时顺带/出现瓶颈再优化） |

---

## 2. 每轮迭代执行情况与结果

| 轮次 | 范围 | 结果 |
|---|---|---|
| 第一轮（2026-09-13 早，提交 `96ee544` 前基础审计） | 分析文档 §4.5 基础审计时点：F-01～F-09 全部开放或部分实现 | 识别缺口与方案；记录于 `archive/20260914/project_analysis_20260913.md` §4.5 |
| 第二轮（契约硬化，提交 `96ee544`） | F-04/F-05/F-06/F-07/F-08 + F-03 训练侧 fail-fast；2623 passed / 1 failed（旧 checkpoint） | 见 `project_repair_report_20260913.md` 上一版本与 L-2026-0913-02；F-01 边界显式化（statistical_evidence=inconclusive） |
| 第三轮（增量归档复核，提交 `3e2166d`/`219b81f`） | 归档复核零新增 | 分析文档 §3.2/§8.1 |
| **第四轮（本轮）** | **F-01 统计接续、F-02 共享 prepare、F-03 生成端 donor 绑定、F-09 文档化、TD-13-13/14、触碰文件 format；新增 16 个测试** | **2639 passed / 1 failed（同一旧 checkpoint 资产失败）/ 21 skipped / 1412.90s；ruff check 通过；mypy 166 文件 0 errors；requirements 274 pins 通过** |

本轮迭代内部顺序（依 L-2026-0913-01"后续顺序"）：先 F-02（执行生命周期，统计
接续的前置），再 F-03（数据契约），最后 F-01（统计接续消费前两者的输出语义）；
每步均先跑定向测试再做下一步，最后全量回归 + 静态检查 + 文档/lessons 同步。

---

## 3. 系统性复核分析：E2E 训练与推理的全部技术要求

### 3.1 代码层要求逐项核对

| # | 技术要求（AGENTS 2026-09-13 约束 / 方案 §4.6-4.7/§7.2） | 代码现状 | 判定 |
|---|---|---|---|
| 1 | Gate-0：真实 normal/disease raw counts、显式 donor、≥3 共享 donor、canonical Ensembl | `_preflight_perturbgen_context` + `prepare_perturbgen_anndata` + 原始输入 canonical ENSG 硬校验（`run_davf_perturbgen_e2e.py:235-292`） | **满足（代码）** |
| 2 | 语义合同：context/intervention/baseline/objective/cohort/来源进合同 | SemanticContext 七字段贯穿 proposal→gate→invocation→E2E/CLI（上轮 F-04） | **满足（代码）** |
| 3 | 三方 gate 串联，runner 不得绕过 | gate pass 才建 invocation；`run_perturbgen_pipeline.py --e2e-gate-report` 强制绑定；runner 边界文档化（本轮 F-09） | **满足（代码+文档）；接口层下沉开放** |
| 4 | donor split 冻结与行级绑定 | E2E/训练 CLI + 训练端 fail-fast（上轮）+ 生成端行绑定（本轮 F-03） | **满足（代码）；真实资产需重建** |
| 5 | checkpoint：64×4018、canonical ENSG、冻结 embedding/manifest 一致 | checkpoint contract 强校验（上轮 F-07）；旧 `latent_davf_perturbgen_4018` 被正确拒绝 | **满足（代码）；本地资产不合规** |
| 6 | 六阶段固定 + 公共 prepare 一次 | 共享 prepare + 候选 only-perturb（本轮 F-02），resume/fingerprint 不变 | **满足（代码）** |
| 7 | 双路径两场景与 AND | `source_intervention=[src]`/`within_state=[tgt]+pert_tps` 分开计划；dual_path 严格 AND（方案 §4.7 七条件） | **满足（代码）** |
| 8 | matched-null ≥99、身份绑定 | NullStageRecord 绑定 candidate/path/mode/seed（上轮 F-05）；GPU 批跑独立 CLI | **满足（代码）；真实运行未执行** |
| 9 | 候选 empirical-p + BH-FDR q | `conservative_max_required_runs` + `benjamini_hochberg`；E2E 自动接续（本轮 F-01） | **满足（代码）；真实统计未跑** |
| 10 | 未扰动质量门（中位数 ≥0.60、最差 ≥0.50、相关性 >0、≥3 seeds） | `extract_unperturbed_quality_from_h5ad`；E2E per-candidate 自动提取（本轮 F-01） | **满足（代码）** |
| 11 | Gate-E：三节证据、canonical coverage ≥.99、collision=0、benchmark ≥200 | 硬边界齐备（上轮 F-06） | **满足（代码）；真实 benchmark 缺** |
| 12 | M6 route/mode 行级表达 + donor 互斥 | FrozenCandidate 行级 modes（上轮 F-08） | **满足（代码）；真实 M6 未跑** |
| 13 | PTM site classifier 只做 site presence；方向来自外部假设/override | contracts 与 candidate_spec 维持（L-2026-0913-01 第 1 条） | **满足** |

**结论：E2E 训练与推理的全部代码层技术要求已满足。** 剩余不满足项全部是
资产/数据/真实运行缺口，不是接口缺口。

### 3.2 资产与真实运行现状（阻塞 formal PASS 的根本原因）

| 阻塞项 | 根本原因 | 现状证据 |
|---|---|---|
| A-01 合规 cohort = 0 | 本机 30 个 scPerturb H5AD 无一满足 normal/disease raw counts + 显式 donor + ≥3 共享 donor | `outputs/perturbgen/spike/20260913_donor_audit/evidence.json`（CURRENT_STATUS Gate-0 复审） |
| 旧 checkpoint 不合规 | `latent_davf_perturbgen_4018` 的 `scvi.gene_names` 非 canonical ENSG，F-07 硬失败（全量回归唯一 failed 即此） | `tests/integration/test_scvi_davf_connection.py` 真实资产测试 |
| 行级 donor 资产未重建 | 既有 NPZ/checkpoint 产出早于 F-03 生成端绑定 | 需用 `build_davf_scperturb_pairs.py --donor-*` 重建 |
| A-05 matched-null 矩阵未跑 | GPU ≥99 null × 双路径 × ≥3 seeds 需真实 cohort + 合规 checkpoint 前置 | `run_matched_null_stages.py` 仅 smoke/dry-run |
| A-04 Gate-E benchmark < 200 | 无固定可追溯 PTM→gene benchmark 资产 | Gate-E 工具硬门槛 `MIN_BENCHMARK_SAMPLES=200` |
| 语义参考轴未冻结 | context/intervention/reference_axis 的研究定义须由统计/生物学负责人确认，代码只强制记录 | L-2026-0913-01 第 2 条 |

### 3.3 E2E 一致性判定

- **工程 E2E（gate → 共享 prepare → 候选 perturb/export/report → 统计组装）**：
  可以用显式输入完整驱动，契约测试与 mocked 管线全部通过。
- **正式 E2E（真实 cohort 上的生物学结论）**：**不可达**，被 §3.2 六项资产缺口
  阻塞。任何 smoke/synthetic/mock/bridge 结果仍不得写成 biology PASS（AGENTS
  验收约束）。

---

## 4. 未解决问题与后续解决策略

时间以"取得合规外部资产并完成负责人确认"的工作日起算（T0 = 资产到位日）；
GPU 墙钟不折算编码工作日。

| 优先级 | 问题 | 解决策略与实施步骤 | 时间节点 |
|---|---|---|---|
| 阻塞 | A-01 合规 cohort | 数据 owner 提供真实 normal/disease raw counts 队列（显式 donor、≥3 共享 donor、canonical Ensembl）→ 过 Gate-0 preflight → 冻结 scVI gene order 与 embedding/manifest | T0–T10 工作日；无数据则停止科学声明 |
| 阻塞 | 合规 checkpoint 重训 | 隔离旧 `latent_davf_perturbgen_4018` → 以 A-01 队列重建 latent pairs（用本轮 `--donor-obs-column --train-donors --held-out-donors` 生成行级绑定 NPZ）→ 按 64×4018/canonical ENSG 契约重训 LatentDAVF → 过 checkpoint contract | A-01 后 T+2–T+8 工作日 + 1–3 GPU 日 |
| 高 | A-02/A-03 held-out 方向指标 | 冻结 ≥2 train / ≥3 held-out donor（SHA 绑定）→ 训练端 `_validate_donor_split_metadata` 通过 → 报告 held-out 方向指标 | 随上一项并行，T+2–T+8 |
| 高 | F-01 真实统计运行（A-05） | 合规 checkpoint 后：`run_matched_null_stages.py` 执行 ≥99 null × 双路径 × ≥3 seeds（含 rescue_extractor）→ E2E `--run-perturbgen --assemble-statistical-evidence --deg-table --null-distribution-manifest` 产出 formal p/q/dual-path | A-01 后 T+3–T+10 GPU 日 |
| 高 | 语义参考轴冻结 | 统计/生物学负责人逐候选确认七字段（含 reference_axis 与 objective），写入 cohort manifest | T0–T+3，先于正式候选验收 |
| 中 | F-09 接口层统一 | 评估方案 B（runner 接收 gate context）对内部 null/engineering 调用方的影响面；若采纳则改造 + 全 caller 回归 | T+4–T+7，约 1.5–3 工作日 |
| 中 | A-04 Gate-E | 提供 ≥200 固定可追溯 benchmark → `evaluate_gate_e.py` 全三节证据 + canonical coverage | 可与 A-02/03 并行，T+5–T+12 |
| 中 | TD-13-05 全仓 format | 一次性机械 PR：`ruff format src scripts tests` + 全量回归 + CI 增加 ruff format/check step | 任意时点，0.5 工作日 |
| 中 | TD-13-07 覆盖率刷新 | 独立环境 `pytest --cov` 刷新 `docs/TEST_COVERAGE.md`（`fail_under=74`） | 下一轮验证，0.5 工作日 |

推荐顺序与 L-2026-0913-01 一致：先冻结语义与 cohort（§4 前两行），再重建资产
并跑真实统计链，最后 Gate-E 与发布讨论。**不新增** hash/调度框架/兼容开关来
"预设"这些目标已完成。

---

## 5. 验证记录（原始结果）

| 命令 | 结果 |
|---|---|
| `python -m pytest -m "not slow and not gpu" --timeout=300` | 2639 passed / 1 failed / 21 skipped / 67 warnings / 1412.90s；唯一 failed = `test_scvi_davf_connection.py::test_real_current_davf_direction_keeps_token_and_decoder_indices_separate`（旧 checkpoint 非 canonical ENSG，与第三轮基线同一已知真实资产失败；本轮 +16 测试全过） |
| `ruff check src scripts tests` | All checks passed |
| `python -m mypy src/ --ignore-missing-imports` | Success: no issues found in 166 source files |
| `python scripts/check_requirements_consistency.py` | OK，274 lock pins |
| `python -m ruff format --check src scripts tests`（触碰文件） | 本轮 12 个触碰文件全部格式化通过；全仓 339 文件待格式化的机械债维持独立 PR 取舍 |

slow/gpu/real_assets 本轮未执行；GPU matched-null 与真实 cohort 验收仍待资产。

## 6. 本轮改动清单

源码：`src/integration/perturbgen/orchestrator.py`（共享 prepare 计划/引用解析/
root 重写提取）、`src/integration/perturbgen/replay_evaluation.py`（新增，重放
评估单一实现）、`src/integration/perturbgen/eval_assembly.py`（per-candidate
质量）、`src/integration/perturbgen/runner.py`（docstring 边界契约）、
`src/data/davf_scperturb.py`（donor 绑定划分与 NPZ/metadata 字段）、
`scripts/run_davf_perturbgen_e2e.py`（共享 prepare + 统计接续 CLI）、
`scripts/evaluate_perturbgen_dual_path.py`（薄封装）、
`scripts/build_davf_scperturb_pairs.py`（donor CLI）。
测试：`tests/integration/test_perturbgen_pipeline_mocked.py`（+3）、
`tests/unit/scripts/test_run_davf_perturbgen_e2e.py`（+2）、
`tests/unit/data/test_davf_scperturb.py`（+4）、
`tests/unit/models/test_davf_losses.py`（新增 7）、
`tests/unit/scripts/test_evaluate_perturbgen_dual_path.py`（8 个维持通过）。
文档：`docs/guides/davf_perturbgen_e2e.md`、`docs/guides/perturbgen_bridge.md`、
`API_DOCUMENTATION.md`、`docs/CURRENT_STATUS.md`、`lessons.md`（L-2026-0913-03）。

最终判定：F-01/F-02/F-03 剩余缺口代码闭合，F-09 与中等级文档/测试债落地；
工程主链接口全部齐备。**正式科学验收仍为 inconclusive，没有生物学 PASS**——
剩余阻塞全部是真实资产与真实运行（A-01/A-04/A-05、合规 checkpoint、语义轴冻结），
其解决策略与时间节点见 §4。
