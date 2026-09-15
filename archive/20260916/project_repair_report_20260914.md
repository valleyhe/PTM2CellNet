# PTM2CellNet 第八轮修复与审计报告（2026-09-14）

本报告更新并替代工作区原标为“第七轮”的 `project_repair_report_20260914.md`。
审计依据是根目录 `project_analysis_20260914.md` §4 的问题 1–6、该报告列出的高/中
技术债，以及本轮对源码、测试、真实冻结资产和操作指南的复核。工作区已有未提交改动
全部保留，未执行 `git reset`、`git checkout` 或覆盖式回滚。

## 1. 结论先行

- **代码与契约修复**：第七轮遗留的 pairing、统计 lineage、state coverage、Diagnosis
  唯一性、DEG 列名、共享 prepare、序列化和审计口径等高/中项已由当前工作区代码与
  回归测试覆盖；本轮又修复了真实词表暴露的 `null_selection._token_values` 静默/过宽
  处理风险，以及 bridge runbook 的参数和 Python 示例错误。
- **真实 CPU/资产结果**：GSE174367 EX `between_donor` 冻结 manifest、SHA 绑定、donor
  split、5 候选、5×99 null selection 和 90-run acceptance plan 均可在当前环境核验；
  `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 的真实冻结资产测试 **1 passed**。
- **GPU 探针结果**：当前代理环境的 `nvidia-smi` 和两个 Python 环境都能看到 Tesla P40；
  本轮没有执行六阶段 PerturbGen、matched-null stage 运行或生物学统计验收。因此不能把
  GPU 可见性写成 GPU E2E PASS，也没有尝试系统驱动修复。
- **最终科学结论**：**生物学 PASS 仍为 0**。正式 candidate_spec、donor-level DEG
  表、实际 perturb h5ad/rescue 结果和 null distribution manifest 尚未在输出目录中
  形成，所以没有伪造 E2E、GPU 或研究输入结果。

## 2. 实际变更文件与根因

以下是本轮累计工作区中与本任务相关的实际变更文件（均未提交；其中已有修改没有被
回滚）。冻结资产位于 git-ignored 的
`outputs/perturbgen/frozen/20260914_gse174367_ex/`，不列作源码变更文件。

文档与状态：

- `CHANGELOG.md`
- `docs/CURRENT_STATUS.md`
- `docs/guides/davf_perturbgen_e2e.md`
- `docs/guides/perturbgen_bridge.md`
- `lessons.md`
- `project_analysis_20260914.md`
- `project_repair_report_20260914.md`

脚本：

- `scripts/audit_ad_cohort_gate0.py`
- `scripts/build_davf_scperturb_pairs.py`
- `scripts/run_davf_perturbgen_e2e.py`
- `scripts/run_frozen_acceptance.py`
- `scripts/standardize_gse174367_ad_cohort.py`

源码：

- `src/data/davf_scperturb.py`
- `src/integration/perturbgen/eval_assembly.py`
- `src/integration/perturbgen/frozen_cohort.py`
- `src/integration/perturbgen/null_selection.py`
- `src/integration/perturbgen/orchestrator.py`
- `src/integration/perturbgen/replay_evaluation.py`
- `src/integration/perturbgen/reports.py`

测试：

- `tests/unit/data/test_davf_scperturb.py`
- `tests/unit/integration/perturbgen/test_frozen_cohort.py`
- `tests/unit/integration/perturbgen/test_null_selection.py`
- `tests/unit/scripts/test_run_davf_perturbgen_e2e.py`
- `tests/real_assets/test_real_frozen_cohort.py`
- `tests/unit/scripts/test_audit_ad_cohort_gate0.py`
- `tests/unit/scripts/test_standardize_gse174367_ad_cohort.py`

主要根因与修复如下：

| 项目 | 根因 | 修复/状态 |
|---|---|---|
| F-10 / 问题 3 | frozen manifest、Gate-0 spec 和 state/donor 校验曾把 `within_donor` 作为隐含前提，无法验收 AD case-control | manifest 显式保存 `pairing`；校验按 `within_donor`/`between_donor` 分支；freeze、E2E 和 verify 互相绑定。GSE174367 真实资产已闭合。 |
| F-14 | AD audit 文案、结构分析和 verdict 是旧的 shared-donor 口径，且存在死函数/空 glob 隐式行为 | 按 pairing 推导 per-cohort 状态与 verdict；空输入硬失败；真实审计结果不再把 GSE174367 判成旧的 blanket blocked。 |
| F-11 | 统计 eval input 没有声明 donor pairing，冻结验收无法核对统计设计 | `cohort_pairing` 写入 eval input、E2E statistical evidence，并由 frozen verifier 复核。 |
| F-12 | held-out donor 的 state 覆盖不是显式契约，单态池可能进入 rescue | 新增显式 `--state-obs-column/--require-state-coverage`，缺 donor split 时硬失败，覆盖分布写入 manifest。 |
| F-13 | standardize 的 per-sample 唯一性检查漏掉 `Diagnosis`，双标注可能污染 donor 分组 | 将 `Diagnosis` 纳入唯一性检查，并补真实输入对应的合成回归测试和三态退出码。 |
| F-15/F-16 | between_donor 统计链覆盖不足，DEG 列名在 E2E 中硬编码 | 新增 between_donor 统计链测试；E2E 暴露四个 DEG 列参数并写入 lineage。 |
| TD-14-02/03/04/05/06 | E2E 校验层过长、序列化实现重复、审计/standardize 有未覆盖或隐式路径 | 拆分校验函数，统一 `reports.to_plain_object`，补脚本测试，删除死函数并明确退出码。 |
| N-17（本轮新发现） | 正式 vocabulary 含特殊 token；旧 `_token_values` 对所有键 canonicalize，且 `except ValueError: continue` 会把任意非法 gene key 静默吞掉 | 仅允许精确特殊 token `<cls>`、`<eos>`、`<mask>`、`<pad>` 跳过；`<unk>`、任意非法 key、负数/非整数特殊 token ID 均 `ValueError` 硬失败。mapping 和 sequence 两种词表都覆盖。 |
| bridge runbook（本轮新发现） | §3 第 2 步漏掉 `_resolve_perturbgen_contract` 强制要求的 config、seed 和 sensitivity 参数；null 示例使用非法 `<...>` 语法及错误关键字 `output` | 补 `--perturbgen-config`、`--seeds 0,1,2`、`--sensitivity-modes pad,delete`；新增从 E2E `statistical_evidence` 读取真实路径的 `--verify` 命令；示例改成语法有效的真实数据绑定接口骨架并使用 `output_path=`。 |

### 2.1 按严重程度汇总

高项中，F-10、F-14 和本轮 N-17 已修复并有测试/真实资产证据；第七轮遗留的冻结前置
数据契约已闭合。问题 1 的真实 GPU matched-null、问题 2 的完整六阶段与独立 `--verify`
尚未执行；问题 4 的 Workflow B/Gate-E ≥200 benchmark 仍未执行。

中项 F-11、F-12、F-13、F-15、F-16 与 TD-14-02/03/04/05/06 已由源码和聚焦测试
闭合。scPerturb 0/30 合规队列仍是外部数据事实，没有用代码补造；GSE174367 是当前
已冻结的替代队列。

低项中，旧 `latent_davf_perturbgen_4018` checkpoint 的 canonical ENSG 资产失败仍存在；
TD-13 的全仓 format/覆盖率/长函数清单保持独立 PR 取舍，没有为本轮扩大修改范围。

## 3. 迭代结果

### 3.1 侦察与真实资产审计

源码核对确认：

- `scripts/run_davf_perturbgen_e2e.py` 的正式 PerturbGen 路径在缺少
  `--perturbgen-config` 时硬失败；parser 默认仍是 `seeds="0"`、`sensitivity_modes=""`，
  所以正式冻结矩阵必须显式传参，不能依赖默认值。
- `build_candidate_stage_plans` 的主 mode 为 `mask`，显式 `pad,delete` 加上
  `0,1,2` 形成每候选 2 path × 3 mode × 3 seed = 18 个 perturb runs；冻结 manifest
  的 5 个候选因此为 90 runs。
- `_assemble_statistical_evidence` 实际写出 `statistical_evidence.eval_input` 和
  `statistical_evidence.report_manifest`，所以新增的 verify 命令不猜统计文件名。
- `run_matched_null_stages` 的真实参数是 `rescue_extractor(request, stage_result)`、
  `output_path=`；rescue extractor 必须由外部研究代码用登记的真实 h5ad、per-donor
  表达和 donor-level DEG 构造 `NullStageRecord`。

真实冻结目录当前核对结果：

- `manifest.json`：`ptm2cellnet.frozen-cohort/v1`、EX、`between_donor`，train 12、
  held-out 6，18 个 donor 覆盖；cohort 路径为
  `data/AD/standardized/GSE174367_ad_cohort.h5ad`。
- 候选：APP、PSEN1、BACE1、MAPT、APOE，均为 KO，ENSG 与 cohort var/embedding
  vocabulary 绑定；每个候选 seeds 为 `[0,1,2]`、modes 为 `[mask,pad,delete]`、
  matched null 为 99。
- `acceptance_plan.json`：`n_runs=90`、`formal_plan_complete=true`、无
  `formal_plan_issues`。
- `null_selection/`：5 份 manifest，每份 99 个 selection；fc 缺少正式 DEG 时标记
  为 `unavailable/default`，未伪造 fc 值。

### 3.2 本轮新增问题的验证

正式 vocabulary 实测为 18,967 个键：18,963 个 gene rank 加精确四个特殊键
`<cls>`、`<eos>`、`<mask>`、`<pad>`。`{"<unk>": 4}` 已实际触发
`ValueError: invalid Ensembl gene id`。新增测试覆盖 mapping/sequence 的未知特殊
token；原有测试覆盖精确四个特殊 token 的跳过和 rank 位置语义。

runbook 静态修复通过新增的 E2E contract 测试：显式 `(0,1,2)` 和
`("pad", "delete")` 被实际解析，`between_donor` 被绑定到 Gate-0 spec；缺 config
仍硬失败。进一步核对 `_resolve_statistical_evidence_inputs` 后修正 §3 顺序：第 2 步
只跑六阶段，第 3 步完成真实 null，第 4 步用同一参数加 `--resume` 才传入统计参数，
第 5 步再 `--verify`。frozen replay 既有测试实际验证 tampered verdict 返回 CLI code `1`。

### 3.3 现有工作区验证结果

实际执行命令和结果：

```text
python -m pytest -m "not slow and not gpu" --timeout=300
# 2675 passed, 1 failed, 22 skipped, 59 warnings, 796.66s
# 唯一失败：
# tests/integration/test_scvi_davf_connection.py::test_real_current_davf_direction_keeps_token_and_decoder_indices_separate

python -m pytest -q --timeout=300 \
  tests/unit/scripts/test_run_davf_perturbgen_e2e.py \
  tests/unit/integration/perturbgen/test_null_selection.py
# 23 passed, 8 warnings, 3.93s

python -m pytest -q --timeout=300 \
  tests/unit/integration/perturbgen/test_frozen_cohort.py \
  -k 'ReplayVerdicts or AcceptancePlan'
# 3 passed, 19 deselected, 8 warnings, 7.93s

PTM2CELLNET_RUN_REAL_ASSET_TESTS=1 python -m pytest -q --timeout=300 \
  tests/real_assets/test_real_frozen_cohort.py -v
# 1 passed, 8 warnings, 18.25s

ruff check src scripts tests
# All checks passed

python -m ruff format --check \
  scripts/audit_ad_cohort_gate0.py \
  scripts/build_davf_scperturb_pairs.py \
  scripts/run_davf_perturbgen_e2e.py \
  scripts/run_frozen_acceptance.py \
  scripts/standardize_gse174367_ad_cohort.py \
  src/data/davf_scperturb.py \
  src/integration/perturbgen/eval_assembly.py \
  src/integration/perturbgen/frozen_cohort.py \
  src/integration/perturbgen/null_selection.py \
  src/integration/perturbgen/orchestrator.py \
  src/integration/perturbgen/replay_evaluation.py \
  src/integration/perturbgen/reports.py \
  tests/unit/data/test_davf_scperturb.py \
  tests/unit/integration/perturbgen/test_frozen_cohort.py \
  tests/unit/integration/perturbgen/test_null_selection.py \
  tests/unit/scripts/test_run_davf_perturbgen_e2e.py \
  tests/real_assets/test_real_frozen_cohort.py \
  tests/unit/scripts/test_audit_ad_cohort_gate0.py \
  tests/unit/scripts/test_standardize_gse174367_ad_cohort.py
# 19 files already formatted；未执行、也未声称全仓 format check 通过

python -m compileall -q src scripts tests
# exit 0

python -m mypy src/ --ignore-missing-imports
# Success: no issues found in 166 source files

python scripts/check_requirements_consistency.py
# requirements/lock consistency OK: 274 lock pins satisfy all core constraints
```

唯一全量失败的根因已单测复现：旧 checkpoint 加载时因
`checkpoint.scvi.gene_names must contain canonical ENSG identifiers` 被拒绝，随后
`predict_expression_direction` 正确拒绝 `zero_fallback`，报
`DAVF checkpoint is not loaded; expression direction evidence cannot be produced from
zero_fallback`。这不是本轮修改引入，也不应通过放宽 hard-fail 或静默 fallback 修复。

## 4. E2E 训练/推理逐项复核

| 环节 | 当前事实 | 证据/结论 |
|---|---|---|
| PTM proposal → scVI/PTM 映射 → DAVF decode → 三方 direction gate → invocation | 代码链、字段约束和 invocation gate 在位 | 既有 E2E/bridge 聚焦测试通过；未用旧 checkpoint 失败资产声称科学方向 PASS |
| Gate-0 数据契约 | GSE174367 `between_donor` 的 EX 真实 cohort、donor split、canonical ENSG 和 pairing 已绑定 | real frozen test 1 passed；这是数据契约 PASS，不是生物学 PASS |
| frozen manifest / donor leakage / 90-run plan | 5 候选 × 2 path × 3 seed × 3 mode，plan complete | manifest 与 acceptance plan 实际字段核验通过 |
| `_token_values` vocabulary | 精确四 token 白名单；非法 key 硬失败 | 真实 vocabulary 统计与 23 项聚焦测试通过 |
| shared prepare / candidate reuse | `tokenise/train_mask/train_decoder` 由 shared prepare 计划执行一次，候选引用缺失硬失败 | orchestrator 源码和现有 mocked pipeline 测试通过 |
| DAVF 推理 | CPU/mock/契约层可测；旧 checkpoint canonical ENSG 失败保持显式 | 全量唯一失败为已知真实资产基线；未将零 fallback 当结果 |
| 六阶段 PerturbGen GPU 执行 | 未执行 | 当前没有 candidate_spec、正式 DEG、E2E report；没有猜路径或伪造运行 |
| matched-null stage 批跑 | selection 前置资产已生成；99 个 null 的实际 perturb/rescue stage 未运行 | 只有 5×99 selection manifest，不能写成 null distribution/statistical PASS |
| `--assemble-statistical-evidence` | 代码和 synthetic/mock 链接有测试；本次没有正式输入可组装 | 未生成正式 p/q、BH-FDR 或 dual-path verdict |
| M6 `run_frozen_acceptance.py --verify` | CLI 会核验 manifest、eval input、report manifest、覆盖、replay 和独立 h5ad 重算；tampered verdict 测试返回 1 | 本次没有 E2E 产出的 eval/report 路径，未执行正式 verify；bridge §3 提供了可复制命令 |
| Workflow B / Gate-E ≥200 | 未执行 | 需要独立 encoder/embedding/LatentDAVF 生命周期和研究输入 |

### 4.1 当前环境与执行边界

GPU 探针实际结果：

```text
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# Tesla P40, 580.178.04, 24576 MiB

SSH_unit Python: torch 2.4.1+cu118, cuda_available=True, device_count=1,
                device=Tesla P40, capability=(6, 1)
PerturbGen direct Python: torch 2.5.1+cu124, cuda_available=True, device_count=1,
                          device=Tesla P40, capability=(6, 1)
```

因此当前记录不是 NVML 错误或驱动不可见；本轮没有做系统驱动修复。GPU 训练仍标记
“未执行”，原因是正式研究输入和运行产物缺失，而不是把环境探针误写成研究运行。
PerturbGen 独立环境的直接解释器可以导入仓库内 `ref/Perturbgen-src/perturbgen`，但这
只证明环境入口可见，不证明六阶段或生物学结果已经运行。

输出目录核查没有发现正式 candidate spec、donor-level DEG、null distribution index
或 E2E report；只有冻结候选/selection 资产和历史 archive candidate scores。因此
没有执行正式 `--run-perturbgen` 命令，也没有使用占位文件名冒充成功。

## 5. `project_analysis_20260914.md` §4 问题 1–6 的后续策略与时间节点

时间节点是输入就绪后的执行计划，不是已完成时间或结果承诺。

| 问题 | 当前状态与根因 | 后续策略 | 时间节点（计划） |
|---|---|---|---|
| 1. 六阶段 GPU + matched-null + 统计验收 | 未执行；缺正式 candidate_spec、donor-level DEG、perturb h5ad/rescue 记录和 null distribution manifest。GPU 探针本身已通过 | 先由研究输入持有者提供 PTM site/方向/semantic context/DEG；按 bridge §3 第 2 步只跑六阶段，随后按 5 候选 × 2 path × 3 seed × 3 mode 跑真实 matched-null，再用同一参数加 `--resume` 和统计参数组装，最后 `--verify` | 输入齐备后 D0 冻结守护；D1–D2 六阶段；D2–D4 null/assemble/verify |
| 2. M6 `--verify` 独立复算 | 未执行；依赖第 1 项产生的 E2E `statistical_evidence` 路径 | 从 E2E JSON 读取 `eval_input`、`report_manifest`，执行 bridge §3 的 `run_frozen_acceptance.py --verify`；缺失/篡改必须保留非零 | 第 1 项组装完成后同日，预计 0.5 天 |
| 3. A-04/Gate-E ≥200 benchmark | 未执行；Workflow B 需要独立 embedding asset、LatentDAVF 重训和 GPU 排期 | 按 `encoder → 冻结 embedding → LatentDAVF retrain → Gate-E` 顺序执行，不把 Workflow A 结果代替 | 正式双路径效用验收后，单独 GPU 周期 |
| 4. 旧 checkpoint 真实资产失败 | 未修复；根因是 checkpoint 的 gene names 非 canonical ENSG，当前 hard-fail 正确 | 用 canonical ENSG 管线重训/替换明确登记资产，重新运行失败测试；不放宽检查、不启用 zero fallback | 下一个真实重训窗口，依赖 GPU 与合法 checkpoint |
| 5. TD-13 存量 | 未处理；全仓格式、覆盖率门禁刷新和长函数清单属于独立维护批次 | 独立 PR 处理；本轮只保持触碰文件格式与静态检查通过 | 下一次维护批次，不纳入本轮 |
| 6. scPerturb 侧 0/30 合规 | **已处置为外部数据阻塞**；缺 donor/cell 注释，没有合法代码补造路径；GSE174367 已作为替代队列冻结 | 外部数据更新时重跑审计；不把 0/30 写成代码失败，也不伪造合规 cohort | 视研究需要 |

完成条件仍是：真实 cohort、显式 donor、canonical Ensembl/scVI order、冻结 manifest、
实际 null/质量/p-q/dual-path lineage、独立 verify 全部通过后，才可讨论正式计算效用；
这些条件仍不等于治疗或临床因果结论。

## 6. 章节与证据引用

- `project_analysis_20260914.md` §4：问题 1–6、剩余技术债和时间策略来源。
- `project_analysis_20260914.md` §1–§3：M6 资产、runbook、E2E 代码/契约现状；其中
  旧的 broad “skip non-ENSG” 和 NVML mismatch 表述以本轮源码/命令实测为准。
- `docs/guides/perturbgen_bridge.md` §3：冻结状态、六阶段命令、90-run 参数、真实
  路径驱动的 `--verify` 与 matched-null 接口骨架。
- `docs/guides/davf_perturbgen_e2e.md`：统计组装、donor split 和不把 smoke/mock 当
  生物学 PASS 的边界。
- `lessons.md` L-2026-0914-01/02/03/04：pairing、M6 资产、special-token hard-fail
  与 runbook 纠偏决策。
- `src/integration/perturbgen/null_selection.py`、`scripts/run_davf_perturbgen_e2e.py`、
  `src/integration/perturbgen/frozen_cohort.py`、`scripts/run_frozen_acceptance.py`：
  本轮关键契约实现。

本轮实际使用 `ponytail` skill；它约束本次只做必要的白名单、测试和 runbook 修复，
没有新增兼容层、猜测性 fallback 或无关重构。
