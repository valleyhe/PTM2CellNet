# PTM2CellNet 技术债修复报告（2026-09-21）

> **修复基线**：`project_analysis_20260921.md`（审计日期 2026-09-21，HEAD `323ab4d`）
> **修复会话**：ZCode 本地修复轮（session `zcode-repair-20260921`）
> **执行环境**：本机 `SSH_unit` conda 环境（GPU Tesla P40 可见，`torch.cuda.is_available()=True`）；KSTAR 独立环境 `kstar`（Python 3.12.14 + kstar 1.2.0）未在本轮触发 full analysis
> **修复范围**：TD-01（重点）及严重/高/中级技术债中可本地闭环验证的项（TD-01/02/04/05/06/07/12/13/14/16/17）；TD-03/08/09/10/11 完成现状量化与策略制定
> **验证口径**：与 `docs/CURRENT_STATUS.md` 验证命令一致（pytest 分层、ruff、mypy、依赖一致性）

---

## 目录

- [1. 执行摘要](#1-执行摘要)
- [2. 问题修复汇总（按严重程度）](#2-问题修复汇总按严重程度)
- [3. 每轮迭代执行情况与结果](#3-每轮迭代执行情况与结果)
- [4. 系统性复核：E2E 训练与推理技术要求](#4-系统性复核e2e-训练与推理技术要求)
- [5. 未解决问题及后续解决策略](#5-未解决问题及后续解决策略)
- [6. 文档引用索引](#6-文档引用索引)

---

# 1. 执行摘要

本轮依据 [`project_analysis_20260921.md`](project_analysis_20260921.md) §6（技术债清单）与 §7（解决策略），在本地工作树完成 **11 项技术债的代码级修复**（1 严重正确性缺陷 + 1 严重科学准入缺口 + 4 高级 + 5 中级），并对 5 项不宜本轮盲动的债务（TD-03/08/09/10/11）完成现状量化与策略制定。

**核心成果**：

| 指标 | 修复前（分析报告口径） | 修复后（本轮实测） |
|---|---|---|
| KSTAR 真实 full run 阻断 | 双方向 metrics 相等断言必失败（§5.2.1） | 断言删除，方向化 metrics 绑定 + 4 项新测试 |
| 传播前 activity 准入 | 无（q=1/0 substrate/0 coverage 可传播，§5.2.2） | 冻结 policy + 逐行拒绝原因 + policy hash，旧无门禁入口删除 |
| fast 测试口径完成度 | 2,945 项收集、188 项超时未完成（§8.1.1） | **2,868 passed / 0 failed / 275.68s**，全部完成 |
| slow/integration 分层 | **0 项**测试带 slow/gpu marker（分层形同虚设） | 17 项重型测试标记 slow，`full-test.yml` 承接（`not gpu`） |
| builder mypy | 4 errors / 3 files（§8.2） | 0 errors（逐文件复验） |
| active network release | TF-only `omnipath-2026-09-16`（§5.1） | `omnipath-kinase+tf-2026-09-21` + manifest 三向 hash 绑定校验 PASS |
| KSTAR 网络 verifier | 空 `INDIVIDUAL_NETWORKS` 可通过（§5.2.3） | ST/Y 各 50 非空文件 + 双 Unique Network ID pin，缺/空/漂移硬失败 |
| Pydantic v1 deprecated | `@validator`/`class Config` 触发 v3 removal 警告 | 迁移 `field_validator`/`model_validator`/`ConfigDict`，警告清除 |
| 根目录污染 | `=1.0.0`、`=3.0.0` 等 7 个 pip 残留不可见（§5.1） | 已删除 |

**不变量（明确未达成，不因本轮修复改变）**：正式 biology PASS 仍为 0；真实 PTM cohort（`PENDING_EXTERNAL_PTM_COHORT`）与独立 activity benchmark 仍未到位；`outputs/` 扫描 `biology_pass=true` 计数仍为 0。本轮所有修复均为工程合同层收紧，**不产生任何新的生物学结论**。

---

# 2. 问题修复汇总（按严重程度）

## 2.1 严重（TD-01、TD-02）——已修复

### TD-01：KSTAR 两方向 metrics equality + 单 metrics 绑定 【已修复】

- **问题**（分析报告 §5.2.1、§6.1 TD-01）：`scripts/run_kstar_activity.py:367-368`（修复前行号）错误要求 increased/decreased 两次 KSTAR 分析的 network metrics 全表 `equals`。两次分析消费不同 evidence column（`data:{contrast}:increased` vs `:decreased`），`_metrics_from_kinact` 的 `evidence_sites` 随方向不同，`n_substrates`/`network_coverage` **合法不同**——该断言使真实 full analysis 确定性失败。同时 adapter 只收 `metrics=increased_metrics`，胜出方向为 decreased 的 kinase 也会被绑上 increased 的 metrics（方向-证据错配）。
- **修复方案**（采纳分析报告 §7.1 TD-01 方案 A）：adapter 显式接收成对方向化 metrics，按胜出方向绑定。
- **改动**：
  - `src/analysis/kstar_adapter.py::convert_kstar_outputs`：签名 `metrics=` → `increased_metrics=`/`decreased_metrics=`（必须成对提供，单只提供硬失败）；两方向 metrics kinase 集必须一致（完整性检查）；每个 kinase 行按 `direction = increased if inc_p < dec_p else decreased` 取**该方向**的 `n_substrates`/`network_coverage`。无 metrics 时保留旧内嵌回退路径（要求两方向内嵌一致，供单元测试构造）。
  - `scripts/run_kstar_activity.py`：删除 equality 断言；两方向 metrics 分别落盘 `<name>_increased_network_metrics.tsv` / `<name>_decreased_network_metrics.tsv`；manifest `directional_analysis` 记录两方向 metrics 文件、行数、`n_substrates` 总量与 `metrics_binding` 语义说明。
  - Manifest schema：`ptm2cellnet.kstar-adapter-manifest/v1` → **`/v2`**（全仓消费方核查：仅 runner 写入，无读取方受影响）。
- **测试**：`tests/unit/analysis/test_kstar_adapter.py` 新增 `test_winning_direction_binds_its_own_metrics`（GSK3B 胜出 increased 绑定 7/0.7，CDK5 胜出 decreased 绑定 5/0.5——正是分析报告要求的"两方向 substrate/coverage 不同"场景）、`test_directional_metrics_must_be_supplied_together`（单只提供/集合不一致均硬失败）；原 3 处 `metrics=` 调用点迁移。
- **验证**：`test_kstar_adapter.py` 9 passed。
- **边界**：真实 KSTAR full analysis 仍待真实 PTM cohort（§2.3 U-04）；本修复使其不再被确定性逻辑缺陷阻断。

### TD-02：无 activity benchmark/admission gate 【已修复（policy+selection 部分；benchmark evaluator 依赖外部资产，见 §5）】

- **问题**（分析报告 §5.2.2、§6.1 TD-02、§4.2.3）：`activities_for_propagation()` 只按 method/contrast 选择全部非零 score，`activity_qvalue`/`n_substrates`/`network_coverage` 被验证但**不参与准入**——q=1、0 substrate、0 coverage 的非零 activity 可进入正式传播，违反执行方案 §4.4/§5.2"预先冻结阈值"与 §8"传播前冻结准入"。
- **修复方案**（采纳分析报告 §7.1 TD-02 方案 A）：`ActivityAdmissionPolicy` + `ActivitySelectionResult`，传播入口强制消费。
- **改动**：
  - `src/analysis/ptm_research_config.py`：新增 frozen dataclass `ActivityAdmissionPolicy(max_activity_qvalue, min_substrates, min_network_coverage)`，带 `policy_hash`（sha256，内容寻址）与范围校验；`PTMResearchConfig` 新增 `activity_admission` 字段（YAML `activity_admission` 段解析，未知键拒绝）。
  - 新模块 `src/analysis/ptm_activity_admission.py`：`select_activities_for_propagation(frame, *, method, condition_or_contrast, policy) -> ActivitySelectionResult`——admitted dict + rejected DataFrame（逐行原因 `activity_qvalue_above_max`/`n_substrates_below_min`/`network_coverage_below_min`）+ policy hash + `benchmark_gate` 登记（无独立 benchmark 时 `available: false`，admitted 明确标注 exploratory-only，**不伪造 benchmark PASS**）；全部被拒时硬失败并输出拒绝明细。
  - `src/analysis/ptm_activity.py`：**删除** `activities_for_propagation`——不再存在绕过 admission 的传播入口（分析报告 §7.1 TD-02 方案 B 的批评点："容易被其他调用入口绕过"）。
  - `scripts/build_ptm_global_gene_scores.py`：阶段 3 改为 `select_activities_for_propagation(..., policy=config.activity_admission)`；manifest `sources.activity_table.admission` 记录完整 selection 审计。
  - `configs/research/ptm_research_config.yaml`：显式登记 `activity_admission` 宽松冻结值（1.0/0/0.0）并注释"待独立 benchmark 校准后显式修订；一经冻结不得事后调整（方案 §8.7）"。
- **测试**：新增 `tests/unit/analysis/test_ptm_activity_admission.py` 10 项（policy hash 稳定性、越界阈值拒绝、宽松默认全收、严格策略逐行拒绝原因、全拒硬失败、缺列硬失败）；`test_ptm_activity.py` 旧选择测试迁移至新入口。
- **验证**：分析层 249 passed。
- **决策说明**：默认宽松阈值是**显式登记**而非缺省隐式——当前无独立 benchmark 校准（方案 §5.2 前提不满足），收紧阈值属研究决策（AGENTS.md 2026-09-14 约束：不得在 AD 结果上反调 PTM 侧阈值）。机器门禁已就位，研究负责人冻结严格值后立即生效。

## 2.2 高（TD-04、TD-05、TD-06、TD-07）——已修复

### TD-04：Formal PTM intake 仅非空校验 【已修复】

- **问题**（分析报告 §6.1 TD-04、§8.2"Formal intake 直接探针"）：`PTMResearchConfig` 接受 `ptm_cohort: PENDING_EXTERNAL_PTM_COHORT`；KSTAR `_require_manifest()` 接受空 `{}`；实测每状态仅 1 donor 仍生成 evidence——"外部输入未到位"只靠人工记忆阻止。
- **修复方案**（采纳 §7.1 TD-04 方案 A 的 typed gate 核心）：
  - `PTMResearchConfig` 新增 `mode: exploratory|formal`（默认 `exploratory`）；**formal 模式拒绝一切 `PENDING*` 占位资产**（`ptm_cohort`/`cohort_h5ad`），exploratory 保持 sentinel 与既有 lineage 边界（`may_enter_lineage=false`）。
  - `kstar_adapter._require_manifest`：要求结构化 stage-1 manifest——`schema_version` 必须精确匹配 `ptm2cellnet.ptm-input-manifest/v1`，必须含 `source.sha256` 溯源与 `standardization` 统计；空 `{}`/自由 JSON 硬失败。与 `outputs/ptm_activity/20260918_ptm_smoke/pipeline/ptm_input_manifest.json` 的真实结构一致（已核对）。
  - `build_kstar_input` 新增 `min_donors_per_state`（默认 1，`__post_init__` 范围校验）；runner CLI 新增 `--min-donors-per-state`——正式运行显式提高 donor 下限，单 donor 表无法进入正式 KSTAR evidence。
- **测试**：`test_structured_input_manifest_is_required`（空 manifest 拒绝）、`test_min_donors_per_state_gate`（下限 3 拒绝/2 通过/0 硬失败）。
- **未纳入**：typed cohort manifest schema（donor 设计/hash 绑定逐字段 schema）——依赖真实 PTM cohort 的研究设计声明（配对语义、donor 下限须预注册，§7.1 TD-04 风险栏），列为后续工作（§5.4）。

### TD-05：KSTAR network verifier 可接受空网络目录 【已修复】

- **问题**（分析报告 §5.2.3、§6.1 TD-05）：`verify_kstar_network_dir` 只查目录存在 + Unique Reference ID；`tests/unit/analysis/test_kstar_resources.py:88-96`（修复前）把"空 INDIVIDUAL_NETWORKS 目录通过"固化成预期行为。
- **修复**：
  - `src/analysis/kstar_resources.py`：新增常量 `KSTAR_ST_UNIQUE_NETWORK_ID`/`KSTAR_Y_UNIQUE_NETWORK_ID`/`KSTAR_EXPECTED_NETWORK_FILES=50`（来自本机实测：`/home/scu/anaconda3/envs/kstar/NETWORKS/`，与分析报告 §8.3 实测一致）；`verify_kstar_network_dir` 重写——RUN_INFORMATION 同时提取并校验 Reference ID **与 per-type Unique Network ID**；每类型必须**恰好 50 个非空文件**；返回新 `KSTARNetworkAudit` dataclass（reference_ids/network_ids/n_files/total_bytes）。
  - `scripts/run_kstar_activity.py`：network audit 全量写入 adapter manifest（`network_audit` 段）。
  - **删除固化弱校验的测试**，改为 `test_network_verifier_rejects_empty_partial_and_truncated_dirs`（空目录/缺 1 文件/0 字节文件均硬失败）+ Network ID 漂移负测试；测试 helper `_write_network_dir` 构造合规 50+50 fixture。
- **验证**：`test_kstar_resources.py` 6 passed。

### TD-06：signed/KSTAR 实体资产不可移植且被忽略 【最小增量已落地；完整 resolver 待外部决策】

- **现状澄清**（比分析报告 §6.1 TD-06 更进一步的事实）：恢复机制**已存在**——`scripts/setup_kstar_env.sh` 一键重建 conda env（`environments/kstar/environment.yml` 钉包）→ 校验资源 hash（`resource_hashes.json`）→ `install_kstar_networks` 从官方 figshare URL（`kstar_resources.py` 钉双 URL）下载或复用本地 archive → 严格 verifier（本轮 TD-05 强化）→ mapping smoke。真正的缺口是**官方 URL 单点**与 signed network 实体资产的对象存储镜像（托管位置是外部决策，§7.2 TD-05/06 方案 A 已明示"需要确定资产托管位置"）。
- **本轮落地**：
  - release manifest `data/manifests/kstar_signed_network_release_20260921.json` 新增 `kstar_networks` 段：archive sha256（实测 `2fb054e6…`，与分析报告 §8.3 一致）、archive 字节数、ST/Y Unique Network ID、每类型文件数、恢复指引——release 与 KSTAR 环境资产绑定可追溯。
  - `scripts/build_kinase_tf_network_release.py`：manifest `output` 段补 `sha256`（此前无 hash，下次重建 release 无法被 binding 校验消费）。
- **遗留**：对象存储/内部镜像 + `resolve_signed_network_release(manifest, cache)` 自动恢复（§5.4 后续工作）。

### TD-07：active config 未绑定 kinase+TF 2026-09-21 release 【已修复】

- **问题**（分析报告 §6.1 TD-07、§5.1）：config 仍为 TF-only `omnipath-2026-09-16`，新 combined release 已生成但正式传播不会使用。
- **修复方案**（采纳 §7.1 TD-07 方案 A）：
  - `configs/research/ptm_research_config.yaml`：`network_release: "omnipath-kinase+tf-2026-09-21"`；新增 `network_release_manifest: data/manifests/kstar_signed_network_release_20260921.json`；`propagation.gene_edge_types` 增加 `"kinase_substrate:signaling"`（kinase activity 主线要求 kinase→substrate 边终止于基因；AGENTS.md 2026-09-18 明确"正式 KSTAR→gene score 必须另冻 kinase–substrate + TF 网"，该冻结即本 release）。
  - `src/analysis/signed_network.py`：新增 `verify_network_release_binding(network_tsv, manifest_path, *, expected_release)`——**三向校验**：config 的 release 名 ↔ manifest `release` 字段 ↔ 实际传播用网络文件的 sha256+数据行数（兼容 manifest 的 `combined` 与 builder 的 `output` 两种段名）。任何漂移（自由文本、manifest 记录、磁盘资产）硬失败。
  - `scripts/build_ptm_global_gene_scores.py`：config 设置 `network_release_manifest` 时启动即执行绑定校验，结果写 manifest `release_binding`。
- **实测**：对真实资产执行 `verify_network_release_binding` **PASS**（`omnipath-kinase+tf-2026-09-21`，28,011 行，sha256 `9483f613…` 与 manifest 一致）。
- **研究语义说明**：release 切换是研究口径变更——config 注释明确"在此绑定变更前生成的 score 与新 release 结果不得混用"；旧 TF-only 文件未改动（符合 AGENTS.md"不得改 2026-09-16 文件原地加点边"）。
- **测试**：`test_signed_network.py` 新增 binding 正例 + 三类漂移负例（release 名/资产 hash/行数），24 passed。

## 2.3 中（TD-12、TD-13、TD-14、TD-16、TD-17）——已修复

### TD-12：测试 marker 与分片策略失效 【已修复】

- **问题实证**（本轮补充的关键事实，比分析报告 §8.1 更精确）：全仓 **2,959 项测试中 0 项带 slow marker、0 项带 gpu marker**——`pytest.ini` 定义了 marker 但从未被使用，`not slow and not gpu` 过滤形同虚设；且 `full-test.yml`（名义"全量"）也排除 slow，**没有任何 CI job 承接 slow 测试**。
- **修复**（采纳 §7.4 TD-12 方案 A）：
  - 17 项重型 integration 测试（实测时长 11–27 秒/项：`test_train_resume_cli.py` 全文件 5 项、`test_davf_pipeline.py` 4 项重型 forward、`test_data_contract_fixes.py` ScperturbRegistration 2 项、`test_lightning_pipeline.py` train smoke、`test_data_manifest_cli.py` 2 项 CLI 校验）加 `pytest.mark.slow`。
  - `.github/workflows/full-test.yml`：`not slow and not gpu` → **`not gpu`**（slow 由 full-suite job 承接）；CI fast job 口径不变。
- **效果实测**：fast 口径（unit+integration+root，`not slow and not gpu`）**2,868 passed / 275.68s 全部完成**（分析报告 §8.1.1 记录 188 项超时未完成）；slow 层 17 passed / 251.77s 单独可跑。

### TD-13：KSTAR/network builder 4 个 mypy 错误 【已修复】

- **修复**（采纳 §7.4 TD-13 方案 A）：`_edge_sign(row: pd.Series)` → `Mapping[Any, Any]`（`to_dict("records")` 产出 dict，两个 builder 共享该函数）；`build_kinase_signed_edges.py`/`build_kinase_tf_network_release.py` manifest 返回注解 `dict[str, object]` → `dict[str, Any]`（消费方需要嵌套索引）；补 `typing.Any` import。
- **验证**：三个 builder + runner 逐文件 mypy `Success: no issues found`；`mypy src/` 183 文件 0 errors 不变。

### TD-14：Pydantic v1 deprecated API 【已修复】

- **修复**（采纳 §7.4 TD-14 方案 A）：`src/api/schemas.py` 的 `@validator` → `@field_validator`（`limit_samples`）、`@validator(always=True)` → `@model_validator(mode="after")`（`fill_predicted_cell_state` 的 alias 回填语义在 v2 下由模型级 after validator 保证恒执行）、`class Config: allow_population_by_field_name` → `model_config = ConfigDict(populate_by_name=True)`。
- **验证**：`test_schemas.py`/`test_api.py`/`test_api_routes.py`/`test_predictions_routes.py`/`test_api_lifespan.py` 共 108 passed；项目源码中 Pydantic v1 API 清零（剩余警告均为第三方 mamba_ssm/Starlette，属 TD-18 低级债）。

### TD-16：requirement/status/guide 状态漂移 【已修复（本轮覆盖项）】

按分析报告 §5.1 漂移清单逐项同步（采纳 §7.4 TD-16 方案 C 止血 + 本轮新增变更一并记录）：

| 文档 | 修复前状态 | 修复后 |
|---|---|---|
| `.planning/REQUIREMENTS.md` P-04 | "no KSTAR/PhosR/OmniPath execution entry exists" | 更新为 KSTAR 执行边界已落地（runner/adapter/resources/builders），主环境仍无 KSTAR 依赖 |
| `.planning/REQUIREMENTS.md` P-05 | "all six remain owner-supplied" | 更新为 KSTAR env+网络+OmniPath release 已到位可验证；PTM cohort/PhosR 输出/benchmark/DEG 冻结仍缺 |
| `.planning/REQUIREMENTS.md` P-06 | "Open; wiring is follow-up work" | **Landed**：`_assemble_downstream_target_evaluation` 已接线（与分析报告 §5.2.4 证据一致），正式真实运行证据仍缺 |
| `.planning/STATE.md` | downstream lineage 列为 next work；active_next_action 过时 | 同步为已落地 + 本轮修复记录 + 新的 next order |
| `docs/guides/kstar_activity_plan.md` | "analysis 仍待 ST/Y network"（§75-76/200-220 区域） | ST/Y 网络已装齐并通过严格 verifier；TD-01 方向化 metrics 语义、v2 manifest、min-donors 参数写入执行合同 |
| `docs/guides/ptm_activity_pipeline.md` | config 示例无 admission/mode/manifest 字段 | config 示例与阶段 3 说明同步（admission 硬门禁、binding 校验、combined release） |
| `API_DOCUMENTATION.md` | 指向 `project_analysis_20260917.md` | 指向 20260921 + 本修复报告 |
| `docs/CURRENT_STATUS.md` | Last updated 为修复前状态 | 新增 2026-09-21 修复轮 Quick Reference 条目（lessons L-2026-0921-02） |

### TD-17：根目录 ignore 隐藏工作区污染 【已清理（残留部分）】

- **已做**：删除 7 个 pip 命令事故残留（`=0.8.0`、`=1.0.0`、`=2.5.0`、`=3.0.0`、`0.7692`、`EOF`；均为 pip install 输出重定向意外创建，内容已核对确认）。注：分析报告提及的 `1852544` 实为 `ls -la` 的 total 行，非文件。
- **未做**（有意保留，见 §5.4）：取消根级 `/*` whitelist 改黑名单（§7.4 TD-17 方案 A）——初次启用会使大量本地大资产（`model_checkpoints.zip` 1.7GB、`methylation.zip` 等）涌入 `git status`，需要先做目录治理决策；本轮以清理存量污染止步，whitelist 改造列为后续工作。

## 2.4 严重/高级但本轮不盲动的项（TD-03、TD-08）——策略见 §5

- **TD-03（严重，依赖/供应链）**：现状复测 `pip check` 3 组冲突依旧（numpy 2.4.3 vs `>=1.24,<2`；scvi-tools 1.4.3 vs `<1.0`；torchaudio 2.4.1+cu118 vs `>=2.5`）；`pip-audit` 本轮口径 **26 包 / 96 条唯一 (package, advisory) 记录**（分析报告口径 18 包/152 条原始记录含重复；两次扫描的 advisory 数据库版本不同）。不盲动理由与策略见 §5.1。
- **TD-08（高，Workflow B）**：组件盘点确认 `build_davf_latent_pairs.py`、`train_latent_davf.py`、`evaluate_latent_davf.py`、`build_gate_e_benchmark.py`、`evaluate_gate_e.py`、`gate_e.py` 均存在，缺统一可重放编排与真实 ≥200 条 benchmark（外部资产）。策略见 §5.3。

---

# 3. 每轮迭代执行情况与结果

本轮按"严重正确性 → 科学准入 → 资产完整性 → 配置绑定 → 工程卫生"顺序分 6 轮迭代，每轮独立验证后进入下一轮。

| 轮次 | 内容 | 涉及文件 | 验证命令与结果 |
|---:|---|---|---|
| R1 | **TD-01 KSTAR 双方向 metrics**：删除 equality 断言；`convert_kstar_outputs` 方向化签名与胜出方向绑定；runner 落盘两方向 metrics + manifest v2 | `kstar_adapter.py`、`run_kstar_activity.py`、`test_kstar_adapter.py` | `pytest tests/unit/analysis/test_kstar_adapter.py`：9 passed |
| R2 | **TD-02 activity 准入**：`ActivityAdmissionPolicy`（config 家族）+ `ptm_activity_admission.py` 新模块 + CLI 接线 + 删除旧入口 + config 显式登记 | `ptm_research_config.py`、`ptm_activity_admission.py`（新）、`ptm_activity.py`、`build_ptm_global_gene_scores.py`、`test_ptm_activity_admission.py`（新） | 分析层 5 个测试文件 68 passed |
| R3 | **TD-04 formal gate**：`mode` 字段 + PENDING 拒绝 + 结构化 manifest 强制 + `min_donors_per_state`；**TD-05 verifier**：50+50 非空 + 双 Network ID pin + audit 入 manifest | `ptm_research_config.py`、`kstar_adapter.py`、`kstar_resources.py`、`run_kstar_activity.py`、两个测试文件 | `test_kstar_resources.py` 6 passed；`test_kstar_adapter.py` 含新负测试全过 |
| R4 | **TD-07/TD-06 release 绑定**：config 切 combined release + manifest 路径；`verify_network_release_binding` 三向校验 + CLI 接线；builder output 段补 sha256；release manifest 登记 KSTAR archive/网络 ID | `signed_network.py`、`ptm_research_config.py`、`build_ptm_global_gene_scores.py`、`build_kinase_tf_network_release.py`、config yaml、release manifest、`test_signed_network.py` | 分析+脚本层 249 passed；真实资产 binding 实测 PASS（28,011 行 sha 一致） |
| R5 | **TD-13 mypy / TD-14 Pydantic / TD-12 marker**：builder 类型修复（4→0）；schemas 迁移 v2 API；17 项重型测试加 slow + full-test.yml 承接 | 3 个 builder、`src/api/schemas.py`、5 个 integration 测试文件、`full-test.yml` | mypy 逐文件 Success；API 测试 108 passed（v1 警告清除）；分层后 collect 17 slow / 110 fast |
| R6 | **TD-16/17 治理**：8 份文档同步（§2.3 表格）；根目录 pip 残留清理；`CURRENT_STATUS.md` + `lessons.md`（L-2026-0921-02） | `.planning/*`、`docs/guides/*`、`API_DOCUMENTATION.md`、`docs/CURRENT_STATUS.md`、`lessons.md` | 人工核对与分析报告 §5.1 漂移清单逐项对应 |
| 全量 | **最终验证**（§4 验证口径） | — | fast 2,868 passed / 275.68s；slow 17 passed；e2e 42 passed / 209.64s；real-assets 15 skipped（gate 预期）；`ruff check`+`format` 全过；`mypy src/` 0 errors；依赖文本一致性 OK |

**迭代中的一次方向修正**（记录供审计）：R4 发现 builder 生成的 manifest（`output` 段）与现存 release manifest 文件（`combined` 段）结构不同且前者无 sha256——若不处理，下次重建 release 后 binding 校验会失败。已同步修正 builder 并让校验兼容两种段名，避免"本轮修复引入下一个隐性断裂"。

---

# 4. 系统性复核：E2E 训练与推理技术要求

按任务要求，对"E2E 训练与推理的全部技术要求"逐层复核。结论先行：

> **工程层（代码 + 接口 + 合成/冒烟验证）：满足。**
> **真实资产验收层：不满足**——但缺口全部是**外部输入与正式执行**，不是代码缺陷；本轮修复已把"外部输入未到位时不得产生正式结果"从文档约定升级为机器硬门禁。

## 4.1 复核矩阵

| # | 技术要求 | 现状证据 | 判定 |
|---:|---|---|---|
| 1 | E2E CLI 与编排器可执行 | `scripts/run_davf_perturbgen_e2e.py`（候选 spec、donor split、`--run-perturbgen` 六阶段、`--assemble-statistical-evidence`、`--downstream-target-sidecar`、`--resume`/`--seeds`）；`src/integration/perturbgen/orchestrator.py` 等 20+ 模块 | ✅ 满足（接口完成） |
| 2 | 六阶段 runner 与公共 prepare/reuse | `run_perturbgen_pipeline.py` + `build_shared_prepare_plans`/`resolve_prepare_artifact_references`（AGENTS.md 2026-09-13 第四轮契约） | ✅ 满足（接口完成） |
| 3 | 训练/推理 CLI 冒烟 | `tests/e2e/`：train/predict CLI、DAVF、Mamba、pretrained、PTM site 五类 e2e | ✅ 42 passed / 209.64s |
| 4 | 单元/合同测试 | fast 口径 2,868 passed / 0 failed（本轮实测，含 KSTAR/PTM/admission 全部新测试） | ✅ |
| 5 | GPU 训练环境 | 本机 Tesla P40 可见，`torch.cuda.is_available()=True`（沙箱内探测）；torch 2.4.1+cu118 | ✅ 可用（正式长时间训练按用户要求须脱离沙箱执行） |
| 6 | 真实 AD cohort 资产 | `data/AD/standardized/`：GSE174367、GSE157827、combined 三个 H5AD + provenance JSON 均在 | ✅ 资产在位 |
| 7 | 已训模型/嵌入资产 | `outputs/models/`：best_model.pt、esm2_35m/8m、artifact_manifest.json 等在位 | ✅ 资产在位 |
| 8 | real-assets 测试 | 15 项全部 gate skip（需 `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` + 真实资产配置） | ⚠️ 按设计 gate（AGENTS.md：跳过≠验收通过） |
| 9 | **上游 PTM 链**（本轮修复对象） | 阶段 0–5 合同 + KSTAR 执行边界 + admission gate + combined release 绑定（§2） | ✅ 工程合同闭合；真实 PTM cohort 未到位 |
| 10 | 正式生物学验收 | `outputs/` 扫描 `biology_pass=true` = 0（分析报告 §8.5；本轮未新增任何正式运行） | ❌ 未满足（外部输入 + 正式执行缺口，见 4.2） |
| 11 | 环境可发布性 | `pip check` 3 组冲突、pip-audit 26 包/96 条记录（§2.4） | ❌ 未满足（TD-03，§5.1） |

## 4.2 主要缺口与根因

1. **真实 PTM cohort 缺失（根因：外部数据未到位）**。`ptm_cohort: PENDING_EXTERNAL_PTM_COHORT`。本轮修复后：formal 模式在 config 层直接拒绝 PENDING（TD-04）；KSTAR 输入 manifest 必须结构化；donor 下限可显式强制。"PTM 侧有真实输入"已不可能靠漏检进入正式链。
2. **独立 activity benchmark 缺失（根因：外部基准数据 + 预注册决策未完成）**。admission gate 已机器化（TD-02），但 benchmark evaluator 的输入（已知 kinase perturbation phosphoproteomics，方案 §5.2/§9.1）不存在；`benchmark_gate.available=false` 显式登记，admitted activity 仅 exploratory。
3. **正式六阶段/matched-null/双场景执行未发生（根因：上述 1/2 阻断上游候选 + GPU 正式 run 未排期）**。接口与统计组装代码齐备（分析报告 §4.2.4 确认），缺正式运行证据。
4. **Workflow B 生命周期未闭环（根因：无统一编排 + 真实 ≥200 条方向 benchmark 缺失）**。组件分散存在（§2.4），A-07/A-08 未关闭（分析报告附录 A）。
5. **环境依赖冲突（根因：当前环境是"能跑全部测试的实用环境"但违反自身声明约束）**。`requirements-lock.txt` 文本自洽 ≠ 运行环境满足约束；修复需隔离重建 + 全量回归，不能在当前环境上就地升级（会连锁破坏 torch/cu118/scgpt，见 §5.1）。

## 4.3 E2E 复核结论

- **"能否训练与推理"**：能——全部 CLI、编排器、训练/评估脚本在合成/冒烟数据上全链路可执行（本轮 2,927 项测试全绿，其中 e2e 42 项为真实 CLI 训练/预测冒烟）。
- **"是否满足正式验收的全部技术要求"**：否——正式验收的剩余条件是外部输入（真实 PTM、benchmark）与正式执行（GPU 六阶段/null/Gate-E/4/5），加上环境治理（TD-03）。这些都不是当前代码库能自产满足的；本轮的价值在于把等待期间的边界从"约定"变成"硬失败"。

---

# 5. 未解决问题及后续解决策略

> 时间为工程工作日估计（与分析报告 §7 口径一致），起点记为 T0（本报告完成日）。

## 5.1 TD-03：环境依赖冲突与漏洞审计（严重）

**为什么不本轮修**：当前环境虽违反自身约束但 2,927 项测试全绿、GPU 训练可用；就地升级 numpy<2 / scvi-tools<1.0 / torchaudio≥2.5 会连锁要求重装 torch（cu118 wheel 与 numpy 2.x 的组合锁定）、scgpt、lightning，极可能把可用环境变成不可用环境，且数值回归验证（checkpoint 兼容）需要冻结资产与 GPU 窗口。分析报告 §7.3 方案 A 自身要求"隔离分支及冻结资产上验证"。

**策略**（分析报告 §7.3 TD-03 方案 A，P0 第 3 位）：

| 步骤 | 内容 | 时间 |
|---:|---|---:|
| 1 | 从 `environment.yml`/`requirements-*.txt` 分 profile（core/API/analysis/KSTAR）空环境重建候选环境 | T0+1d |
| 2 | 逐 profile 升级有 fix 版本的包（`urllib3` 2.7.0、`transformers` ≥5.x、`uv` 0.11.15、`aiohttp`/`anyio`/`pillow`/`pypdf`/`lightning` 等，本轮 pip-audit 输出已给出各条 fix 版） | T0+2d |
| 3 | 无 fix 的 advisory（`accelerate`/`mamba-ssm` 部分）：按调用面评估隔离/功能关闭/时限化风险接受，登记 triage 表 | T0+2.5d |
| 4 | 冻结资产数值回归：现有 checkpoint + e2e 冒烟 + 关键单测对比 | T0+3.5d |
| 5 | 重新 lock、`pip check` + 去重 audit 进 CI gate | T0+4d |

**风险控制**：torch/cu118 升级单独评估（GPU 驱动约束），不与纯 Python 包升级混在同一批。

## 5.2 U-01/TD-02 遗留：独立 activity benchmark evaluator（严重缺口的外部依赖部分）

admission policy/selection 已机器化；剩余是 `evaluate_activity_benchmark(activity, benchmark, criteria)` evaluator + benchmark 数据本身：

| 步骤 | 内容 | 时间 |
|---:|---|---:|
| 1 | 研究负责人预注册 benchmark 来源（已知 kinase perturbation phosphoproteomics）与判据 | 外部决策 |
| 2 | 实现 benchmark schema + evaluator + CI fixture（接口已在分析报告 §3.2 定义） | T0+1d（数据到位后） |
| 3 | benchmark PASS 前传播结果固定 exploratory（本轮已由 `benchmark_gate.available=false` 机器保证） | 已生效 |
| 4 | 校准后显式修订 config `activity_admission` 阈值（研究决策，须记录理由） | 随 2 |

## 5.3 TD-08：Workflow B 统一可重放生命周期（高）

采纳分析报告 §7.1 TD-08 方案 A：

| 步骤 | 内容 | 时间 |
|---:|---|---:|
| 1 | 新增 `scripts/run_workflow_b.py`：asset verify → latent pair 构建 → donor split → train → evaluate → Gate-E 的单一编排，产出不可变 run manifest（checkpoint/指标/资产 hash/环境/seed） | T0+3d |
| 2 | 真实同坐标系 latent pairs + 冻结旧 checkpoint（外部资产）到位后首跑 | 外部依赖 |
| 3 | ≥200 条可追溯真实 PTM→gene benchmark（A-08）+ Gate-E 正式 verdict | 外部依赖 + T0+1d |

## 5.4 其余未修项（中低级）

| 项 | 现状量化（本轮实测） | 策略 | 时间 |
|---|---|---|---:|
| TD-09 复杂度 | Ruff C901/PLR：src+scripts 372 项（C901 175/分支 106/语句 78/返回 13）；热点 `build_scperturb_latent_pairs` C901=69、`train.py::main` 56 | 分析报告 §7.4 方案 B：先拆 6 个最高风险函数（外部签名不变的纯 helper + dataclass context 拆分），配合 characterization test | T0+4d |
| TD-10 broad exception | BLE001=34、S110=6、S112=2（src+scripts） | §7.4 方案 A：formal pipeline 逐路径改具体异常 + silent pass 记录原因；先从 release evidence/数据导入路径开始 | T0+3d |
| TD-11 安全规则分流 | S 规则 127 项（src+scripts）：S101 assert 57（多为断言合同）、S311 20、S603 13、S301 pickle 10、S310/S607 8/8、S324 4 | §7.4 方案 A：按调用面建 triage 表——重点：不可信 pickle load、URL 下载、subprocess 路径；测试断言/随机数与生产风险分流 | T0+3d（与 TD-10 并行） |
| TD-15 OpenAPI golden | 无 versioned schema diff | 生成 versioned `openapi.json` + CI breaking-change diff | T0+1d |
| TD-17 whitelist 改造 | 存量垃圾已清；`/*` whitelist 仍隐藏新增根文件 | 先 inventory 根目录 → 决策大资产规则 → 改目标化 ignore 或加 root guard | T0+1d |
| TD-04 深化 typed cohort manifest | mode+结构化 manifest+donor 下限已落地 | 真实 PTM cohort 到位后按其研究设计（pairing/donor 下限预注册）补 typed schema | 随外部数据 |
| TD-06 完整 resolver | archive/network ID 已登记；`setup_kstar_env.sh` 恢复链已验证 | 对象存储/内部镜像决策后实现 `resolve_signed_network_release` + clean checkout CI | 外部决策 + T0+2d |

## 5.5 外部依赖清单（代码无法自产满足）

1. 真实位点级 PTM 定量 cohort（species/contrast/donor/provenance 可追溯，方案 §10）；
2. 独立 kinase perturbation benchmark（§5.2）；
3. PhosR sensitivity 独立 R 环境输出；
4. Workflow B 的真实 latent pairs、冻结旧 checkpoint、≥200 条方向 benchmark；
5. 正式 GPU 运行窗口（六阶段 × 多 seed × matched-null 99 轮，按用户要求脱离沙箱执行）。

在 1–5 到位前，任何"PTM 导致某 cell type 表达变化"或"正式生物学 PASS"的表述继续禁止（与主方案 §10.3 结论一致）；允许的结论上限是"工程合同通过 + 方向一致性探索证据"。

---

# 6. 文档引用索引

| 引用 | 位置 |
|---|---|
| 审计报告总表（TD-01…TD-19） | [`project_analysis_20260921.md`](project_analysis_20260921.md) §6.1 |
| 严重/高/中债解决策略 | 同上 §7.1–§7.4；资源/风险矩阵 §7.5 |
| TD-01 代码证据（修复前） | 同上 §5.2.1（`run_kstar_activity.py:349-376` 旧行号） |
| admission 缺失证据 | 同上 §5.2.2；U-01 §3.1 |
| 空 verifier 测试证据 | 同上 §5.2.3 |
| 测试/静态检查基线 | 同上 §8.1–§8.2 |
| KSTAR 实体资产实测 | 同上 §8.3 |
| signed release 实测 | 同上 §8.4 |
| 正式证据扫描（biology_pass=0） | 同上 §8.5 |
| 优先级路线图 | 同上 §9.1–§9.3 |
| 执行方案准入条款 | `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §4.4/§5.2/§8（行 152/188/208-209/232） |
| KSTAR 执行合同（已同步） | `docs/guides/kstar_activity_plan.md` |
| 管线命令/数据契约（已同步） | `docs/guides/ptm_activity_pipeline.md` §1 config 示例、§4 阶段 3 |
| 项目当前状态（已同步） | `docs/CURRENT_STATUS.md` Quick Reference 2026-09-21 修复轮 |
| 本轮决策记录 | `lessons.md` L-2026-0921-02 |
| 需求追踪（已同步） | `.planning/REQUIREMENTS.md` P-04/P-05/P-06 |
| 规划状态（已同步） | `.planning/STATE.md` |
| 代码引用 | `src/analysis/{kstar_adapter,kstar_resources,ptm_activity,ptm_activity_admission,ptm_research_config,signed_network}.py`、`src/api/schemas.py`、`scripts/{run_kstar_activity,build_ptm_global_gene_scores,build_kinase_signed_edges,build_kinase_tf_network_release}.py`、`configs/research/ptm_research_config.yaml`、`data/manifests/kstar_signed_network_release_20260921.json`、`.github/workflows/full-test.yml` |

---

## 附：验证命令与完整结果（2026-09-21 本地实测）

```text
# 分层测试
pytest tests/unit tests/integration tests/test_*.py -m "not slow and not gpu"
  → 2868 passed, 7 skipped, 17 deselected in 275.68s
pytest tests/integration -m "slow"
  → 17 passed, 110 deselected in 251.77s
pytest tests/e2e -m "not slow and not gpu"
  → 42 passed in 209.64s
pytest tests/real_assets（无环境变量）
  → 15 skipped（gate 预期）

# 静态检查
ruff check src scripts tests        → All checks passed
python -m ruff format --check ...   → 536 files already formatted
python -m mypy src/                 → Success: 183 files, 0 errors
mypy（逐 builder 文件）             → Success（TD-13 的 4 错误清零）
python scripts/check_requirements_consistency.py → 274 lock pins OK

# 真实资产绑定（TD-07）
verify_network_release_binding(network, manifest, omnipath-kinase+tf-2026-09-21)
  → PASS: 28011 rows, sha256 9483f613…

# 环境健康（未修复项，如实报告）
pip check → 3 组冲突（TD-03）
pip-audit --local → 26 包 / 96 条唯一 (package, advisory) 记录
```
