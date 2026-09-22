# PTM2CellNet 代码修复报告（2026-09-22）

> **修复范围**：依据 [`project_analysis_20260922.md`](project_analysis_20260922.md) §5（任务三：技术债识别、分类与解决策略）的方案与优先级，本轮修复 **ND-01、ND-02、TD-03（本地可闭环部分）、TD-08、TD-09（六热点）、TD-10/11（分流）、TD-15**，并完成 E2E 训练与推理技术要求的系统性复核。<br>
> **基线**：`dee3d8c`（main，与 origin/main 同步）<br>
> **执行环境**：本地 `/home/scu/PTM2CellNet`（非 devspace 只读环境）；主环境 conda `SSH_unit`（Python 3.12 / torch 2.4.1+cu118 / CUDA Tesla P40 可见）<br>
> **验证口径**：`pytest -m "not slow and not gpu" --timeout=300` 全目录 **2,947 passed / 0 failed / 457.94s**（基线 2,922 + 本轮新增 25 项），ruff check / format（541 files）/ mypy（183 文件 0 错误）/ 依赖一致性（274 pins OK）/ compileall 全绿。<br>
> **结论底线**：本轮全部为工程合同与债务修复，**正式 biology PASS 仍为 0**；smoke/synthetic/bridge 证据不因本轮升级为生物学结论（AGENTS.md 2026-09-13 约束）。

---

## 目录

- [1. 问题修复汇总（按严重程度）](#1-问题修复汇总按严重程度)
- [2. 每轮迭代的执行情况与结果](#2-每轮迭代的执行情况与结果)
- [3. 系统性复核：E2E 训练与推理技术要求](#3-系统性复核对-e2e-训练与推理技术要求)
- [4. 未解决问题与后续解决策略](#4-未解决问题与后续解决策略)
- [5. 文档引用索引](#5-文档引用索引)

---

# 1. 问题修复汇总（按严重程度）

严重程度标准沿用分析报告 §2.3（严重/高/中/低四级）。本轮覆盖中、高、严重三级中**可本地代码闭环**的全部目标项（分析报告 §1.3 排名 2–5 的阻断项与 §5.2/§5.3 全部中高级未修债）。

| ID | 严重度 | 债务（分析报告出处） | 本轮处置 | 验证证据 |
|---|---|---|---|---|
| **TD-03** | 严重 | 环境依赖冲突 + advisory 面 + "冲突消失"实为元数据脱节（§1.1 第 3 条、§5.3） | **本地闭环**：①ptm2cellnet 元数据以 editable 恢复在位（`pip install -e . --no-deps` → SSH_unit），`pip check` 重获真实检测；②advisory 重扫（pip-audit，325 包）并安全升级 13 包，18 包/152 条 → **5 包/41 条**；③`requirements-lock.txt` 同步 11 项钉扎；④CI 加 pip-audit 可见性 gate；⑤澄清根因（见 §2 R2） | `pip check` 仅剩 2 组混居冲突（见 §4）；lock 一致性 274 pins OK；全量测试后无回归 |
| **TD-08** | 高 | Workflow B 无统一可重放生命周期，`run_workflow_b` 不存在（§3.1 U-02、§5.3） | **已修复**：新增 `scripts/run_workflow_b.py`（编排器）+ 23 项单元测试；真实冻结 embedding asset 的 `verify_assets` 实测 PASS（`--dry-run` 全链计划生成，退出码 0） | `tests/unit/scripts/test_run_workflow_b.py` 23 passed；真资产 probe 见 §2 R4 |
| **TD-09** | 高 | 复杂度热点 372 项，六大热点 69/56/44/41/40/40（§5.3、0921 §7.4 方案 B） | **六大热点全部拆至 C901<10**，外部签名不变、RNG 顺序保持；护栏测试逐函数全绿（§2 R5） | C901/PLR 总量 372→371（热点消除被新 helper 项部分抵消，见 §2 R5 说明） |
| **TD-10** | 中 | broad exception / silent pass：BLE001=33、S110=6、S112=2（§5.3） | **分流完成 41→27**：1 处 `except BaseException` 收窄（吞 KeyboardInterrupt 是真缺陷）；2 处静默 pass 改为记录；10 处正当降级点补 `noqa: BLE001` 显式声明 | `ruff --select BLE001,S110,S112` 复测 27；受影响模块 93 项测试 passed |
| **TD-11** | 中 | 静态安全规则 127 项未分流（§5.3） | **triage 完成**（§2 R6）：S101（测试断言）/S311（种子随机）/S301（受控 torch.load·safe_torch_load 已治理）/S603（受控 subprocess）/S324（非安全 hash）全部归类；结论为无可信输入面暴露，不引入新防御 | triage 表（§2 R6）；CI 的 ruff select 口径不变（pyproject §lint） |
| **TD-15** | 中 | 无 versioned OpenAPI golden（§5.3） | **已修复**：`tests/api/openapi_golden.json`（OpenAPI 3.1.0 快照）+ `tests/unit/test_openapi_golden.py`（path 集合 + schema 全量对比）+ `scripts/update_openapi_golden.py`（有意变更时重生成并 review） | 2 项对比测试 passed |
| **ND-01** | 中 | `scripts/` 无包身份，console_scripts 与 `ptm_smoke.py:506-509` 反向 import 在 pip 安装后即坏（§5.2、§5.2 ND-01 方案 A） | **已修复（方案 A）**：`scripts/__init__.py` 包身份；`tools/`、`experimental/` 有意留在发行版外 | SSH_unit editable 实测三件套：repo 外 `from scripts.* import main` ✓、`ptm2cellnet-train/predict --help` exit 0 ✓、`python -m scripts.train --help` exit 0 ✓ |
| **ND-02** | 中 | CHANGELOG `[Unreleased]` 漏记 0921 修复轮 11 项，属 09-14 同类修复后的回归（§4.3、§5.2） | **已修复**：0921 轮 11 项按 Added/Changed/Fixed 分节补记（标注 "backlogged 2026-09-22 — ND-02"），并新增本轮 0922 分节 | `CHANGELOG.md` [Unreleased] 两段分节齐备 |

**范围外（低级债，未动，与分析报告 §5.2 ND-03~06 一致）**：ND-03 modules.rst 过期、ND-04 iterrows、ND-06 `_edge_sign` 私有 import、TD-17 whitelist 改造、TD-18/19 第三方警告。TD-16 文档同步随本轮变更落地（CHANGELOG/guide/REQUIREMENTS/CURRENT_STATUS/lessons，见 §5）。

---

# 2. 每轮迭代的执行情况与结果

本轮按依赖顺序分 6 轮迭代（R1→R6），每轮完成即跑对应护栏测试；全量验证在 R6 后统一执行。

## R1：ND-02 CHANGELOG 补记 + ND-01 包身份（中级×2）

| 动作 | 结果 |
|---|---|
| 从归档 `archive/20260922/project_repair_report_20260921.md` §2.1–§2.3 提取 11 项准确语义，写入 `[Unreleased]`（Added 3/Changed 3/Fixed 4/治理 1），标注 backlogged | `CHANGELOG.md` 补记完成，与 0921 报告 §2 逐项对应 |
| 新建 `scripts/__init__.py`（docstring 声明包身份与 tools/experimental 发行边界） | `find_packages()` 收集 scripts |
| `pip install -e . --no-deps` 验证 | **首次安装误入 Python 3.10 用户目录**（`pip` 解析到 `~/.local/bin/pip`），撤销后以 SSH_unit 的 `python -m pip` 重装——该失误本身成为 TD-03 根因的直接证据（§2 R2） |
| 验证三件套 | repo 外反向 import、两个 console_scripts、`python -m scripts.train --help` 全部 exit 0 |

## R2：TD-03 依赖治理（严重）

| 动作 | 结果 |
|---|---|
| editable 恢复 ptm2cellnet 元数据 | `pip check` 从"仅 PyNaCl"（0922 分析口径）变为**实报 2 组冲突**：scgpt↔scvi-tools 1.4.3、ssh-unit↔torchaudio 2.4.1 |
| 根因澄清 | **分析报告 §1.1 第 3 条与 §5.3 TD-03 的"冲突消失"结论需要修正**：`pip show` 无输出的原因是审计用了 `~/.local` 的 3.10 pip，而 scgpt/ssh-unit 元数据一直在 SSH_unit。冲突方从未消失，是审计看错了环境 |
| 混居定性 | ssh-unit（用户的另一项目，editable 安装，依赖 scgpt）与 PTM2CellNet 共用 SSH_unit；`requirements-analysis.txt:24-29`（TD-M05）明确禁止 scgpt 与 scvi-tools≥1.2 混装——冲突主体是 ssh-unit 生态，**不动用户另一项目**，登记隔离建议（§4） |
| pip-audit 扫描（升级前） | 325 包 / 18 包 / 152 条 advisory |
| dry-run 升级集 → 执行升级 | 13 包升级成功（aiohttp 3.14.3、anyio 4.15.1、idna 3.20、msgpack 1.2.2、pillow 12.3.0、pypdf 6.19.0、sqlparse 0.6.0、urllib3 2.8.0、lightning 2.6.6、pytorch-lightning 1.9.5→2.6.6、hydra-core 1.3.7、pip 26.2.1、uv 0.12.17） |
| transformers 升级回退 | 4.48.1→4.53.0 触发新冲突：**esm 3.2.3 钉 `transformers<4.48.2`**；esm 3.4.1 的替代路径会拖入 torch 2.11 + CUDA13 全栈（dry-run 证实）。回退至 4.48.1，22 条 transformers advisory 登记为**上游阻塞**（lessons L-2026-0922-02 第 2 条） |
| 升级后复扫 | **5 包 / 41 条**（accelerate/datasets/mamba-ssm/setuptools/transformers，各剩理由见 §4） |
| lock 同步 + CI gate | `requirements-lock.txt` 更新 11 项钉扎（transformers 由漂移值 4.57.6 对齐环境实际 4.48.1）；`check_requirements_consistency.py` 274 pins OK；`ci.yml` dependencies job 新增 pip-audit 步骤（`continue-on-error` 至无 fix 项清零，之后转硬门禁——非静默吞掉，失败在 CI 可见） |

## R3：TD-08 Workflow B 统一编排器（高）

- 新增 `scripts/run_workflow_b.py`：固定阶段序 `verify_assets → build_pairs → train → evaluate → gate_e`；每阶段调既有组件 CLI 的 `main()`（`build_davf_latent_pairs`/`train_latent_davf`/`evaluate_latent_davf`/`evaluate_gate_e`），捕获 stdout JSON、记录 argv/退出码/耗时/输出 sha256；
- **不可变 run manifest**：`ptm2cellnet.workflow_b_run/v1`，run 目录非空即硬失败；写毕以 `run_manifest.sha256` 封存；
- **Gate-E 语义**：显式 opt-in（`--stages` 含 gate_e 或 `all`，必须给 `--benchmark`）；非 PASS verdict = 退出码 **2** + `run_status=gate_failed`（不伪装成执行错误，不隐藏）；
- 组件异常：记录进 manifest（status=failed + error）后 re-raise（let it crash，但留痕）；
- 测试 23 项（argv 构造、verify 六分支、不可变目录、dry-run、crash 记录、gate_failed 全链、seal、CLI 退出码）；
- **真实资产实证**：以 `checkpoints/scvi/ibd_norman_model` + 冻结 `outputs/perturbgen/embedding_asset_20260822` 执行 `--dry-run`，`verify_assets`（含 asset manifest sha256 校验）**PASS**，计划 manifest 正确生成、退出码 0。
- 文档：`docs/guides/perturbgen_bridge.md` §5 新增"统一生命周期入口"段（含示例命令与边界声明）；`.planning/REQUIREMENTS.md` A-07 → **Partially landed**。

## R4：TD-09 六大热点拆分（高）

按 0921 §7.4 方案 B（外部签名不变的纯 helper + dataclass context + characterization test）执行：

| 函数（原复杂度） | 文件 | 拆分方式 | 护栏测试 |
|---|---|---|---|
| `build_scperturb_latent_pairs`（69） | `src/data/davf_scperturb.py` | 12 个私有 helper（校验/加载/donor 标签/scVI 编码/四分支 split 构造/control 分配/逐 split 导出）+ `_SplitExportContext` dataclass；**RNG 构造与调用顺序逐处保持**（确定性输出不变） | `test_davf_scperturb.py` 17 passed |
| `train.py::main`（56） | `scripts/train.py` | 11 个 helper（seeds/数据加载/demo 判定/质量门禁/balanced loader/精确 resume（内部再拆 `_safe_restore_state`/`_restore_callback_states`/`_log_state_dict_diff`）/推理导出/`_sanitize_nans`/manifest 门禁/CLI 覆盖/通路分析）+ `from __future__ import annotations` + TYPE_CHECKING | `test_train_resume_cli.py` + `test_artifacts.py` 28 passed |
| `check_release_evidence`（44） | `scripts/check_perturbgen_release_evidence.py` | `_check_evidence_identity` / `_resolve_evidence_paths` / `_check_benchmark_header` / `_audit_benchmark_samples`（内拆 formal per-sample 与 h5ad 输出检查） | 6 passed |
| `_assemble_downstream_target_evaluation`（41） | `scripts/run_davf_perturbgen_e2e.py` | `_validate_candidate_sidecar_binding` / `_collect_gated_candidates` / `_collect_result_runs` / `_load_candidate_delta_frames` | `test_run_davf_perturbgen_e2e.py` 20 passed |
| `_validate_frozen_run`（40） | `src/integration/perturbgen/frozen_cohort.py` | `_validate_stage_manifest_binding` / `_validate_output_binding` / `_validate_run_cohort_reference` / `_validate_bound_null_distribution` | `test_frozen_cohort.py` + `test_formal_verification.py` 32 passed |
| `build_eval_input_payload`（40） | `src/integration/perturbgen/eval_assembly.py` | `_EvalModeConfig`(NamedTuple) + `_validate_eval_mode_inputs` / `_resolve_eval_paths` / `_build_candidate_run_records` / `_assemble_eval_candidate` | perturbgen 单测目录 294 passed |

**过程事故与处置**（lessons L-2026-0922-02 第 5 条）：拆分 train.py 时一次基于子串锚点的脚本替换命中了 helper 定义体内的相同文本，把 `_export_inference_artifact` 变成自递归——由护栏测试前的一次 ruff/目视检查发现，随即以 `git checkout` 恢复该文件并改用带唯一性 assert 的精确替换。最终所有文件 ruff/format/mypy 全绿。

## R5：TD-10/11 异常与安全规则分流（中）

- BLE001/S110/S112 复测 41 项（与分析报告 §5.3 口径一致）；
- **真缺陷修复 3 处**：`src/utils/dependency_check.py:127` `except BaseException` → `Exception`（原写法会吞 KeyboardInterrupt/SystemExit）；`scripts/finetune_davf.py` AUC 静默 pass → 打印原因；`scripts/integrate_data_v2.py` 单条 fetch 静默 pass → 计数 + 打印（≤5 条）；
- **正当降级显式声明 10 处**（`noqa: BLE001` + 理由）：API 无模型启动（autoinit）、gene_mapper 线程转发、ibd_qc Scrublet per-sample、genki batch 逐请求×3 / 回调不阻断×3、geneformer 多候选路径、import_scperturb best-effort close（原有）；
- 复测 41→**27**（剩余为 scripts/ 辅助工具的可选功能降级，triage 为"有意降级、无正式路径影响"，见 §4）；
- 受影响模块回归：93 项测试 passed。

## R6：TD-15 OpenAPI golden（中）+ 全量验证

- golden 三件套（快照/对比测试/重生成脚本）落地，首轮生成 OpenAPI 3.1.0 文档快照；
- 全量验证（见报告头部验证口径）：**2,947 passed / 0 failed**（较 0922 基线 +25 = run_workflow_b 23 + openapi golden 2，数目精确对账）、ruff/format/mypy/lock/compileall 全绿。

---

# 3. 系统性复核：对 E2E 训练与推理技术要求

按用户要求，对"当前代码是否满足 E2E 训练和推理的全部技术要求"逐层复核。判定依据：本轮实测 + 分析报告 §3/§4 + AGENTS.md 主线约束。

## 3.1 分层判定总表

| 层 | 技术要求 | 判定 | 证据 |
|---|---|---|---|
| L1 基础训练/推理 CLI | `scripts/train.py`/`predict.py` 可跑、断点续训、产物分离、demo 标记 | ✅ **满足** | 28 项 resume/artifacts 测试 + console_scripts 实测（ND-01 修复后 pip 安装亦可用） |
| L2 API 服务 | schema 稳定、有契约防回归 | ✅ **满足**（本轮补齐 TD-15 后） | OpenAPI golden 2 项测试；API 测试 108 passed（0921 口径维持） |
| L3 DAVF/LatentDAVF 训练链 | pair 构建/训练/评估合同 + donor split + 资产绑定 | ✅ **工程合同满足**；正式重训待外部资产 | davf_scperturb 17 / perturbgen 294 项测试；真实冻结 asset verify PASS（本轮 run_workflow_b probe） |
| L4 Workflow B 生命周期 | 统一可重放编排 + 不可变 manifest | ✅ **工程侧满足**（本轮 TD-08 落地）；真实 latent pairs/旧 checkpoint/≥200 benchmark 为外部输入 | run_workflow_b 23 项测试 + 真资产 dry-run exit 0 |
| L5 PerturbGen 六阶段 | tokenise→train_mask→train_decoder→perturb→export→report 可执行 + 共享 prepare + invocation gate | ✅ **接口与 frozen plan 满足**（分析报告 §3.1 U-06 与 0921 §4.2.4 口径维持）；正式 GPU 执行未发生 | e2e 42 项（含本轮全量）；真实执行产物 PASS 标志 = 0 |
| L6 统计验收链 | matched-null/质量/候选 p-q/双场景 AND | ✅ **接口齐备**（`--assemble-statistical-evidence` 等）；真实 GPU null 未执行 | 分析报告 §3.1 U-06；outputs/ 无 PASS 标志 |
| L7 环境 | 可重复、advisory 受控、无隐藏冲突 | ✅ **本地闭环**（本轮 TD-03）；发布级清洁仍差 2 组混居冲突（§4） | pip check / pip-audit 5 包 41 条 / lock 274 pins |
| L8 正式生物学证据 | 真实 cohort、独立 benchmark、正式 verdict | ❌ **不满足（外部输入）** | 分析报告 §1.1 第 2 条维持：biology PASS = 0 |

## 3.2 结论

**"是否满足 E2E 训练和推理的全部技术要求"——工程侧满足、科学验收侧不满足，且两者边界在本轮后已完全机器化。**

1. **训练与推理的工程链路（L1–L6）无已知确定性缺陷**：分析报告 §4.2.2 在 0921 修复轮后已判定"当前生产路径上无已知确定性缺陷"，本轮拆分 6 个巨型函数（含 E2E 编排、冻结验证、统计装配核心路径）在 414 项护栏测试全绿下完成，未引入回归；新增 Workflow B 编排器补齐了 A-07 的工程缺口。
2. **不满足项的根因全部是外部输入与执行窗口，不是代码缺口**（与 0921 修复报告 §3 结论一致并延续）：
   - 真实 PTM cohort 缺失 → formal 模式已被 config 层硬拒绝（分析报告 §3.1 U-04），等待姿态正确；
   - 独立 activity benchmark（evaluator 实现仅剩 1 天工作量，输入数据外部）→ `benchmark_gate.available=false` 显式登记，不伪造 PASS；
   - Workflow B 真实资产（同坐标系 latent pairs、冻结旧 checkpoint、≥200 方向 benchmark）→ 编排器已就绪，资产到位即可首跑；
   - 正式六阶段/matched-null/双场景 → 接口齐备，待上述输入 + 脱离沙箱的 GPU 窗口。
3. **环境侧残留一处真实缺口**：SSH_unit 混居用户另一项目 ssh-unit（连带 scgpt），产生 2 组 `pip check` 冲突。这不阻断本项目任何测试/CI（CI 在干净环境从 lock 装配），但"发布级环境清洁"未达。

---

# 4. 未解决问题与后续解决策略

| # | 问题 | 严重度 | 根因 | 解决策略与步骤 | 建议时间节点 |
|---|---|---|---|---|---|
| 1 | 真实 PTM cohort 缺失（U-06/主线起点） | 严重（外部） | 外部数据未到位 | 到位后：按其研究设计补 typed cohort schema（pairing/donor 下限预注册，分析报告 §3.1 U-04 剩余）→ formal config 解锁 → 真实 KSTAR full analysis | 外部数据到位后 +1 天 |
| 2 | 独立 activity benchmark + evaluator（U-01 剩余） | 高（外部+1 天代码） | 基准数据外部；evaluator 未实现 | 数据到位后实现 `evaluate_activity_benchmark`（消费已知 kinase perturbation phosphoproteomics，方案 §5.2）+ config 阈值校准 | 数据到位后 +1 天 |
| 3 | Workflow B 真实资产 + Gate-E 正式执行（U-02/U-03 后半） | 高（外部资产） | 真实同坐标系 latent pairs、冻结旧 checkpoint、≥200 方向 benchmark 均外部 | 资产到位 → `run_workflow_b --stages all --benchmark …` 首跑 → Gate-E verdict → Gate-4/5 | 资产到位后 +2 天（编排已就绪） |
| 4 | ssh-unit/scgpt 与 PTM2CellNet 混居 SSH_unit（2 组 pip check 冲突） | 中（用户决策） | 用户另一项目 editable 安装于主环境 | 建议为 PTM2CellNet 建独立 conda env（从 `requirements-lock.txt` 装配，五步计划的 profile 隔离形态），或迁移 ssh-unit 至独立 env；**不擅动用户另一项目** | 用户决策后半天（环境重建 + 数值回归抽查） |
| 5 | 剩余 advisory 5 包/41 条 | 低–中 | transformers 受 esm 3.2.3 pin（22 条，上游）；mamba-ssm/accelerate 无 fix（2 条）；setuptools 83 需放宽项目 `<81` pin（1 条，仅影响 sdist 构建）；datasets 属 ssh-unit 链（1 条，fix 在 5.0.1 大版本跳） | 跟踪上游：esm 放宽 pin 或项目决策升 torch/CUDA 后一次性升级；无 fix 项维持登记；CI pip-audit 步骤保持可见性，无 fix 项清零后转硬门禁 | 随上游/随 torch 升级窗口 |
| 6 | 正式六阶段/matched-null/双场景/Gate-4/5 执行 | 高（排期） | 依赖 #1–#3 + 脱离沙箱的 GPU 窗口 | 输入齐备后按 `docs/guides/perturbgen_bridge.md` §5 执行链运行；runbook 已齐 | #1–#3 后 +3–5 天 GPU 窗口 |
| 7 | 复杂度总量 371 项（六大热点已清，新榜首 `validate_manifest` 38） | 中（持续） | 历史积累 | 下一批 6 函数（`validate_manifest` 38、`cross_scale_dataset._validate` 38、`build_ad_deg_tables` 35、`run_davf_perturbgen_e2e._run` 35 等）按本轮同方法拆分 | 下一轮 +4 天 |
| 8 | BLE001 剩余 27 处（scripts 辅助工具） | 低 | 有意的可选功能降级 | 随触随改（改动到的脚本顺手收窄 + 补声明），不专项批量改 | 持续 |
| 9 | 低级债 ND-03/04/06、TD-17 whitelist、TD-18/19 | 低 | 局部卫生 | ND-03 重跑 `docs/generate_api_docs.sh`；ND-04 逐处向量化；ND-06 提公共 `edge_sign`；按分析报告 §5.2 估时随手清 | 空闲窗口合计 2 天 |

**推进顺序建议**（延续分析报告 §7.3）：#4 用户决策（半天）→ #7 下一批拆分（可与等待期并行）→ #1/#2/#3 外部输入到位即插即用 → #6 GPU 窗口集中执行。在 #1–#3/#6 完成前，任何"PTM 导致某 cell type 表达变化"或"正式生物学 PASS"的表述继续禁止（分析报告 §7.3 收口约束维持）。

---

# 5. 文档引用索引

| 引用 | 出处 |
|---|---|
| 修复依据与方案 | `project_analysis_20260922.md` §5.2（ND-01/02 方案表）、§5.3（TD-03/08/09/10/11/15 现状与策略）、§1.3（阻断项排名）、§7.3（推进顺序） |
| 0921 修复轮语义（CHANGELOG 补记源） | `archive/20260922/project_repair_report_20260921.md` §2.1–§2.3、§5.3（TD-08 方案 A）、§5.4（TD-09/10/11/15 策略行） |
| Workflow B 需求与组件 | `.planning/REQUIREMENTS.md` A-07/A-08；`docs/guides/perturbgen_bridge.md` §5（本轮新增"统一生命周期入口"段） |
| 依赖契约 | `requirements-analysis.txt:24-29`（TD-M05 scgpt 隔离决策）；`requirements-lock.txt:231-234`（ssh-unit 移除记录）；`setup.py:139-140`（console_scripts）；`.github/workflows/ci.yml` dependencies job（本轮新增 pip-audit 步骤） |
| 本轮决策沉淀 | `lessons.md` L-2026-0922-02（多解释器错位/transformers 上限/ND-01 方案/manifest 契约/拆分纪律/TD-10 分流结论） |
| 状态同步 | `docs/CURRENT_STATUS.md`（Quick Reference 新增 0922 修复轮条目）；`CHANGELOG.md` [Unreleased]（0921 backlogged + 0922 本轮两段） |
| 验证口径 | `AGENTS.md` 验证节（`pytest -m "not slow and not gpu" --timeout=300` 等五命令）；本轮全部实测输出见 §2 各轮 |

---

*报告生成：ZCode 主会话（子代理系统本环境不可用，见分析报告 §8；全部调查、修复、验证由主会话完成）。*
