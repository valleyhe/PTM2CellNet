# PTM2CellNet 综合分析报告（2026-09-22 环境隔离轮）

> **任务来源**：依据 [`archive/20260922/project_repair_report_20260922.md`](archive/20260922/project_repair_report_20260922.md) §4（未解决问题与后续解决策略）的用户指令复执行：主攻"为 PTM2CellNet 单独建立 conda env"（§4-4）、U-01/02/03（§4-1/2/3）与严重/高/中级技术债（§4-7 复杂度次批、§4-5 advisory）。<br>
> **基线**：`dee3d8c`（main）+ 本轮工作区改动<br>
> **执行环境**：本地 `/home/scu/PTM2CellNet`；**conda env `ptm2cellnet`**（Python 3.12 / torch 2.4.1+cu118 / Tesla P40）。开发期 SSH_unit 于本日 11:38 前后被用户删除（磁盘 90% 占用下的清理，与本轮目标方向一致），`ptm2cellnet` 自此为唯一主环境。<br>
> **验证口径**（全部在 `ptm2cellnet` 实测）：全量 fast **2,968 passed / 0 failed / 27 skipped / 440.91s**；ruff check / format（543 files）全绿；mypy（184 文件）**0 错误**；`pip check` **No broken requirements found**；依赖一致性 274 pins OK；compileall 通过；GPU 侧 conv1d fwd/bwd、causal-conv1d/mamba-ssm 自编译 kernel、scvi-tools/lightning 导入与 forward、训练确定性抽查全部实测通过。<br>
> **结论底线**：本轮为环境隔离与工程合同收尾，**正式 biology PASS 仍为 0**（修复报告 §3.2 结论维持；AGENTS.md 2026-09-13 约束）。

---

## 目录

- [1. 问题修复汇总（按严重程度）](#1-问题修复汇总按严重程度)
- [2. 每轮迭代的执行情况与结果](#2-每轮迭代的执行情况与结果)
- [3. 系统性复核：E2E 训练与推理技术要求](#3-系统性复核对-e2e-训练与推理技术要求)
- [4. 未解决问题与后续解决策略](#4-未解决问题与后续解决策略)
- [5. 文档引用索引](#5-文档引用索引)

---

# 1. 问题修复汇总（按严重程度）

| ID | 严重度 | 任务（修复报告 §4 出处） | 本轮处置 | 验证证据 |
|---|---|---|---|---|
| **TD-03 终局** | 严重 | ssh-unit/scgpt 混居 SSH_unit，2 组 pip check 冲突（§4-4） | **已解决**：独立 conda env `ptm2cellnet` 建成（Python 3.12，`requirements-lock.txt` 全量装配）；用户随后删除 SSH_unit 及 5 个闲置环境，混居主体不复存在 | `pip check` 零冲突；全量测试 2,968 passed；一致性 274 pins OK（§2 R1） |
| **U-01（代码侧）** | 高 | `evaluate_activity_benchmark` 不存在（§4-2；0921 分析 §3.2 接口表） | **已实现**：新模块 `src/analysis/ptm_activity_benchmark.py` + config 预注册 `activity_benchmark` 段 + admission gate 集成 + CLI `--activity-benchmark` opt-in（§2 R2） | 28 项单测 + 4 项 CLI 集成全绿；guide §4 已同步 |
| **TD-09 次批** | 中 | 复杂度新榜首六函数 38/38/35/35/33/32（§4-7） | **六函数全部拆至 C901<10**（外部签名不变、RNG/调用顺序保持） | 护栏测试 1,033 项全绿；C901 总量 178→172（§2 R3） |
| **TD-03 附带发现一** | 高（新） | lock 两处 fresh-resolver 不可自洽：anyio 4.15.1 要求 typing_extensions≥4.16.0 而 lock 钉 4.15.0；transformers 4.48.1 要求 tokenizers<0.22 而 lock 钉 0.22.2 | **已修复**：lock 两行更新（typing_extensions 4.16.0、tokenizers 0.21.1） | 新环境 fresh 装配成功；一致性 274 pins OK（§2 R4） |
| **TD-03 附带发现二** | 高（新） | nvidia cu11/cu12 cudnn 包共享 `nvidia/cudnn/lib` 目录文件级互相覆盖；cu12 后装时 torch 在 sm_61 上 conv1d 无 engine（首跑 9 failed + 12 errors） | **已修复**：装配收尾显式重装 `nvidia-cudnn-cu11==9.1.0.70`；runbook 写入安装指南 §2a 与 lessons L-2026-0922-04 | 修复后 2,968 passed / 0 failed；`cudnn.version()==90100` + conv fwd/bwd 实测（§2 R5） |
| **TD-M08（存量）** | 中 | SSH_unit 增量缓存掩盖的 9 个 mypy 存量错误（candidate_spec/pmads_ridge/external_tools） | **已清零**：int(None) 收窄、Returning Any 加 cast、requests 重导入 ignore | 新环境干净跑 mypy 184 文件 0 错误（§2 R5） |
| **§4-5 setuptools 项** | 低–中 | CVE-2025-47273 需放宽 `<81` pin（§4-5 第 5 项） | **决策：维持 pin**——lightning_utilities/pandas/scipy 运行时 import `pkg_resources`（81 起移除），放宽即破坏训练栈；advisory 暴露面（`package_index` 远程获取）本项目不可达 | 环境内实测 grep 依赖面（§2 R3 判定）；lessons L-2026-0922-03 第 4 条 |

**范围外维持**（修复报告 §4-8/9 的低级债，未动）：BLE001 剩余 27 处、ND-03/04/06、TD-17/18/19。

---

# 2. 每轮迭代的执行情况与结果

本轮按依赖顺序分 6 轮（R1 环境装配贯穿全程，与 R2/R3 并行）。

## R1：独立 conda env `ptm2cellnet` 装配（§4-4 主任务）

| 动作 | 结果 |
|---|---|
| `conda create -n ptm2cellnet python=3.12` + lock 全量装配（三源：PyPI + `download.pytorch.org/whl/cu118` + `data.pyg.org/whl/torch-2.4.1+cu118.html`） | 本机 3Mbps 带宽（用户确认）下 pip 对 >500MB 文件无提示 stall；改 curl `-C -` 断点续传 + `--speed-limit` 自动重试，串行拉取 5.7GB wheel（torch cu118 857MB、cu11 全家 ~1.8GB、cu12 链 ~2.6GB、triton 209MB）后本地 `--find-links` 安装 |
| mamba-ssm 2.2.2 / causal-conv1d 1.4.0 | PyPI sdist **不含 csrc**（纯 Python 壳）；从 GitHub tag 源码 + setup.py 追加 `arch=compute_61,code=sm_61` 编译（CUDA 11.8 nvcc / `TORCH_CUDA_ARCH_LIST=6.1`）；打过补丁的源码副本留 `~/.cache/mamba-build/` |
| editable 安装 + `pip check` | `ptm2cellnet-1.0.0` editable 就位；**pip check 零冲突**（SSH_unit 的 scgpt↔scvi-tools、ssh-unit↔torchaudio 两组混居冲突随 SSH_unit 删除而消失） |
| GPU 实测 | `torch.cuda.is_available()=True`（Tesla P40, sm_61）；conv1d fwd+bwd（cudnn 90100）；causal-conv1d kernel 前向实测；mamba-ssm `Mamba` block 完整 forward；scvi-tools 1.4.3 / lightning 2.6.6 / anndata 0.11.4 导入 |
| **事件**：SSH_unit 连同 5 个闲置 env 于 11:38 前后被用户删除（`envs/` 目录 mtime 证据；磁盘 90% 占用） | 本轮目标环境成为唯一主环境；原修复报告 §4-4"迁移 ssh-unit 至独立 env"的另一半方案随环境删除直接消解 |

## R2：U-01 activity benchmark evaluator（高）

- `src/analysis/ptm_activity_benchmark.py`：`load_activity_benchmark_table`（两列契约 `regulator_id` + signed `perturbation_effect`，regulator 唯一、有限值硬失败）+ `evaluate_activity_benchmark`（method/contrast 过滤后的全量 activity 与 benchmark 的 paired 集：方向 concordance、Spearman 秩相关、固定 seed bootstrap CI、PASS/FAIL、输入内容 sha256；与 admission policy 正交——校准的是方法而非 admitted 子集）；
- `ActivityBenchmarkCriteria` 冻结于 `ptm_research_config.yaml` 可选 `activity_benchmark` 段：`min_paired_regulators`/`min_direction_concordance`/`min_abs_spearman` 三判据**必填、无宽松默认**（方案 §8.7 预注册纪律）；非法值与未知键 fail-fast；
- admission gate 集成：`select_activities_for_propagation(..., benchmark_report=...)`，manifest `benchmark_gate` 由恒 `available=false` 升级为真实 verdict（PASS→可进 formal lineage；FAIL→显式 exploratory，不删行、不伪造）；
- CLI：`build_ptm_global_gene_scores.py --activity-benchmark` opt-in——无预注册 criteria 硬失败（不替用户挑阈值）、exploratory FAIL 保留输出、**formal mode + FAIL 硬失败**（方案 §5.2 第 5 条的可执行化）；manifest 登记 benchmark 文件 sha256；
- 指南同步：`docs/guides/ptm_activity_pipeline.md` §4 新增"独立 benchmark 评估"段；
- 测试：`tests/unit/analysis/test_ptm_activity_benchmark.py` 28 项（契约/PASS/FAIL/CI/确定性/hash/gate 反映/config 解析）+ CLI 集成 4 项（无预注册拒绝/PASS 登记/FAIL exploratory/formal 拒绝 FAIL）。

**U-01 剩余**：benchmark 数据（已知 kinase perturbation phosphoproteomics）仍外部；阈值预注册为研究决策（config 注释段已给出模板）。

## R3：TD-09 次批六热点拆分（中）

| 函数（原复杂度） | 文件 | 拆分方式 | 护栏测试 |
|---|---|---|---|
| `validate_manifest`（38） | `src/data/data_manifest.py` | `_activate_profile`/`_validate_dataset_fields`/`_validate_dataset_metadata`/`_check_dataset_files`/`_check_single_file_entry`/`_check_file_hash` + `_FileCheckContext` | `test_data_manifest.py` 24 passed |
| `_validate`（38） | `src/data/cross_scale_dataset.py` | schema/required/embeddings/graph/optional/targets 六段模块级校验 + `_validate_cell_edge_weight` | `tests/unit/data/` 109 passed |
| `build_ad_deg_tables`（35） | `src/analysis/ad_deg_table.py` | `_validate_deg_request`/`_load_deg_matrices`/`_cell_type_donors`/`_compute_donor_means`/`_center_donor_means_by_cohort`/`_append_cell_type_rows`/`_deg_normal_reference` | `test_ad_deg_table.py` 19 passed |
| `_run`（35） | `scripts/run_davf_perturbgen_e2e.py` | `_validate_candidate_direction_fields`/`_load_context_and_prepare`/`_build_initial_payload`/`_resolve_and_bind_donor_split`/`_execute_perturbgen_stages`/`_bind_downstream_target_evaluation`/`_bind_statistical_evidence` | `test_run_davf_perturbgen_e2e.py` 20 passed |
| `evaluate`（33） | `src/evaluation/evaluators.py` | `_map_ptm_type_ids`/`_resolve_model_outputs`/`_run_eval_batches`/`_enrich_classification_metrics` | `tests/unit/evaluation/` 92 passed |
| `_validate_report_against_eval_input`（32） | `src/integration/perturbgen/frozen_cohort.py` | `_index_eval_candidates`/`_index_report_candidates`/`_compare_candidate_replay`/`_compare_replay_runs` | frozen_cohort + formal_verification 32 passed |

拆分纪律沿用 lessons L-2026-0922-02 第 5 条（外部签名不变、RNG/调用顺序逐处保持、阶段化私有 helper 不引入抽象层）。**C901 总量 178→172，榜首 38→32（`latent_davf_dataset._load_and_validate_metadata`）**；受影响五目录回归 1,033 项全绿。同轮判定：setuptools `<81` pin 维持（见 §1 表）。

## R4：全 lock 装配与两处 lock 自洽性修复（高）

- 全 lock 装配在 fresh resolver 下两次 `ResolutionImpossible`，暴露 0922 lock 的隐藏矛盾（增量安装历史掩盖）：`typing_extensions` 4.15.0→**4.16.0**（anyio 4.15.1 要求）、`tokenizers` 0.22.2→**0.21.1**（transformers 4.48.1 要求）；两行修正后装配通过，`check_requirements_consistency.py` 274 pins OK；
- 缺口分析脚本核对：lock 279 项全部落地（277 直接安装 + mamba/causal 编译安装；2 个"缺口"为脚本伪迹——lock 注释行与过期临时清单）。

## R5：首跑回归 → cuDNN 覆盖缺陷修复 → 全绿（高）

- 新环境全量首跑：2,947 passed + **9 failed + 12 errors**，全部指向 `F.conv1d` 的 `FIND was unable to find an engine`；
- 最小复现定位：`torch.backends.cudnn.version()==91900`——cu12 的 cudnn **9.19.0.56** 覆盖了 cu11 9.1.0.70 的文件（两 pip 包共享 `nvidia/cudnn/lib`，torch 2.4.1+cu118 RPATH 硬链 `libcudnn.so.9`）；9.19（CUDA 12.8 线）在 sm_61 无 conv engine；
- 修复：显式重装 `nvidia-cudnn-cu11==9.1.0.70`（90100），conv1d fwd/bwd 实测通过；**重跑全量 2,968 passed / 0 failed**；
- 同轮清掉 mypy 存量 9 错（见 §1 表 TD-M08 行），mypy 184 文件 0 错误；
- runbook 沉淀：安装指南新增 §2a（lock 重建权威口径，含 cudnn 终态步骤与 mamba 源码编译）、lessons L-2026-0922-04。

## R6：数值回归抽查与文档同步（高）

- **训练确定性**：`scripts/train.py` 同 seed=42 在 GPU 上两次训练（cnn encoder, 3 epochs），`best_model.pt` 逐字节一致（sha16 `81ec3598ca34f063`）、state_dict 张量 bitwise 一致；
- **真实资产探针**：`run_workflow_b.py --stages verify_assets --dry-run`（`checkpoints/scvi/ibd_norman_model` + 冻结 `outputs/perturbgen/embedding_asset_20260822`）在新环境 **PASS**（含 asset manifest sha256 校验），`run_status=completed`——0922 修复报告 §2 R4 的探针在新环境等价复现；
- 文档：CHANGELOG 新增 environment-isolation round 分节（Added/Changed/Fixed）；lessons 追加 L-2026-0922-03/04；CURRENT_STATUS 头部与 Quick Reference 更新；报告链归档（旧 analysis + repair report → `archive/20260922/`，MANIFEST 补记）；`ptm_activity_pipeline.md` §4 与 `installation.md` §2a 同步。

---

# 3. 系统性复核：对 E2E 训练和推理的全部技术要求

沿用修复报告 §3.1 的分层判定并按本轮事实更新（判定依据：本轮全部实测 + 代码事实）。

## 3.1 分层判定总表

| 层 | 技术要求 | 判定 | 本轮证据（与 0922 修复报告 §3.1 的差异） |
|---|---|---|---|
| L1 基础训练/推理 CLI | train/predict 可跑、断点续训、产物分离、demo 标记 | ✅ 满足 | e2e 42 项 + resume/artifacts 测试在新环境全绿；**新增 GPU 确定性实证**（同 seed 两次训练逐字节一致） |
| L2 API 服务 | schema 稳定、契约防回归 | ✅ 满足 | OpenAPI golden 对比测试全绿；API 相关测试（含 auto-init GPU 推理路径）在 cudnn 修复后全绿 |
| L3 DAVF/LatentDAVF 训练链 | pair 构建/训练/评估合同 + donor split + 资产绑定 | ✅ 工程合同满足 | perturbgen 294 项全绿；**U-01 evaluator 落地后 admission→benchmark→propagation 合同链完整**（benchmark 数据仍外部） |
| L4 Workflow B 生命周期 | 统一可重放编排 + 不可变 manifest | ✅ 工程侧满足 | run_workflow_b 23 项测试全绿；**真实冻结 asset verify_assets 在新环境 PASS 复现** |
| L5 PerturbGen 六阶段 | tokenise→…→report + 共享 prepare + invocation gate | ✅ 接口与 frozen plan 满足 | e2e 全量通过；正式 GPU 执行未发生（外部输入） |
| L6 统计验收链 | matched-null/质量/候选 p-q/双场景 AND | ✅ 接口齐备 | 接口层测试全绿；真实 GPU null 未执行 |
| L7 环境 | 可重复、advisory 受控、无隐藏冲突 | ✅ **由"本地闭环"升级为"满足"** | 独立 env `pip check` 零冲突；lock 自洽（2 处矛盾修复后 fresh 装配成功）；GPU 栈完整实测（cudnn conv/mamba kernel/scvi-tools）；SSH_unit 混居主体已删除 |
| L8 正式生物学证据 | 真实 cohort、独立 benchmark、正式 verdict | ❌ 不满足（外部输入） | biology PASS = 0 维持（修复报告 §3.1 L8 口径） |

## 3.2 结论

**"是否满足 E2E 训练和推理的全部技术要求"——工程侧满足且环境侧本轮闭环，科学验收侧仍不满足（外部输入）。**

1. **训练与推理工程链路（L1–L6）在新环境完整复现**：2,968 项测试 + GPU 确定性抽查 + 真实资产探针三重证据；此前 SSH_unit 与 lock 的四处隐性脱节（pip 混居 2 组、typing_extensions/tokenizers 矛盾、mypy 缓存掩盖、cudnn 文件覆盖序）全部暴露并修复——**lock 现在是 fresh-resolver 可自洽的**，这是"环境可重复"从声明变为实测性质的变化。
2. **U-01/02/03 本轮收口情况**：U-01 代码侧全部落地（evaluator + criteria 预注册 schema + gate 集成 + CLI），剩余为外部 benchmark 数据与阈值研究决策；U-02 编排器在位且新环境真资产 verify 复现，剩余为真实同坐标系 latent pairs/冻结旧 checkpoint；U-03（≥200 方向 benchmark 正式执行）完全依赖 U-01 数据与 U-02 资产。三者当前无代码缺口。
3. **不满足项根因全部为外部输入**（与修复报告 §3.2 结论一致并强化）：真实 PTM cohort、独立 benchmark 数据、Workflow B 真实资产、正式 GPU 执行窗口。代码侧等待姿态正确（formal 模式硬拒绝、`benchmark_gate.available=false` 不伪造、run_workflow_b 就绪即插即用）。

---

# 4. 未解决问题与后续解决策略

| # | 问题 | 严重度 | 根因 | 解决策略与步骤 | 建议时间节点 |
|---|---|---|---|---|---|
| 1 | 真实 PTM cohort 缺失（U-06/主线起点） | 严重（外部） | 外部数据未到位 | 到位后按其研究设计补 typed cohort schema（pairing/donor 下限预注册）→ formal config 解锁 → 真实 KSTAR full analysis（维持修复报告 §4-1 方案） | 外部数据到位后 +1 天 |
| 2 | U-01 剩余：benchmark 数据 + 阈值预注册 | 高（外部+研究决策） | 数据外部；阈值是研究负责人决策 | 数据到位 → config `activity_benchmark` 段取消注释并预注册阈值 → `--activity-benchmark` 首评；evaluator/CLI/测试已就绪（本轮 §2 R2） | 数据到位后 +0.5 天（代码侧已清零） |
| 3 | U-02/U-03 剩余：Workflow B 真实资产 + Gate-E 正式执行 | 高（外部资产） | 真实同坐标系 latent pairs、冻结旧 checkpoint、≥200 方向 benchmark 均外部 | 资产到位 → `run_workflow_b --stages all --benchmark …` 首跑（编排器就绪，新环境 verify 已复现）→ Gate-E verdict → Gate-4/5 | 资产到位后 +2 天 |
| 4 | 正式六阶段/matched-null/双场景/Gate-4/5 执行 | 高（排期） | 依赖 #1–#3 + GPU 窗口 | 输入齐备后按 `docs/guides/perturbgen_bridge.md` §5 执行链运行（`ptm2cellnet` env 已验证可承载 GPU 训练） | #1–#3 后 +3–5 天 GPU 窗口 |
| 5 | 剩余 advisory 5 包/41 条 | 低–中 | transformers 受 esm 3.2.3 pin（22 条，上游）；mamba-ssm/accelerate 无 fix；datasets 属已删 ssh-unit 链（新环境实际不装，可从 lock 剔除登记）；setuptools 维持 `<81`（本轮判定） | 跟踪上游；torch/CUDA 升级窗口一次性处理；CI pip-audit 可见性维持 | 随上游/随 torch 升级窗口 |
| 6 | 复杂度总量 172 项（榜首 `validate_manifest` 已清，新榜首 `_load_and_validate_metadata` 32） | 中（持续） | 历史积累 | 下一批 6 函数（`_load_and_validate_metadata` 32、`replay_verdicts` 31、`verify_eval_input_against_manifest` 30、orchestrator `__post_init__` 30、`ptm_research_config.__post_init__` 29、`predict_expression_direction` 28）按本轮同方法拆分 | 下一轮 +4 天 |
| 7 | BLE001 剩余 27 处 / ND-03/04/06 / TD-17/18/19 | 低 | 有意降级与局部卫生 | 随触随改（维持修复报告 §4-8/9 口径） | 空闲窗口 |
| 8 | `ptm2cellnet` env 的 CI/复现口径固化 | 低（工程） | 本轮 runbook 已写入指南但未进 CI | 可选：CI 增加 lock fresh-装配 smoke job（发现 typing_extensions/tokenizers 类矛盾） | 下一轮 +1 天（可选） |

**推进顺序建议**：#2/#3 数据与资产到位即插即用（代码零缺口）→ #1 真实 cohort → #4 GPU 窗口集中执行；#6/#8 可在等待期并行。在 #1–#4 完成前，任何"PTM 导致某 cell type 表达变化"或"正式生物学 PASS"的表述继续禁止（修复报告 §4 收口约束维持）。

---

# 5. 文档引用索引

| 引用 | 出处 |
|---|---|
| 本轮任务依据 | `archive/20260922/project_repair_report_20260922.md` §4（未解决项清单）、§3.1（分层判定基线）、§5（文档索引） |
| U-01 接口契约 | `archive/20260922/project_analysis_20260921.md` §3.2（`evaluate_activity_benchmark` 接口表）；`archive/20260922/project_repair_report_20260921.md` §5.2（U-01 步骤表）；方案 `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §5.2 第 4–5 条、§8.7 |
| 本轮新代码 | `src/analysis/ptm_activity_benchmark.py`（模块 docstring）；`src/analysis/ptm_research_config.py`（`ActivityBenchmarkCriteria`）；`scripts/build_ptm_global_gene_scores.py`（`--activity-benchmark`）；`tests/unit/analysis/test_ptm_activity_benchmark.py` |
| 环境契约 | `docs/guides/installation.md` §2a（lock 重建 runbook，本轮新增）；`requirements-lock.txt`（typing_extensions/tokenizers 两行修正）；`setup.py` L40（setuptools `<81` pin） |
| 指南同步 | `docs/guides/ptm_activity_pipeline.md` §4（独立 benchmark 评估段，本轮新增）；`docs/guides/perturbgen_bridge.md` §5（Workflow B 生命周期，0922 落地） |
| 状态与决策 | `lessons.md` L-2026-0922-03（装配口径/pip stall/index 命名陷阱/setuptools 判定/evaluator 契约）、L-2026-0922-04（cudnn 覆盖/lock 自洽/mypy 缓存/mamba 源码）；`CHANGELOG.md` [Unreleased] environment-isolation round 分节；`docs/CURRENT_STATUS.md`（头部 + Quick Reference 首条） |
| 归档 | `archive/20260922/MANIFEST.md`（追加归档记录：旧 analysis + repair report） |
| 验证口径 | `AGENTS.md` 验证节五命令；本轮全部实测见 §2 各轮 |

---

*报告生成：ZCode 主会话（子代理本环境不可用，见 0922 分析报告 §8；全部调查、装配、修复、验证由主会话完成）。GPU 相关工作全部在沙箱外执行。*
