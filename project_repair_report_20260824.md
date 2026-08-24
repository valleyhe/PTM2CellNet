# PTM2CellNet 项目修复报告（2026-08-24）

> **执行日期**：2026-08-24　**修复基线**：`main` = `7e4b66d`（2026-08-24 docs: add executive summary）
> **依据文档**：`project_analysis_20260824.md` v1.0（第五轮周期性复核，下称【分析报告】）
> **任务性质**：按【分析报告】§5.3 技术债清单与 §6 下一步建议执行代码修复 + 修复后系统性复核
> **验证环境**：conda `SSH_unit`（Python 3.12.13，torch 2.7.1+cu118，numpy 2.2.6；环境在本轮开始前已被外部刷写，见 §4.3）

---

## 目录

- [0. 任务判断与输入依据](#0-任务判断与输入依据)
- [1. 问题修复汇总（按严重程度分类）](#1-问题修复汇总按严重程度分类)
- [2. 每轮迭代的执行情况与结果](#2-每轮迭代的执行情况与结果)
- [3. 系统性复核分析结果](#3-系统性复核分析结果)
- [4. 未解决问题及后续解决策略](#4-未解决问题及后续解决策略)
- [5. 相关文档引用](#5-相关文档引用)
- [6. 可复现命令](#6-可复现命令)

---

## 0. 任务判断与输入依据

| 项 | 内容 |
|---|---|
| 任务类型 | Bug 修复 + 技术债清偿 + 依赖/配置治理 + 文档补全 + 代码审查（修复后复核） |
| 输入 | 【分析报告】（尤其 §5.3 TD-N-26~34 与 §6 建议②③）、当前代码现状、全量实测 |
| 范围口径 | 「E2E 训练和推理」按现行需求基线 = ① 传统 PTM→cell state 训练/推理链 + ② DAVF×PerturbGen 双路径（`lessons.md` L-2026-0821-01：自研 E2E KO/KD/OE 已终止，不再作为目标） |
| 关键假设 | 无。所有判定以本机实测命令输出与 file:line 证据为准 |

**修复项选取原则**：【分析报告】§6 给出的代码侧第一优先（TD-N-34、TD-N-26-B）+ 第二批可落地项（TD-N-27/32/33 + U-15）+ 文档债（TD-N-19/31）。U-01~U-04 为外部数据/真环境依赖，按方案 §7.3 停止条件有意挂起，**不得以伪造数据方式"解决"**（AGENTS.md 工程规范 1；`lessons.md` L-2026-0822-06）。

---

## 1. 问题修复汇总（按严重程度分类）

分级口径沿用【分析报告】§5.2：严重=阻塞核心功能或威胁数据正确性；高=确定性缺陷；中=可维护性/效率受损；低=完善项。

### 1.1 中级债（2 项，全部闭环）

| 编号 | 问题 | 修复方案 | 落地证据 |
|---|---|---|---|
| TD-N-34 | 主环境依赖漂移：① datasketch 声明未装（大规模同源划分静默回退稠密 OOM 路径）；② lock 内 scgpt==0.2.4 与 scvi-tools>=1.2 互斥（pip check 永久冲突）；③ lock 缺 datasketch/uvicorn 钉扎 | ① SSH_unit 安装 datasketch 2.0.0 并实测 MinHash+LSH 路径产出正确相似度矩阵；② 从 `requirements-lock.txt` 移除 scgpt（对齐 TD-M05 决策，`requirements-analysis.txt` 注释明示 scgpt 需独立环境）；③ 按验证环境实际版本补钉 `datasketch==2.0.0`、`uvicorn==0.50.0`（字母序插入） | `python -c "from datasketch import MinHashLSH"` OK；近似矩阵实测（相同序列 1.0 / 无关 0.0 / 单突变序列 0.4≈3-mer Jaccard 理论值 5/11）；`scripts/check_requirements_consistency.py` 归零 |
| TD-N-26（B 方案） | `pretrain_masked_ptm` 286 行超长函数（【分析报告】§5.3 TD-N-26 首项，`src/training/self_supervised.py:276→562`），自监督预训练主入口认知复杂度与回归风险高 | 拆为 6 个职责单一辅助函数：`_split_train_val`（数据划分）/`_build_lr_schedulers`（调度器）/`_restore_checkpoint`（N20 resume）/`_compute_masked_loss`（掩码损失，保留空 mask 跳过 forward 的原语义）/`_train_one_epoch`（单 epoch，AMP/grad-clip/warmup）/`_save_best_checkpoint`（检查点）；`_validate` 复用 `_compute_masked_loss` 消除三处重复准备逻辑；主函数变纯编排，签名与返回值不变 | `tests/unit/test_self_supervised.py` + `tests/integration/test_pretrain_resume.py` 13 passed（覆盖 resume/scheduler/warmup/checkpoint 路径） |

### 1.2 低级债（6 项闭环，1 项维持）

| 编号 | 问题 | 修复方案 | 落地证据 |
|---|---|---|---|
| TD-N-32 | 多轨 requirements `>=` 区间与 lock 钉扎零校验 | 新增 `scripts/check_requirements_consistency.py`：core 依赖必须在 lock 中存在且版本满足区间；可选轨（pretrained/mamba/analysis）在 lock 中则须满足；CI 新增 `dependencies` job 跑 `pip install -e .` + `pip check` + 一致性脚本 | 首跑即抓到 2 处真实漂移（datasketch/uvicorn 缺钉）→ 补钉后 OK（274 pins）；`tests/unit/test_ci_workflow_contract.py::test_ci_has_dependency_consistency_job` 锚定 |
| TD-N-27 | `uv.lock` 空壳（52B 仅 3 行头部）误导钉扎源 | 删除（未被 git 跟踪，无历史影响）；README「安装」节新增钉扎源说明（唯一权威 = requirements-lock.txt） | `ls uv.lock` 不存在；`README.md` 安装节 |
| TD-N-33 | 同名测试双份并存（`tests/unit/test_gene_mapper.py` 159 行退避专项 vs `tests/unit/analysis/test_gene_mapper.py` 331 行主功能） | 退避专项 8 个测试以 `TestUniProtPollBackoff` 类并入 analysis 版，helper 更名防冲突（`_response`→`_poll_response`），`gm`→`gene_mapper` 统一；删除旧文件；`docs/CURRENT_STATUS.md:16` 回归锚点路径同步刷新 | 合并后 `tests/unit/analysis/test_gene_mapper.py` 27 passed（19 主功能 + 8 退避，断言全保留） |
| TD-N-19 | `docs/guides/perturbgen_bridge.md` 缺失（【分析报告】§4.2 docs 85.7% 的唯一缺口） | 新写 v1.0 桥接指南：双工作流架构、主进程/独立环境边界、环境准备三脚本、Gate-0 donor 契约、六阶段 pipeline CLI、schema v2 嵌入资产导出与注入、评估与发布证据、门禁状态速查、常见陷阱——全部 CLI 参数逐一 `--help` 核实 | `docs/guides/perturbgen_bridge.md`（280 行） |
| TD-N-31 | CHANGELOG 4 个月未更新（v2.x 无发布记录） | 按 git 历史核实补记 `[2.1.0] - 2026-08-24`（2026-05-04 v2.1 milestone 起 106 commits 收口），Added/Changed/Fixed 分组、Known Issues 更新为当前口径；Unreleased 清空 | `CHANGELOG.md` |
| U-15 | CI analysis job 仅显式覆盖 3 个 PerturbGen 测试文件，方案 §4.5「PerturbGen 单测必跑」语义打折（`test_data_prep.py:5` importorskip(anndata) 在主 job 静默 skip） | analysis job pytest 目标从 3 个点名文件扩展为整个 `tests/unit/integration/perturbgen/` 目录（10 文件）；契约测试断言同步更新为目录级 | 本地实测该目录 + mocked pipeline 91 passed；`test_ci_workflow_contract.py` 3 passed |
| TD-N-30 | TEST_COVERAGE.md 基线过时 | **维持登记**（受 TD-N-09 阻塞无法本地复测，【分析报告】§2.2 已判定保留） | 不变 |

### 1.3 事实修正（相对【分析报告】§5.3 TD-N-34 原始表述）

复核发现 TD-N-34 三条证据中的两条在本轮开始前已被外部变化部分解决，本次按现状收敛：

| 原证据（【分析报告】§1.3） | 本轮实测现状 | 处置 |
|---|---|---|
| ① `ptm2cellnet requires datasketch, which is not installed` | SSH_unit 确实未装 → **仍成立** | 安装 2.0.0 + lock 补钉（见 §1.1） |
| ② `requirement numpy<2,>=1.24, but you have numpy 2.4.3` | `setup.py:19` 已是 `numpy>=1.24,<3`（D1，2026-08-16 变更，`requirements-core.txt:14-16` 注释）；当前各活跃环境无旧 metadata 残留 | 已过时，无需动作 |
| ③ `scgpt 0.2.4 requires scvi-tools<1.0` | scgpt 包已不在任何活跃环境（仅 lock 文件残留钉扎） | 从 lock 移除（根因修复） |

---

## 2. 每轮迭代的执行情况与结果

本轮为单会话四阶段顺序执行，无失败迭代；两处中途修正如实记录。

### 迭代一：现状核查（只读）

- 核查 `src/training/self_supervised.py`（pretrain_masked_ptm 全文 286 行）、双份 test_gene_mapper、`requirements*.txt` ×8、`ci.yml`、`uv.lock`、CHANGELOG、docs/guides。
- **环境漂移发现**：当前 shell 激活 `SSH_unit`（torch 2.7.1+cu118 / numpy 2.2.6），与【分析报告】§1.3 基线（torch 2.4.1 / numpy 2.4.3）不同——两环境均在本轮开始前被外部刷新（`ptm` 环境同样刷至 2.7.1）。本轮全部验证在 SSH_unit 完成，与 `requirements-core.txt:14` 注明的验证环境口径一致。
- **结果**：修复清单确认（§1 全部条目），范围排除外部依赖项（U-01~04/06/07）。

### 迭代二：代码与配置修复

1. TD-N-26-B 拆分落地（§1.1）。中途修正 ×2：① 空 mask 批次恢复"跳过 forward"的原语义（`_compute_masked_loss` 返回 `None`）；② `_validate` 复用同一辅助函数消除重复。
2. TD-N-33 合并落地（§1.2），import 补 `MagicMock`。
3. TD-N-34/32 依赖治理落地：lock 移除 scgpt（272→271 条）→ 补钉 datasketch/uvicorn（→274 条，含字母序一次修正）→ 一致性脚本首跑抓漂移→归零。
4. CI 更新：`dependencies` job + analysis job 目录级覆盖；契约测试同步（中途修正：首轮契约断言旧文件名失败 → 更新为更强语义断言，3 passed）。
5. TD-N-27 uv.lock 删除 + README 说明；TD-N-19 指南（CLI 逐一 `--help` 核实后成文）；TD-N-31 CHANGELOG（git log 106 commits 逐条核实）。

### 迭代三：全量验证（实测）

| 检查 | 命令 | 结果 |
|---|---|---|
| 编译 | `python -m compileall -q src scripts tests` | exit 0 ✅ |
| 静态风格 | `ruff check src scripts tests` | All checks passed ✅ |
| 类型检查 | `mypy src` | **38 errors in 18 files** ⚠️ ——经 `git stash` 对照实验证实为环境刷写（torch 2.4.1→2.7.1 类型 stub 变严）导致的既有债显形扩大（修改前同为 38），**非本轮引入**；本轮改动文件 `self_supervised.py` 0 错误。详见 §4.2 |
| 全量测试 | `unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest tests -q` | **2329 passed / 16 skipped / 0 failed（574.18s）** ✅ 较【分析报告】§1.3 基线 2328 +1（新增 CI 契约测试；TD-N-33 合并无净增减） |
| 一致性 | `python scripts/check_requirements_consistency.py` | OK（274 pins） ✅ |

### 迭代四：修复后系统性复核（E2E 实测，见 §3）

> 注：原计划两只读侦察子代理因模型供应商认证失败不可用（与 cognee 记忆服务同源故障），复核全部改由主代理直接实测完成，结论反而更强（全部为可复现命令而非静态阅读）。

---

## 3. 系统性复核分析结果

### 3.1 E2E 训练与推理技术要求 vs 现状判定表

| # | 技术要求 | 出处 | 现状（本轮实测/证据） | 判定 |
|---|---|---|---|---|
| T1 | 传统训练链端到端可跑 | `configs/smoke/cnn_cpu.yaml:7-10` 用法注释 | `python scripts/train.py --config configs/smoke/cnn_cpu.yaml --data data/processed/pmads_test_5k.csv` 1 epoch 完整跑通：训练→评估（test_metrics.json + bootstrap CI）→artifact manifest；数据契约 fail-fast 正确（provenance 缺失 → `deployable=False` 警告而非静默） | ✅ 满足（实测） |
| T2 | CLI 推理链 | 同上 | `scripts/predict.py --model .../best_model.pt --sequence ... --ptm-sites ...` 输出概率分布 + PTM 解析（2 个）+ 落盘 predictions.csv | ✅ 满足（实测） |
| T3 | API 服务链（冷启动→初始化→预测→溯源） | AGENTS.md §0.6 | TestClient 实测：`POST /api/v1/initialize`（自动发现 checkpoint 同目录 config，P1-1 行为）→ 200；`POST /api/v1/predict` → 200（cell_state/confidence/probabilities/model_kind=real）；`GET /api/v1/model/info` → 200（encoder/dim/cell_states/supported_ptm_types）；未初始化时 /predict 正确 503 状态机 | ✅ 满足（实测） |
| T4 | API 安全链 | AGENTS.md §0.6（速率限制/请求体上限/CORS/路径白名单） | 实测全程按设计工作：速率限制 600 rpm 启用日志；`/tmp` 路径 checkpoint 被路径穿越防护 403 拒绝（移入 outputs/ 后放行）；生产模式 /initialize 要求 API key | ✅ 满足（实测） |
| T5 | DAVF×PerturbGen 工程链（M1-M3 + M4 注入） | 方案 §4.4/§4.5 | `tests/unit/integration/perturbgen/` + mocked pipeline 91 passed；M4 schema v2 注入链（`davf_inference.py:271-301`→`latent_davf.py:196-216`→`architectures.py:244-254` 迁移闸门）由既有回归锚定（【分析报告】§4.3 八子项 6/8 ✅） | ✅ 满足（工程口径） |
| T6 | 双路径判定反假阳性 | `lessons.md` L-2026-0821-01 §1.3 | `src/integration/perturbgen/results.py:26,261` `excluded_target_gene` 实现「rescue 排除目标基因本身」；`bootstrap_confidence_interval`（`results.py:372`）支撑 CI 硬门；`evaluate_perturbgen_dual_path.py` 候选级评估入口 + sha256 输入指纹 | ✅ 满足（代码证据） |
| T7 | PerturbGen **科学验收链**（M4 真实重训/Gate-E/M6/Gate-4/Gate-5） | 方案 §7.2 M4-M6、§5.4 | **未达成**：Gate-0 M0⑥ donor cohort 0 合规候选（【分析报告】§3.1 U-01，`outputs/perturbgen/spike/20260823_donor_audit/evidence.json`）→ 按方案 §7.3 有意挂起，非代码缺口 | ❌ 外部数据阻断 |
| T8 | E2E 测试家族 | 方案 §5.1 测试矩阵 | `tests/e2e/` 46 个测试覆盖 train-predict CLI（19）/API initialize（4）/DAVF（3）/Mamba（4）/pretrained（2）/PTM-site（2）/矩阵（12）；全量 2329 passed | ✅ 满足 |
| T9 | GPU 训练支持 | 本机 P40 单卡 | `train.py`/`predict.py` 均有 `--device`；AMP 链路（`amp_compat.py`）被 self_supervised 回归覆盖。注：本轮全部验证为 CPU 口径（沙箱限制），GPU 实测待 M4 重训时脱离沙箱执行 | ✅ 接口满足（GPU 实测待外部数据解锁） |

### 3.2 复核结论

1. **代码侧 E2E 训练与推理的技术要求全部满足**：传统链（T1-T4）本轮实测端到端通过；PerturbGen 工程链（T5/T6/T8）测试锚定全绿；静态门禁（编译/风格）全绿。
2. **唯一未达成项 T7 是外部数据依赖而非代码缺口**：合规 donor cohort（≥3 donor、normal/disease 配对、raw counts、ENSG）供给前，M4 重训及下游全部科学验收按方案 §7.3 有意挂起——这是防伪造数据的正确行为，不是待修 bug。
3. **新发现的环境层问题**（详见 §4.2/§4.3）：mypy 既有债显形扩大（38 处）与 `ptm` conda 环境残缺，均源于本轮开始前的环境外部刷写，与仓库代码无关，但需登记跟踪。

---

## 4. 未解决问题及后续解决策略

### 4.1 阻塞 E2E 科学验收链的外部依赖（不可代码解决）

| # | 问题 | 根因 | 解决策略 | 时间节点 |
|---|---|---|---|---|
| N1 | U-01：Gate-0 M0⑥ donor cohort（0 合规候选） | 本地 30 个 scPerturb h5ad 无一满足「显式 donor/patient + normal/disease 配对 + raw counts + ENSG + ≥3 donor」（`audit_perturbgen_cohort.py` 实测） | ① 数据侧：从 GEO/EGA/hCA 检索含 donor 元数据的配对队列（如肾脏/肝脏等 normal vs disease 配对研究），用 `audit_perturbgen_cohort.py` 预审后再入 preflight；② 决策侧备选：若 8 周内无合规队列，评估将 Gate-1 降级为「合成 cohort 工程验证 + 真实数据后补科学验收」的双轨方案（需方案 v2.1 修订评审） | 数据供给后 T0+2 工作日出 Gate-0 结论（方案 §7.1 排期口径）；U-02/U-03/U-04 随之解锁 |
| N2 | U-06/U-07：replogle/scgenescope controlled 数据 | 数据库访问受控（`data/manifests/datasets.yaml:596-670` status: controlled） | 同 N1 ② 或申请数据访问；或决策 cross_scale_training profile 降级 | 与 N1 独立，随数据供给 |
| N3 | M4 真实重训 + Gate-E + Gate-4/5 | 依赖 N1；GPU 训练须脱离沙箱执行（用户约束） | N1 解锁后：`scripts/export_perturbgen_gene_embeddings.py` 导出真实资产 → `finetune_davf_e2e.py --embedding-asset` 重训（GPU 脸前先 dry-run）→ Gate-E 基准（≥200 PTM→gene + bootstrap CI + 下游非劣） | 方案 §7.2：M4 5 人天、M5 3 人天、M6 3-8 人天 |

### 4.2 环境层技术债（本轮新发现/显形）

| # | 问题 | 证据 | 解决策略 | 时间节点 |
|---|---|---|---|---|
| N4 | mypy 既有债显形扩大：23→38 errors（`git stash` 对照证实非本轮引入；增量 15 处主要在 `src/integration/perturbgen/runner.py` 17 处、`model_utils`/`long_sequence` 等） | torch 2.4.1→2.7.1 后类型 stub 变严；分布见 `mypy src | awk` 实测 | 按 TD-N-06/11 同族处理：先清 runner.py 17 处（新增代码，注释缺失为主，~0.5 日），其余并入既有清偿批次 | 下一批次（建议 2026-08-26 前） |
| N5 | `ptm` conda 环境残缺：torch 被外部刷至 2.7.1+cu118 且缺 psutil/idna/msgpack 等系统包（`pip check` 26 条缺失告警） | `/home/scu/anaconda3/envs/ptm/bin/pip check` 实测 | 二选一：① 弃用 ptm 环境统一用 SSH_unit（本轮已验证全量绿）；② 重建。属环境运维决策，超出代码修复范围 | 用户决策 |
| N6 | torch 2.4.1→2.7.1 跨版本升级未经项目侧评审（本轮实测全量测试绿，但与 lock 钉扎 `torch==2.4.1` 口径不一致——lock 未随环境刷写更新） | `requirements-lock.txt` torch 行 vs SSH_unit 实测 | 待用户确认升级意图后统一：更新 lock 至 2.7.1+cu118 或回退环境；一致性脚本已能守住后续漂移 | 与 N5 同批决策 |

### 4.3 仓库侧剩余登记债（维持【分析报告】§5 口径，不因本轮变化）

| 编号 | 内容 | 策略 | 估算 |
|---|---|---|---|
| TD-N-26 余量 | 另 4 个超长函数（`Evaluator.cross_validate` 202 行、模块级 `evaluate` 171 行、`_load_model` 168 行、`_stream_url_to_dir` 153 行） | 方案 A 分段抽取，行为等价由既有测试锚定 | 1.5~2.0 日 |
| TD-N-30 | TEST_COVERAGE.md 基线过时 | 受 TD-N-09 阻塞，解除后在 CI 复测刷新 | 外部依赖 |
| TD-N-08 | torch CVE（CVE-2024-48063/6577）随版本升级评估 | 与 N6 升级决策合并评估（2.7.1 已含修复，若确认升级则自动闭环） | 随 N6 |
| U-05/M7 | Geneformer 主链删除 | Gate-E 通过后执行（门控顺序正确保持） | 2 日（随 N3） |
| U-08~U-14 | 分支项（MultiPLM/PTMTokenAdapter/cross-scale 真实训练、ESM-3/CPTAC/外部服务/多 GPU 验收） | 按【分析报告】§3.2 阻断链，随 N2 数据供给或独立排期 | 各 0.5~3 日 |

---

## 5. 相关文档引用

| 引用 | 章节/位置 | 本报告用途 |
|---|---|---|
| `project_analysis_20260824.md` | §1.3 验证矩阵、§3.1 U-01~15、§3.2 阻断依赖链、§4.2 模块完成度、§4.3 M4 八子项、§5.3 TD-N-26~34、§6 结论建议 | 修复任务来源与基线（本报告 §0/§1/§4） |
| `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | §4.1 双工作流、§4.5 修改文件、§4.6 数据语义、§4.7 rescue 判定、§5.4 科学质量门、§7.1 排期、§7.2 M0-M7、§7.3 停止条件、§9 API 边界 | E2E 需求口径与挂起依据（§3.1 T5-T7、§4.1） |
| `lessons.md` | L-2026-0821-01（E2E 终止+双路径 AND 标准）、L-2026-0821-02（底座切换）、L-2026-0822-04（manifest 绑定）、L-2026-0822-05（证据可提交）、L-2026-0822-06（cohort 反伪造） | 范围口径与反假阳性依据（§0/§3.1 T6/§4.1） |
| `docs/CURRENT_STATUS.md` | :16（TD-N-24 回归锚点，本轮刷新路径） | §1.2 TD-N-33 |
| `requirements-core.txt` | :14-16（D1 numpy 口径）、:31（datasketch 声明） | §1.3 事实修正 |
| `requirements-analysis.txt` | TD-M05 注释块（scgpt 互斥决策） | §1.1/§1.3 scgpt 移除依据 |
| `docs/guides/perturbgen_bridge.md` | 全文（本轮新增） | §1.2 TD-N-19 |
| `archive/20260824/MANIFEST.md` | 归档批次权威清单 | 历史追溯 |

---

## 6. 可复现命令

```bash
# 静态与编译门禁
python -m compileall -q src scripts tests && ruff check src scripts tests
mypy src                                   # 预期 38 errors（§4.2 N4，非本轮引入）

# 依赖一致性（TD-N-32/34 门禁，CI dependencies job 同款）
python scripts/check_requirements_consistency.py

# 全量测试（网络隔离 + HF 离线口径）
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest tests -q -p no:cacheprovider    # 预期 2329 passed / 16 skipped

# E2E 训练→推理冒烟（§3.1 T1/T2，CPU，~30s）
python scripts/train.py --config configs/smoke/cnn_cpu.yaml \
    --data data/processed/pmads_test_5k.csv --output outputs/e2e_smoke
python scripts/predict.py --model outputs/e2e_smoke/models/best_model.pt \
    --sequence ACDEFGHIKLMNPQRSTVWYACDEFGHIK \
    --ptm-sites '[{"position":5,"type":"phosphorylation"}]' --device cpu

# PerturbGen 单测目录（§3.1 T5，CI analysis job 新覆盖口径）
python -m pytest tests/unit/integration/perturbgen/ tests/integration/test_perturbgen_pipeline_mocked.py -q

# MinHash 近似路径（TD-N-34 ①验证）
python -c "from datasketch import MinHash, MinHashLSH; print('OK')"
```

---

**报告版本**：v1.0（2026-08-24）
**执行方式**：主代理顺序执行（侦察子代理因模型供应商认证故障不可用，全部验证改为主代理实测）
**提交批次**（本地 `main`，未 push）：代码修复 = `1227371`；依赖与 CI 治理 = `be69a45`；文档与报告 = `131fc83`（本文件所在提交）
