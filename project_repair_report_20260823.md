# PTM2CellNet 项目代码修复报告（2026-08-23，v2.0）

> **版本说明**：本文件 v2.0 为当日**第二轮**修复的完整报告，覆盖并取代同日 12:36 的 v1.0。
> v1.0 的交付（M0⑤ 真实 perturb smoke + 资源基线 + 6 组隐式依赖补装）已全部继承并
> 在 §2 标注；本轮（v2.0）以 v1.0 §6「未解决问题及后续解决策略」为直接输入。

## 1. 任务判断与输入依据

本任务类型：Bug 修复、技术债清理、数据处理/脚本任务、GPU 训练/推理链路验证、E2E 阻塞清除、系统性复核。

依据当前日期最新分析文档 `[A1]` §4「未解决问题及后续解决策略」与当日第一轮修复报告
（v1.0，下称 `[R4]`）§6 的可执行项执行修复。本轮锁定四项：

| 输入条款 | 本轮动作 | 结果 |
|---|---|---|
| `[R4]` §6 第 1 行：M0⑥ donor 审计「①可立即执行」 | 对 DatlingerBock2021 与全部本地 h5ad 做字段级 donor/状态语义审计 + M1 preflight 真实执行 | ✅ 审计完成，结论：**本地 0 合规候选，必须外部获取**（§2.1） |
| `[R4]` §6 第 3 行：M2 上游三坑规避固化 | 写入 `config_builder.py` 契约校验 + golden 测试 + 模板修复；过程中确证**上游第 4 坑** | ✅ 完成（§2.2） |
| `[R4]` §6 第 5 行：独立环境锁定快照漂移 | 8 组依赖（6 组继承 + 本轮新增 protobuf/tensorboard 版本天花板）并入 `setup_perturbgen_env.sh`，evidence 重采 | ✅ 完成（§2.3） |
| `[R4]` §6 第 2 行：decoder 端无训练权重（原排期 08-31 M2 窗口） | **提前执行**最小 train-mask：真实数据 1 epoch 官方训练 → ckpt 前缀转换 → 官方 val.py 真实权重推理，训练→推理闭环 | ✅ 工程链路闭环（§2.4，科学有效性仍待合规队列） |

引用文档：

| 标记 | 文档 | 使用章节 |
|---|---|---|
| `[A1]` | `project_analysis_20260823.md` | §1、§3、§4（修复依据）、§6 |
| `[R4]` | 本文件 v1.0（12:36 版，已被覆盖；其内容以 `[R3]`/`[A1]` 交叉引用可溯） | §2.2、§4.2、§4.3、§6 |
| `[R3]` | `project_repair_report_20260822.md` | §2、§4.2（M0 六项 evidence）、§5（排期） |
| `[R2]` | `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | §4.5、§4.6（donor 契约）、§5.3、§7.2（M0-M7）、§7.3（停止条件） |
| `[CS]` | `docs/CURRENT_STATUS.md` | Quick Reference 全节 |

---

## 2. 问题修复汇总（按严重程度分类）

### 2.1 严重——M0⑥ donor 合规队列：审计闭环，结论为外部数据依赖（Gate-0 最后缺口）

**修复状态：✅ 审计执行完毕（策略①），结论确定性否定本地候选；策略②（外部获取）为唯一路径。**

| 项目 | 结果 |
|---|---|
| 审计脚本 | ✅ `scripts/audit_perturbgen_cohort.py`（新增，fail-fast 无兜底，可复现） |
| 审计范围 | ✅ 本地 `data/raw/scperturb/` 全部 30 个 h5ad（26 可读 + 4 截断，与 `[R3]` §4.2 一致） |
| 审计标准 | `[R2]` §4.6 第 1 条：raw counts、无版本 ENSG、唯一 obs_names、cell_type/state/donor 标注、目标 cell type 每状态 ≥3 可评估 donor；donor 判定严格排除 sample/batch/replicate/cell_line/孔位编号（`[R3]` §5 M1 行禁令） |
| **DatlingerBock2021 深度审计** | ❌ 不合规（3 项独立理由）：`cell_line` 唯一值 "Jurkat cells"（永生化细胞系非供体）；`disease` 全部 "acute T cell leukemia"（**无 normal 状态**）；`sample` 为 384 孔板孔位编号（A01…，非 donor）。这给 `[R4]` §4.3-2 的遗留猜测（「sample 与 donor 的关系待审计」）提供了数据级否定答案 |
| **全库补充扫描**（超出 `[R4]` 计划的增量） | 22/26 可读文件 `tissue_type=cell_line`（细胞系）；仅 3 个 primary（ShifrutMarson2018：healthy 单状态 + 仅 2 patient + **本轮新发现还缺 ensembl_id**；LaraAstiasoHuntly2023_exvivo：healthy 单状态；leukemia：leukemia 单状态；WeinrebKlein2020：healthy 单状态）——**无一文件内存在 normal/disease 状态配对** |
| **M1 preflight 真实执行**（`[R2]` §7.2 M1「真实 cohort preflight 尚未执行」→ 已执行） | ✅ 调用真实工程代码 `prepare_perturbgen_anndata()` 于真实数据切片（5000 细胞，真实 X counts layer + 真实 obs/var）：契约拒绝，报错 `adata.obs is missing required columns: ['state', 'donor']`——**这是 M1 Gate-1 要求的 preflight 报告首个真实记录，失败原因完全可解释**（非硬错误，是数据语义缺失） |
| Evidence | ✅ `outputs/perturbgen/spike/20260823_donor_audit/evidence.json`（30 文件逐项审计明细 + policy 声明 + preflight 结果），verdict = `NO_LOCAL_COMPLIANT_COHORT` |

**结论（影响后续排期）**：M0⑥ 从「审计一份数据」收敛为「纯外部数据获取」，与 `[R4]` §6 第 1 行策略②对齐：需获取 raw counts、无版本 ENSG、cell type/state、**明确 donor 标注**、normal/disease 配对、≥3 共享 donor 的 AnnData（`[R2]` §4.6）。本地数据对 M2 **工程链路**验证仍有效（§2.4 已用），但对科学验收（Gate-E/M6）无效。

### 2.2 严重——M2 runner 上游坑固化：3 坑契约校验 + 新确证第 4 坑

**修复状态：✅ 完成（`[R4]` §6 第 3 行），并在执行中新确证上游第 4 坑（§2.4 迭代 9）。**

| 坑（上游 val.py/train.py 实证行号） | 固化方式 | 测试 |
|---|---|---|
| (a) `ckpt_masking_path=None` 静默跳过推理（val.py:253，exit 0） | `_validate_perturb_candidate_contract` 强制 `model.ckpt_masking_path` 为非空字符串 | `test_perturb_contract_ckpt_masking_path_matrix`（3 参数化：absent/none/blank） |
| (b) 缺显式维度时 `max(input_id)` 对变长 list 字典序比较（val.py:53 要求 tgt_vocab_size 与 max_seq_length **两者同时**存在才走显式分支，:63 为坑） | 强制 `trainer.tgt_vocab_size`/`max_seq_length` 同时为显式正整数 | `test_perturb_contract_requires_both_explicit_dimensions`（2 参数化：任一缺失） |
| (c) 运行时对 config 显式值 +50/+100 buffer 且重写 `datamodule.max_len` 为 base 值（val.py:216-218） | docstring 固化 base 值语义 + `datamodule.max_len` 若提供必须 == `trainer.max_seq_length`；**新增跨 stage 校验** `_validate_stage_dimension_consistency`（train_mask/train_decoder/perturb 三处 tgt_vocab_size 显式时必须一致——M0 smoke 迭代 11 size-mismatch 教训的泛化防御） | `test_perturb_contract_datamodule_max_len_must_equal_trainer_base` + `test_stage_dimension_consistency_rejects_tgt_vocab_drift` |
| **(d) 新确证**：训练 ckpt 前缀 `transformer.*`（PerturbGenTrainer）与推理 restore 目标 `pretrained_model.*`（PerturberTrainer）不匹配，strict load 全量失败 | 转换器实测（§2.4 迭代 9-10：288/288 张量 `transformer.* → pretrained_model.*` 重映射 + 逐元素断言）；**已记录于 evidence，M2 runner 实现时需把该转换固化为 stage**（未解决问题 §5） | 尚无单测（转换逻辑在 spike 流程中，见 §5） |

**模板修复（活体坑 b 清除）**：`configs/integration/perturbgen.yaml` perturb stage 此前**没有**显式 `tgt_vocab_size`/`max_seq_length`——正是坑 (b) 会让官方 val.py 落入字典序分支的活体场景；本轮补 `tgt_vocab_size: 2002`、`max_seq_length: 1024`（base 值，含语义注释）+ `datamodule.max_len: 1024`。

**测试同步**：`tests/unit/integration/perturbgen/test_config_builder.py` 新增 7 项契约用例；既有 `test_build_stage_plans_rejects_output_path_escape` 与 mocked pipeline fixture 按新契约补齐合法字段（测试意图不变）。

### 2.3 严重——独立环境锁定快照漂移：8 组依赖并入 + 版本天花板

**修复状态：✅ 完成（`[R4]` §6 第 5 行 + 本轮 2 组新增）。**

| 组 | 包 | 版本 | 触发原因 |
|---|---|---|---|
| 1-6（继承 `[R4]` §2.2） | einops / geneformer@`04c2b2e8`（HF 源，`--no-deps`）+ tdigest/peft/optuna/tensorboard / scvi-tools / jax+jaxlib / POT / rouge-score+nltk | 见脚本 | 上游顶层 import 硬依赖 |
| **7（本轮）** | protobuf | **4.25.8**（自 7.36.0 降级） | wandb 0.17.9 预生成 pb2 与 protobuf≥5 不兼容（训练触发 WandbLogger import） |
| **8（本轮）** | tensorboard | **2.18.0**（自 2.21.0 降级） | tensorboard≥2.19 的 pb2 要求 protobuf≥5.26，与组 7 冲突；2.18+4.25.8 是唯一工作组合，已实测 wandb/SummaryWriter 双 import 通过 |

`scripts/setup_perturbgen_env.sh` 已更新为 6 步全锁定清单（含版本天花板注释与 `TORCHDYNAMO_DISABLE=1` 的 P40 约束说明）；`outputs/perturbgen/env_evidence_20260823.json` 重采（python 3.11.15 / commit `a9a9375` / cuda_available=true / 离线 import 全过 / overall=OK）。

### 2.4 严重——M2 最小 train-mask：训练→推理工程闭环（原排期 2026-08-31，本轮提前）

**修复状态：✅ 工程链路闭环（真实数据、官方 CLI、真实训练权重推理）；科学有效性明确不宣称（1 epoch / 750 细胞子集，loss 7.34 远未收敛）。**

| 项目 | 结果 |
|---|---|
| 训练数据 | M0⑤ 已 tokenised 的真实 raw-counts 子集（400 control + 350 LCK_2 细胞，2000 hvg，DatlingerBock2021） |
| 训练命令 | ✅ 官方 `perturbgen train-mask` CLI（batch 16、epochs 1、lr 5e-05、wd 1e-06、mask_scheduler pow、seed 42、encoder=scmaskgit + foundation encoder_path）——**foundation 162 encoder keys 严格加载**（scmaskgitwrapper strict 路径） |
| 训练指标 | train/masking_loss_epoch 7.3428、val/loss_epoch 7.20233、19 steps、wandb offline run 落盘 |
| 产出 checkpoint | ✅ `ref/T_perturb/res/datlinger2021_m0smoke_trainmask/checkpoints/20260823_2107_cellgen_train_masking_...-epoch=00.ckpt`（664MB、288 keys = **162 foundation encoder + 126 真实训练 decoder**、token_embedding (2032,768)=1982+50 buffer）——**这正是 val.py 需要的下游 `ckpt_masking_path`，M0⑤ 时的 seed-42 随机 decoder 缺口被真实权重补上** |
| ckpt 前缀转换 | ✅ 坑 (d) 的应对：构造 `PerturberTrainer` 并将训练 ckpt 全部 288 张量 `transformer.* → pretrained_model.*` 重映射，`torch.equal` 逐元素断言全过，导出 582 keys |
| 推理验证 | ✅ 官方 `python -m perturbgen.Perturb.val --config`：25/25 batches、输出 `20260823-21:13_minference_adata_gLCK_ssrc_tmask.h5ad`（227 细胞 × 2001 基因、obsm: mean_cos_similarity/perturbed_cls/true_cls、varm: gene_cos_similarity）——**schema 与 M0⑤ 输出完全一致，且 decoder 为真实训练权重** |
| Evidence | ✅ `outputs/perturbgen/spike/20260823_m2_trainmask/evidence.json`（超参、指标、ckpt/输出 sha256、三坑实证、语义边界声明）+ `val_full.log` + 转换 ckpt + 推理 YAML |
| 语义边界 | 训练权重真实但仅 1 epoch 小样本；数值是**工程闭环证据**，不是科学扰动质量（后者属 Gate-E/M4/M6，且依赖合规队列 §2.1） |

**新确证的上游坑与硬件约束**（全部已入 evidence 与 setup 脚本注释，供 M2 runner 固化）：
- train.py 默认 `split_obs`/`var_list` 引用不存在的 `cell_type_cellgen_harm` 等列 → 必须显式传真实列（`split_obs=perturbation`、`var_list=perturbation`）；
- 默认 `pred_tps=[1,2,3]` 在单时间点数据上 KeyError → `--pred_tps 1`；
- Tesla P40（CUDA capability 6.1）< triton 最低 7.0 → 训练入口必须 `TORCHDYNAMO_DISABLE=1`；
- shell 管道 `| tail` 会掩盖真实退出码（本轮实际踩过：管道后 EXIT=0 为假象，重定向后真实 EXIT=1）。

### 2.5 明确保留边界（非问题，不处理）

- `geneformer` 主链 67 处引用：M7 处理（`[A1]` §1.3 / `[R2]` §7.2 M7 停止条件）。
- mypy 既有 23 错误、共享环境 pip check 冲突：`[A1]` §4 低优先级并行项，本轮 0 新增（见 §5 验证）。
- masking 推理路径输出无 `true_counts/pred_counts` layers（cos-sim 矩阵输出）：该 schema 要求属 count-decoder 路径（train-decoder 阶段，`[R2]` §7.2 M2 的第二个训练 stage），本轮 masking 闭环不覆盖（见 §5）。

---

## 3. 每轮迭代的执行情况与结果

### 迭代一（v1.0，上午）：M0⑤ 真实 perturb smoke —— 15 轮驱动迭代，完成（详见 `[R4]` §3，此处不重复）

### 迭代二（v1.0，上午）：环境隐式依赖 6 组补装 —— 完成

### 迭代三（本轮）：donor 审计脚本 —— 3 轮修复后通过

| 轮 | 失败点 | 修复 |
|---|---|---|
| 1 | 截断文件提前返回缺 `compliant_candidate` 键 → KeyError | 列表推导改 `.get()` |
| 2 | anndata 0.11.4 backed 切片 `to_memory()` 丢 X（上游怪癖） | 全量读入后切真实前 5000 行（CSR ~700MB） |
| 3 | backed CSR 直接切片撞 `_validate_indices` 缺失 | 同上（一并绕开两个 backed 坑） |

**通过后产出**：30 文件审计 evidence + M1 preflight 真实 REJECTED 记录（§2.1）。

### 迭代四（本轮）：三坑契约校验 —— 2 轮修复后通过

| 轮 | 失败点 | 修复 |
|---|---|---|
| 1 | 内部复用函数被 pytest 收集（缺 mutation fixture → ERROR）；既有 escape 测试的最小 config 被新校验先拒绝 | 复用函数改下划线命名；escape 测试 fixture 补合法维度字段（测试意图不变） |
| 2 | — | 88 passed（`tests/unit/integration/perturbgen/` 全套） |

### 迭代五（本轮）：train-mask 训练 —— 9 轮失败修复后成功

| 轮 | 失败点（真实 Traceback 根因） | 修复 |
|---|---|---|
| 1 | `stratified_split` KeyError `cell_type_cellgen_harm`（默认 split_obs 列不存在） | `--split_obs perturbation`（真实列） |
| 2 | encoder 相对路径 FileNotFoundError | 改绝对路径 |
| 3 | wandb pb2 ImportError（protobuf 7.36） | 降级 protobuf 4.25.8（§2.3 组 7） |
| 4 | tensorboard pb2 ImportError（runtime_version 需 protobuf≥5.26） | 降级 tensorboard 2.18.0（§2.3 组 8） |
| 5 | datamodule KeyError `tgt_dataset_t2`（默认 pred_tps=[1,2,3]） | `--pred_tps 1` |
| 6 | collate KeyError `celltype`（var_list 混入 dataset 不存在的键） | `--var_list perturbation` |
| 7 | triton RuntimeError（P40 capability 6.1 < 7.0） | `TORCHDYNAMO_DISABLE=1` |
| **8** | — | **训练成功**：1 epoch、loss 7.34/7.20、664MB ckpt 落盘（162+126 keys） |

### 迭代六（本轮）：真实权重推理 —— 3 轮修复后闭环

| 轮 | 失败点 | 修复 |
|---|---|---|
| 1 | 管道 `\| tail` 吞掉真实退出码（EXIT=0 假象、无产物） | 改重定向到 log 文件读真实 exit 1 |
| 2 | strict restore 失败：训练保存 `transformer.*` vs 推理期望 `pretrained_model.*`（**第 4 坑**） | 写转换器：构造 PerturberTrainer + 288 张量重映射 + 逐元素断言（坑 d 固化入 §2.2） |
| 3 | 转换器自身 2 次路径错误（相对路径基准是 `ref/` 而非 `ref/Perturbgen-src`） | 对齐 M0⑤ driver 的 `chdir(ref/)` 口径 |
| **4** | — | **推理闭环**：25/25 batches、输出 h5ad schema 与 M0⑤ 一致 |

### 迭代七（本轮）：回归与静态门禁 —— 通过（见 §5）

---

## 4. 系统性复核分析：E2E 训练与推理技术要求

### 4.1 主进程（PTM2CellNet 自身）E2E：**满足工程要求（维持 `[A1]` §3.1 判定）**

- 训练链路：`scripts/train*.py`、`finetune_davf*.py`（含 `--embedding-asset`）可用；本轮对 `src/` 主进程模型代码零改动（仅 PerturbGen 集成层 `config_builder.py`）。
- 推理链路：`predict*.py` + FastAPI + DAVF schema v2 行为不变。
- 门禁：全量离线测试 **2328 passed / 16 skipped / 0 failed / 567.02s**（v1.0 基线 2321，+7 为本轮新增契约测试）；`ruff check src scripts tests` 全过；`compileall` 0 错；`mypy src` 23 errors in 8 files 与 `[R3]` §4.1 基线一致、0 新增。

### 4.2 PerturbGen 独立链路 M0 六项 evidence（对照 `[R3]` §4.2 / `[R2]` §7.2 M0 done 条件）

| M0 evidence | v1.0 状态 | 本轮后状态 |
|---|---|---|
| ① 真实 encoder checkpoint + tensor key | ✅ | ✅ 并新增「foundation 在真实训练中被 strict 加载」证据（train-mask 走 scmaskgitwrapper strict 路径成功） |
| ② token↔row 对齐 | ✅ 端到端实测 | ✅ 维持（本轮训练/推理沿用同一词表闭环） |
| ③ Python 3.11 独立环境 | ✅ | ✅ 锁定清单升级为 8 组依赖 + 2 个版本天花板 + evidence 重采（§2.3） |
| ④ 离线初始化/inference | ✅ 真实 inference | ✅ 维持（本轮训练亦全程离线三变量） |
| ⑤ 真实 perturb smoke + 资源基线 | ✅（decoder 随机） | ✅ **升级**：decoder 换真实训练权重后同 schema 输出（§2.4） |
| ⑥ 合规 normal/disease donor 队列 | ❌ 阻塞（待审计） | ❌ **审计完毕后仍阻塞，且收敛为纯外部数据缺口**：本地 30 文件 0 合规候选（字段级证据 `20260823_donor_audit/evidence.json`），M1 preflight 对本地唯一 raw-counts 候选真实执行并 REJECTED（§2.1） |

**M0 判定**：5/6 完成；Gate-0 维持 BLOCKED（`[R2]` §7.3），剩余缺口**唯一**且为外部数据依赖（donor 队列）。

### 4.3 M2（外部环境调度与可恢复执行）工程前置状态

| M2 子项（`[R2]` §7.2） | 状态 |
|---|---|
| train-mask 官方训练链路真实执行（含 foundation strict 加载、wandb offline、GPU） | ✅ 本轮（§2.4） |
| 训练产物 → val.py restore 的 ckpt 转换 | ✅ 本轮实测（288/288 张量映射）——**转换逻辑目前在 spike 流程中，未固化为 runner stage**（§5 遗留） |
| 上游坑 (a)(b)(c) 契约校验固化 | ✅ 本轮（§2.2，7 项测试） |
| 坑 (d)（前缀转换）与训练入口三坑（默认列/pred_tps/P40 torch.compile）的固化 | 🟡 evidence 已记录，runner stage 化待 M2 正式实现（§5） |
| train-decoder（count 路径，产出 true/pred_counts layers） | ❌ 未开始（masking 路径已闭环；count 路径是 `[R2]` 模板 required_layers 的来源） |
| 全指纹 manifest / GPU lock / dry-run 估算（`[R2]` §7.2 M2 完整范围） | 工程实现已存在（`runner.py`/`env_guard.py`），**真实数据全流程演练**待合规队列 |

### 4.4 DAVF × PerturbGen 集成链路（M4 前置）

| 子项 | 状态 |
|---|---|
| schema v2 / 注入链路 / checkpoint 版本化 / 脚本参数（`[A1]` §1.2 全部项） | ✅ 维持（本轮零改动，全量回归通过） |
| runtime 默认链路切换 | ⛔ 有意不做（`[R2]` §7.3 Gate-0 未过） |
| decoder 端真实权重 | ✅ **工程存在性已证明**（本轮 1-epoch 产物即 `ckpt_masking_path` 合法输入）；科学可用版本仍依赖合规队列 + 正式训练 |
| DAVF 重训 + Gate-E | ❌ 未开始（依赖队列，`[A1]` §4 排期不变） |

### 4.5 结论

> 主进程工程 E2E 全部满足；PerturbGen 侧 **M0 工程五项全部闭环且其中两项（⑤、①训练侧）在本轮升级为真实训练权重级证据，M2 训练→推理工程链路已打通**（原排期 08-31 的核心工程验证提前完成）；剩余阻塞收敛为**唯一外部数据缺口**（M0⑥ donor 队列，本轮已用字段级审计+真实 preflight 排除本地全部候选）；M4/Gate-E 按 `[R2]` §7.3 停止条件有意等待。

---

## 5. 验证结果（可复现命令）

```bash
# 1. donor 审计 + M1 preflight（主环境，本轮实测 NO_LOCAL_COMPLIANT_COHORT）
python scripts/audit_perturbgen_cohort.py
# 产物：outputs/perturbgen/spike/20260823_donor_audit/evidence.json

# 2. M2 三坑契约测试（本轮实测 7 项新用例全过）
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest tests/unit/integration/perturbgen/ tests/integration/test_perturbgen_pipeline_mocked.py -q

# 3. train-mask 最小训练（独立环境 + GPU，本轮实测 1 epoch 成功；注意 TORCHDYNAMO_DISABLE）
cd ref/Perturbgen-src && TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 WANDB_MODE=offline \
  /home/scu/anaconda3/envs/perturbgen/bin/perturbgen train-mask \
    --train_mode masking --output_dir T_perturb/res/datlinger2021_m0smoke_trainmask \
    --src_dataset T_perturb/tokenized_data/datlinger2021_m0smoke/dataset_2000_hvg_src/control.dataset \
    --tgt_dataset_folder T_perturb/tokenized_data/datlinger2021_m0smoke/dataset_2000_hvg_tgt \
    --src_adata T_perturb/tokenized_data/datlinger2021_m0smoke/h5ad_pairing_2000_hvg_src/control.h5ad \
    --tgt_adata_folder T_perturb/tokenized_data/datlinger2021_m0smoke/h5ad_pairing_2000_hvg_tgt \
    --mapping_dict_path T_perturb/tokenized_data/datlinger2021_m0smoke/token_id_to_genename_2000_hvg.pkl \
    --batch_size 16 --epochs 1 --ckpt_every_n_epochs 1 --pred_tps 1 \
    --split_obs perturbation --var_list perturbation \
    --encoder scmaskgit \
    --encoder_path /home/scu/PTM2CellNet/perturbgen_ckpt/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt \
    --cellgen_lr 5e-05 --cellgen_wd 1e-06 --mask_scheduler pow \
    --pos_encoding_mode time_pos_sin --seed 42 --wandb_mode offline
# 注意：退出码必须脱离管道读取（| tail 会返回 tail 的退出码）

# 4. 真实权重推理（独立环境 + GPU；ckpt_masking_path 用 §2.4 转换产物）
cd ref/Perturbgen-src && TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 WANDB_MODE=offline \
  /home/scu/anaconda3/envs/perturbgen/bin/python -m perturbgen.Perturb.val \
    --config /home/scu/PTM2CellNet/outputs/perturbgen/spike/20260823_m2_trainmask/perturb_eval_trained.yaml

# 5. 全量离线回归（本轮实测 2328 passed / 16 skipped / 567.02s / 0 failed）
unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest tests -q -p no:cacheprovider

# 6. 静态门禁（本轮实测全过；mypy 23 errors 为既有基线，0 新增）
python -m compileall -q src scripts tests && ruff check src scripts tests && python -m mypy src

# 7. 独立环境复建（幂等，含 8 组依赖锁定 + 版本天花板）
bash scripts/setup_perturbgen_env.sh
```

---

## 6. 未解决问题及后续解决策略

| 优先级 | 问题 | 策略与实施步骤 | 时间节点 |
|---|---|---|---|
| 高 | M0⑥ donor 合规队列（**唯一 Gate-0 缺口，纯外部数据依赖**） | 本地已排除（§2.1 字段级审计 + preflight REJECTED）；按 `[R4]` §6 策略②外部获取：raw counts、无版本 ENSG、cell type/state、明确 donor 标注（禁 sample/replicate/cell_line 冒充）、normal/disease 配对、≥3 共享 donor 的 AnnData；到手后先跑 `scripts/audit_perturbgen_cohort.py`（已具备该数据集字段审计能力，需按新数据集微调路径）+ M1 preflight | 2026-08-26～08-30（M1 窗口；外部依赖） |
| 高 | 坑 (d) 转换逻辑与训练入口三坑的 runner stage 固化 | M2 runner 实现时：① 把 `transformer.* → pretrained_model.*` 转换固化为 train_mask 与 perturb 之间的 stage（含 288 张量逐元素断言）；② 把 `--split_obs/--var_list/--pred_tps` 真实列校验与 `TORCHDYNAMO_DISABLE=1`（P40）写入 `config_builder.py`/runner 契约；③ 补坑 (d) 契约单测（ckpt 前缀不变式） | 与 M2 正式实现并行（2026-08-31～09-03 窗口内） |
| 中 | train-decoder（count 路径）未执行 | masking 路径已闭环（§2.4）；count 路径产出 `true_counts/pred_counts` layers（模板 required_layers 的来源），按 `[R2]` §7.2 M2 在合规数据上执行 `train-decoder` + 重跑 val.py 验证 layers schema | M2 窗口（数据到位后） |
| 中 | DAVF 重训 + Gate-E | 维持 `[A1]` §4 计划：`embedding_asset_path` 指向真实资产重训、≥200 PTM→gene 基准、bootstrap CI | 2026-09-10～09-16（M4） |
| 中 | Gate-4 真实资产 evidence / M5 | 双路径真实 smoke + P50/P95/RSS/显存冻结阈值；以 `[R4]` §2.1 P40 基线为首版参照（10k cells P95<30s 门需 batch≥64 重测） | 2026-09-17～09-21 |
| 低 | Geneformer 主链物理删除（M7） | Gate-E + Gate-5 双过后执行（`[A1]` §4 原计划不变） | 2026-10-05～10-06 |
| 低 | mypy 23 基线 / 共享环境 pip check / 90s daemon 监控 | 维持 `[A1]` §4 低优先级策略不变 | 并行收尾 |

---

## 7. 最终判定

| 判定项 | 状态 |
|---|---|
| M0⑥ donor 审计（30 文件字段级 + M1 preflight 真实执行） | ✅ 完成（结论：本地 0 候选，外部获取为唯一路径） |
| M2 上游三坑契约固化（7 项新测试 + 模板活体坑修复） | ✅ 完成 |
| 上游第 4 坑（ckpt 前缀不匹配）确证与转换实测 | ✅ 完成（288/288 张量逐元素断言） |
| 独立环境锁定快照（8 组依赖 + 版本天花板 + evidence 重采） | ✅ 完成 |
| M2 最小 train-mask（官方 CLI 真实训练 + 产物转换 + 真实权重推理闭环） | ✅ 工程闭环（科学有效性不宣称） |
| 主进程回归（全量 2328 passed / 0 failed + 静态门禁全过 / mypy 0 新增） | ✅ 完成 |
| M0⑥ donor 合规队列本体 | ❌ 阻塞（外部数据；本地候选已全部字段级排除） |
| train-decoder count 路径 / DAVF 重训 / Gate-0 / Gate-E / Gate-4 / Gate-5 | ❌ 按停止条件有意等待（`[R2]` §7.3） |
| 报告 | ✅ 本文件 v2.0（覆盖同日 v1.0；v1.0 交付已在 §2 继承标注） |

**本轮边界声明**：不把「1 epoch 小样本训练权重的推理输出」写成「PerturbGen 科学有效」；不把「本地 30 文件审计 0 候选」写成「donor 队列不可能获得」（是外部获取问题，不是可行性问题）；不把「M2 工程链路闭环」写成「Gate-0/Gate-2 通过」。

**文档版本**：v2.0（2026-08-23）
