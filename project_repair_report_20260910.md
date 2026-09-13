# PTM2CellNet 代码修复报告（2026-09-10）

> 本报告依据权威分析 [`project_analysis_20260910.md`](project_analysis_20260910.md)
> 执行修复，覆盖其 §4.1 登记的 N-01～N-05 未实现功能项与 §6.2 登记的高/中级
> 技术债（TD-NEW-01～16），并在修复后对 E2E 训练与推理的技术要求做系统性复核。
> 需求契约以
> [`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
> （下称"方案"）为准；主线架构以 `lessons.md` L-2026-0901-01 为准。

## 目录

- [1. 执行摘要](#1-执行摘要)
- [2. 问题修复汇总（按严重程度）](#2-问题修复汇总按严重程度)
- [3. 迭代执行情况与结果](#3-迭代执行情况与结果)
- [4. 验证证据](#4-验证证据)
- [5. 系统性复核：E2E 训练与推理技术要求](#5-系统性复核e2e-训练与推理技术要求)
- [6. 未解决问题与后续策略](#6-未解决问题与后续策略)
- [7. 文档引用索引](#7-文档引用索引)

## 1. 执行摘要

本轮按分析报告 §9.3 建议顺序完成 **5 项未实现功能 + 4 项高级技术债 + 6 项中级技术债 + 判定链两条腿**的修复：

| 维度 | 修复前 | 修复后 |
|---|---|---|
| 主线衔接（N-01～N-05） | 5 项零代码，候选 JSON 靠手写 | 4 个库模块 + 4 个 CLI 全部落地，含 57 项新单元/集成测试 |
| 正式判定链（方案 §4.7） | 多 seed、pad/delete 敏感性无编排端 | `--seeds`/`--sensitivity-modes` 进入 E2E CLI，上游 `val.py --seed` 已接线，每 (path, mode, seed) 独立输出目录 |
| mypy | 80 errors / 17 文件（TD-NEW-16） | **0 errors / 160 文件**（优于 23 基线） |
| Geneformer 词汇语义 | `{str(i):i}` 哈希陷阱（TD-NEW-02） | vocab.json/官方 token_dictionary 加载，未知基因抛 `KeyError`，哈希写回删除，缺词汇 fail-fast 不降级 |
| UniProt 映射 | >500 基因静默截断 + organism 静默忽略 | Link 游标分页拉全页；非 9606 显式 `NotImplementedError` |

全量离线回归与静态检查结果见 §4。系统性复核结论（§5）：**工程工具面 E2E 已闭合；
科学执行面仍被外部资产（合规 donor cohort、Gate-E 基准集）与剩余 3 项代码缺口
（matched null 生成端、`davf_score` 生产、旁路 CLI 收敛）阻断**。

## 2. 问题修复汇总（按严重程度）

### 2.1 未实现功能（分析 §4.1，全部完成）

| # | 功能 | 实现 | 关键文件 | 测试 |
|---|---|---|---|---|
| N-01 | PTM site 预测 CSV → `PTMSiteDirectionProposal` 自动转换器（§4.1 N-1；方案 §4.1 工作流 A） | `load_ptm_site_predictions`/`load_gene_map`/`load_direction_map` + 显式 `--proposed-direction`（或位点级 direction map CSV）；方向主张绝不从 observed 证据推导，避免三方 gate 循环验证 | `src/integration/perturbgen/candidate_spec.py`、`scripts/build_candidate_spec.py` | `test_candidate_spec.py` 16 项 |
| N-02 | `direction_evidence.csv` → gate `observed_*` 自动 join（§4.1 N-2；方案 §4.3） | `load_direction_evidence` 校验（符号一致性、FDR 域、(cell_type, ensembl) 唯一）+ `build_spec_candidates` 按 ensembl+cell_type join；neutral/无证据/未映射逐项跳过并审计；donor<3 显式标记 `low_donor_support` | 同上 | 同上 |
| N-03 | Gate-E 词表迁移与非劣基准评估（§4.1 N-3；方案 §5.4、§7.2 M4） | 基准集契约（≥200 样本 + sha256 冻结）、词表迁移（覆盖率≥99%/token 碰撞=0/action code 100% 不变，经 `PTMDirectionMapper` 端到端比对）、DAVF 非劣（配对 bootstrap 95% CI，下降≤1pp 且 CI 不劣化） | `src/integration/perturbgen/gate_e.py`、`scripts/evaluate_gate_e.py` | `test_gate_e.py` 16 项 |
| N-04 | M6 冻结队列科学验收编排（§4.1 N-4；方案 §7.2 M6） | 冻结 cohort manifest（h5ad sha256 + train/held-out donor 划分 + 候选计划）、donor 泄漏审查（train∩heldout=∅ 且 ≥3 held-out）、验收矩阵（候选×路径×seed×mode）、独立重算（从报告 manifest 重放 `evaluate_dual_path_candidate` + 重算 BH-FDR 对比 verdict） | `src/integration/perturbgen/frozen_cohort.py`、`scripts/run_frozen_acceptance.py` | `test_frozen_cohort.py` 14 项 |
| N-05 | E2E 输出 → dual-path 评估输入组装器（§4.1 N-5；方案 §7.2 M3） | 从成功 perturb stage manifest 绑定 h5ad（status/artifacts/result_h5ad），解析上游文件名 `g{gene}_s{src\|tgt}_t{mode}` 并交叉校验 path↔sequence，seed 从 manifest `fingerprint_material.random_seed` 读取；DEG 表/null 分布/p 值/质量门状态必须显式提供（E2E 报告不含这些证据，禁止编造） | `src/integration/perturbgen/eval_assembly.py`、`scripts/build_dual_path_eval_input.py` | `test_eval_assembly.py` 14 项 |

### 2.2 判定链补腿（分析 §5.4 ④①、§9.3-1，完成 3 条中的 2 条）

| 缺口 | 修复 |
|---|---|
| KO 三模式断点：E2E 只产 `mask`，正式判定要求 mask/pad/delete 至少 2/3 同方向（方案 §4.7 条件 5） | `build_candidate_stage_plans` 增加 `sensitivity_modes`；KO+mask 主模式时可展开 pad/delete；`PerturbGenInvocation` 增加 `is_sensitivity`/`seed` 字段（敏感性仅限 KO pad/delete，主模式缺失即拒绝） |
| 多 seed 无编排端（方案 §4.7 条件 4） | `--seeds` 进入 e2e CLI；`_apply_mode_and_seed` 写入 stage `seed` 并经 `config_builder` 传上游 `val.py --seed`（上游原生支持，`ref/Perturbgen-src/scmaskgit/val.py:221`）；`pipeline.random_seed` 进 fingerprint（不同 seed 不可复用 resume）；每 (path, mode, seed) 独立输出目录保证 artifact 发现唯一 |
| matched null 生成端 | **未修复**（见 §6） |

### 2.3 高级技术债（分析 §6.2，4/4 完成）

| 编号 | 修复 |
|---|---|
| TD-NEW-01 UniProt REST 单页 500 条截断 | `_RequestsUniProtMapper.get` 跟随响应 `Link rel="next"` 游标循环拉全页（`src/analysis/gene_mapper.py`）；新增 501 基因两页合并测试 |
| TD-NEW-02 Geneformer 词汇无语义 + 哈希写回 | 新增 `_load_vocabulary`：safetensors/HF 三分支均加载 `vocab.json`（或官方 `geneformer/token_dictionary_gc104M.pkl`），越界 token 拒绝；`GeneformerVocabularyError` 不落入随机回退（fail-fast）；真实词汇下未知基因 `get_gene_embedding`/`create_gene_id_mapping` 抛 `KeyError`；fallback 模式哈希保留但**不再写回词汇表** |
| TD-NEW-03 organism 参数静默忽略 | `organism` 透传 `_get_mapper_results`；非 9606 在两个后端统一 `NotImplementedError`（REST idmapping 无 taxId 过滤能力，返回人类数据是契约欺骗） |
| TD-NEW-16 mypy 80→基线 | **80 → 0**（清零 9 月新增 57 + 旧基线 23 内的 api/routes 5 处：`Optional` 收窄、`cast`、Literal 类型化、变量遮蔽重命名；全部为类型层修改，运行时行为由 2544+ 回归守护） |

### 2.4 中级技术债（分析 §6.2，6/9 完成）

| 编号 | 修复 | 状态 |
|---|---|---|
| TD-NEW-04 跨模块私有属性访问 | `GeneformerEmbeddingLoader.gene_to_idx` 与 `DAVFInferenceModule.embedding_symbol_to_ensembl` 公开 property；mapper 优先公开 property（Mapping 类型校验）；orchestrator 要求公开访问器否则显式报错 | 完成 |
| TD-NEW-06 常量三处定义 | `gse_normal_disease.py`/`davf_scperturb.py` 改 import `src/models/davf_checkpoint_contract.FORMAL_DAVF_NUM_GENES`（data→models 方向无环）；`test_dimensions.py` 加同源断言 | 完成 |
| TD-NEW-08 PTM loader 三胞胎 | `_load_generic_ptm_table` 统一 download/parse/fallback 样板，三个公开方法变为参数化委托（API 不变） | 完成 |
| TD-NEW-09 路径猜测链 | `_resolve_model_path` 锚定 `checkpoint_base_dir`（默认项目根 `outputs/ptm_pretrain`，不再依赖 cwd）；保留两个规范产物名与显式 `model_path` 回退（显式参数非猜测），回退时 loud warning | 完成 |
| TD-NEW-10 iterrows 热路径 | `predict.py` 批量行循环、`predict_ptm_sites.py` 变体循环改 `to_dict("records")`；`ESMTokenizedDataset` 预 tokenize 缓存加上限（200k 行）+ 惰性按需填充 | 部分完成（`pmads_ridge._feature_matrix` 向量化未做，见 §6） |
| TD-NEW-05 可选依赖四风格统一 | 未做（分析 §6.3 标注分批清偿项） | 未做 |
| TD-NEW-07/15 超长函数拆分 | 未做（同上） | 未做 |

## 3. 迭代执行情况与结果

本轮为单主线多批次迭代，每批次独立验证：

| 轮次 | 内容 | 即时验证 | 结果 |
|---|---|---|---|
| 1 | 探索：方案文档契约（§4/§5/§7）+ 主线代码数据格式（e2e CLI/gate/orchestrator/评估 CLI/两 CSV 生产端） | — | 确认 N-01~N-05 全部接口契约与既有测试风格；子代理因环境不可用改为主上下文直读 |
| 2 | N-01+N-02 转换器与 join | `pytest test_candidate_spec.py` | 16 passed（中途修正 2 处测试预期：过滤顺序短路语义） |
| 3 | N-05 组装器 | `pytest test_eval_assembly.py` | 11 passed |
| 4 | N-03 Gate-E 工具 | `pytest test_gate_e.py` | 16 passed（修正浮点边界容差 1e-9 与 `--min-samples` 透传） |
| 5 | N-04 M6 编排 | `pytest test_frozen_cohort.py` | 14 passed（修正 replay 测试的 path_result 结构与三模式齐全性） |
| 6 | TD-NEW-01/03 gene_mapper | `pytest test_gene_mapper.py` | 31 passed（修复测试 Mock 缺 `links` 属性导致的分页循环误判） |
| 7 | TD-NEW-02 geneformer 词汇 | `pytest test_geneformer_embedding.py` + DAVF/mapper 四套件 | 19 + 84 passed（发现并接入官方 `token_dictionary_gc104M.pkl` 真实文件名；mapper 公开 property 加 Mapping 校验兼容 Mock fixture） |
| 8 | 判定链两腿（seed/敏感性） | `pytest test_perturbgen_pipeline_mocked.py` + perturbgen 全套件 | 190 passed（修正一处无效语法与 payload 结构） |
| 9 | TD-NEW-16 mypy | 逐文件 `mypy` + 对应单测 | 80→0；相关套件 262+96 passed |
| 10 | TD-NEW-08/09/10 | loader/variant/dataset/predict 测试 | 2+65+81+199+245 passed |
| 11 | 全量回归 #1 | 全量 pytest | 2544 passed / 6 failed——6 个失败全部为本轮引入（3 个 fixture 用 `__new__` 绕过 `__init__` 未初始化新增公开 property、1 个旧哈希写回守护测试、2 个同因参数化变体），当轮修复 fixture 并同步测试期望 |
| 12 | 全量回归 #2（收口） | 全量 pytest + 五项静态检查 | **2550 passed / 15 skipped / 0 failed**（791.72s，exit 0）；mypy 0 / ruff 全绿 / compileall / requirements OK |

## 4. 验证证据

| 检查 | 命令 | 结果 |
|---|---|---|
| 编译 | `python -m compileall -q src scripts tests` | 通过（exit 0） |
| 静态检查 | `ruff check src scripts tests`（0.15.15） | All checks passed |
| 格式 | `ruff format`（本轮新增/重写的 12 个文件） | 已格式化 |
| 类型检查 | `python -m mypy src/ --ignore-missing-imports` | **Success: no issues found in 160 source files**（修复前 80 errors / 17 files） |
| 依赖契约 | `python scripts/check_requirements_consistency.py` | OK（274 lock pins） |
| 聚焦回归 | perturbgen/geneformer/gene_mapper/DAVF/variant/dataset/predict/api 各套件 | 全部通过（§3 各轮） |
| 全量离线回归 | `python -m pytest -m "not slow and not gpu and not real_assets" --timeout=600 -q` | 见下方最终结果 |
| CLI 冒烟 | `build_candidate_spec.py`（合成 PTM CSV + gene map + GSE evidence） | 产出合法 candidate spec（含 donor 审计字段）；`evaluate_gate_e.py` 冒烟通过 |

**全量回归最终结果**：

```
2550 passed, 15 skipped, 7 deselected, 54 warnings in 791.72s (13:11), exit 0
```

对比 2026-09-10 基线（2475 passed / 15 skipped，`docs/CURRENT_STATUS.md`）：净增
75 个通过用例（本轮新增 61 项 N-01~N-05/判定链/词汇语义/分页测试 + 既有套件
受修复影响的用例）。第一轮全量曾出现 6 个失败，全部为本轮修改引入（3 个测试
fixture 用 `__new__` 绕过 `__init__` 未初始化新增的公开 property、1 个旧哈希
写回守护测试、2 个同因参数化变体），修复 fixture 与同步测试期望后第二轮全量
零失败。

证据边界：以上为离线工程契约证据（synthetic fixture/mock 口径）；真实跨尺度数据、
DAVF 生物学方向准确率、外部 UniProt/KEGG、PerturbGen 外部环境与多 GPU DDP 验收
仍需 `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1` 与真实资产（同分析报告 §7.2 边界）。
本轮未执行真实资产测试与 `--cov` 覆盖率刷新，不宣称生物学验收。

## 5. 系统性复核：E2E 训练与推理技术要求

按方案 §4/§5/§7 与分析 §5.1 完成度口径逐项复核：

### 5.1 已满足项

| 要求 | 方案出处 | 现状证据 |
|---|---|---|
| 主线五环自动衔接：PTM site → proposal → 三方 gate → invocation → 评估重放 | §4.1 工作流 A；§7.2 M3 done | 本轮 N-01~N-05；`run_davf_perturbgen_e2e.py` → `build_dual_path_eval_input.py` → `evaluate_perturbgen_dual_path.py` 全链有代码与测试 |
| 三方方向 gate 强制（PTM proposal/DAVF/独立表达证据一致才 pass） | §4.3；L-2026-0902-02 | `direction_gate.py` 既有逻辑未动，回归全绿 |
| 正式判定链统计内核（排目标基因 rescue、donor 一致率、加一经验 p、BH-FDR、严格 AND 三态） | §4.7 | `results.py`/`dual_path.py` 既有；本轮补 `PathKind`/`PerturbationMode` 类型化 |
| 多 seed（≥3）与 KO pad/delete 敏感性编排 | §4.7 条件 4/5 | 本轮补腿：`--seeds`/`--sensitivity-modes` + 上游 `--seed` 接线 + fingerprint/输出目录隔离 |
| M6 冻结验收契约（manifest、泄漏审查、独立重算、bootstrap CI） | §7.2 M6 | 本轮 N-04 |
| Gate-E 工具（词表迁移三指标 + DAVF 非劣 CI） | §5.4；§7.2 M4 | 本轮 N-03 |
| 数据预检（raw counts/ENSG/≥3 donor/resolver 禁哈希） | §5.2 | `data_prep.py`/`gene_vocabulary.py` 既有；本轮 mypy 清零后契约不变 |
| scVI decoder index 唯一来源、token/decoder index 分离 | L-2026-0902-01 | `scvi_adapter.py` 既有；`predict_expression_direction` 强校验回归全绿 |
| 真实 KO/KD LatentDAVF checkpoint（schema v2） | L-2026-0902-01 | `checkpoints/davf/davf_{ko_dixit,kd_nadig}` 既有；本轮在 schema v2 分支增加 `LatentDAVF` 类型断言 |

### 5.2 未满足项与根因

| # | 缺口 | 严重度 | 根因 | 证据 |
|---|---|---|---|---|
| G-1 | **matched null 生成端**：≥99 个表达量/FC/检测率/token rank 匹配 null 的候选选择与批量运行无代码；eval CLI 只消费外部 `null_distribution_path` | 高（正式 PASS 必需） | 9 月批次只交付统计消费端，生成端属未排期范围 | 全仓 grep `matched_null`/`null_distribution` 仅消费端（`results.py:398-430`、`dual_path.py`、`evaluate_perturbgen_dual_path.py:196`） |
| G-2 | **`davf_score` 空转**：`CandidateEvidence.davf_score` 恒 None、`DAVFDirectionEvidence.confidence` 从不填充 | 中 | 方案 §4.3 契约要求 score 随候选记录，但 DAVF 侧无置信度推导实现 | `davf_inference.py` 无 confidence 生产；`direction_gate.py:164`（修复前行号） |
| G-3 | **gate 旁路面**：`run_perturbgen_pipeline.py` 可从任意 YAML 直跑六阶段；库级手工构造 `PerturbGenInvocation` 不校验 gate | 中（规范执行不一致） | 旁路 CLI 服务既有六阶段重放功能，收敛需产品决策 | 分析 §5.5 三条绕过面，本轮未动 |
| G-4 | **合规 donor cohort = 0**：30 个本地 scPerturb h5ad 无一满足正式契约 | 阻塞性（科学验收） | 外部数据资产由用户提供（AGENTS 约束） | `outputs/perturbgen/spike/20260903_donor_audit/evidence.json`；L-2026-0822-06 |
| G-5 | **Gate-E 基准数据缺失**：≥200 可追溯 PTM→gene 样本与旧冻结基线未提供 | 中（M7 前置） | 外部资产；工具已就绪 | 本轮 N-03 工具；方案 §5.4 |
| G-6 | T4 科学验收 release-gate 测试不存在 | 中 | 依赖 G-4 冻结 cohort | 方案 §5.1 |
| G-7 | `pmads_ridge._feature_matrix` iterrows 未向量化 | 低 | 需 golden 数值等价测试先行（防行为漂移） | TD-NEW-10 余项 |
| G-8 | TD-NEW-05/07（可选依赖统一、超长函数） | 低 | 分批清偿项 | 分析 §6.3 |
| G-9 | dual_path 条件 2 严格度高于文档（单 seed rescue≤0 即 fail vs 文档 median>0） | 低 | 实现从严，文档未同步 | 分析 §5.3；`dual_path.py` |

### 5.3 复核结论

1. **工程工具面**：E2E 训练与推理的代码链（数据契约 → DAVF gate → 双路径编排 → 统计判定 → 报告重放 → 冻结验收）在工具层已无断点；N-01～N-05 与判定链两腿补齐后，从 PTM 位点预测 CSV 到双路径评估输入可全程无手写 JSON。
2. **科学执行面**：正式生物学 PASS 仍不可能在当前本机达成——G-4（合规 cohort）是硬阻塞，G-1（null 生成端）与 G-2（davf_score）是代码侧剩余必要条件。与 L-2026-0902-03 的"真实桥接通过不等于生物学验收"口径一致。
3. **质量基线**：mypy 0 错误、ruff 全绿、全量回归见 §4；类型修复未引入行为变化（由回归守护），但 Geneformer 词汇语义修复是**行为变更**（未知基因由静默哈希改为抛错/降级显式标记），真实资产工作流需按 §2.3 提供 vocab.json。

## 6. 未解决问题与后续策略

| 优先级 | 问题 | 策略与实施步骤 | 预估 | 前置 |
|---|---|---|---|---|
| 1 | G-1 matched null 生成端 | 新增 `null_selection.py`：按表达量/FC/检测率/token rank 分层匹配从 cohort 选择 ≥99 null 基因 → 复用 `build_candidate_stage_plans`（mode=mask、排除候选基因）批量编排 → null rescue 汇总输出 `null_distribution.json` → 集成进 N-05 组装器自动绑定 | 1.5 天 | G-4（执行）；选择器与 dry-run 契约可先建 |
| 2 | G-4 合规 cohort | 用户提供满足 normal/disease + raw counts + 显式 donor + ≥3 共享 donor + Ensembl 的队列；接入走既有 `prepare_gse_normal_disease.py` + `summarize_gse_directions.py` + N-01/N-02 新链路 | 外部依赖 | — |
| 3 | G-2 davf_score 生产 | `predict_expression_direction` 输出方向置信（如 |delta| 归一化或预测分布熵）→ 填充 `DAVFDirectionEvidence.confidence` → gate 记入 `davf_score`；补契约测试 | 0.5 天 | — |
| 4 | G-3 旁路收敛 | `run_perturbgen_pipeline.py` 增加默认 gate 证据校验（`--require-direction-gate`，含 perturb 的计划无 e2e invocation 即拒绝）；库级 `run_perturbgen` 校验 invocation 来源 | 1 天 | 产品确认默认口径 |
| 5 | G-7 pmads_ridge 向量化 | 先写 golden 数值等价测试（固定种子对照 iterrows 输出）→ 向量化 `_feature_matrix` → 对拍 | 1 天 | — |
| 6 | G-5/G-6 Gate-E 数据 + T4 | 基准集由用户构建（≥200 可追溯 PTM→gene 样本，GeneMap/CPTAC 来源）→ 跑 `evaluate_gate_e.py` → 通过后执行 M7 退役 Geneformer；T4 测试随冻结 cohort 落地 | 外部依赖 + 0.5 天 | G-4 |
| 7 | G-8/G-9 | TD-NEW-05 按 `LazyImport` 约定分 3~4 个小 PR；TD-07 按 `train.py` main 四段拆分；`dual_path` 严格度差异在方案 §4.7 勘误标注 | 2~3 天（分批） | — |

时间节点建议：代码侧 1/3/4/5 合计 4 天内可完成（无外部依赖）；G-1 执行与 G-5/G-6
验收随合规 cohort 到位后 1 周内收口。

## 7. 文档引用索引

| 引用 | 章节/条目 | 本报告用途 |
|---|---|---|
| `project_analysis_20260910.md` | §4.1 未实现功能模块表（N-1~N-5） | §2.1 修复范围 |
| 同上 | §4.2 缺失接口清单 | §2.1 接口设计依据 |
| 同上 | §4.3 未实现业务流程图 | §5.1 链路复核 |
| 同上 | §5.4 三分类清单（实现有缺陷项） | §2.2 判定链补腿 |
| 同上 | §6.2 新增债务清单（TD-NEW-01~16） | §2.3/§2.4 |
| 同上 | §6.3 高/中等级债务解决策略 | 各项修复方案选型 |
| 同上 | §9.2 不能宣称完成 / §9.3 建议顺序 | §5.3 复核结论、§6 策略排序 |
| `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | §4.1 两条工作流 | §2.1/§5.1 |
| 同上 | §4.3 候选与方向契约（`CandidateEvidence`/方向规则表） | §2.1 N-02 匹配规则、§6 G-2 |
| 同上 | §4.7 rescue 与正式判定（条件 1-7） | §2.2 判定链、§5.2 G-1 |
| 同上 | §5.1 测试矩阵（T1-T5） | §5.2 G-6 |
| 同上 | §5.4 科学质量门（Gate-E 标准） | §2.1 N-03 阈值 |
| 同上 | §7.2 M3/M4/M6 阶段定义 | §2.1 N-03/N-04 验收口径 |
| `lessons.md` | L-2026-0901-01（DAVF 方向筛选/PerturbGen 效用分工） | §1 主线口径 |
| 同上 | L-2026-0902-01（LatentDAVF/scVI/asset 契约） | §5.1 已满足项 |
| 同上 | L-2026-0902-02（串联门控不可绕过） | §5.2 G-3 |
| 同上 | L-2026-0902-03（真实桥接≠生物学验收） | §5.3 结论 2 |
| 同上 | L-2026-0822-06（通用 scPerturb 不能冒充冻结队列） | §5.2 G-4 |
| `docs/CURRENT_STATUS.md` | 测试基线与验证环境（2026-09-10 更新） | §4 证据边界 |

---

**报告方法说明**：修复由主上下文单线执行（计划中的 3 个探索子代理因运行环境
`Model provider is not configured` 失败，改为主上下文直读代码）；每轮修改后运行
聚焦回归，最终以全量离线回归 + ruff + mypy + compileall + requirements 五项收口。
本轮未使用外部 skill；未执行 git 提交（工作树保留待用户审阅）。
