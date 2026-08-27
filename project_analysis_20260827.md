# PTM2CellNet 项目代码与文档综合审计报告

> 审计日期：2026-08-27<br>
> 审计范围：当前工作树、main 分支、origin/main、src、scripts、tests、docs、.planning、outputs 与历史报告<br>
> 当前结论：工程基线可复现，但 DAVF × PerturbGen 的真实科学闭环仍未完成；本报告不把 synthetic、mock 或 zero fallback 证据写成真实生物学验收。

## 目录

- [1. 任务判断与摘要](#1-任务判断与摘要)
- [2. 审计范围、方法与版本基线](#2-审计范围方法与版本基线)
- [3. 文档与检测报告归档](#3-文档与检测报告归档)
- [4. 需求对照与未实现功能](#4-需求对照与未实现功能)
- [5. 部分实现、接口差异与业务流程](#5-部分实现接口差异与业务流程)
- [6. 模块完成度](#6-模块完成度)
- [7. 技术债清单与解决策略](#7-技术债清单与解决策略)
- [8. 验证结果](#8-验证结果)
- [9. 子智能体执行统计](#9-子智能体执行统计)
- [10. 结论与建议](#10-结论与建议)

## 1. 任务判断与摘要

### 1.1 任务类型

本任务同时属于：

| 类型 | 本次交付 |
|---|---|
| 代码审查与技术分析 | 结构调用链、接口契约、测试和真实资产证据审计 |
| 文档处理 | 识别过时材料，归档旧报告，修订活动文档 |
| 配置与版本控制 | 远程同步检查、主分支提交、合并和验证 |
| 测试补充/质量检查 | 执行收集、全量离线测试、编译、lint、依赖一致性和类型检查 |
| 需求转实现审计 | 对照 v2.1/v2.2 正式需求和 PerturbGen 后续提案记录缺口 |

### 1.2 结论摘要

1. v2.1/v2.2 的 24 项正式工程需求仍可按需求文件标记为完成；这只代表工程契约和离线验证完成，不代表真实权重、真实 donor 队列或科学效果验收完成。依据为 [.planning/REQUIREMENTS.md](.planning/REQUIREMENTS.md#L30-L83)。
2. 当前全量离线测试为 2329 passed、16 skipped、47 warnings；收集到 2345 项。编译和 Ruff 规则检查通过，mypy 仍有 23 个错误，依赖环境仍有 3 个冲突。
3. Gate-0 donor 条件未通过：30 个文件中 0 个合规候选；真实 DAVF smoke 的证据显示 model_source=zero_fallback；因此不能声称真实 DAVF 或双路径科学结论已成立。证据见 [donor audit evidence](outputs/perturbgen/spike/20260823_donor_audit/evidence.json#L27-L29)、[real DAVF evidence](outputs/real_assets/test_real_davf_e2e_subprocess.json#L6-L14)。
4. 发现并记录 22 项新技术债（TD-N-35 至 TD-N-56，含本轮已修复的文档债）；其中无“严重”项，存在支持路径确定失败、可能输出错误语义或发布门禁失败的“高”项，以及可通过文档、契约或维护工作解决的“中/低”项。
5. 已把确认过时的 2026-08-24 完整分析报告、修复报告和旧测试覆盖率报告迁入 [archive/20260827/](archive/20260827/)，保留 Git 重命名历史，并在 [MANIFEST.md](archive/20260827/MANIFEST.md) 记录原始版本和迁移依据。

### 1.3 范围边界

以下已取消的内容不列为未实现功能，也不重新规划：实时质谱流、自定义 PTM 数据库、GUI、API key 功能扩展。该边界与 [AGENTS.md](AGENTS.md#L297-L299)、[.planning/REQUIREMENTS.md](.planning/REQUIREMENTS.md#L95-L108) 一致。

## 2. 审计范围、方法与版本基线

### 2.1 证据来源

- 结构关系：CodeGraph 索引；状态健康，392 个文件、8315 个节点、9603 条边。
- 文字和路径：rg、find、stat、nl、Git 状态和历史。
- 需求与计划：.planning/REQUIREMENTS.md、ROADMAP.md、STATE.md、v2.0 phases、PerturbGen 方案文档。
- 运行证据：pytest、compileall、ruff、mypy、pip check、requirements consistency、CLI help 和真实资产只读检查。
- 历史基线：当前 main 中的 project_analysis_20260824.md、project_repair_report_20260824.md 及已有 archive。

### 2.2 版本控制基线

| 项目 | 结果 |
|---|---|
| 审计前分支 | main |
| 审计前 HEAD | a8c480ca97a9902e7f34871358102b18bfebdd3f |
| 远程来源 | origin = git@github.com:valleyhe/PTM2CellNET.git |
| fetch | git fetch origin main 成功 |
| origin/main | c76fff881f268f9bd0b39d68db8ba547115ec4cf |
| main...origin/main | 43 0；本地包含远程全部提交并领先 43 个提交 |
| 冲突 | fetch 和后续 fast-forward 合并均无冲突 |
| 审计后 HEAD | 由最终交付命令 git rev-parse HEAD 记录；未推送远程 |

审计期间所有修改均集中在文档、规划记录、归档清单和本报告；没有擅自创建 worktree，没有修改 src、scripts 或 tests 的生产逻辑。

### 2.3 当前环境

| 项目 | 实测值 |
|---|---|
| Python | 3.12.13，解释器为 /home/scu/anaconda3/envs/SSH_unit/bin/python |
| PyTorch | 2.4.1+cu118 |
| NumPy | 2.4.3 |
| pytest | 9.0.3 |
| Ruff | 0.15.15 |
| mypy | 2.1.0，通过 python -m mypy 调用 |
| CUDA | 可用 |
| 包版本 | ptm2cellnet 1.0.0 |

注意：裸 mypy 和裸 pip 指向另一套 Python 3.10 用户环境。本报告只采信 python -m mypy、python -m pip 的同一解释器结果；这本身已登记为 TD-N-44。

## 3. 文档与检测报告归档

### 3.1 判定规则

- 文档“过时”：内容与当前代码、正式需求或当前状态存在实质差异；仅日期较早不足以判定。
- 报告“过期”：生成时间超过 30 天，或其测试基线、提交状态、技术债和结论已被后续状态取代。
- 已在 archive/ 或 docs/archive/ 的历史材料不重复迁移。
- 活动文档若只是可修正的命令或链接错误，优先修订，不把仍需使用的文档直接归档。

### 3.2 本批次归档清单

| 原路径 | 归档路径 | 原始版本/时间 | 归档原因 |
|---|---|---|---|
| project_analysis_20260824.md | archive/20260827/reports/project_analysis_20260824.md | commit 7e4b66d，2026-08-24 02:34:05 +0800 | 2328/16 旧基线、旧债务列表和旧文档状态被本报告及 2026-08-27 实测取代 |
| project_repair_report_20260824.md | archive/20260827/reports/project_repair_report_20260824.md | commit a8c480c，2026-08-24 20:22:40 +0800 | 点时修复记录已被当前综合审计吸收 |
| docs/TEST_COVERAGE.md | archive/20260827/reports/TEST_COVERAGE_20260809.md | commit 366e691，2026-08-09 18:49:29 +0800 | 旧覆盖率/测试基线不再代表 2026-08-27 的测试状态 |

迁移使用 Git rename，原始历史可由 git log --follow 追溯。归档批次说明、原始 HEAD、远程 HEAD、判定标准和文件列表见 [archive/20260827/MANIFEST.md](archive/20260827/MANIFEST.md)；批次说明见 [archive/20260827/README.md](archive/20260827/README.md)。

### 3.3 保留为活动入口或已修订的材料

| 文件 | 处理 |
|---|---|
| 根目录 20260816、20260820、20260821、20260822、20260823 报告入口 | 保留为历史指针；它们已指向更早 archive，不重复迁移 |
| project_analysis_20260824.md、project_repair_report_20260824.md 根入口 | 保留短指针，指向本批次 archive 和当前 20260827 报告 |
| docs/TEST_COVERAGE.md | 重建为当前测试治理页，旧基线保留归档链接 |
| docs/CURRENT_STATUS.md | 更新为 2026-08-27 基线、当前报告和 Gate 状态 |
| docs/guides/installation.md | 修正核心依赖、extra 和验证命令 |
| docs/guides/quickstart.md | 修正默认 API 前缀为 /api/v1/predict |
| docs/guides/training.md | 修正 --batch_size、移除不存在的 --plot 和 --device |
| docs/guides/perturbgen_bridge.md | 修正资产 schema 说法、当前报告链接和真实 Gate 状态 |
| README.md | 修正 API + Lightning 安装 extra 表述 |
| .planning/PROJECT.md、REQUIREMENTS.md、ROADMAP.md | 区分已完成正式里程碑和未完成的后续科学验收 |

仍发现但没有直接改动的活动文档问题会进入第 5、7 节，包括 AGENTS.md 的旧模块映射、docs/api/*.rst 仅有 automodule 骨架，以及历史 v2.0 UAT/VERIFICATION 记录不完整。它们没有被误归档，因为仍承担当前约束或历史审计作用。

## 4. 需求对照与未实现功能

### 4.1 正式需求状态

| 需求组 | 数量 | 当前状态 | 证据 |
|---|---:|---|---|
| v2.1 TEST/DEPS/CODE/CONF | 13 | 需求层面完成 | [.planning/REQUIREMENTS.md](.planning/REQUIREMENTS.md#L8-L29) |
| v2.2 DATA/BASE/MODEL/VERIFY | 11 | 需求层面完成 | [.planning/REQUIREMENTS.md](.planning/REQUIREMENTS.md#L32-L83) |
| 合计 | 24 | 工程验收完成，真实科学验收不等同完成 | 需求文件与 Phase 15、18、19、20 traceability |

### 4.2 后续提案中未完成项目

下表用 U-01 至 U-14 记录当前活动的 DAVF × PerturbGen 后续方案和真实资产验收缺口。优先级表示对主流程的阻断程度，影响表示业务影响范围。

| 编号 | 未实现或未验收项 | 需求/方案章节与原文摘要 | 优先级 | 影响 | 代码/证据 |
|---|---|---|---|---|---|
| U-01 | 合规的 normal/disease paired donor 队列 | PerturbGen bridge §7：要求 ≥3 donor 合规 cohort；当前标记“0 合规候选” | 高 | 核心功能 | [donor audit evidence](outputs/perturbgen/spike/20260823_donor_audit/evidence.json#L27-L29) |
| U-02 | M4 resolver/Protocol 接入、真实 DAVF 重训和 Gate-E | CURRENT_STATUS：M4 代码前置 6/8，余 2/8 延后 | 高 | 核心功能 | [CURRENT_STATUS](docs/CURRENT_STATUS.md#L20)、[davf_inference.py](src/models/davf_inference.py#L271) |
| U-03 | 真实 PerturbGen release evidence | bridge §7：Gate-4 等真实资产 workflow 运行；real-assets 测试默认 skip | 高 | 核心功能 | [test_real_perturbgen_smoke.py](tests/real_assets/test_real_perturbgen_smoke.py#L39) |
| U-04 | 3 seeds、held-out donors、≥99 null、BH-FDR 冻结队列验收 | 方案 §5/§7：双路径 AND 标准和统计门禁 | 高 | 核心功能 | [方案文档](docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md#L250) |
| U-05 | Gate-E 通过后删除 Geneformer 旧主链 | bridge §5：Gate-E 未过前不删除 Geneformer 主链 | 中 | 核心功能 | [ptm_direction_mapper.py](src/models/ptm_direction_mapper.py#L207-L229) |
| U-06 | Replogle 真实 loader、路径和训练输入 | v2.2 DATA-01/02 要求 manifest 与本地导入契约；当前 loader/path 为 null | 中 | 核心功能 | [datasets.yaml](data/manifests/datasets.yaml#L596-L610) |
| U-07 | scGeneScope 真实 loader、路径和训练输入 | 同上；当前仅有 local_import_contract_only | 中 | 核心功能 | [datasets.yaml](data/manifests/datasets.yaml#L636-L648) |
| U-08 | 真实 Ankh39/ESM-2/ProtT5 权重和跨尺度科学验收 | README 明确 synthetic fixtures 不代表真实扰动数据或生物学效果 | 中 | 核心功能 | [README](README.md#L89-L93)、[cross_scale.py](src/models/cross_scale.py#L233-L250) |
| U-09 | 真实 PMADS/PTM 训练等价性 | CURRENT_STATUS 只证明工程契约和 synthetic batch | 中 | 核心功能 | [ptm_modules.py](src/models/ptm_modules.py#L30-L70) |
| U-10 | 真实 PPI/kinase-substrate graph 训练与因果验收 | v2.2 MODEL-02/03 要求 typed graph、sensitivity matrix 和 cell decoder | 中 | 核心功能 | [.planning/REQUIREMENTS.md](.planning/REQUIREMENTS.md#L45-L57)、[cross_scale.py](src/models/cross_scale.py#L861-L920) |
| U-11 | ESM-3 真实 checkpoint 验收 | real-assets 测试要求显式 checkpoint，默认跳过 | 低 | 次要功能 | [test_real_esm3.py](tests/real_assets/test_real_esm3.py#L28-L36) |
| U-12 | CPTAC/PDC 真实模型和数据验证 | 测试明确绿色 CI 不等于真实 PDC 验证 | 中 | 次要功能 | [test_real_cptac_validation.py](tests/real_assets/test_real_cptac_validation.py#L25-L49) |
| U-13 | UniProt/KEGG/Reactome 真实服务验收 | real external service tests 为 opt-in | 低 | 次要功能 | [test_real_external_services.py](tests/real_assets/test_real_external_services.py#L8-L27) |
| U-14 | 至少两张 GPU 的真实 DDP 验收 | 测试在少于两张 GPU 时 SKIPPED | 低 | 边缘功能 | [test_real_distributed.py](tests/real_assets/test_real_distributed.py#L15-L50) |

上述 U-01 至 U-14 是“已定义但未完成/未验收”的后续项，不是 24 项正式需求的反向否定。正式需求自己声明真实资产和科学等价性边界，见 [.planning/REQUIREMENTS.md](.planning/REQUIREMENTS.md#L85-L108)。

### 4.3 缺失或未闭合的接口

| 接口 | 当前状态 | 预期参数/返回 | 用途与缺口 |
|---|---|---|---|
| API → DAVF PTM site contract | 缺失 | 输入应为与 PTMDirectionMapper 一致的 PTMSite 对象，含 position、type、gene_symbol；输出为方向和 gene ids | [predictions.py](src/api/routes/predictions.py#L373-L380) 生成 dict，mapper 在 [ptm_direction_mapper.py](src/models/ptm_direction_mapper.py#L338-L385) 读取 .type，合法 DAVF 请求可能 500 |
| request use_davf → model runtime switch | 缺失 | 输入 use_davf；预期启用 DAVF 分支或明确拒绝；返回应标记是否真的使用 DAVF | 请求 flag 只决定收集字段，模型分支由构造时 self.use_davf 决定，[architectures.py](src/models/architectures.py#L512-L535) |
| gene_symbol required validation | 缺失 | DAVF 请求必须要求每个 PTM site 有 gene_symbol，失败返回 4xx | schema 仍 Optional，缺失值在 [predictions.py](src/api/routes/predictions.py#L81-L106) 被静默跳过 |
| PerturbGen vocabulary resolver boundary | 未接入 | resolve(symbol) -> compact token row；resolve_batch(symbols) -> aligned rows | GeneVocabularyResolver 已存在，但生产 mapper 仍使用 Geneformer 私有映射，[gene_vocabulary.py](src/models/gene_vocabulary.py#L1-L120)、[ptm_direction_mapper.py](src/models/ptm_direction_mapper.py#L320-L330) |
| embedding asset → DAVF | 未闭合 | loader 应同时传递 embeddings 和 gene_to_token，LatentDAVF 应按 compact row 取向量 | [davf_inference.py](src/models/davf_inference.py#L277-L301) 丢弃 gene_to_token，LatentDAVF 后续按数值 ID 直接索引，[latent_davf.py](src/models/latent_davf.py#L451-L492) |
| run_formal_gate_evidence | 未形成统一入口 | 输入候选、src/tgt 输出、donor、seed、null；输出 self-contained evidence manifest 和 verdict | runner 只有 stage manifest，evaluator 需人工另传 JSON/H5AD/DEG/null |
| load_replogle / load_scgenescope | 未实现为可运行数据源 | 输入 manifest entry，返回符合 CrossScaleDataset 的 typed dataset | manifest 只有受控 null/local import contract |
| public DualPathVerdict | 未成为实际返回契约 | 输入双路径指标；返回 stable、rescue、FDR、verdict 和 provenance | 公共 [contracts.py](src/integration/perturbgen/contracts.py#L264-L281) 与内部 [dual_path.py](src/integration/perturbgen/dual_path.py#L49-L59) 类型不一致 |
| PTM virtual perturbation mode | 默认值不符合契约 | mode 应为 hard_ko 或 soft_ptm，adapter 应接收 node_decay/edge_scale | [ptm_virtual_perturbation.py](src/integration/ptm_virtual_perturbation.py#L151-L215) 默认 soft_ko，且计算参数没有传给 adapter |

### 4.4 缺失业务流程

#### API → DAVF → 预测

~~~mermaid
flowchart LR
    A[POST /api/v1/predict] --> B[PredictionRequest]
    B --> C[preprocess_request]
    C --> D[davf_sites]
    D --> E[PTMDirectionMapper.map_batch]
    E --> F[DAVFInferenceModule]
    F --> G[PTM2CellNet predictor]
    D -. dict fields do not match PTMSite .type .-> X[AttributeError / 500]
    F -. checkpoint load failure .-> Y[zero_fallback]
~~~

缺失点：API 没有把自身生成的字典转换为 mapper 契约；use_davf 不能动态启用模型；checkpoint 失败后仍可能返回看似正常的预测。

#### DAVF candidate → PerturbGen dual path → evidence

~~~mermaid
flowchart LR
    A[DAVF candidate evidence] --> B[strict preflight]
    B --> C[runner source_intervention]
    B --> D[runner within_state]
    C --> E[stage manifests]
    D --> E
    E --> F[dual-path evaluator]
    F --> G[AND verdict]
    G --> H[formal release evidence]
    A -. runner has no candidate input .-> X[人工拼接]
    E -. runner does not invoke evaluator .-> Y[人工拼接]
    B -. no compliant donor cohort .-> Z[Gate-0 blocked]
~~~

缺失点：runner 不接收 DAVF 候选、不自动使用 observed_direction/davf_action、不强制严格 Anndata 预处理、不自动运行三 seed/多 mode evaluator，也没有 self-contained formal evidence。

#### PerturbGen embedding → DAVF

~~~mermaid
flowchart LR
    A[PerturbGen encoder checkpoint] --> B[embedding asset]
    B --> C[embeddings plus gene_to_token]
    C --> D[GeneVocabularyResolver]
    D --> E[LatentDAVF compact rows]
    B -. current code discards gene_to_token .-> X[default Geneformer IDs]
    X -. row semantics not guaranteed .-> E
~~~

缺失点：文件资产可独立读取，但词表语义没有进入 DAVF 运行时，且 configs/davf_integration.yaml 的 embedding_asset_path 默认仍为 null。

## 5. 部分实现、接口差异与业务流程

### 5.1 分类标准

| 类别 | 判断标准 |
|---|---|
| 部分实现但可用 | 主流程入口存在，正常受支持输入可运行；缺的是可选资产、扩展分支或正式科学验收 |
| 实现但有缺陷 | 入口可达，但存在确定性异常、静默截断、状态污染、错误返回或资源语义错误 |
| 实现但不符合规范 | 代码能执行，但与需求、数据契约、文档原文、发布门禁或科学验收条件不一致 |

### 5.2 关键差异清单

| 编号 | 文档/需求原文摘要 | 实际实现与证据 | 分类、影响 |
|---|---|---|---|
| F-01 | DAVF API 请求应把 PTM 信息送入方向映射和推理 | API 在 [predictions.py](src/api/routes/predictions.py#L373-L380) 生成 dict；mapper 在 [ptm_direction_mapper.py](src/models/ptm_direction_mapper.py#L338-L385) 读取 ptm_site.type | 实现但有缺陷；合法 DAVF 请求可 500 |
| F-02 | DAVF 开关应表示请求是否使用 DAVF | use_davf 只控制收集字段，真正模型开关来自构造参数；无 gene_symbol 时静默使用零特征 | 实现但不符合规范；会把未使用 DAVF 伪装成成功请求 |
| F-03 | variant response 公开 cell_state_prediction | [predictions.py](src/api/routes/predictions.py#L715-L749) 对 DAVF 字段 list 调用 unsqueeze，异常被转成 None | 实现但有缺陷；变体结果不完整 |
| F-04 | M4 方案要求公共 resolver 和 PerturbGen token 对齐 | loader 只把 embeddings 传给 DAVF，gene_to_token 被丢弃；mapper 仍按 Geneformer id 取行 | 实现但不符合规范；真实资产行语义不可靠 |
| F-05 | 缺少真实 checkpoint 时应能区分不可判定和有效模型结果 | [davf_inference.py](src/models/davf_inference.py#L355-L364) 和 #L524 允许 zero_fallback；配置也允许 | 实现但不符合规范；真实 smoke 证据已显示 zero_fallback |
| F-06 | checkpoint 验证应反映完整权重加载 | [davf_inference.py](src/models/davf_inference.py#L442-L464) 使用 strict=False，只要命中 key 就标记 loaded | 实现但有缺陷；部分权重可能被当作完整模型 |
| F-07 | Phase 11 summary 声称包含 mismatched lengths validation | [ptm_direction_mapper.py](src/models/ptm_direction_mapper.py#L405-L453) 使用 zip(..., strict=False)，batch 长度不一致会静默截断 | 实现但有缺陷 |
| F-08 | PTM 类型应有统一语义 | API 未知类型映射为索引 0，mapper 未知类型默认 OE；同一输入两条分支语义不同 | 实现但不符合规范 |
| F-09 | 训练 release gate 不应放过无效 PTM 位点 | data contract 默认 warning，scripts/train.py 未启用 require_all_rows_valid_ptm=True | 实现但不符合规范；坏数据可能进入训练 |
| F-10 | 独立 cross_validate 应返回各折和汇总指标 | [evaluators.py](src/evaluation/evaluators.py#L236-L271) 只保留顶层标量，类方法实际返回嵌套 mean_metrics | 实现但有缺陷；消费者可能只得到 n_splits |
| F-11 | 梯度累积应在 epoch 尾部处理剩余 batch | [trainers.py](src/training/trainers.py#L242-L263) 只在完整 accumulation 周期 step | 实现但有缺陷；尾批梯度跨 epoch 残留 |
| F-12 | GenKI adapter mode 只接受 hard_ko/soft_ptm | engine 默认 soft_ko，且 node_decay/edge_scale 没传入 adapter，[contracts.py](src/integration/contracts.py#L35-L54) | 实现但有缺陷；默认真实路径可能直接失败 |
| F-13 | fail_fast=False 应收集每个样本错误 | [cross_scale.py](src/api/routes/cross_scale.py#L735-L812) 只捕获两类异常，ValueError/KeyError/RuntimeError 可逃逸 | 实现但不符合规范 |
| F-14 | 现有 API 兼容中间件应对受保护入口保持一致 | [middleware.py](src/api/middleware.py#L357-L362) 未覆盖 /predict/batch、/model_info 和 /cross-scale/* | 实现但不符合规范；属于现有兼容行为一致性，不构成新增 API key 需求 |
| F-15 | initialize 的 strict=False/加载统计应反映真实状态 | 显式 config path 校验不一致；后续又拒绝 missing/unexpected；返回 loaded_parameters 等为固定值，[initialize.py](src/api/routes/initialize.py#L214-L352) | 实现但不符合规范 |
| F-16 | 双路径正式流程应强制 raw counts、ENSG、state、donor、配对和候选门禁 | 严格函数位于 [data_prep.py](src/integration/perturbgen/data_prep.py#L138-L302)，但 runner/evaluator 主入口没有自动调用 | 实现但不符合规范；正式门禁可被人工绕过 |
| F-17 | 对外 verdict 应使用公共 DualPathVerdict | 实际 evaluator 返回内部 DualPathDecision | 实现但不符合规范；序列化/消费者接口不稳定 |
| F-18 | DAVF config 的 scVI/Geneformer 资产字段应进入推理链 | scvi_model_path、gene_names_path 只读取或告警，没有完成真实加载，[davf_inference.py](src/models/davf_inference.py#L122-L130) | 部分实现但可用；不能据此宣称资产已接入 |
| F-19 | 文档中的工程完成需与真实验收边界一致 | CURRENT_STATUS 声称 M1-M3 工程链完成，但同页说明 real path mocked；设计文档预期 resolver，生产链仍无接入 | 实现但不符合规范；容易误判发布状态 |
| F-20 | Gate-0/Gate-E/Gate-4 需要真实 evidence | donor audit 为 0 合规候选，M0 decoder 为随机初始化，real DAVF 为 zero_fallback | 部分实现但可用；工程 smoke 可用，科学结论不可用 |

### 5.3 其他文档差异

| 文件 | 证据 | 当前判断 |
|---|---|---|
| AGENTS.md | §0 路由表只列标准路由，未完整列 cross-scale；脚本表也未完整列 PerturbGen/cross-scale | 活动约束文件部分过时；本次报告记录，未改写用户提供的核心指令 |
| README.md | 安装说明原来把 API + Lightning 写成 .[api,lightning]，setup.py 没有 lightning extra；已改为 .[api,pretrained] | 本轮已修复 |
| docs/guides/quickstart.md | Docker curl 原来调用 /predict，而默认前缀是 /api/v1 | 本轮已修复 |
| docs/guides/training.md | 原命令使用 --batch-size、--plot、--device；train/evaluate help 不支持这些组合 | 本轮已修复 |
| docs/guides/perturbgen_bridge.md | 原文把嵌入 manifest 写成 schema v2；exporter/loader 实际使用 schema_version 1，DAVF 配置才是 schema v2 | 本轮已修复 |
| .planning/PROJECT.md | 历史 active checklist 与当前已闭环债务混在一起；原“Multitask unsupported w/ DAVF”与架构不一致 | 本轮改为 Historical Debt Checklist 和 Active Follow-up |
| .planning/REQUIREMENTS.md | 原更新时间仍为 2026-08-08，未区分 24 项正式需求和 M0-M7 后续提案 | 本轮已修订 |
| .planning/ROADMAP.md | v2.2 complete 与 Gate-E/Gate-4/Gate-5 follow-up 容易混淆 | 本轮增加 post-v2.2 acceptance boundary |
| docs/api/*.rst | 只有 automodule，没有端点、schema、错误语义、CLI 和真实资产门禁 | 活动文档覆盖不足，建议另行补齐，不应归档为历史文件 |
| v2.0 Phase 11 UAT | status=testing、passed=0、pending=10，与 summary 的实现完成叙述冲突；Phase 12-14 还缺 VERIFICATION.md | 历史审计证据不完整，纳入 TD-N-54 |

## 6. 模块完成度

### 6.1 评估方法

以下百分比是代码闭合度估算，精确到小数点后一位，不是 pytest 覆盖率、模型准确率或科学结论。估算维度为：入口存在、核心组件、依赖/资产契约、错误处理、测试证据和文档/验收边界。真实资产门禁单独列出，避免把代码数量当成生物学完成度。

### 6.2 完成度表

| 模块 | 完成度 | 分类 | 缺失组件/依赖和主要证据 |
|---|---:|---|---|
| src/data | 86.0% | 部分实现但可用 | 加载、特征、Dataset 存在；PTM 严格校验和 Replogle/scGeneScope 真实 loader 未闭合 |
| src/models | 68.5% | 实现但有缺陷 | 编码器和级联架构齐全；DAVF API 契约、checkpoint、resolver、真实 embedding 接入缺失 |
| src/training | 88.0% | 实现但有缺陷 | 训练/checkpoint/artifact 存在；梯度累积尾批问题，真实 DAVF 重训未完成 |
| src/evaluation | 78.5% | 实现但有缺陷 | 评估指标存在；独立 cross_validate 返回契约和真实双路径证据缺失 |
| src/utils | 100.0% | 部分实现但可用 | 主要工具和安全加载层存在；类型债务来自调用方 |
| src/api | 62.5% | 实现但有缺陷 | 标准路由存在；DAVF payload、runtime flag、variant、跨尺度异常和状态原子性有缺陷 |
| analysis | 100.0% | 部分实现但可用 | 变体、基因和通路模块存在；真实外部服务测试为 opt-in |
| GenKI/PTM integration | 61.0% | 实现但有缺陷 | adapter/fixture 存在；默认 mode 与契约不符，计算参数未真正传递 |
| PerturbGen 工程代码 | 68.0% | 实现但不符合规范 | runner、contract、asset loader 存在；preflight、candidate、双路径和正式报告未统一接入 |
| scripts | 75.0% | 部分实现但可用 | 入口齐全；部分入口依赖 fixture、zero fallback 或非 canonical 数据处理 |
| configs | 100.0% | 部分实现但可用 | 配置齐全；embedding_asset_path 默认 null，版本/schema 命名容易误导 |
| 活动文档 | 91.0% | 部分实现但可用 | 本轮修订主要入口；AGENTS 映射、API RST、历史状态仍有维护债务 |
| tests | 69.0% | 部分实现但可用 | 单元和集成规模大；关键 API→DAVF、真实权重、Gate-E/Gate-4 断言不足 |
| 真实资产/Gate-0/Gate-E/Gate-4/Gate-5 | 31.0% | 部分实现但可用 | 文件和 smoke 部分存在；合规 donor、训练 decoder、真实 DAVF、正式 evidence 未闭环 |

本表的“代码闭合度”与历史报告的模块聚合分值不是同一量尺；例如历史报告中的 models 98.6% / PerturbGen 71.4% 反映已提交组件的聚合完成度，本表把真实调用边、契约、错误语义和验收证据纳入评分。为避免状态漂移，当前指南已改用本表口径，历史数字仍只在归档报告中保留。

## 7. 技术债清单与解决策略

### 7.1 严重度标准

| 等级 | 判定标准 |
|---|---|
| 严重 | 会造成数据不可恢复丢失、生产安全边界失效、核心结果系统性错误且没有可接受绕行 |
| 高 | 支持的 API/CLI/CI/release 路径确定失败，或可能返回语义上错误的核心预测/科学证据 |
| 中 | 可选或扩展路径、安装发布、类型契约、计划证据或文档存在实质问题，但有明确人工绕行 |
| 低 | 不阻断当前受支持结果，只增加维护、测试隔离或环境使用风险 |

本轮没有发现满足“严重”的债务。高/中债务都给出至少两种可行方案、工期按 0.5 工作日计，并说明资源和风险。

### 7.2 新增债务总表

最新完整分析报告只登记到 TD-N-34；以下为本轮未在既有报告中登记的新发现。已修复项仍保留记录，便于审计闭环。

| 编号 | 严重度 | 状态 | 问题与证据 | 影响范围 |
|---|---|---|---|---|
| TD-N-35 | 高 | 开放 | API STATE 逐字段更新，[state.py](src/api/routes/state.py#L117-L135) 与线程池初始化 [initialize.py](src/api/routes/initialize.py#L313-L331) 并发，预测同时读取 [predictions.py](src/api/routes/predictions.py#L455-L486) | 热加载期间模型、特征器、序列长度可混搭 |
| TD-N-36 | 高 | 开放 | ruff check 通过但 ruff format --check src/ tests/ 报 252 files would be reformatted；format workflow 还被 .gitignore 忽略，CI 门禁不稳定 | CI 和合并质量 |
| TD-N-37 | 高 | 开放 | setup.py 使用 find_packages()，scripts 无 __init__.py，但 console_scripts 指向 scripts.train/scripts.predict | 非 editable wheel 安装后的 CLI 可能无法导入 |
| TD-N-38 | 中 | 已修复 | installation.md 原文与 setup.py extras/requirements 聚合不一致 | 新环境安装误导 |
| TD-N-39 | 中 | 开放 | CHANGELOG 为 2.1.0，而 package、src、FastAPI metadata 和 docs/conf.py 仍为 1.0.0 | 发布溯源、客户端兼容判断 |
| TD-N-40 | 中 | 开放 | GSD 工具检测 milestone v2.1/18 phases/17 complete，但 STATE/ROADMAP 已是 v2.2 complete；若干 phase 缺 VERIFICATION | 计划审计可信度 |
| TD-N-41 | 中 | 开放 | 24 项正式 requirements 与 M0-M7 supplemental proposal 的追踪边界近期才补充，仍没有机器可读 gate matrix | 后续验收容易把工程完成当科学完成 |
| TD-N-42 | 中 | 开放 | Pydantic v1 validator/class Config 在 schemas.py 触发 v2 deprecation warnings | 依赖升级后的 API 兼容和维护成本 |
| TD-N-43 | 低 | 开放 | test_lightning_pipeline.py 缺文件时写入被 .gitignore 忽略的 data/raw/sample_data.csv | 测试改变 clone 工作区外部状态 |
| TD-N-44 | 低 | 开放 | 裸 pip/mypy 与项目 Python 不同；同一命令名得到不同错误数和环境 | 验证结果不可复现 |
| TD-N-45 | 低 | 已修复 | full-test workflow 注释仍写“1910 项”，实际套件为 2345 collected | 维护者误读测试规模 |
| TD-N-46 | 高 | 开放 | API DAVF PTMSite dict 与 mapper 对象契约不一致，且 use_davf 不能动态启用模型 | 支持的 DAVF API 请求确定失败或静默零特征 |
| TD-N-47 | 高 | 开放 | PerturbGen gene_to_token 被丢弃，Geneformer numeric id 直接当 compact embedding row；checkpoint 失败还能 zero_fallback | 真实 DAVF 可能输出错误语义 |
| TD-N-48 | 高 | 开放 | runner 不接 DAVF candidate、不强制 preflight、不自动 evaluator/report，正式 evidence 需人工拼接 | 双路径主流程断开 |
| TD-N-49 | 高 | 开放 | PTM virtual engine 默认 soft_ko，不在 GenKI 合约允许值中，node_decay/edge_scale 未传递 | 默认真实扰动路径失败或参数无效 |
| TD-N-50 | 中 | 开放 | cross-scale fail_fast 异常范围、variant list tensor、initialize 统计契约不完整 | 扩展 API 的错误响应不稳定 |
| TD-N-51 | 中 | 开放 | PTM unknown type、越界位点 warning、batch zip strict=False 导致输入数据可静默失真 | 训练和 DAVF 结果语义 |
| TD-N-52 | 中 | 开放 | 独立 cross_validate 只保留顶层标量，公共 DualPathVerdict 与内部 Decision 不一致 | 评估消费者和报告序列化 |
| TD-N-53 | 中 | 开放 | v2.0 需求与当前 PTM direction mapping 语义冲突：requirements 要求 methylation→KD/sumoylation→KO/succinylation→KO，代码和 Phase 11 记录采用另一套映射 | 需求追踪和科学解释 |
| TD-N-54 | 中 | 开放 | Phase 11 UAT 仍 testing/0 passed，Phase 12-14 缺 verification 文件，历史 evidence 不能独立证明真实梯度/权重 | 里程碑审计 |
| TD-N-55 | 高 | 开放 | donor 0 候选、随机 decoder、zero fallback 仍可产生“通过”的 smoke 记录 | release/scientific gate 可能误放行 |
| TD-N-56 | 中 | 开放 | PMADS 脚本跳过 PTMD 并在 API 失败时生成 placeholder/random sequence；variant CLI 将 accession 当 gene symbol | 数据处理和变体分析结果可能不符合输入语义 |

TD-N-06、TD-N-08、TD-N-09、TD-N-10、TD-N-11、TD-N-25、TD-N-26 至 TD-N-34 等既有债务不重复登记；本轮只引用其当前状态。

### 7.3 高/中债务解决策略

#### TD-N-35：API runtime 状态原子切换

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：引入不可变 ModelRuntime 快照，初始化全部成功后一次替换 | 1.5 天 | 读请求始终拿到同一组模型/特征器/配置，语义最稳 | 需要改造 state/predictions 读取；临时 GPU 内存峰值增加；需 API/PyTorch 开发者和并发回归测试 |
| B：局部构造后用 RLock 统一提交和读取 | 1.0 天 | 改动小、保留现有接口 | 容易漏锁；锁住推理会增加延迟；需锁竞争测试 |
| C：禁止在线热加载，只允许进程重启换模型 | 0.5 天 | 最简单、无混合状态 | 失去当前 initialize 热加载能力；需明确运维契约 |

#### TD-N-36：Ruff format 门禁

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：按当前 Ruff 版本格式化 252 个文件并跟踪 lint workflow | 1.0 天 | 直接恢复格式门禁，一致性最好 | diff 大，需维护者复核合并冲突 |
| B：建立一次性 format baseline，只对新增/变更文件检查 | 0.5 天 | 小 diff，快速恢复 CI | 历史格式债保留，长期规则复杂 |
| C：将被 .gitignore 忽略的 workflow 纳入版本控制并合并到 ci.yml | 0.75 天 | CI 规则可被审计和执行 | 仍需决定历史格式范围；需 CI 维护者 |

#### TD-N-37：wheel CLI 打包

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：新增 scripts/__init__.py，保留入口并补 wheel smoke | 0.5 天 | 最小改动，find_packages 能收录 scripts | 暴露顶层脚本包；需验证安装后导入 |
| B：将 CLI 移到 src/cli/，更新 entry_points 和文档 | 1.5 天 | 包边界清晰，发布产物稳定 | import 和文档改动较多 |
| C：改用显式 py_modules/脚本打包配置 | 1.0 天 | 不必把 scripts 当 package | setuptools 配置更复杂，需跨平台安装测试 |

#### TD-N-39、TD-N-42：版本与 Pydantic 兼容

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：建立单一版本源，统一 setup.py、src metadata、FastAPI、Sphinx、CHANGELOG；同时迁移 Pydantic v2 API | 2.0 天 | 发布溯源和依赖升级长期稳定 | 可能影响客户端版本断言和 schema 行为；需 API/packaging 测试 |
| B：先固定 pydantic<3，明确 1.0.0 为 package 版本并把 2.1.0 标成 milestone | 0.5 天 | 快速消除误读和 warning 之外的升级风险 | 保留 v1 API 技术债；版本语义不够统一 |
| C：只统一版本，不迁移 Pydantic | 1.0 天 | 范围小 | warnings 继续存在，未来升级仍可能破坏 |

#### TD-N-40、TD-N-41、TD-N-54：计划和验收证据治理

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：补齐 Phase 12-14 verification，增加正式需求/提案/gate 矩阵并统一 GSD milestone 元数据 | 2.0 天 | 审计链完整，能区分 24 项工程需求和 M0-M7 科学 gate | 需要同时核对历史证据；需项目维护者 |
| B：建立 waiver/evidence manifest，明确缺失 verification 的理由、证据位置和失效日期 | 0.5 天 | 快速消除“空白即完成”的误读 | 历史阶段文件仍不完整 |
| C：迁移或重建旧 GSD phase 目录后再审计 | 1.5 天 | 结构长期一致 | 目录迁移风险和历史链接维护成本高 |

#### TD-N-46、TD-N-50、TD-N-52：API、评估和公共契约统一

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：定义单一 typed request/batch contract，在 preprocess 结束前构造 PTMSite；补 API→DAVF、variant、cross-scale fail_fast、cross_validate 回归测试 | 2.5 天 | 一次修正多条调用边，错误尽早暴露 | 需同步 schema、Dataset、mapper 和测试 |
| B：在各路由增加局部转换和校验 | 1.5 天 | 改动快 | 类型转换分散，未来仍容易漂移 |
| C：让 mapper 接受 dict 和 PTMSite 两种输入 | 1.0 天 | 对旧调用兼容 | 增加兼容层，语义边界不清；不符合 let it crash 的长期方向 |

#### TD-N-47、TD-N-48、TD-N-49、TD-N-55：DAVF × PerturbGen 真实闭环

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：先取得合规 donor，再把 resolver、asset vocabulary、strict checkpoint、candidate preflight、runner、evaluator、formal evidence 串成一个可重放入口 | 5.0 天 | 覆盖真实主流程，能形成可审计 Gate-0→Gate-5 链 | 依赖外部 donor、真实 checkpoint、PerturbGen 环境；需要生信/ML/工程协作 |
| B：保留 runner/evaluator 分离，但生成机器可读 candidate manifest 和 stage manifest，禁止人工拼接 | 3.0 天 | 工程改动较小，报告可重放 | 仍需操作多个命令；跨工具失败定位较复杂 |
| C：先做 fail-fast/INCONCLUSIVE gate，只允许真实资产通过才进入 evaluator | 1.5 天 | 立即阻止 zero fallback 和假绿色证据 | 不会立即完成真实科学结果；需要明确用户提供资产格式 |

#### TD-N-50、TD-N-51、TD-N-53、TD-N-56：数据和语义契约

| 方案 | 工期 | 优点 | 缺点、资源与风险 |
|---|---:|---|---|
| A：统一 PTM ontology、方向映射、严格位点验证、gene symbol/accession 解析和 PMADS 输入；未知/placeholder 直接失败 | 2.5 天 | 结果语义最可靠，符合项目 let it crash | 可能暴露现有脏数据；需更新 fixtures 和文档 |
| B：增加 release 模式 strict validation，保留探索模式 warning | 1.0 天 | 兼顾研究探索和正式发布 | 两种模式可能被误用；需要清晰入口命名 |
| C：只修文档和测试断言 | 0.5 天 | 快速 | 不能消除实际错误数据和方向冲突，不推荐作为最终方案 |

### 7.4 低债务处理

- TD-N-43：将测试缺省数据写入 pytest tmp_path，预计 0.5 天。
- TD-N-44：文档和 CI 统一使用 python -m pip、python -m mypy、python -m pytest，预计 0.5 天。
- TD-N-45 已在本轮修复。

## 8. 验证结果

### 8.1 已执行命令与结果

| 命令 | 结果 |
|---|---|
| python -m pytest --collect-only -q | 通过，2345 tests collected in 4.21s |
| unshare -rn env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest tests -q -p no:cacheprovider | 通过，2329 passed、16 skipped、47 warnings，551.77s |
| python -m compileall -q src scripts tests | 通过 |
| ruff check src scripts tests | 通过，All checks passed |
| python scripts/check_requirements_consistency.py | 通过，274 lock pins satisfy all core constraints |
| python -m mypy src | 未通过，23 errors in 8 files；既有类型债和本轮调用契约债并存 |
| python -m pip check | 未通过，3 项环境冲突：ptm2cellnet numpy<2、scgpt scvi-tools<1、ssh-unit torchaudio>=2.5 |
| ruff format --check src/ tests/ | 未通过，252 files would be reformatted，88 already formatted |
| python setup.py --name --version | 通过，ptm2cellnet 1.0.0 |
| python scripts/finetune_davf_e2e.py --help | 通过，显示 data/checkpoint/embedding-asset 入口 |
| python scripts/run_perturbgen_pipeline.py --help | 通过，支持 source_intervention、within_state、both、resume、dry-run |
| git fetch origin main | 通过 |
| git rev-list --left-right --count main...origin/main | 43 0 |
| git merge --ff-only origin/main | 最终闭环执行，预期 Already up to date；若输出不同，以最终交付记录为准 |

### 8.2 真实资产和科学门禁

| 检查 | 结果 | 不能推出的结论 |
|---|---|---|
| PerturbGen embedding loader | 可加载 [18967, 768] | 不能推出 gene_to_token 已进入 DAVF |
| DAVF checkpoint | safe_torch_load 因 LatentDAVFConfig pickle global 失败 | 不能推出真实 DAVF forward 成功 |
| real DAVF smoke | 返回码为 0，但 model_source=zero_fallback | 不能推出真实权重生效 |
| donor audit | 30 files、26 readable、0 compliant candidates | 不能进入 Gate-0 正式队列 |
| 本地 PerturbGen H5AD | shape (227, 2001)，无 layers，obs 只有 perturbation/cell_idx | 不能进入 formal result extractor |
| real-assets tests | 默认环境变量未设置，相关测试 skip | 不能把绿色离线套件当真实资产发布证据 |

### 8.3 警告处理

47 个 pytest warnings 主要来自 Pydantic v2 deprecation、Mamba future warning、Lightning worker/checkpoint、anndata implicit modification 和取消范围测试。它们没有使当前测试失败，但 Pydantic、测试数据写入和环境选择分别已登记为 TD-N-42、TD-N-43、TD-N-44。

### 8.4 失败时的下一步

- 若 full test 失败：先确认使用 SSH_unit Python、HF 离线环境和 unshare -rn，再按首个失败测试定位。
- 若 mypy 失败：使用 python -m mypy src，禁止用裸 mypy 对比不同解释器结果。
- 若 pip check 失败：不要临时覆盖版本；先按 TD-N-34 和当前环境冲突拆分 core、scgpt、torchaudio 环境。
- 若真实 DAVF 失败：先修复 checkpoint pickle 可加载性和 resolver/token 语义，再禁止 zero_fallback 参与 Gate-E。
- 若文档链接失败：从当前报告和 archive/20260827/MANIFEST.md 反查，不恢复旧报告为活动权威入口。

## 9. 子智能体执行统计

### 9.1 统计口径

平台未提供可靠的每个子智能体开始/完成时间字段；因此不伪造平均耗时。调用次数按本次显式 spawn 任务调用计数，四个子智能体各 1 次；wait/close 管理调用不计入任务调用次数。结果均为只读审计，没有子智能体修改、提交或创建 worktree。

| 智能体 | 调用次数 | 主要执行任务 | 平均执行时长 |
|---|---:|---|---|
| Halley | 1 | 需求、设计文档、活动文档、未实现功能和 Gate-0/Gate-5 缺口 | 无法计算：平台未返回完成时长 |
| Cicero | 1 | API 路由、DAVF/PerturbGen 调用链、接口契约和 E2E 断点 | 无法计算：平台未返回完成时长 |
| Dalton | 1 | 新技术债、严重度、CI/打包/状态/版本/测试维护审计 | 无法计算：平台未返回完成时长 |
| Gauss | 1 | gsd integration checker：Phase 10-14/18-20 集成和真实资产证据 | 无法计算：平台未返回完成时长 |
| 合计 | 4 | 并行完成需求、调用链、技术债和跨阶段集成复核 | 不计算；缺少可审计时长字段 |

简要分析：四个角色覆盖了需求、实现、质量和跨阶段证据四个互补面；统计结果可以确认每个角色都实际被调用一次，但不能从平台缺失的时间字段推导平均耗时。

## 10. 结论与建议

### 10.1 已完成

- 完成主分支与远程主分支同步检查，确认本地没有落后远程。
- 完成旧报告和旧测试页归档，保留 Git 历史和版本清单。
- 修订当前状态、安装、Quickstart、训练、PerturbGen bridge、README 和 planning 边界文档。
- 完成综合代码/接口/需求/技术债审计，生成本报告。
- 完成测试、编译、Ruff、依赖一致性、CLI help 和真实资产边界验证。

### 10.2 不应宣称完成

- 不应宣称 Gate-0 donor 队列完成。
- 不应宣称真实 DAVF checkpoint 已生效。
- 不应宣称 PerturbGen 双路径已自动接入 DAVF candidate 和正式 evidence。
- 不应宣称 synthetic 或 zero fallback 结果代表科学效果。
- 不应因 24 项正式工程需求为 [x] 就把 M0-M7 后续提案当作完成。

### 10.3 建议执行顺序

1. 先修 TD-N-46，建立真实 API→DAVF 回归测试；这条链当前对合法 DAVF 请求存在确定性失败。
2. 再修 TD-N-47、TD-N-48、TD-N-55，取得合规 donor 和真实 checkpoint 后，形成 fail-fast/INCONCLUSIVE 的正式 evidence 入口。
3. 同步修 TD-N-35、TD-N-37，避免在线模型混合状态和发布 wheel CLI 失效。
4. 补齐 TD-N-40、TD-N-41、TD-N-54 的规划证据，恢复阶段验证记录的可审计性。
5. 最后处理格式、版本、Pydantic、数据处理和低风险维护债务。

报告完成后，最终提交 hash、git merge --ff-only origin/main 的实际输出和工作树状态以交付消息为准；本地不执行 push。

---
报告生成：2026-08-27<br>
报告权威入口：project_analysis_20260827.md<br>
归档批次：archive/20260827/
