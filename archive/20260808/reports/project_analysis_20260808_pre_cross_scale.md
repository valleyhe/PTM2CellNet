# PTM2CellNet 项目代码与文档综合分析报告

> 报告日期：2026-08-08
> 审计基线：本地 `main`，审计开始时 HEAD 为 `6c64117537b6e4921b9bdd1ee66a32641336d628`
> 最终提交：以交付时本地 `main` 的 `git log -1 --format=%H` 为准
> 适用范围：项目源代码、测试、配置、脚本、活动文档、历史报告、需求原件及检测产物

## 1. 任务判断与执行摘要

### 1.1 本任务类型

本任务同时属于：

1. 代码审查与系统性技术分析；
2. 未实现功能识别与需求追踪；
3. 文档与检测报告归档；
4. Bug 修复（本轮审计过程中发现并修复类型检查错误）；
5. 测试与质量验证；
6. 版本控制提交及本地主分支整理。

### 1.2 结论先行

项目当前的工程骨架较完整：数据加载、特征工程、模型训练、评估、API、Docker 健康检查、速率限制、模型 provenance、测试分层等均有实际代码和自动化测试支撑。本次验证得到：

- 单元测试：1558 passed，4 skipped；
- 集成测试：67 passed，1 skipped；
- E2E 测试：42 passed；
- real-assets 测试：8 skipped（未提供真实模型/外部资产）；
- 总收集测试数：1680；
- `mypy src`：124 个源文件无错误；
- `ruff check src scripts tests`：通过；
- `compileall src scripts tests`：通过。

但原始需求文档定义的“PTM 状态 → 信号传导图 → 单细胞扰动状态”的核心科学闭环尚未完成。主要缺口不是普通 API 或训练脚本，而是以下三个跨尺度组件：

1. Ankh39、ESM-2、ProtT5 的指定多预训练模型融合；
2. 以 PPI/激酶-底物图为基础的 CIGNN 信号桥和敏感性矩阵；
3. Cell-Graph-Compass 风格的虚拟细胞状态/表达变化解码头。

当前实现可提供可运行的 PTM 分类/回归、DAVF 条件特征、通路启发式映射、GenKI/虚拟扰动和 API 推理，但这些能力不能据此宣称已经实现原需求中的生物学跨尺度模型。该结论有源代码位置和原始需求章节作为证据，详见第 4、5 节。

### 1.3 本轮已完成的实际修改

- 将过时的活动版 API/项目/技术/文件说明文档、2026-08-04 旧报告、原始需求文档原件和覆盖率检测产物统一归档到 `archive/20260808/`；
- 通过 `MANIFEST.md` 记录来源路径、版本/日期、Git 历史或文件哈希；
- 在原活动路径保留指向归档原件和当前权威文档的入口页，避免旧链接直接失效；
- 更新部署文档中的当前配置项；
- 修复 DAVF legacy checkpoint 分支的 `mypy` 类型错误，并保留用户已有的 legacy DAVF 兼容实现及 PTM module 开关修改；
- 新增本报告；
- 已完成测试、静态检查、编译检查，待本报告落盘后提交全部变更。

## 2. 审计依据、方法与边界

### 2.1 直接依据

本报告只使用本地项目内容和实际执行结果，未使用网络搜索替代代码分析。主要依据包括：

- 原始需求原件：[archive/20260808/requirements/蛋白质PTM与细胞状态预测项目需求.docx](archive/20260808/requirements/蛋白质PTM与细胞状态预测项目需求.docx)；
- 当前代码：`src/` 下 124 个 Python 文件；
- 当前测试：`tests/` 下单元、集成、E2E、real-assets 测试；
- 当前配置与依赖：`requirements-*.txt`、忽略的环境锁定文件、`Dockerfile`；
- 当前范围边界：[src/project/scope.py](src/project/scope.py#L1-L115)；
- 当前状态页：[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)；
- CodeGraph 状态：308 files、6392 nodes、7146 edges，索引健康；
- 实际执行的 pytest、mypy、ruff、compileall 结果（第 10 节）。

### 2.2 原始需求章节抽取

原始需求文档的主要可追踪要求如下：

| 需求章节 | 需求内容摘要 | 本报告追踪编号 |
|---|---|---|
| §一.1 Protein State Encoder | Ankh39、ESM-2、ProtT5 固定/融合蛋白语言模型，并接入 PTM Adapter/PTM tokens | R-01、R-02 |
| §一.2 Signal Transduction Bridge | CIGNN、STRING/PPI 与激酶-底物图、敏感性矩阵 `S=(1-alpha)(I-alpha G')^-1` | R-03、R-06 |
| §一.3 Perturbation Head | Cell-Graph-Compass 风格输出 Δ expression spectrum 或 perturbed cell state | R-04 |
| §二 数据集 | PTMAtlas、dbPTM/PhosphoSitePlus、ProteomeTools/PROSPECT；scPerturb、Norman/Adamson、Replogle；PMADS、scGeneScope；STRING/BioPlex、RegNetwork | R-05、R-06 |
| 实现建议 | 先用 PMADS + Ridge Regression 建立基线，再实现复杂模型 | R-07 |

原始文档还包含 Python 3.9.16、Torch 1.13.0 等基线版本。这些版本已明显落后于当前环境，不能直接作为当前运行时约束；需求语义仍然有效，版本约束应由当前依赖策略重新固化。

### 2.3 评估口径

- “完成度百分比”是基于代码路径、接口、测试和需求覆盖的工程估算，不等价于科学有效性或生产可用性证明；
- “已实现”只表示存在可调用代码，不表示使用了真实生物学数据；
- 真实模型/外部数据库需要显式资产门控，默认 mock/synthetic 测试通过不等于真实资产验证通过；
- 原项目范围已明确取消实时质谱流、自定义 PTM 数据库、GUI、API key 扩展等功能。本报告不把这些取消项重新列为待开发需求，依据见 [src/project/scope.py](src/project/scope.py#L28-L108)。

## 3. 当前系统实现地图与完成度

下表按“需求覆盖 + 代码完整度 + 测试证据”给出估算。百分比用于排序技术债，不是产品验收分数。

| 功能模块 | 估算完成度 | 当前证据 | 分类 |
|---|---:|---|---|
| 数据加载与基础预处理 | 85% | [src/data/loaders/](src/data/loaders/)、[src/data/preprocess.py](src/data/preprocess.py)、[src/data/validation.py](src/data/validation.py)；支持 UniProt、PhosphoSitePlus、dbPTM、CPLM、CSV/FASTA | 部分实现但可用 |
| 特征工程与 PTM 数据集 | 90% | [src/data/features.py](src/data/features.py)、[src/data/datasets.py](src/data/datasets.py)、DAVF augmentation；已有单元/集成测试 | 部分实现但可用 |
| 序列编码器 | 75% | [src/models/encoders.py](src/models/encoders.py)、[src/models/pretrained_encoders.py](src/models/pretrained_encoders.py) 提供 CNN/Transformer/RNN/ESM/ProtBERT/ProtT5 等；缺少 Ankh39 及需求规定的多 backbone 融合 | 实现但不符合原规范 |
| PTM 模块 | 80% | [src/models/ptm_modules.py](src/models/ptm_modules.py) 与 [src/models/architectures.py](src/models/architectures.py#L137-L213)；当前还支持关闭 PTM module 的兼容开关 | 部分实现但可用 |
| 主模型与 DAVF | 80% | [src/models/architectures.py](src/models/architectures.py)、[src/models/davf_inference.py](src/models/davf_inference.py#L241-L535)、[src/models/legacy_davf.py](src/models/legacy_davf.py) | 部分实现但可用 |
| CIGNN/PPI 信号桥 | 25% | [src/models/signaling_network.py](src/models/signaling_network.py#L276-L460) 是 KEGG/Reactome/启发式路径映射；[src/analysis/pathway_integration.py](src/analysis/pathway_integration.py#L484-L568) 缺数据时使用 fully-connected approximation；没有 CIGNN 和需求中的敏感性矩阵 | 实现但不符合规范 |
| 虚拟细胞扰动头 | 35% | [src/integration/genki/perturbation.py](src/integration/genki/perturbation.py)、[src/integration/ptm_virtual_perturbation.py](src/integration/ptm_virtual_perturbation.py) 提供图扰动/排名输出；没有 Cell-Graph-Compass Δ expression decoder | 实现但不符合规范 |
| 训练与 checkpoint | 90% | [src/training/](src/training/)、[scripts/train.py](scripts/train.py)、callbacks/artifacts；有单元和集成验证 | 部分实现但可用 |
| 评估指标与交叉验证 | 90% | [src/evaluation/metrics.py](src/evaluation/metrics.py#L329-L430)、[src/evaluation/evaluators.py](src/evaluation/evaluators.py#L236-L430)，含排序指标、置信区间、`cross_validate` | 部分实现但可用 |
| API 与可观测性 | 90% | `/predict`、`/predict/batch`、`/batch_predict`、`/predict/variant`、`/initialize`、`/health`、`/live`、`/ready`、`/model/info`、`/metrics` 均有路由 | 部分实现但可用 |
| 外部工具集成 | 70% | [src/models/external_tools/](src/models/external_tools/)、[scripts/enrich_with_external_tools.py](scripts/enrich_with_external_tools.py)；部分工具在不可用时返回 synthetic/fallback 结果 | 部分实现但可用 |
| 指定原始数据源覆盖 | 35% | 已覆盖 UniProt、PhosphoSitePlus、dbPTM、CPLM、PMADS 路径；原需求中的 PTMAtlas、ProteomeTools/PROSPECT、scPerturb、scGeneScope、BioPlex、RegNetwork 没有对应完整接入 | 未完全实现 |
| 测试与真实资产验证 | 80%（mock）/40%（real assets） | 默认测试通过；[tests/real_assets/](tests/real_assets/) 默认全部门控并跳过真实资产测试 | 部分实现但可用 |
| 文档与需求追踪 | 70% | 现已添加当前状态和本报告；旧文档已归档；仍需建立版本化需求矩阵和 OpenAPI 自动同步 | 部分实现但可用 |

## 4. 未实现功能项、缺失接口与缺失业务流程

### 4.1 未实现或未达到原始需求的核心功能

| 编号 | 原需求章节 | 优先级 | 当前实现与证据 | 影响范围 | 分类 |
|---|---|---|---|---|---|
| R-01 | §一.1 | P0 | [src/models/pretrained_encoders.py](src/models/pretrained_encoders.py#L1-L65) 提供通用 Hugging Face encoder 及 ESM/ProtBERT/ProtT5 类，但全仓库没有 Ankh39 实现，也没有三模型联合融合模块 | 核心蛋白状态表征不符合需求，实验结果不可直接对标原设计 | 未实现/部分实现 |
| R-02 | §一.1 | P0 | [src/models/ptm_modules.py](src/models/ptm_modules.py) 有 PTMEmbedding、attention/fusion 等组件；[src/models/architectures.py](src/models/architectures.py#L200-L213) 通过配置接入，但没有证据表明它是需求所指的 Ankh/ESM/ProtT5 统一 PTM Adapter | PTM 输入可用，但与规定 backbone 的契约未闭合 | 部分实现但不符合规范 |
| R-03 | §一.2 | P0 | [src/models/signaling_network.py](src/models/signaling_network.py#L377-L460) 通过 pathway database 和启发式分数映射；[src/analysis/pathway_integration.py](src/analysis/pathway_integration.py#L531-L568) 在缺边表时构造近似图；未发现 CIGNN 层、PPI/激酶-底物图消息传递或 `S=(1-alpha)(I-alpha G')^-1` 实现 | PTM 到信号传播的因果/图计算缺失，无法复现需求桥接模块 | 未实现 |
| R-04 | §一.3 | P0 | [src/integration/genki/perturbation.py](src/integration/genki/perturbation.py) 提供 KO/PTM 图扰动和 gene ranking；[src/integration/ptm_virtual_perturbation.py](src/integration/ptm_virtual_perturbation.py) 提供虚拟扰动接口，但没有 Cell-Graph-Compass decoder、Δ expression spectrum 或经过训练的 cell-state head | 无法完成“PTM 变化 → 单细胞表达/状态变化”的原始主输出 | 未实现/替代实现 |
| R-05 | §二 | P0 | [src/data/loaders/ptm_database_loaders.py](src/data/loaders/ptm_database_loaders.py#L54-L205) 覆盖 PhosphoSitePlus、dbPTM、CPLM；[src/data/loaders/uniprot_loader.py](src/data/loaders/uniprot_loader.py#L173-L220) 覆盖 UniProt；原需求指定的 PTMAtlas、ProteomeTools/PROSPECT、scPerturb、scGeneScope 等没有同等完整加载器和版本化 fixture | 训练数据分布与需求定义不一致，核心结论可能不可复现 | 未完全实现 |
| R-06 | §二 | P1 | [src/analysis/pathway_integration.py](src/analysis/pathway_integration.py#L488-L568) 可读取预处理 STRING/BioGRID 边表；没有发现 BioPlex、RegNetwork 的完整加载/版本校验/统一图构建流程 | 图先验覆盖不完整，信号桥输入不稳定 | 部分实现 |
| R-07 | 实现建议 | P1 | 原始需求建议先以 PMADS + Ridge Regression 建立基线；当前脚本和模型路径中没有明确的、可复现的 Ridge baseline 命令、artifact manifest 和对比报告 | 无法以简单基线判断复杂模型增益，增加研发风险 | 未实现 |

### 4.2 当前内部接口是否存在缺失

原始蓝图中描述的多数“工程接口”已经有对应实现：

- 数据接口：`DataLoader`、CSV/FASTA/UniProt/数据库加载器、预处理、特征工程、`PTMDataset`；
- 模型接口：序列 encoder、PTM modules、predictors、`PTM2CellNetBase`/`PTM2CellNet`/`PTM2CellNetLarge`；
- 训练接口：Trainer、Lightning wrapper、loss、optimizer、callback、artifact；
- 评估接口：metrics、`evaluate()`、`cross_validate()`、解释器；
- API 接口：预测、批量预测、variant、初始化、健康检查、ready、model info、metrics。

因此，本轮没有证据支持“当前 API 缺少某个既定端点”的结论。真正的接口缺口是科学模型内部契约：

1. “三种 pLM 的融合输出”没有统一的 encoder interface；
2. “图节点/边/权重/敏感性矩阵”没有 CIGNN 输入输出契约；
3. “cell graph + Δ expression/state”没有 decoder schema、loss 和 metric 契约；
4. “指定数据源版本、下载清单、许可和哈希”没有统一 data manifest 接口。

API 文档过期曾造成表面上的端点缺失：实际代码已提供 `/live`、`/ready`、`/metrics`、`/initialize` 等路由，旧文档已移入 [archive/20260808/docs/API_DOCUMENTATION.md](archive/20260808/docs/API_DOCUMENTATION.md)，根路径现保留入口页 [API_DOCUMENTATION.md](API_DOCUMENTATION.md)。

### 4.3 未实现业务流程

#### 原始需求主流程（未完成）

```text
PTMAtlas/ProteomeTools/PSP 等版本化数据
  -> Ankh39 + ESM-2 + ProtT5 融合及 PTM Adapter
  -> STRING/BioPlex/RegNetwork/PPI/激酶-底物图
  -> CIGNN 消息传递 + 敏感性矩阵
  -> Cell-Graph-Compass 虚拟细胞解码器
  -> Δ expression spectrum / perturbed cell state
```

当前代码在第一段使用单模型/可选预训练 encoder，在中间使用 pathway/GenKI 近似，在最后使用扰动排名或现有 predictor；上述链路不是同一个端到端训练和推理流程。

#### 数据准备分支（未完成）

- 指定数据库下载、版本锁定、许可记录、字段统一和可复现快照未形成一条完整流水线；
- 原始单细胞数据源和 PTM 数据源没有统一 schema/manifest；
- 缺失数据时部分路径会退化为 synthetic/fallback，不能替代真实数据。

#### 基线分支（未完成）

PMADS + Ridge Regression 的建议基线没有对应明确 CLI、固定数据切分、artifact、指标表和与主模型的自动对照流程。

#### 当前可运行主流程（已实现）

```text
CSV/FASTA/结构化 PTM 输入
  -> preprocess/features/dataset
  -> PTM2CellNet 或 DAVF 条件推理
  -> 分类/回归/多任务输出
  -> evaluate/cross_validate/API response
```

该流程由训练、集成、E2E 和 API 测试覆盖，但它覆盖的是当前实现契约，不等于原始跨尺度科学主流程已完成。

## 5. 未完全实现功能的分类与规范差异

### 5.1 部分实现但可用

| 功能 | 已具备 | 主要限制 |
|---|---|---|
| DAVF 推理 | checkpoint 加载、legacy checkpoint 兼容、SCVI/gene 条件、严格资产模式 | 默认允许零特征 fallback；真实资产测试默认跳过 |
| 通路/信号映射 | KEGG/Reactome、本地 pathway 数据库、PTM-to-pathway 结果 | 启发式评分，非 CIGNN，缺失图时会近似建图 |
| GenKI/虚拟扰动 | KO/PTM 扰动、VGAE/图缓存、排名结果 | 不是 Cell-Graph-Compass 表达变化解码器 |
| API 服务 | 预测、批量、variant、初始化、健康检查、ready、metrics | Pydantic v1 风格弃用警告；速率限制是进程内状态 |
| 外部工具 | AlphaFold/BLAST/ClustalW/PSIPRED 客户端和 CLI 桥接 | 外部工具不可用时部分路径返回 synthetic/fallback 结果 |

### 5.2 实现但有缺陷

1. [tests/real_assets/test_real_external_services.py](tests/real_assets/test_real_external_services.py#L66-L103) 中存在 `assert result is not None or True`，该断言恒为真，不能验证 external service 的返回质量。
2. `requirements-core.txt` 已声明 `torchmetrics`，但 [requirements-pretrained.txt](requirements-pretrained.txt#L15-L22) 仍保留“core does not declare torchmetrics”的过期说明，说明依赖文档和实现已漂移。
3. 忽略的 `requirements-lock.txt` 记录了 `numpy==2.4.3`，而 [requirements-core.txt](requirements-core.txt#L14-L15) 约束为 `numpy>=1.24,<2`；该 lock 文件既不纳入版本控制，也不能作为可复现安装依据。
4. [src/api/schemas.py](src/api/schemas.py#L82-L136) 使用 Pydantic v1 `@validator` 和旧 `Config` 写法；测试虽通过，但会产生 deprecation warnings。
5. `tests/real_assets` 默认跳过全部 8 个真实资产测试，mock/synthetic 通过不能证明真实 checkpoint、外部数据库和服务可用。

### 5.3 实现但不符合原始规范

- 当前 `pretrained_encoders.py` 的 ESM/ProtBERT/ProtT5 是可选的单 backbone 编码器，不是 §一.1 指定的 Ankh39 + ESM-2 + ProtT5 融合架构；
- 当前 `pathway_integration.py` 缺少真实边表时使用 fully-connected approximation，这可以作为开发 fallback，但不能作为 §一.2 的 PPI/CIGNN 科学实现；
- 当前 GenKI/PTM virtual perturbation 输出图扰动排名或潜变量变化，不是 §一.3 要求的 Cell-Graph-Compass 训练 decoder 及 Δ expression spectrum；
- 当前外部工具 fallback 的 synthetic 结果适合接口测试，不应进入需要真实结构/比对结果的科学结论。

## 6. 技术债识别、分级与证据

严重程度定义：

- **严重**：阻断原始核心目标、会使科学结论失效，或造成不可接受的复现/安全风险；
- **高**：显著影响生产、真实数据验证、部署一致性或质量门禁；
- **中**：影响维护性、可观测性或长期迭代效率，但有可用绕行方案；
- **低**：局部清理、性能优化或警告治理，不立即阻断功能。

| ID | 严重度 | 技术债 | 具体证据 | 影响 |
|---|---|---|---|---|
| TD-01 | 严重 | 原始跨尺度核心模型缺失 | [src/models/signaling_network.py](src/models/signaling_network.py#L377-L460) 是启发式 pathway mapping；未发现 CIGNN/敏感性矩阵；[src/integration/genki/perturbation.py](src/integration/genki/perturbation.py) 是扰动执行器 | 无法验证 PTM→信号→单细胞状态的主科学假设 |
| TD-02 | 严重 | 需求指定数据源和 PMADS Ridge 基线未闭环 | [src/data/loaders/ptm_database_loaders.py](src/data/loaders/ptm_database_loaders.py#L54-L205) 覆盖范围有限；缺少统一 data manifest 和 Ridge baseline | 复杂模型没有可靠基线和数据可复现性 |
| TD-03 | 高 | 依赖声明、lock 和运行环境不一致 | `requirements-core.txt` 的 NumPy `<2` 与 ignored `requirements-lock.txt` 的 NumPy `2.4.3` 冲突；lock 不入 Git | 新环境可能安装出不同且不兼容的依赖集合 |
| TD-04 | 高 | DAVF 零特征 fallback 与真实资产验证不足 | [src/models/davf_inference.py](src/models/davf_inference.py#L493-L535) 支持 zero fallback；[tests/real_assets/](tests/real_assets/) 默认跳过 | 资产缺失时 API 仍可返回结果，容易把占位输出误当科学结果 |
| TD-05 | 高 | API 速率限制/指标为进程内状态 | [src/api/middleware.py](src/api/middleware.py#L215-L340) 使用进程内 `_RateLimitState`；[Dockerfile](Dockerfile#L80-L87) 默认 4 Uvicorn workers | 多 worker 部署下限流和计数不全局一致 |
| TD-06 | 高 | 测试存在真实资产盲区和恒真断言 | [tests/real_assets/test_real_external_services.py](tests/real_assets/test_real_external_services.py#L66-L103) 的 `or True`；真实资产测试默认 skip | 绿色测试不能充分证明外部服务和真实模型链路 |
| TD-07 | 中 | 活动文档和需求追踪长期漂移 | 原项目文档曾引用 93+ 文件、旧测试数和旧配置；本轮已归档并更新入口，但尚无自动化文档同步 | 使用者可能按旧接口/旧指标操作 |
| TD-08 | 中 | Pydantic/Starlette/AMP 等弃用警告未治理 | [src/api/schemas.py](src/api/schemas.py#L82-L136) 使用旧 validator；测试输出 Pydantic、Starlette/httpx、Mamba AMP 警告 | 升级依赖时可能转为错误，增加维护成本 |
| TD-09 | 中 | 大文件和职责聚集 | `scripts/tools/create_tech_doc.py` 1479 行、`src/models/pretrained_encoders.py` 851 行、`src/integration/genki/perturbation.py` 822 行、`src/models/architectures.py` 692 行 | 修改风险高，测试定位和代码审查成本高 |
| TD-10 | 中 | 长序列窗口逻辑重复 | [src/models/long_sequence.py](src/models/long_sequence.py#L33-L127) 与同文件 `SlidingWindowESM2Encoder`/`_create_windows` 另有一套窗口实现 | 边界修复可能只覆盖一个实现，造成行为不一致 |
| TD-11 | 中 | 安全/科学 fallback 与真实结果边界不够强 | [src/models/external_tools/alphafold.py](src/models/external_tools/alphafold.py#L34-L113) 存在 synthetic fallback；[src/data/validation.py](src/data/validation.py#L22-L31) 具有可选格式 fallback | 结果 provenance 不清晰时可能误用占位数据 |
| TD-12 | 低 | 性能和警告技术债 | AGENTS 记录的 Mamba 串行实现、`estimate_max_batch_size` 硬编码层数，以及当前 AMP deprecation warnings | 长序列/大 batch 场景效率和升级稳定性受影响 |

说明：旧文档曾提到“10 个文件 `mypy: ignore-errors`”。本次直接执行 `mypy src` 已通过 124 个文件；本报告不把已不存在的旧标记重复计入当前债务。当前报告只记录可由现代码或本次测试直接支持的问题。

## 7. 严重/高等级技术债解决策略

以下时间为 1 名熟悉项目的 ML 工程师、1 名生物信息学工程师并行工作的研发估计，不含外部数据许可等待和大规模训练时间。

### 7.1 TD-01：补齐跨尺度科学模型（严重）

**问题与影响**：当前代码可以从 PTM 输入得到模型输出或图扰动排名，但没有将需求中的蛋白状态、信号图传播和单细胞状态解码串成同一可训练模型，无法对原始主假设做端到端验证。

**建议方案**：

1. 冻结输入/输出契约：蛋白节点 embedding、PTM site embedding、PPI/kinase-substrate edge schema、cell graph schema、Δ expression/state target schema；
2. 增加 `MultiPLMEncoder`，明确 Ankh39、ESM-2、ProtT5 的加载、维度投影、冻结策略和缺失资产行为；
3. 增加 `CIGNNSignalBridge`，实现图消息传递和可单测的敏感性矩阵计算，禁止静默使用 fully-connected approximation 作为生产路径；
4. 增加 `CellGraphCompassHead` 或等价 decoder，明确表达变化输出、loss、masked gene 处理和评估指标；
5. 先接入 PMADS + Ridge baseline，再逐步接入复杂模型；
6. 用固定小型 fixture 验证维度、梯度、确定性和 checkpoint round-trip，再使用真实数据做端到端实验；
7. 为模型输出写 provenance：模型版本、数据 manifest、是否 fallback、图版本、随机种子。

**实施估计**：4–8 周，分为契约/基线 3–5 天、数据与图桥 1–2 周、decoder 1–2 周、集成训练与验证 1–3 周。

**所需资源**：预训练模型权重、PPI/激酶-底物图、单细胞扰动数据、GPU、数据许可确认；至少 1 名 ML 工程师和 1 名生物信息学工程师。

**潜在风险**：三种 pLM 的 tokenizer/维度/许可不同；图数据版本差异会改变结果；Cell-Graph-Compass 的原始论文/数据契约需要进一步确认。若原始资料不足，应先将实现目标明确为“项目内可复现等价模型”，避免宣称论文架构复刻。

### 7.2 TD-02：补齐数据清单和 Ridge baseline（严重）

**问题与影响**：当前加载器并未覆盖原需求列出的全部数据源，且没有一个可复现的 PMADS + Ridge 对照管线。没有数据版本和简单基线，复杂模型的收益无法判断。

**建议方案**：

1. 建立 `data/manifests/<dataset>.yaml`，记录来源 URL、发布日期/版本、许可、下载文件 SHA-256、字段映射和预处理版本；
2. 为每个实际接入的数据源实现最小 loader、schema validator 和脱机 fixture；
3. 对不可公开下载或许可不明的数据只提供 manifest schema 和用户本地导入接口，不伪造数据；
4. 实现 `scripts/baseline_pmads_ridge.py`，固定 split、标准化、训练、评估、artifact manifest 和结果表；
5. 将 baseline 纳入 CI 的小 fixture smoke test，并在报告中与主模型使用同一 split 和指标。

**实施估计**：1–2 周；若数据许可或下载受限，额外等待时间单独记录。

**所需资源与风险**：数据管理员/生物信息学工程师、数据存储空间和许可确认。最大风险是源数据库变更或无法合法分发；应保留 manifest 和用户本地路径，不把敏感/受限原始数据提交到仓库。

### 7.3 TD-03：统一依赖和锁定策略（高）

**问题与影响**：核心 requirements 与 ignored lock 的 NumPy 约束冲突，lock 也未进入版本控制，无法保证部署复现。

**建议方案**：

1. 选定 Python 版本和 CPU/CUDA 安装 profile；
2. 用当前支持的依赖解析器从 `requirements-core.txt`、`requirements-pretrained.txt` 等重新生成按 profile 分开的 lock 文件；
3. 明确 torch CUDA index/source，避免把本机 CUDA wheel 隐式当成通用依赖；
4. 将 lock 文件纳入 Git，或在项目规范中明确可重建的 hash-pinned 生成流程；
5. CI 执行 clean-install、import smoke、pytest smoke 和版本冲突检查；
6. 删除/更新 `requirements-pretrained.txt` 中关于 torchmetrics 的过时说明。

**实施估计**：1–2 个工作日；CUDA 矩阵验证另计。

**风险**：升级 NumPy/Pydantic/Lightning 可能触发行为变化，应使用分 profile、逐步升级和回归测试。

### 7.4 TD-04：收紧 DAVF 资产策略（高）

**问题与影响**：`davf_inference.py` 为开发可用性提供 zero-feature fallback；在真实资产缺失时仍可能返回结构化结果，存在把占位结果误用为科学结果的风险。

**建议方案**：

1. 生产默认启用 strict asset mode，checkpoint/latent model 缺失时返回明确错误；
2. 仅在显式 `development/mock` 配置下允许零特征 fallback；
3. 响应和 artifact 增加 `asset_status`、`fallback_used`、`checkpoint_sha256`、`model_kind`；
4. 为 legacy checkpoint 和新 checkpoint 分别建立固定 fixture，测试输入 shape、输出 shape、方向和缺失资产行为；
5. 在可用的受控环境运行 real-assets 测试，并将资产要求写入部署检查清单。

**实施估计**：3–5 个工作日加真实模型资产准备时间。

**风险**：严格模式可能使旧部署启动失败；应提供显式迁移开关和清晰错误信息，不能静默退回占位输出。

### 7.5 TD-05：修复多 worker 可观测性和限流一致性（高）

**问题与影响**：限流状态在进程内，而 Docker 默认 4 workers；请求可能被不同 worker 接收，限流和指标不是实例级全局值。

**建议方案**：

1. 短期：生产默认单 worker 或在文档中明确限流语义；
2. 中期：引入 Redis/共享存储实现原子计数和窗口过期；
3. Prometheus 使用 multiprocess mode 或外部聚合；
4. 增加多 worker integration test，验证跨 worker 的限流和指标语义；
5. 将限流算法、窗口、burst、异常时行为写入 API 文档。

**实施估计**：2–4 个工作日，取决于部署基础设施。

**风险**：引入共享存储增加运维依赖；若部署规模小，保留单 worker 作为明确的低复杂度 profile。

### 7.6 TD-06：消除测试盲区和恒真断言（高）

**问题与影响**：real-assets 默认 skip，且 external service 测试有恒真断言，绿色结果不足以证明真实服务链路。

**建议方案**：

1. 把 `assert result is not None or True` 改为对明确返回结构、状态码或错误类型的断言；
2. 将“服务不可用时应 graceful fallback”和“服务可用时应解析真实结果”拆成两个测试；
3. 为真实模型/外部服务提供受控 nightly profile，不把凭据、受限数据或大模型权重提交到仓库；
4. 默认 CI 运行覆盖率和关键模块阈值检查；
5. 将 skip 原因、所需环境变量和资产版本记录到测试报告。

**实施估计**：1–3 个工作日完成测试修复和 CI 接线；真实资产环境另计。

**风险**：外部服务不稳定、网络和许可导致 nightly 波动；应采用 fixture/VCR 或本地服务替代，真实测试只做低频验证。

## 8. 文档与检测报告归档处理

### 8.1 本次识别并归档的过时文件

归档目录：[archive/20260808/](archive/20260808/)。完整记录见 [archive/20260808/MANIFEST.md](archive/20260808/MANIFEST.md)。

| 类别 | 归档内容 | 原因 |
|---|---|---|
| API 文档 | 旧版 `API_DOCUMENTATION.md` | 端点、配置和当前 API 行为已演进，旧文档缺失 `/live`、`/ready`、`/metrics`、`/initialize` 等当前信息 |
| 项目/技术/文件说明 | 3 份 `docs/PTM2CellNet_*.md` | 文件数、测试数、配置和架构说明与当前代码不同 |
| 旧检测/评估报告 | 2026-08-04 系统复核报告、E2E 能力评估 v2 | 报告基线和测试统计已被本轮实际验证结果替代 |
| 原始需求原件 | `蛋白质PTM与细胞状态预测项目需求.docx` | 原件是未纳入 Git 的根目录输入，保留在日期归档中用于需求追踪 |
| 检测产物 | `.coverage`、`coverage.json` | 环境相关的覆盖率快照，不作为当前质量结论；保留用于审计溯源 |

项目原有的 `docs/archive/` 已包含更早的历史报告、设计文档和技术记录，本轮没有重复搬迁，而是在其 README 中增加了 2026-08-08 根归档批次说明。治理文件 `AGENTS.md`、`CLAUDE.md`、`coding-guide.txt` 不属于过时检测报告，未移动。

### 8.2 版本和原始记录保留

- 已跟踪文件先使用 `git mv` 移入归档，再在原路径保留当前入口页；最终提交分别记录归档原件和入口页，原始提交、版本、路径和 SHA-256 已写入 manifest；
- 原本被忽略的 DOCX 和检测产物没有 Git 历史，因此在 manifest 中记录原路径、文件大小、修改时间和 SHA-256；
- 原活动路径保留入口页，指向归档原件和当前权威文档；
- `.gitignore` 已增加对 `archive/` 及归档检测产物的例外，确保本次归档可以提交。

## 9. 版本控制处理

本次工作在本地 `main` 分支完成。审计开始时没有创建临时分支，原因是用户要求将所有已完成修改合并到本地主分支，而当前工作分支本身就是 `main`。已完成以下检查：

1. 所有用户已有代码修改均保留并纳入同一提交；
2. 归档文件、入口页、状态文档、部署文档和报告均纳入提交；归档原件与原路径入口页分别保留；
3. `git diff --cached --check -- . ':(exclude)archive/**'` 对本轮新增/修改内容无空白错误；归档正文保留原始历史空白；
4. 本地 `main` 上执行 `git merge --ff-only main` 作为无冲突的自合并确认；
5. 不执行远端 push，也不将落后的 `origin/main` 强行合并进本地，因为用户要求的是本地版本库操作，且远端分支不是本次审计目标。

提交已完成于本地 `main`；交付时以 `git log -1 --format=%H` 读取最终哈希。执行 `git merge --ff-only main` 返回 `Already up to date.`，确认无需额外合并且无冲突。

## 10. 验证方式与实际结果

### 10.1 已实际执行

| 命令 | 实际结果 |
|---|---|
| `python -m pytest -q tests/unit/test_architecture_davf_integration.py tests/unit/test_legacy_davf_compat.py tests/unit/test_ptm_module_switch.py` | 26 passed，13 warnings，6.69s |
| `python -m mypy src --show-error-codes` | `Success: no issues found in 124 source files` |
| `python -m ruff check src scripts tests` | `All checks passed` |
| `python -m compileall -q src scripts tests` | 退出码 0 |
| `python -m pytest tests/unit -q` | 1558 passed，4 skipped，17 warnings，74.65s |
| `python -m pytest tests/integration -q` | 67 passed，1 skipped，16 warnings，94.14s |
| `python -m pytest tests/e2e -q` | 42 passed，21 warnings，177.46s |
| `python -m pytest tests/real_assets -q` | 8 skipped，0.02s；未启用真实资产环境变量 |

### 10.2 警告与未覆盖边界

测试没有失败，但输出包含：

- Pydantic v1 validator/Config 弃用警告；
- Starlette TestClient/httpx 相关弃用警告；
- Mamba `torch.cuda.amp.custom_fwd/custom_bwd` 弃用警告；
- Lightning dataloader worker 数量提示。

这些警告已记录为 TD-08/TD-12，不被误报为通过条件。real-assets 测试未实际加载真实 checkpoint、外部数据库或外部服务，因此真实资产结论仍为“未验证”。

## 11. 子智能体调用统计

本次按用户要求启动 3 个只读探索子智能体，分别负责需求/接口、完成度、技术债方向的独立扫描。它们在等待阶段未返回可用 artifact，随后被安全关闭；没有修改文件，也没有将未返回的推测作为本报告证据。报告中的结论全部由主智能体直接读取代码、文档并执行验证得出。

| 智能体名称 | 调用次数 | 主要执行任务 | 结果/是否修改文件 |
|---|---:|---|---|
| Boyle | 1 | 需求章节、接口缺口、业务流程扫描 | 未返回可用报告；0 文件修改 |
| Socrates | 1 | 功能完成度和实现-需求差异扫描 | 未返回可用报告；0 文件修改 |
| Nietzsche | 1 | 技术债、依赖和质量风险扫描 | 未返回可用报告；0 文件修改 |
| **合计** | **3** | **只读并行探索** | **0 文件修改；未作为证据来源** |

## 12. 最终结论与后续优先级

### 12.1 当前可交付状态

当前工程可以作为“现有 PTM 预测、DAVF/图扰动扩展和 API 服务”的可测试版本交付；代码质量门禁和默认 mock 测试均通过，历史文档和检测快照已经归档，当前状态和分析入口已经更新。

### 12.2 不应做出的结论

不能把当前的 pathway heuristic、GenKI ranking、synthetic fallback 或单模型 pLM encoder 宣称为原始需求中完整的 CIGNN + Cell-Graph-Compass 跨尺度模型。也不能把 real-assets 测试的 skip 解释为真实模型资产验证通过。

### 12.3 推荐顺序

1. 先冻结 R-01～R-07 的数据、模型和输出契约；
2. 建立 PMADS + Ridge baseline 和数据 manifest；
3. 再实现 CIGNN/敏感性矩阵与 cell-state decoder；
4. 同步收紧 DAVF fallback、补真实资产测试；
5. 最后治理依赖 lock、多 worker 限流和弃用警告。

本报告是当前审计基线；后续若需求、数据许可或模型论文定义发生变化，应新增带日期的报告或更新需求矩阵，不应直接覆盖本文件。
