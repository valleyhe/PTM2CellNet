# PTM2CellNet 项目经验与决策记录（lessons.md）

> **文件版本**：2.0.0（SemVer，主版本.次版本.修订号）
> **创建时间**：2026-08-21 22:29 ｜ **最近更新**：2026-08-21 23:35
> **维护规则**：
> 1. 本文件为追加式日志（append-only）：每条记录一旦写入不再改写历史，勘误以新增条目方式补充。
> 2. 每条记录必须包含：唯一编号、精确时间戳（精确至分钟）、SemVer 版本号、决策级别、决策依据与可追溯引用。
> 3. 版本号规则：战略级决策 → 主版本 +1；模块级/流程级决策 → 次版本 +1；笔误勘误 → 修订号 +1。
> 4. 涉及代码执行的重大决策，必须先形成正式方案文档并通过审核后方可动代码。

---

## L-2026-0821-01｜项目架构与研究思路重大调整：终止 E2E KO/KD/OE 模型研发，建立 DAVF × PerturbGen 双路径分析框架

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0821-01 |
| 时间戳 | 2026-08-21 22:29 |
| 条目版本 | v1.0.0（本文件首条记录，同时为文件初始版本） |
| 决策级别 | **战略级**（影响项目 roadmap、模块规划、实验验证候选筛选标准） |
| 决策来源 | 用户正式指令（2026-08-21 晚间下达，本时间戳同步记录入档） |
| 状态 | 已决策；代码修改方案待审核，**未开始任何代码实施** |

### 1. 决策内容

#### 1.1 战略方向调整：终止 E2E KO/KD/OE 模型研发

终止原计划中"训练端到端（E2E）推理的 KO/KD（基因敲除/敲低）或 OE（过表达）模型"的研发工作。

- 该方向的历史背景与修复记录见 `docs/E2E训练与推理能力评估报告_2026-08-04_v2.md`、`docs/E2E训练与推理代码修复报告_2026-08-08_v2.md`。
- 终止含义：不再投入新的训练与迭代成本；已有代码不据此规划新增功能。

#### 1.2 新研究框架：DAVF × PerturbGen 双路径整合

将项目现有的 **DAVF 模型**与 **PerturbGen 模型**进行系统性整合，构建双路径分析体系。

**实施方法**：在明确候选基因的表达变化方向后，针对需要预测的基因，在指定的候选细胞类型上执行以下两套独立分析流程：

##### 分析路径 1：早期/预防性干预模拟

| 要素 | 内容 |
|---|---|
| 实验条件 | `normal`（正常状态）→ `disease`（疾病状态）的状态转移 |
| 干预方式 | source KO/KD 或 OE：在疾病发生前，对 source（正常状态）中的源基因执行敲除/敲低或过表达（PerturbGen `perturbation_sequence: "src"`） |
| 分析目标 | 评估早期干预对疾病发生发展的预防效果 |

##### 分析路径 2：治疗性干预模拟

| 要素 | 内容 |
|---|---|
| 实验条件 | `disease within-state`（疾病状态内，不跨状态转移） |
| 干预方式 | KO/KD 或 OE：在疾病已形成状态下进行基因扰动（PerturbGen `perturbation_sequence: "tgt"` + `pert_tps`） |
| 分析目标 | 评估疾病状态下的治疗性干预效果 |

**扰动方向选择逻辑**：候选基因的表达变化方向（上调/下调）由 DAVF 侧的 PTM → 基因表达方向映射给出（现有实现：`src/models/ptm_direction_mapper.py`）；疾病中高表达的基因优先做 KO/KD 模拟（PerturbGen `mask` 模式为主，`pad`/`delete` 做敏感性分析），疾病中低表达的基因做 OE 模拟（PerturbGen `overexpress` 模式）。

#### 1.3 候选基因判定标准（决策标准）

**当且仅当**候选基因在上述两套分析流程中，经过模拟扰动后**均能稳定地**促使细胞状态向正常方向转变时，方可判定该 PTM 对该基因的表达扰动具有潜在的实验验证价值，纳入后续实验验证候选清单。

"稳定地向正常方向转变"的操作性要求（依据参考资料的讨论结论）：

1. rescue score（疾病分数下降）在两条路径均为正；
2. 计算 rescue 时必须**排除目标基因本身**，避免"KO 后目标基因表达自然下降"造成的假阳性；
3. 多数 donor 方向一致，效应不由单一 donor 驱动；
4. 不依赖特定随机种子、配对方式或扰动模式（`mask`/`pad`/`delete` 方向一致）；
5. 评估以疾病 signature 逆转、DEG 方向逆转、通路/GSEA 方向为主，UMAP 位移仅作展示、不作为主要证据。

### 2. 决策依据与可追溯性记录

#### 2.1 关键讨论节点（按时间线）

| # | 时间 | 节点 | 内容摘要 |
|---|---|---|---|
| 1 | 2026-08-21（日间，见参考资料 A/B 整理时间） | PerturbGen 技术调研 | 完成 PerturbGen 输入数据要求梳理（参考资料 A）与官方资源、预训练权重使用方式、双路径 KO 分析框架论证（参考资料 B） |
| 2 | 2026-08-21（参考资料 B 第一节 Q&A） | 双路径有效性标准讨论 | 讨论中曾论证"两套结果都有效通常是较强优先级信号，但并不要求二者必须同时有效"（宽松 OR 标准，适用于一般性靶点排序）；同时明确"排除目标基因后仍向正常恢复才能称为系统性状态恢复"的反假阳性原则 |
| 3 | 2026-08-21 22:29（本条目时间戳） | 用户正式决策 | 采用**比讨论稿更严格的 AND 标准**：两条路径均稳定向正常转变才进入实验验证候选清单。此为项目筛选门槛的最终口径，与讨论稿的 OR 标准差异属有意决策，后续分析报告不得混用两种口径 |
| 4 | 2026-08-21 22:29 | 执行约束确立 | 代码修改方案与测试方案须先成文、提交审核，**审核通过前不得动代码**；PerturbGen 预训练权重由用户手动下载保存，agent 不得自动拉取大权重文件 |

#### 2.2 决策依据摘要

1. **终止 E2E 路线的原因（第一性原理）**：自训 E2E KO/KD/OE 模型需要大规模配对扰动数据与训练成本，而 PerturbGen 已在超过 1 亿单细胞转录组上完成预训练，并提供冻结 encoder + 数据集特异下游适配（transition decoder + count decoder）的官方工作流；复用它比自训 E2E 更优。
2. **PerturbGen 预训练权重的正确用法**：不能零样本直接推理；官方工作流 = 冻结预训练 scMaskGIT encoder → 在自己的 normal/disease 配对数据上训练 masking/transition model → 训练 count decoder → 用自己数据的 checkpoint 做 KO/OE 推理（参考资料 B 第三节）。
3. **双路径设计依据**：`src` 扰动评估早期干预对后续状态的影响，`tgt` 扰动用于 within-state 分析，均为 PerturbGen 官方支持的扰动施加方式（`perturbation_sequence: src/tgt`）。
4. **KO 执行前提**：目标基因必须实际出现在被扰动状态的 token 序列中；"疾病中特异诱导、正常中几乎不表达"的基因，其 normal-source KO 阴性不应直接否定靶点（参考资料 B 第一节第 4/7 节）——分析报告解读时必须携带该警示。

#### 2.3 参考资料（决策输入）

| 编号 | 文件 | 位置 |
|---|---|---|
| A | `PerturbGen对于输入数据的要求.txt` | 项目根目录（`/home/scu/PTM2CellNet/PerturbGen对于输入数据的要求.txt`） |
| B | `PerturbGen_分析与预训练模型使用指南.md` | 项目根目录（`/home/scu/PTM2CellNet/PerturbGen_分析与预训练模型使用指南.md`） |

参考资料 B 中汇总的 PerturbGen 官方资源：

| 资源 | 地址 |
|---|---|
| Web Portal | https://cellatlas.io/perturbgen |
| 官方文档 | https://perturbgen.cog.sanger.ac.uk/ |
| GitHub 仓库 | https://github.com/Lotfollahi-lab/Perturbgen |
| Hugging Face 权重 | https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main |
| bioRxiv 预印本 | https://www.biorxiv.org/content/10.64898/2026.03.04.709254v1（2026-03-05 发布，"Predicting how perturbations reshape cellular trajectories with PerturbGen"） |

### 3. 前期准备工作与执行状态（截至本条目时间戳）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 访问 PerturbGen 官方文档与 GitHub 仓库 | 已完成（2026-08-21 22:2x，GitHub 仓库结构与扰动教程已核实） |
| 2 | 下载并分析预印本论文 | 进行中（bioRxiv 对 PDF 下载限流 HTTP 429，已重试，继续通过其他途径获取） |
| 3 | 获取源代码 | 已完成（`ref/Perturbgen-src`，浅克隆，commit `a9a9375`，2026-08-21 22:2x） |
| 4 | 制定项目代码修改方案（模块整合、接口设计、数据流转、异常处理） | 进行中，方案文档将提交审核 |
| 5 | 制定测试方案（单元/集成/性能/结果验证） | 进行中，方案文档将提交审核 |

**执行约束（再次强调）**：上述方案文档审核通过之前，不执行任何项目代码修改。

### 4. 影响范围

| 范围 | 影响 |
|---|---|
| Roadmap | 移除 E2E KO/KD/OE 相关规划；新增 DAVF × PerturbGen 双路径整合路线 |
| 代码 | 新增 `src/integration/` 下的 PerturbGen 适配层（仿照 GenKI 集成模式）；DAVF 侧输出作为候选基因与扰动方向的输入源 |
| 数据 | 需要符合参考资料 A 全部要求的 normal/disease 配对 scRNA-seq AnnData（raw counts + Ensembl ID + cell_type/state/donor metadata） |
| 实验验证 | 候选清单筛选门槛变更为 1.3 节的双路径 AND 标准 |

---

### 补充记录：前期准备工作完成情况（2026-08-21 22:44，追加不改写上文）

| # | 任务 | 最终状态 |
|---|---|---|
| 1 | 访问 PerturbGen 官方文档与 GitHub | ✅ 完成（仓库结构/扰动教程/训练教程已核实） |
| 2 | 预印本论文分析 | ✅ 完成：PDF 直链持续限流（HTTP 429，三次重试），已通过全文 HTML 渠道完成核心内容分析并存档 `paper/perturbgen_preprint_notes_20260821.md` |
| 3 | 源代码获取与研究 | ✅ 完成：`ref/Perturbgen-src`（commit `a9a9375`），源码研究报告（入口/配置/扰动实现/依赖/输出物/坑位）已并入方案文档 |
| 4 | 代码修改方案 | ✅ 已成文：`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`（PROPOSAL-20260821-01），**待审核** |
| 5 | 测试方案 | ✅ 已成文（同上文档第三部分），**待审核** |

> 审核通过并开始实施后，以新条目记录实施情况。

---

## L-2026-0821-02｜DAVF 基因表示底座切换：Geneformer → PerturbGen encoder，移除 Geneformer 依赖

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0821-02 |
| 时间戳 | 2026-08-21 23:35 |
| 条目版本 | v2.0.0（战略级决策，文件主版本 +1） |
| 决策级别 | **战略级**（改变 DAVF 核心基础设施依赖） |
| 决策来源 | 用户正式指令（2026-08-21 23:3x，作为对底座问题分析的直接批复） |
| 状态 | 已决策；已并入待审核方案 PROPOSAL-20260821-01 v1.1（§2.11），**未开始实施** |

### 1. 决策内容

1. DAVF 的基因嵌入底座由 ctheodoris/Geneformer 切换为 PerturbGen 预训练 encoder（scMaskGIT）的 token embedding；
2. 切换完成后移除 Geneformer 模型依赖（两步走：M3 链路切换后旧文件标 deprecated，Gate-E 回归验证通过后于 M5 删除 `src/models/geneformer_embedding.py` 与专属测试；`transformers` 等库因 ESM-2/ProtBERT 保留）；
3. 该决策不作为远期课题，纳入本次整合方案范围一并实施。

### 2. 决策依据（可追溯）

| # | 节点 | 内容 |
|---|---|---|
| 1 | 2026-08-21 22:5x | 分析确认：Geneformer 是 DAVF 链的基因表示底座（`davf.py:21` 持有 loader、`davf_inference.py:521` 推理取嵌入），双路径整合后架构上仍需保留 |
| 2 | 2026-08-21 22:5x | 同轮分析发现既存缺陷：`geneformer_embedding.py:227` vocab 为数字字符串索引，真实基因名全部落入 sha256 哈希取行——"Geneformer 嵌入"的实际贡献名不副实 |
| 3 | 2026-08-21 23:35 | 用户决策：放弃"远期评估统一底座"的保守路线，本次修改即把 DAVF 底座统一到 PerturbGen encoder，省掉 Geneformer 依赖 |

### 3. 关键实现约束（已在方案 §2.11 落实）

1. PerturbGen 依赖不可进主环境（torch 2.5.1 等硬冲突）→ 嵌入资产"独立环境一次性导出（safetensors + token_dict.json + manifest）+ 主环境轻量加载"，新 `PerturbGenEmbeddingLoader` 零 PerturbGen 依赖；
2. 嵌入维度变化 → 已训 DAVF checkpoint 不兼容，须 `scripts/finetune_davf.py` 重训；
3. **Gate-E 回归验证门**：固定 PTM 基准集上方向映射一致率与下游指标不低于旧底座（或达约定阈值，方案 Q7），未通过不得删除旧依赖；
4. 顺带收益：以真字典（gc95M）替换哈希空转 vocab，修复节点 2 所述缺陷。

### 4. 影响范围

| 范围 | 影响 |
|---|---|
| 代码 | `davf.py` / `davf_inference.py` / `ptm_direction_mapper.py` 的 loader 注入点切换（数值逻辑不动）；新增 `src/models/perturbgen_embedding.py` 与 `src/integration/perturbgen/embedding_export.py` |
| 资产 | Geneformer 模型权重不再需要；新增 encoder ckpt 嵌入导出物（用户已承诺手动下载 encoder 权重） |
| 训练 | DAVF 需在新底座上重训/微调一次 |

---

## L-2026-0822-03｜DAVF × PerturbGen M1–M3 工程主干完成，M4 严格受 Gate-0 阻断

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0822-03 |
| 时间戳 | 2026-08-22 00:52 |
| 条目版本 | v2.1.0 |
| 决策级别 | 模块级/流程级 |
| 决策来源 | 用户要求按 PROPOSAL-20260821-01 v2.0 重新分析并实施 |
| 状态 | M1–M3 工程主干完成；M0 BLOCKED；M4 未开始 |

实施确认：主进程已具备严格数据/候选契约、隔离 runner、双路径 src/tgt 计划、rescue/null/FDR/AND 判定和可追溯报告；聚焦回归 146 项通过。真实 encoder 权重、Python 3.11 独立环境和合规 cohort 缺失，因此不得猜 embedding key、接入 DAVF runtime、删除 Geneformer 或声称 Gate-E 完成；PerturbGen 动态输出 discovery 也必须等 Gate-0 真环境后冻结。

---

## L-2026-0822-04｜PerturbGen 产物契约必须按上游路径公式与 manifest 绑定

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0822-04 |
| 时间戳 | 2026-08-22 01:35 |
| 条目版本 | v2.2.0 |
| 决策级别 | 模块级/流程级 |
| 状态 | 已实施 |

实施确认：PerturbGen tokenise 产物不在本地 stage 目录，checkpoint 与 perturb h5ad 也是动态文件名。桥接层现在按上游公式精确登记 tokenise 六个具名产物，对训练/扰动产物采用“隔离目录 + 唯一 glob”，并把真实路径与 hash 写入 manifest 供下游显式引用；零匹配、多匹配、旧外部产物或 hash 变化都直接失败，禁止使用 latest-mtime 猜测。

---

## L-2026-0822-05｜正式证据必须可提交、可留存且单次运行自洽

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0822-05 |
| 时间戳 | 2026-08-22 |
| 条目版本 | v2.3.0 |
| 决策级别 | 流程级 |
| 状态 | 已实施 |

实施确认：正式 Gate-4 不接受多次残缺 benchmark 的并集；每个样本必须独立包含双路径成功、两个 h5ad 与有效资源指标。workflow 必须被 `.gitignore` 精确放行，benchmark JSON 必须写入 `outputs/real_assets/` 而非 pytest 临时目录，否则即使本机测试通过也无法形成可提交、可上传、可审计的发布证据。

---

## L-2026-0822-06｜通用 scPerturb 数据不能冒充冻结 normal/disease donor 队列

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0822-06 |
| 时间戳 | 2026-08-22 |
| 条目版本 | v2.4.0 |
| 决策级别 | 数据契约级 |
| 状态 | 已确认 |

实施确认：本地存在 h5ad 不等于 Gate-0 cohort 已具备。只有显式 donor/patient、normal/disease 配对、raw counts、ENSG 和至少 3 个可评估 donor 同时成立才可进入 preflight；禁止把 `sample`、`batch`、`replicate` 或 CRISPR control 自动解释成 donor/疾病状态。当前 30 个 scPerturb h5ad 中 4 个截断，唯一显式 patient 队列仅 2 人且全为 healthy，因此合规候选为 0。

---

## L-2026-0901-01｜DAVF 负责方向筛选，PerturbGen 负责下游效用

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0901-01 |
| 时间戳 | 2026-09-01 |
| 条目版本 | v2.5.0 |
| 决策级别 | 主线架构级 |
| 状态 | 已确认 |

实施确认：项目主线采用“PTM site 预测 → DAVF 方向预测/筛选 → DAVF × PerturbGen 双路径效用评估”。`DAVFInferenceModule` 主要输出基因表达变化方向及置信证据，不作为最终表达效用预测器；方向验证必须依赖独立的 normal/disease 表达证据，避免循环验证。PerturbGen 负责 `source_intervention` 与 `within_state` 两条路径的 rescue、DEG、signature 和 pathway 效用评估；最终候选必须同时满足方向证据门槛与两条 PerturbGen 路径通过。

## L-2026-0902-01｜正式 DAVF 只接受当前 LatentDAVF + 当前 scVI/asset 契约

正式方向证据必须来自 schema v2 的当前 `LatentDAVF` 完整 state dict，固定 `latent_dim=64`、`num_genes=4018`，并逐值使用冻结的 PerturbGen embedding asset。旧 `delta_mlp`/legacy checkpoint 只能保留历史 feature path；训练 pair 必须由同一 scVI 模型生成并携带 scVI gene order 与 PerturbGen manifest，运行时目标 decoder index 只能从 `adapter.gene_names` 解析，不能复用 PerturbGen token index。

## L-2026-0902-02｜E2E 必须是串联门控，不能让 runner 绕过 DAVF

当前主线新增严格编排器：显式 KO/KD route 的 DAVF 方向证据先经过 PTM proposal、DAVF 输出和独立表达方向三方 gate，只有通过的候选才生成 PerturbGen 两路径 invocation；外部六阶段仍由既有 runner 管理。KO/KD 继续使用独立 scVI 坐标系，最终只按 canonical Ensembl ID 合并；缺少合规 normal/disease donor cohort 时只能验证桥接和执行契约，不能把 PerturbGen smoke 或不完整 artifact 宣称为生物学 PASS。

## L-2026-0902-03｜真实桥接通过不等于 PerturbGen 生物学验收

实施确认：真实 KO/KD LatentDAVF、scVI 与 embedding asset 已通过 schema/shape/方向桥接和 CUDA 冒烟；E2E CLI 只在三方方向 gate 通过后创建 PerturbGen invocation。六阶段 PerturbGen 仍必须使用满足 normal/disease、raw counts、至少 3 个 donor 的正式 cohort，当前本机 smoke 数据只能验证 runner 契约，不能推出效用 PASS。

## L-2026-0903-01｜RustDesk systemd 环境修复不是远端输入法的充分条件

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0903-01 |
| 时间戳 | 2026-09-03 00:03 |
| 条目版本 | v2.6.0 |
| 决策级别 | 流程级 |
| 状态 | 已确认，继续排障 |

实施确认：用户执行 RustDesk/IBus systemd 修复脚本后问题仍存在，说明“RustDesk 客户端继承本机 IBus 环境”不能直接推出“远端文本框可用中文输入”。后续必须严格区分本机组合输入链路与远端输入法链路，单独验证 RustDesk 键盘模式、XTEST/按键传输以及远端 IBus/Fcitx 状态；在完成远端真实文本框回归前，不得把服务环境修复宣称为端到端解决。

## L-2026-0913-01｜研究契约复审：方向语义、证据独立性与双路径执行边界

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0913-01 |
| 时间戳 | 2026-09-13 |
| 决策级别 | 主线研究契约级 |
| 决策来源 | 2026-09-13 研究讨论与当前代码/方案复核 |
| 状态 | 已确认；文档与后续验收入口已同步，正式科学证据仍待真实 cohort |

### 决策

1. 候选准入固定为外部 PTM proposal/candidate_spec → scVI/PTM 映射 → DAVF
   方向与置信证据 → 独立 donor-level 表达方向三方 gate → evidence/invocation。
   `src/models/contracts.py` 的 classifier 只表示 site presence；
   `scripts/build_candidate_spec.py` 接收外部假设或逐 site direction override，
   不能把 rawsite 分类器写成自动因果表达方向推断。
2. 观测方向按 donor-level disease−normal 记录（`gse_normal_disease.py`），
   DAVF 方向按干预后 decode−当前 context decode 记录
   (`davf_inference.py`)，二者在比较前必须明确 `context`、`intervention`、
   比较基准和研究目标（关联、复现或逆转）。不得用全局同号/取反规则，
   不得把病程签名、normal/disease 对比或来源复用写成 KO/KD 干预效应
   ground truth 或三个统计独立证据；结果只能称方向筛选、模型预测或计算效用。
3. `source_intervention=[src]` 是状态转移前场景，
   `within_state=[tgt]+pert_tps` 是目标状态内场景；`dual_path.py` 的正式 AND
   保留。两个场景同时通过才可进入正式候选清单，单场景只能报告相应场景的
   探索结果，不外推为普遍治疗疗效。
4. 六阶段固定为 `tokenise → train_mask → train_decoder → perturb →
   export_gene_embeddings → report`。`orchestrator.py` 当前按 candidate 重做
   tokenise/训练/两场景 perturb/export/report，E2E 按候选目录运行，resume 只
   在同目录身份范围内有效。固定 cohort、词表、训练配置和资产版本后公共
   prepare 一次、候选只做两场景 perturb/效用是目标，不能写成已实现，也不
   新增 hash、调度框架或兼容开关来假设完成。
5. Workflow A 是 gate 后效用评估；Workflow B 是固定基础 encoder → 冻结
   embedding asset → LatentDAVF 重训 → Gate-E 的独立资产生命周期。基础
   encoder export 不消费候选结果、不训练本次新 checkpoint、不回灌本次 DAVF；
   Gate-E 前不删除 Geneformer。

### 当前实现与待验收边界

`run_perturbgen_pipeline.py` 的正式 `perturb`/`--path` 入口要求绑定通过的
E2E gate report，invocation 拒绝 non-pass；低层 `runner.py` 当前只执行
StagePlan，不自行重查 gate。matched-null 生成、候选 empirical-p 聚合、formal
输入隔离、未扰动质量提取、donor split 和 dual-path AND 接口已存在，但 E2E
report 目前只汇总 `stage_manifest`，尚未自动接续真实统计。

2026-09-13 本机 Gate-0 donor audit 为 30 个 scPerturb H5AD 中 26 个可读、0 个
满足真实 normal/disease raw counts、显式 donor、至少 3 个共享 donor、canonical
Ensembl 契约。正式 PASS 还需冻结 scVI gene order、embedding/manifest、真实
matched null、未扰动质量和双场景统计；smoke、synthetic、mock、bridge 和四队列
IBD 规划只证明工程/数据来源准备，不能写成生物学验收。

### 后续顺序

先冻结方向语义、研究目标、来源和 train-only/held-out donor 划分；再落实公共
prepare/reuse 与 invocation 边界；接续现有统计接口；最后在真实 cohort 上完成
Gate-0、Gate-E、Gate-4、Gate-5。相关活动入口为
`docs/CURRENT_STATUS.md`、`project_analysis_20260913.md`、
`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` 和
`.planning/REQUIREMENTS.md`；旧 `project_analysis_20260901.md` 仅作历史快照。

## L-2026-0913-02｜语义上下文与资产身份契约代码化（提交 96ee544）

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0913-02 |
| 时间戳 | 2026-09-13 |
| 决策级别 | 工程契约级（L-2026-0913-01 的代码落地） |
| 决策来源 | 2026-09-13 第三轮契约硬化复核（project_analysis_20260913.md §4.5a/§6.3a） |
| 状态 | 已提交入库并 push；正式科学证据仍待真实 cohort |

L-2026-0913-01 要求的方向语义记录已从文档约束升级为 fail-fast 代码契约：
`src/integration/perturbgen/contracts.py` 的 `SemanticContext` 七字段
（context、intervention、comparison_baseline、reference_axis、
research_objective、evidence_source、cohort）为 formal candidate/invocation
必填，`research_objective` 限 association/replication/reversal；E2E 候选与
`run_perturbgen_pipeline.py` invocation 均校验。同批落地的身份契约：
`NullStageRecord` 绑定 candidate Ensembl/path/mode/seed 并在收集与执行两路
重验；Gate-E 要求 benchmark/vocab/davf 三节证据齐全且 coverage 只认 canonical
ENSG；checkpoint `scvi.gene_names` 强制 canonical ENSG 无后缀无重复（本地旧
`latent_davf_perturbgen_4018` 因此被正确拒绝，对应 pytest 唯一已知失败）；
frozen M6 支持 candidates CSV 行级 modes。E2E 统计接续（F-01）、跨候选公共
prepare（F-02）与 runner 边界（F-09）仍未实现，E2E report 显式写
`statistical_evidence=inconclusive`。

## L-2026-0913-03｜统计接续、共享准备与 donor 行绑定代码化（F-01/F-02/F-03 剩余）

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0913-03 |
| 时间戳 | 2026-09-13 |
| 决策级别 | 工程契约级（L-2026-0913-01 第 4 条与后续顺序的代码落地） |
| 决策来源 | project_analysis_20260913.md §4.5/§6.2（F-01/F-02/F-03 方案 A）与当前代码复核 |
| 状态 | 代码与测试落地；正式科学证据仍待真实 cohort/GPU 资产 |

1. **F-02 共享准备**：`orchestrator.build_shared_prepare_plans` 为每条 route 在
   `<root>/<KO|KD>/_prepare/` 公共执行一次 `tokenise → train_mask → train_decoder`
   （runner manifest/resume 语义不变）；`build_candidate_stage_plans(...,
   skip_prepare_stages=True, prepare_artifact_paths=...)` 使候选循环只执行
   perturb/export/report，`@artifact:tokenise/train_*` 引用由
   `resolve_prepare_artifact_references` 解析到共享产物，缺失引用硬失败。E2E
   (`run_davf_perturbgen_e2e.py`) 单候选与多候选统一走共享准备，报告记录
   `perturbgen_prepare` 与每 run 的 `prepare_root`。未新增 hash、调度框架或
   兼容开关。证据：tests/integration/test_perturbgen_pipeline_mocked.py
   `test_shared_prepare_plans_run_once_and_candidates_reuse_artifacts`。
2. **F-03 生成端 donor 绑定**：`build_scperturb_latent_pairs` 新增
   `donor_obs_column` + `donor_split` payload（`ptm2cellnet.donor_split/v1`），
   train/val 行只来自 train_donors、test 行只来自 held_out_donors，未列入池的
   donor 硬失败；NPZ 写 `target_donors`/`control_donors` 数组，metadata 写
   `donor_split` 与 `dataset.donor_rows`（训练端 `_validate_donor_split_metadata`
   消费的同一契约）。CLI `build_davf_scperturb_pairs.py --donor-obs-column
   --train-donors --held-out-donors` 生成 canonical split。旧 NPZ/checkpoint 无
   行级 donor provenance，正式 held-out 声明仍需重建资产。证据：
   tests/unit/data/test_davf_scperturb.py `test_donor_bound_pairs_*`。
3. **F-01 统计接续**：E2E 新增 `--assemble-statistical-evidence --deg-table
   --null-distribution-manifest [--statistical-output-dir]`，在六阶段完成后自动
   串接 `extract_unperturbed_quality_from_h5ad`（per candidate，primary-mode
   within_state h5ad）→ `build_eval_input_payload`（formal，新增
   `candidate_unperturbed_quality` per-candidate 质量入口）→
   `replay_evaluation.replay_dual_path_evaluation`（自
   `evaluate_perturbgen_dual_path.py` 抽离的单一实现：empirical-p
   conservative_max_required_runs 聚合、BH-FDR、dual-path AND），
   lineage 写回 `statistical_evidence`。null 分布必须显式提供
   （`perturbgen_null_distribution/v1` 索引），GPU matched-null 批跑仍由
   `run_matched_null_stages.py` 独立执行（A-05 资产边界未变）。
   证据：tests/unit/scripts/test_run_davf_perturbgen_e2e.py
   `test_assemble_statistical_evidence_chains_null_quality_pq_dual_path`。
4. **F-09 边界文档化（方案 A）**：`runner.py` 模块契约与两份指南明确 formal
   invocation wrapper（`run_perturbgen_pipeline.py --e2e-gate-report` /
   `PerturbGenInvocation`）是唯一公开正式入口；`StagePlan` 层仅供内部
   null/engineering 复用，runner 不自行重查 gate，该边界未下沉到接口层。
5. **TD-13-13/TD-13-14**：API_DOCUMENTATION.md 增补 PerturbGen 集成合同
   （SemanticContext 七字段等）；`tests/unit/models/test_davf_losses.py` 用精确
   数值断言锚定 DAVFLoss/DirectionConsistencyLoss（幅度项、符号、padding mask、
   自适应权重）。全仓 ruff format 债维持上轮取舍（触碰文件已 format，全仓
   一次性 format 留独立 PR），真实结果记录于修复报告。

## L-2026-0914-01｜Gate-0 增加 between_donor 配对：AD case-control 队列准入（选项 B 决策）

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0914-01 |
| 时间戳 | 2026-09-14 |
| 决策级别 | 研究契约级（用户显式选定选项 B；L-2026-0822-06 契约语义的显式扩展，不是放宽） |
| 决策来源 | data/AD 四队列 Gate-0 审计（outputs/perturbgen/spike/20260914_ad_cohort_audit/evidence.json，verdict GATE0_BLOCKED_SEMANTICS_AND_LABELS）+ 用户选项 B 指令 |
| 状态 | 代码、测试与真实数据 preflight 落地；正式生物学结论仍待完整 E2E 与统计验收 |

1. **配对语义冻结**：`PerturbGenDataSpec` 新增 `pairing` 字段
   （`within_donor` 默认 / `between_donor`），Gate-0
   （`data_prep._validate_obs_contract`）按模式分支：`within_donor` 仍要求
   ≥`min_donors` 个跨态共享 donor（扰动配对语义，默认行为不变，回归
   244 项通过）；`between_donor` 要求两组 donor 不相交（同 donor 双态出现
   即标签错误硬失败）且每组 ≥`min_donors`。E2E 以
   `--perturbgen-cohort-pairing {within_donor,between_donor}` 显式声明，
   汇入 `perturbgen_gate0.data_spec` 序列化证据。观察性队列加载器
   `gse_normal_disease.py` 的 between-donor 语义（显式 normal/disease 样本
   清单、每样本唯一 donor）与本次 Gate-0 扩展对齐，二者互补。
2. **donor 推导证据化**：GSE174367 `SampleID` 被接受为 donor 的唯一依据是
   作者 cell_meta 中每样本一组唯一的供体级协变量（Age/Sex/PMI/RIN/
   Tangle/Plaque，18/18 组合唯一、样本内恒定），由
   `scripts/standardize_gse174367_ad_cohort.py` 在运行时重验，任何协变量
   碰撞即中止；推导记录写入 provenance。标题/`sample`/`batch` 重释为
   donor 的禁令（L-2026-0822-06）不变。版本化 ENSG（含 10x `_PAR_Y` 副本）
   在标准化层去版本并按同基因求和合并（45 条），Gate-0 仍拒收原始输入的
   非 canonical ID。
3. **真实资产**：`data/AD/standardized/GSE174367_ad_cohort.h5ad`
   （61,472 cells × 58,676 canonical ENSG，7 Control + 11 AD donor 不相交，
   298 个无元数据 barcode 剔除并记录）；between_donor Gate-0 preflight
   7/7 细胞类型 PASS，证据
   `outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json`。
   这是数据契约验收（Gate-0），不构成生物学 PASS；正式效用仍需完整
   六阶段、matched null、质量与双路径统计。
4. **不随本次改动的边界**：GSE157827/GSE188545 仍缺 cell-level 注释与
   donor 证据、GSE147528 仍为 raw droplets 且诊断缺失，四队列未合并；
   train/held-out donor split 由既有 `donor_split/v1` 工具在配对生成时
   施加，Gate-0 本身不指定 split。

## L-2026-0914-02｜第七轮修复：between_donor 贯穿 frozen/统计层与 AD 审计口径闭合

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0914-02 |
| 时间戳 | 2026-09-14 |
| 决策级别 | 工程修复级（L-2026-0914-01 契约的代码贯穿完成，语义未变） |
| 决策来源 | project_analysis_20260914.md §6.2（F-10～F-16、TD-14）+ 用户第七轮修复指令 |
| 状态 | 代码、测试、文档与真实 data/AD 重验完成；正式生物学验收仍待 GPU 资产 |

1. **F-10 frozen×pairing（方案 A+B 组合）**：`FrozenCohortManifest` 新增
   `pairing` 字段（默认 within_donor 向后兼容），`_validate_frozen_cohort_asset`
   以 `pairing=manifest.pairing` 构造 Gate-0 spec，missing 检查按 pairing 分支
   （within_donor 逐态全量覆盖；between_donor 并集存在性，每态 ≥3 与不相交仍由
   `prepare_perturbgen_anndata` 强制）；生成端 `run_frozen_acceptance.py
   --freeze --cohort-pairing` 写入，E2E 加载 frozen manifest 时硬校验与
   `--perturbgen-cohort-pairing` 一致。旧 manifest（无 pairing 字段）按
   within_donor 消费；between_donor 冻结验收必须重新 --freeze。证据：
   tests/unit/integration/perturbgen/test_frozen_cohort.py TestBetweenDonorPairing。
2. **F-11 统计 lineage pairing**：eval input 顶层新增 `cohort_pairing`、E2E
   `statistical_evidence.pairing` 与 `deg_columns`；`verify_eval_input_against_manifest`
   新增声明一致性检查（不一致仅使 covered=false，不掩盖其余 issue）。
   证据：test_run_davf_perturbgen_e2e.py
   test_assemble_statistical_evidence_records_between_donor_pairing_in_lineage。
3. **F-12 state 覆盖为 opt-in 契约而非默认**：`build_scperturb_latent_pairs
   --state-obs-column X --require-state-coverage` 仅在显式开启时要求 held-out
   池覆盖 ≥2 个 state 值并记录 `held_out_state_coverage`；无 donor_split 时
   开启该选项为硬失败（不做静默降级）。单态评估设计仍合法，未被禁止。
4. **F-14 审计口径**：audit_ad_cohort_gate0.py 重跑产出
   `GATE0_UNLOCKED_FOR_BETWEEN_DONOR_GSE174367_ONLY`（每队列 gate0_status +
   compute_verdict 动态推导）；20260914_ad_cohort_audit/evidence.json 已被
   本次重跑覆盖为新口径，旧 BLOCKED 结论作废。standardize 脚本 per-sample
   基数检查纳入 Diagnosis，退出码三态 0/2/1。
5. **回归基线**：全量 2671 passed / 1 failed（已知真实资产 checkpoint 基线，
   与第三～五轮同一失败）/ 21 skipped；mypy 166 文件 0 errors。TD-14-01 按
   分析报告 §7.4 建议以 CHANGELOG Versioning note 处置（不为对齐发版），
   下次真实 release 统一 setup.py/__version__。

## L-2026-0914-03｜第八轮：GSE174367 between_donor M6 数据契约冻结与 GPU 前置资产

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0914-03 |
| 时间戳 | 2026-09-14 |
| 决策级别 | 资产冻结级（第七轮修复报告 §4 #1/#2/#3 的 CPU 可执行前置 + 1 项真实缺口修复） |
| 决策来源 | project_repair_report_20260914.md §4 + 用户第八轮指令 |
| 状态 | M6 数据契约冻结完成（真实数据校验 PASS）；六阶段 GPU 执行与统计验收仍未运行 |

1. **M6 冻结（§4 #3，高，闭合）**：GSE174367 EX 完成 between_donor 冻结
   （outputs/perturbgen/frozen/20260914_gse174367_ex/）。donor split 采用
   分层（state 组内）固定 seed 抽样：seed=2 是 0..99 中第一个满足
   train/held-out 四组（held Control/AD、train Control/AD）均含两性的
   seed；held-out Control 3 + AD 3（between_donor 每组 ≥3 的最小可评估
   设计），train Control 4 + AD 8，并集覆盖全部 18 donor（frozen 契约
   要求）。候选为 AD 三通路 5 基因（APP/PSEN1/BACE1/MAPT/APOE，KO，
   外部假设）：ENSG 一律从 cohort h5ad var 查证（禁止凭记忆写 ID），
   token index 在 embedding_asset_20260822 词表全部有效。真实资产上
   `_validate_frozen_cohort_asset` PASS（6,369 cells、无 issue）——F-10
   修复（L-2026-0914-02 第 1 条）在真实数据的首次验证。
2. **真实词表暴露的缺口**：`null_selection._token_values` 原样拒绝含
   `<cls>`/`<pad>` 等特殊 token 的正式 embedding 词表（此前该路径从未
   被真实词表消费）。修复为跳过非 ENSG 键（特殊 token 无基因 rank
   语义），基因 rank 保持词表原始位置（与 mapping 形式的绝对 index
   一致）；版本化 ENSG 仍经 `_canonical_ensembl_id` 归一。教训：
   "合成测试全绿"不覆盖"正式资产首次消费"，match-null/六阶段等
   GPU 前接口应以真实资产预演为准。
3. **matched-null selection 前置资产**：per 候选 99 个表达特征匹配
   null 已用真实 EX 子集生成（mean/detection + token_rank 三真实特征；
   fc 特征在 DEG 表缺省时显式标记 `unavailable/default`，不是静默
   填充）。DEG 表（donor-level disease−normal）就绪后可重生成以启用
   fc 维度。selection manifest 只绑定 cohort 特征与候选 ENSG，不依赖
   E2E 报告，故可先于六阶段冻结。
4. **接口事实（GPU runbook 要点）**：① E2E context 直接用标准化 cohort
   h5ad（Gate-0 在内存副本按 cell_type 校验，tokenise 输入保持原样
   canonical）；② matched-null CLI 只提供 --dry-run 计划，GPU 执行走
   `null_generation.run_matched_null_stages` Python API，rescue 分数用
   `results.summarize_rescue_by_donor`（per-donor held-out signature，
   R=S(unperturbed)−S(perturbed)）绑定 perturb 输出；③ candidate_spec
   的 position(≥1)/ptm_type 是契约必填——AD 基因级候选需文献 PTM site
   锚点 + 外部方向假设 + DEG observed 证据，属研究输入不可自动生成。
5. **口径不变**：M6 冻结、selection 与 preflight 都是数据契约验收；
   生物学 PASS 仍需六阶段 GPU 真实运行、≥99 matched nulls、未扰动
   质量与双路径统计（沙箱 NVML mismatch，GPU 须脱离沙箱执行）。

## L-2026-0914-04｜第八轮补充：词表特殊 token 必须精确白名单，runbook 示例必须可解析

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0914-04 |
| 时间戳 | 2026-09-14 |
| 决策级别 | 硬失败边界与文档契约修复 |
| 决策来源 | 用户第八轮补充核查 + `project_analysis_20260914.md` §4 |
| 状态 | 代码、最小测试、bridge runbook 已修复并验证；正式 GPU/生物学执行仍未运行 |

1. **`null_selection._token_values`**：正式词表实际包含且仅包含
   `<cls>`、`<eos>`、`<mask>`、`<pad>` 四个特殊键；只能跳过这四个明确 token。
   `<unk>`、任意非法 gene key、负数或非整数特殊 token ID 均硬失败，不能用
   `except ValueError: continue` 静默吞掉。该约束同时适用于 mapping 和 sequence
   词表；mapping 的特殊 token ID 仍校验为非负整数。真实词表核对为 18,967 个键、
   18,963 个 canonical gene rank + 4 个特殊键；`{"<unk>": 4}` 已实测硬失败。
2. **`docs/guides/perturbgen_bridge.md` §3**：第 2 步补齐真实 parser 要求的
   `--perturbgen-config`、`--seeds 0,1,2`、`--sensitivity-modes pad,delete`，与
   5 候选 × 2 path × 3 seed × 3 mode = 90-run frozen plan 对齐；新增从 E2E
   `statistical_evidence.eval_input/report_manifest` 读取实际路径的
   `run_frozen_acceptance.py --verify` 命令。缺失或篡改资产由 CLI 非零退出，不能
   解释成 PASS。
3. 同一节的 matched-null Python 片段改成语法有效的接口骨架，使用真实签名的
   `rescue_extractor(request, stage_result)` 与 `output_path=`。骨架明确要求外部
   研究代码绑定登记的真实 perturb h5ad、per-donor 表达和 donor-level DEG，未实现
   时显式 `NotImplementedError`，不伪造 rescue 分数或研究输入。

## L-2026-0914-05｜第八轮后续：PTM activity → AD 交集主线契约层落地

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0914-05 |
| 时间戳 | 2026-09-14 |
| 决策级别 | 新主线代码契约层（方案 §6.2 全量） |
| 决策来源 | 用户 /goal 指令 + `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` |
| 状态 | 5 模块 + 4 CLI + 84 新测试全绿（合成契约级）；真实 PTM/网络/benchmark 资产未到位 |

1. **落地范围**：方案 §6.2 清单全部实现——`src/analysis/ptm_research_config.py`
   （阶段 0 冻结契约，含七字段 semantic_context 模板与 `{cell_type}` 渲染）、
   `ptm_activity.py`（§4.1/§4.2/§5.1）、`signed_network.py`（§4.3/§5.3）、
   `ptm_gene_score.py`（§4.4/§5.4/§4.6）、
   `src/integration/perturbgen/downstream_target_evaluation.py`（sidecar +
   delta 评估），CLI `run_ptm_activity.py`/`build_ptm_global_gene_scores.py`/
   `build_ptm_ad_intersections.py`/`build_celltype_candidate_specs.py`。
   KSTAR/PhosR 运行、OmniPath signed 网络导出、E2E 六阶段执行不在本轮范围。
2. **传播算法冻结**：有符号简单路径枚举，路径在 `gene_edge_types` 边终止于
   基因，per (source,target) 聚合 `activity × Π(sign_e×confidence_e) ×
   decay^length`；平行边同号取最强 confidence、异号整体剔除并计数（禁止静默
   选边）；score 表一行一个 (source_activity, target_gene) 对，不聚合成
   per-gene 单值——多 source 同 gene 不同向是真实状态，聚合规则不得隐式发明。
   `network_coverage`=该 source 全部 gene 路径中到达该 target 的比例，
   `degree_normalized`=每路径平均贡献。无独立 null 前 `prediction_status`
   保持 `direction_only`，禁止写 PTM 侧 q 值。
3. **语义边界落实在代码**：① source 无该 cell type DEG → 只进 exploratory
   清单，不生成正式候选行（target-set 一致性不能替代 source 三方 gate）；
   ② `context_cell_index` 绑定该 cell type 在 cohort 的第一个 cell 位置
   （positional，非行标签）；③ `--embedding-vocab` 下 source 无 token 硬失败；
   ④ matches_predicted/matches_observed 分字段，不合并；⑤ 无 total protein 的
   行归一化列保持 NaN 不回填原始值。candidate spec 与现有
   `ptm2cellnet.candidate-spec/v1` 及 E2E `downstream_required` 字段完全兼容。
4. **本轮自检修复的实现缺陷**（测试驱动暴露）：① TSV gene map 的 csv.DictReader
   漏 `delimiter="\t"`；② 传播聚合漏乘 `decay^length`（docstring 声明了但代码
   没做，单元测试手算期望值抓出）；③ groupby 聚合 `reset_index(drop=True)` 丢
   key 列；④ `context_cell_index` 误取行标签（anndata obs 字符串 index）而非
   位置；⑤ 无 total 行的归一化列误回填原始值。教训延续 L-2026-0914-03：
   期望值必须手算，不能复用被测代码的输出。
5. **待办边界**：downstream target 评估结果写入 E2E report lineage、driver–target
   gate contract、真实 PTM 资产接入（§10 六项外部输入）均为后续工作；正式
   DEG 表（GSE174367 donor-level）生成前 fc 维度与交集只能用合成数据验证。
   synthetic 契约 PASS ≠ 生物学 PASS 口径不变。

## L-2026-0914-06｜文档同步轮：全仓文档对齐 PTM activity 主线指南

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0914-06 |
| 时间戳 | 2026-09-14 |
| 决策级别 | 文档契约同步（无代码改动） |
| 决策来源 | 用户指令 + `docs/guides/ptm_activity_pipeline.md` + 执行方案 v1.0 |
| 状态 | 12 处文档同步完成；本轮未触碰代码与测试 |

1. **同步范围**：README（新主线研究路径置于 DAVF × PerturbGen 路径之前、
   模块表/目录注释）、`.planning/task_plan.md`（整表重写为主线执行顺序，
   已闭合项打勾、外部资产/lineage/GPU 项保持未勾）、STATE/ROADMAP/PROJECT/
   progress/findings/REQUIREMENTS（新增 P-01～P-06 主线需求行）、
   CURRENT_STATUS（研究边界加第 0 项上游主线 + 文档入口）、`index.rst`
   toctree、E2E/bridge 指南（candidate spec 上游生成入口交叉引用）、
   2026-08-21 方案（头部上游注记）、TEST_COVERAGE（新测试清单 + 口径）、
   `project_analysis_20260914.md` §6 增补、AGENTS.md（新主线约束节）。
2. **修正的文档漂移**：① ROADMAP/TEST_COVERAGE/PROJECT/progress/findings
   仍写"Gate-0 cohort 为 0/统计接续未完成/每候选重复准备"——第五/四轮已
   分别落地 GSE174367 between_donor preflight+M6 冻结、
   `--assemble-statistical-evidence`、`build_shared_prepare_plans`；② 多处
   引用已归档的 `project_analysis_20260913.md`（权威为 20260914）；③
   TEST_COVERAGE 的 `../task_plan.md` 失效链接（实际在 `.planning/`）。
   教训：每轮落地后 living docs 若只更新 CURRENT_STATUS/CHANGELOG，
   `.planning/` 层会在下一轮形成反向漂移。
3. **AGENTS.md 新增「2026-09-14 PTM activity 主线约束」**：方向字段分字段、
   source/target 不合并、`direction_only` 无 q 值、外部标准表输入、
   exploratory 不生成候选行、lineage/driver–target gate 是待办——后续代码
   修改与测试以该节 + 指南为执行口径。
## L-2026-0915-01｜PTM activity A/B/C 批次接线与定向验证

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0915-01 |
| 时间戳 | 2026-09-15 |
| 状态 | A/B/C 代码契约与定向验证完成；真实外部资产/GPU 生物学验收未运行 |

- A 新增 AD donor-level 聚合与 CLI：复用 donor log2 normalization、Welch/BH，同 run 产出 aggregate 八列和 donor-level 表，配置/队列/obs 契约硬失败。
- B 将显式 --downstream-target-sidecar 接入 E2E：对 gated 候选的 result.h5ad 计算 pred_counts→X donor delta，写入 payload/lineage；matches_predicted 与 matches_observed 分开，source 三方 gate 仍是唯一 pass/fail。
- C 增加 max_paths_per_seed：正整数校验，超限硬失败不截断，manifest/diagnostics 记录 per-seed 路径计数。
- 新增 A 核心与 CLI 测试；相关回归实际结果为 80 passed、8 warnings，ruff check/format check/compileall 通过。
- 外部 KSTAR/PhosR/OmniPath/benchmark、真实 GPU 六阶段及生物学 PASS 仍待真实输入；synthetic/contract PASS 不等于 biology PASS；本批次不新增 hash 或 fallback。

## L-2026-0915-02｜全量测试债收口与 E2E 真实数据复核的三项硬发现

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0915-02 |
| 时间戳 | 2026-09-15 |
| 状态 | 工程修复完成；DEG 统计口径与 DAVF 基因轴覆盖属研究设计决策，未擅自更改 |

- 全量非 slow/gpu pytest 真实耗时约 818s（2795 passed/3 failed/22 skipped），
  此前 420s timeout 属外层截断而非挂起；3 个失败已修：dual_path 两个测试
  fixture 补齐 formal `evidence_class/pvalue_source/contract` 新必填声明，
  `test_scvi_davf_connection` 方向测试从 Norman 旧链（legacy checkpoint 被
  canonical-ENSG gate 正确拒绝）迁移到 KD 正式链 `davf_kd_nadig`
  （sumoylation→KD code 1、LCK ENSG00000182866 token 328/decoder 2260）。
- 全仓 ruff format 一次性收口（324 files），format/lint/mypy/requirements 全绿。
- E2E dry-run 在显式工程 context（EX 分层 donor 轮转、KO scVI 4018 基因轴、
  33 基因缺失显式填零、davf_batch 绑定 Dixit batch）上全链路跑通：Gate-0
  （counts 层/ensembl_id var 列/双 state≥3 donor）→ DAVF 真实 decode
  （finite delta）→ 三方 gate（三方向同号但 observed FDR=1.0 →
  inconclusive "observed_expression_not_significant"，正确拦截）→ 无
  invocation。工程 context 不是正式资产，批次绑定语义待研究冻结。
- 发现一（阻塞性）：E2E DAVF 推理契约要求 context_h5ad 与 scVI 4018 基因轴
  完全一致 + davf_batch 列，但仓库没有 GSE174367→scVI-aligned context 的
  准备工具；KO 链 33/4018 基因在 GSE174367 缺失，batch 跨数据集绑定语义
  未冻结。D1 前必须补 context 准备步骤并冻结绑定决策。
- 发现二（阻塞性）：AD 冻结候选 5 基因中仅 APOE 在 KO route（Dixit K562）
  scVI/alias 中存在，APP/PSEN1/BACE1/MAPT 在 KO/KD 两链均缺席；DAVF 现有
  训练资产（血液细胞系 perturbation）不覆盖 AD 主线候选基因轴。需要
  Workflow B（按 AD 队列重训 scVI+LatentDAVF）或候选集与资产对齐研究决策。
- 发现三（高）：`_donor_log2_means` 为每细胞 log2(CPM+1) 再 donor 平均；
  该口径下 GSE174367 EX Welch t（7 vs 11 donor）min p≈6.3e-5，BH 后 0 个
  显著基因（全表 FDR=1.0，117,352 行）；而 donor pseudobulk counts 先聚合
  再变换的口径 min p 可达 ~1e-13。三方 gate observed 方在当前口径下数学上
  不可通过；改口径（pseudobulk counts + 检验选择）是研究设计决策，须与
  deg_max_fdr 语义一起冻结后重算，不得静默更换。
