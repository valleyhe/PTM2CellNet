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
## L-2026-0916-01｜U2–U7 修复批次：estimand 冻结、context 入口、轴审计与 P40 eager 实测

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0916-01 |
| 时间戳 | 2026-09-16 |
| 状态 | U2–U7 工程侧落地；observed gate 阻塞确认为队列统计功效限制，非代码缺陷 |

- **U4 修正**（更正 L-2026-0915-02 发现三的预期）：donor pseudobulk counts
  口径已实现并冻结（`deg_donor_aggregation`，config + manifest），但 GSE174367
  EX（7 vs 11 between_donor，donor 文库 15.6 万–2374 万 reads）在 logCPM
  pseudobulk + Welch 下 min p=1.56e-5、BH 后 min FDR 0.915；表达过滤 universe
  16k–19k 仍 0.28–0.94；raw counts t（library confound 口径）也仅 1.9e-4。
  上轮"pseudobulk min p≈1e-13"不可复现，结论作废——**observed 三方 gate 的
  FDR≤0.05 在本队列任何无偏口径下不可达**，解法只能是扩大队列/预注册小基因
  面板/修订 gate 语义（研究决策），不得调阈值。
- **U2**：`src/data/scvi_context.py` + `scripts/prepare_scvi_context.py` 是唯一
  context 准备入口：missing_gene_policy（zero_fill/fail）、`davf_batch`（必须
  ∈ route scVI batch registry）、binding rationale 三者显式必填并逐字进
  manifest；KO context 已真实生成（61,472×4018，33 缺失轴基因零填，batch=
  DixitRegev2016_K562_TFs_7_days:168）并经 E2E 同款加载路径 encode 验证
  （(256,64) finite）。KD 无候选消费者时不强行绑定 55-batch 决策。
- **U3**：轴覆盖审计（`scripts/audit_davf_axis_coverage.py`）：KO={APOE}、
  KD=∅；APP/PSEN1/BACE1/MAPT 在 PerturbGen vocab 全覆盖但本地 30 个
  scPerturb 数据集**从未被扰动**——Workflow B 重训无训练输入，本地不可行；
  4 个 scPerturb h5ad 下载截断损坏（Gasperini at-scale、Lara-Astiaso invivo、
  Nadig hepg2、Sunshine 2023），manifest 记录，现有 KO/KD 资产不受影响。
- **U7**：Gate-E vocabulary-migration benchmark 从真实 CPLM Homo sapiens +
  dbptm Phosphorylation（acc→symbol 用 CPLM 自带配对，symbol→ENSG 用
  PerturbGen ensembl mapping）组装 60,030 行/25 类；identity 对照 coverage
  0.9874 / collisions 0 / action agreement 1.0。本机 uniprot 辅助文件
  （human_proteome.fasta 等）实为酵母数据、idmapping 无 Ensembl 行——不可用。
- **D3 实测**（替换 40GB 估计）：PerturbGen trainer `compile_model=True` 的
  triton 后端不支持 Tesla P40（CC 6.1<7.0），`TORCHDYNAMO_DISABLE=1` eager
  是 P40 唯一路径；750-cell 三阶段探针 8.7/43.0/37.4s、VRAM 峰值 4,359 MiB。
- 验证：全量 2818 passed/0 failed/22 skipped（771.67s）；mypy 174 files 0
  errors；触碰文件 ruff/format 通过；requirements 274 pins 一致。
## L-2026-0916-02｜T1–T6 扩展批次：队列扩充、signed network、Frangieh KO 链与 P40 训练

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0916-02 |
| 时间戳 | 2026-09-16 |
| 状态 | T1/T2/T3/T5 完成，T4 下载中，T6 数据供给 blocked；observed gate 两队列不可达结论固化 |

- **T1 队列扩充（GSE157827）**：donor 推导 = sample title 跳号 subject 池编号
  （AD1,2,4..21/NC3,7..18，每编号一库；与 GSE174367 的"唯一供体协变量向量"
  证据强度同级，依据记入 provenance）；cell type 无官方注释，用 Leiden(0.5,
  seed 0)+固定 marker 模块（EX/INH/ASC/ODC/OPC/MG/PER.END）数据驱动注释，
  177,654 nuclei、7 类全部 12+9 donors、unassigned 仅 1,382（0.8%）。
  **合并统计三口径实测全败**：157827 单队列 min FDR 0.1365（HES4）、朴素池化
  min FDR 1.0（队列基线偏移稀释）、per-cohort centering（16N+23D）EX 0.51/
  INH 0.75——**observed gate FDR≤0.05 在本地两队列任何无偏口径下不可达**；
  不得调阈值，出路是第三/更大队列或 gate 语义修订（研究决策）。
- **T2 signed network**：OmniPath REST `datasets=dorothea,tf_target` 默认只回
  1,258 行——必须显式 `dorothea_levels=A,B,C,D,E`（全量 13,565）；符号语义
  consensus 优先、is_ 后备、双真/双假剔除；confidence 公式
  `min(1, 0.5+0.1×(n_unique_resources−1))` 为冻结校准选择；release 12,878 边
  （761 TF→3,727 target）经 `load_signed_network(expected_release=...)` 验证。
  本地旧 omnipath/regnetwork/string 表全部无符号列，不可冒充 signed。
- **T3 Frangieh KO 链**：Frangieh var['ensembl_id'] 5,649 行是 symbol/别名
  （scPerturb 数据质量），用 ensembl_mapping_dict 重映射（492 成功、5,157 无
  映射丢弃）+ 重复 ENSG 列 COO 聚合后才能进 prepare；新链 240,646×4,018、
  216 targets（Dixit-only 仅 10）**含 APOE 真实 KO 扰动信号**；4018 新轴被
  GSE174367 零缺失覆盖（原链缺 33）。P40 上 scVI(40ep)+pairs+LatentDAVF
  （best_epoch=4, early stop 19）全链数小时完成；APOE 解码 finite 非零、
  新正式测试通过。**注意**：新链与旧 KO 轴不同，资产不可混用；
  context batch 绑定取最大训练份额 batch（frangieh_remapped:batch_0, 82%）。
- **T4**：Zenodo record 13350497 并发限速会静默 stall——断点续传（curl -C -）
  + 完成后 size 比对 + SHA256SUMS 校验 + .new 原子替换。
- **T5**：22 formal + 9 local 环境变量 runbook 入 bridge guide §2.1。
- **anndata.concat（0.11+）丢弃 var 列**：axis=0 concat 后 var 只剩索引，
  ensembl_id/gene_symbol 必须在 concat 后从索引重建（两个脚本各踩一次）。
- argparse `nargs="+"` 重复 flag 会覆盖不追加；列表参数用 `action="append"`。

## L-2026-0916-03｜T4 收尾：scPerturb 重下的并发 append 损坏事故与完整性口径

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0916-03 |
| 时间戳 | 2026-09-16 |
| 状态 | Lara/Nadig/Sunshine 完成并通过全量读验证；Gasperini 单写者全新重下中（size 匹配+全量读通过才替换） |

- **后台续传任务的句柄查不到 ≠ 进程已死**：旧续传 shell 在 TaskOutput 报
  "No task found" 后仍存活，与新任务的两个 curl 以 O_APPEND 并发写同一
  `.new`，字节区间交错损坏（size 超过期望值是最明显信号；本例 2.02GB >
  期望 1.87GB）。HDF5 B-tree signature 也损坏时逐 chunk 挖补不可行，只能
  弃文件全新下载。**长文件下载必须保证唯一写者**：启动前 `pgrep -af`
  确认无残留写者，并在下载脚本里 size 匹配后内联验证再 mv。
- **anndata backed 模式读 shape 不是完整性验证**：backed 惰性只读
  superblock 与部分元数据，X 数据 chunk 损坏（gzip filter 失败）时
  backed 读照样通过。h5ad 完整性 gate = size 精确匹配 + **非 backed 全量
  read_h5ad** 成功。本例损坏文件 backed 读出 (200,000+, 40,000+) shape
  "正常"，全量读才暴露。
- **Zenodo record 13350497 限流实测**：高并发（16×range 并行）触发惩罚性
  限流，聚合速度 23–88KB/s 且段反复失败重传；单连接 `curl -C -` 反而稳定
  ~200KB/s；figshare ndownloader 仅 11KB/s 不可用。大文件从 Zenodo 拉取
  用单连接 + `--speed-time/--speed-limit` stall 检测 + 外层重试循环。
- **版本差异口径**：4 个文件实际 SHA256 全部与本地 `SHA256SUMS.txt`
  （figshare 基准）不一致但 size 与 Zenodo 元数据精确一致——Zenodo 修订版
  重打包。处理：实际 hash 与基准差异逐文件显式记入
  `data/raw/scperturb/redownload_verify.json`，不篡改基准、不把版本差异
  写成下载失败。
- `pkill -f "curl.*<文件名>"` 会匹配到包含该模式文本的自身命令行而自杀；
  清理残留进程用 `pgrep -af` 先查 PID 或让模式不含自身命令行文本。

## L-2026-0916-04｜T4 终态与带宽归因修正

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0916-04 |
| 时间戳 | 2026-09-16 |
| 状态 | T4 完成：4/4 文件 size 精确一致 + anndata 全量读取通过 |

- **T4 终态**：Lara invivo / Nadig hepg2 / Sunshine 2023 / Gasperini at-scale
  （207,324 × 13,135）全部完成三重验证（size 对 Zenodo 元数据、SHA256
  实测记录、非 backed 全量 read_h5ad），实际 hash 与差异逐文件在
  `data/raw/scperturb/redownload_verify.json`。
- **归因修正（L-2026-0916-03 的补充）**：用户确认本机出口带宽约 3MB，
  实测单流 ~200-250KB/s 与之吻合——此前"Zenodo 高并发惩罚性限流"的归因
  不成立，真实瓶颈是本机带宽；大文件下载直接单连接 + 断点续传即可，
  并行分片在窄带宽下无收益且引入段管理复杂度。

## L-2026-0916-05｜M10 driver–target gate 与 U4 多队列 centered 口径的实现边界

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0916-05 |
| 时间戳 | 2026-09-16（第三修复批次） |
| 状态 | 已落地：`evaluate_driver_target_gate` + E2E `driver_target_gate` lineage；`pseudobulk_counts_centered` 口径 + 真实合并队列实测 |

- **driver–target gate 的 verdict 是事实分类不是判据**：`driver_target_status`
  只有 not_evaluable / all_concordant_observed / mixed / none_concordant_observed
  四值（参照 target 自身 `observed_direction`），无阈值、无发明判据；该记录
  永不产生 pass/fail——source 三方 gate 仍是 formal 候选唯一准入门，source 无
  自身 DEG 的候选仍只进 exploratory（方案 §5.5）。E2E 的
  `--downstream-target-sidecar` 路径同时写
  `downstream_target_evaluation.driver_target_gate`（schema
  `ptm2cellnet.driver-target-gate/v1`），pass 候选缺 `direction_gate` 段硬失败。
- **centered 口径单队列不变性是设计性质**：`pseudobulk_counts_centered` 在单
  cohort 下与 `pseudobulk_counts` 的 fdr/log2fc 数值一致（减同一基线对 Welch t
  平移不变）；差异只在多队列——cohort 组成性偏移（library 归一化无法吸收）
  被移除后方差收缩。构造合成数据验证时偏移必须打在单基因上（打全基因会被
  library 归一化约掉，测试会假阴）。
- **centered 真实实测（GSE174367+GSE157827 合并，`dataset` 列）**：min FDR
  EX 0.5254 / INH 0.7908（vs 朴素池化 1.0），仍无 FDR≤0.05 行——与临时实验
  结论一致，observed gate 本地不可达不因合并口径改变；工具化的价值是把
  多队列合并从临时代码固化成正式 CLI（manifest audit 含 cohort_column/
  cohorts/donor_counts_by_cohort，任一 (cell_type, cohort) 缺 normal donor
  硬失败）。

## L-2026-0917-01｜外部扰动证据源选型实测与单源 LPM 最简起步

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0917-01 |
| 时间戳 | 2026-09-17 |
| 状态 | 契约与脚本已落地；LPM 实际训练/回测待外部环境（figshare 下载受限） |

- **词表覆盖必须实测不能从论文推断**：STATE 论文口径"遗传扰动来自
  Replogle-Nadig"，实际 HF 发布的 State-Replogle-Filtered 经 on-target 效应
  过滤后只剩 2,024 个扰动——5 个 AD 候选全部不在（远程 HDF5 range 读 obs
  categories 直测，19 个 HTTP 请求即可读 30GB h5ad 的词表，不必下载全文件）。
  GEARS 的 dataverse norman/adamson 数据是组合子集（284/87 conditions），与
  Norman 原始全量不同源——"训练词表"要看实际加载的文件不是论文名。
- **LPM 的 Replogle 覆盖是推断链不是直测**：figshare 对本机出口 TLS 不稳定
  （202 排队 + SSL EOF），无法直接读 perturblib 下载源词表；覆盖结论 =
  perturblib 源码无过滤加载 + Replogle 全表达基因文库设计 + 本地 Frangieh
  K562 表达实测（5/5 阳性）。首次外部环境训练时必须再 grep 词表闭环。
- **单源起步的边界**：LPM 证据永远是"K562 基线 context 的 crispri_kd 外推"，
  `token_mask_ko` 在契约里被显式拒绝（防语义混同）；Geneformer 第二源只在
  ①出现 OE 候选（LPM 词表仅 PSEN1 有 OE）或 ②APOE 锚点回测可疑时触发。
  先证后用：正式消费前必须过 APOE 锚点回测（真实 KO 方向 + mean-shift
  baseline 对照）。
- **HDF5 远程 range 读的最小实现**：h5py fileobj driver 要求 file-like 对象
  实现 readinto 且 read 不允许短读（循环填满），fsspec 的 aiohttp 在本机代理
  环境会连接失败——requests + 4MB readahead 缓存的 ~40 行实现即可读远端 h5ad
  的 obs/var 词表。

## L-2026-0917-02｜LPM 执行链的外部数据阻塞与 Nadig 词表修正

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0917-02 |
| 时间戳 | 2026-09-17 |
| 状态 | perturblib 环境就绪（torch cu124 下载中）；训练被 figshare 不可达阻塞，移交人工下载 |

- **Nadig 重处理版词表 = 2,024，与 STATE 相同**：HF
  `arcinstitute/Replogle-Nadig-Preprint/replogle.h5ad`（22.3GB 原始级，GSE264667）
  的 obs gene categories 远程直读同为 2,024，5 个 AD 候选（含 APOE）全
  MISS——**2,024 过滤是 Nadig 处理链固有的（low on-target efficacy）**，
  不是 STATE 的选择；上一轮"STATE 过滤掉候选"的表述据此修正为"Nadig 链
  过滤掉候选"。LPM 的词表内覆盖推断**只**依赖 figshare 原版
  （perturblib 无过滤加载 + 全表达基因文库设计 + K562 表达阳性）。
- **figshare 三路径全阻**：默认代理（202 排队 / SSL EOF）、api.figshare.com
  （400）、备用代理 192.18.2.170（超时）。无公开镜像：Nadig 生态（GEO/
  HF）全是 2,024 词表，GWPS 门户是 Dash 应用无静态端点，zenodo 无。**唯一
  前进路径 = 人工在外网可达环境下载
  `https://plus.figshare.com/ndownloader/files/35773075`**（Replogle K562
  处理版，LPM `replogle_k562_paper_lpm` 训练源）。
- **pip 装 torch 最新版会拉 CUDA 13 全家桶**（cudnn_cu13 等 2GB+），而 P40
  （sm_61）不被 cu13 支持——对旧卡必须显式
  `--index-url https://download.pytorch.org/whl/cu124` 装对应版本。
- `pkill -f "<pip 模式>"` 会匹配含该模式文本的自身命令行而自杀（L-2026-0916-03
  同款坑，复踩一次）——清理进程先 `pgrep -af` 拿 PID。

## L-2026-0917-03｜LPM 词表直测定案：默认 K562 context 全 MISS，GWPS context 3/5

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0917-03 |
| 时间戳 | 2026-09-17 |
| 状态 | 词表争议直测闭环；LPM 本机训练被数据体积结构性阻塞 |

- **下载突破**：figshare 的 202 反爬是 plus.figshare.com 域特有；换
  **ndownloader.figshare.com 域 + 浏览器 UA + 断点续传**经默认代理实测稳定
  （~320KB/s）。API v2（api.figshare.com/v2/articles/20029387）不受限，可拿
  完整文件清单。直连（无代理）403——代理是必经路径。
- **perturblib 的 K562 context 是 essentialome 不是全基因组**（figshare
  35773075 = K562_essential_normalized_singlecell，bulk 版直测 2,285 个扰动
  行，5 个 AD 候选 5/5 MISS）——上一轮"LPM 词表内覆盖 5/5"的推断链**被直接
  证伪**；论文配置 `replogle_k562_paper_lpm` 训练出的模型对我们的候选无
  查询入口，该训练不再执行。
- **GWPS context（全基因组）直测 3/5 命中**：K562_gwps_normalized bulk
  （35773217，11,258 行）实测 APOE/MAPT/PSEN1 命中、APP/BACE1 未命中
  （normalized 处理弃掉或文库设计时低表达）。注意 bulk 版只能做词表验证，
  LPM 训练需 singlecell 版 65.83GB。
- **本机存储是硬墙**：65.83GB > 根分区空闲 56GB——GWPS 训练数据在本机
  放不下；曾误启动下载已即时终止并清理（下载前必须先查目标分区空闲）。
- **.obs 行名格式**：Replogle 处理版为 `{编号}_{SYMBOL}_{P1P2}_{ENSG}`，
  解析 symbol 是 `split('_')[-3]` 不是 `[-2]`（踩过一次导致假 MISS）。
- **选型结论修正**："单源 LPM 覆盖大多数"实测为"LPM(GWPS) 3/5，且需先解决
  65.83GB 获取；APP/BACE1 除 GEARS(GO 通道)/Geneformer 外无任何外部模型
  词表入口"。策略一（GEARS+Geneformer 双源）重新成为唯一全覆盖选项。

## L-2026-0917-04｜策略 B 执行链闭环：GEARS/Geneformer 资产产出，APOE 锚点回测 fail 且归因于锚点不可比

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0917-04 |
| 时间戳 | 2026-09-17 |
| 状态 | 双源资产已冻结；按预注册判据回测 fail，**不接入 lineage**；归因修正为锚点不可比 |

- **执行链全部走通**（外部环境 perturblib conda + torch 2.6 cu124/P40 + cell-gears 0.1.2 +
  geneformer 官方 HF 版）：GEARS 扩 graph 训练（gene_set 9,978 = 默认 9,976+缺失候选，
  `gene_set_path` 是官方扩词表通道）→ 5 候选 unseen 预测 → `gears_predictions.h5ad`；
  Geneformer V2-104M ISP（GSE174367 EX 疾病细胞 250 个，delete）→ `geneformer_isp_EX.h5ad`
  （cos 影响谱）；两资产均通过主环境 `external-perturbation-prediction/v1` 契约校验。
- **回测 fail（预注册判据）**：GEARS APOE top-100 方向一致率 0.25（阈 0.59，binom
  p=5.6e-7 显著**低于随机**）；Spearman −0.04；判据 C1/C2 fail → 不接入。
- **归因诊断（重要修正）**：B2M/CD59/CTSD 三个独立锚点同管线一致率 0.25/0.32/0.27
  全部显著反向，mean-shift baseline（与模型无关）也 0.22 反向——**系统性低于随机的
  模式指向锚点数据集不可比**（Frangieh co-culture 的组成效应/受体-效应细胞混合 vs
  Norman 单培养 K562 的内在转录响应），而非 GEARS unseen 单独失效。
- **结构性边界结论**：unseen 候选（APP/BACE1 等无真实扰动数据的基因）的方向证据
  在本机数据条件下**没有干净的回测 ground truth**——"先证后用"只能挡住不可靠证据，
  无法为这类证据发可信证明。GEARS 有效性需按其标准口径（Norman 测试集扰动）另行
  评估，与候选方向证据的可信度是两个问题。
- **工程坑（外部环境适配实录）**：①cell-gears 0.1.2 为 pandas1.x 代码（`Series.nonzero`
  /bool-Series 索引 scipy sparse 全崩）——降 pandas<3 + 打两行补丁（gears.py:87 的
  sparse 掩码 `.to_numpy()`、predict 的 unseen 检查放宽到 node_map_pert）；②GEARS
  predict 期望 **list of lists**（`[["APOE"]]`），字符串会被逐字符迭代；③Geneformer
  官方包源在 HF（GitHub 原仓库 2025 下线，jkobject fork 的 main 分支有语法错误勿用）；
  pip 从 HF clone 安装时包内 pkl 是 **LFS 指针**，需手工用真实字典覆盖 site-packages；
  ④Geneformer ISP `genes_to_perturb` 用 **Ensembl ID** 键、V2 模型 emb_mode 必须
  `cls_and_gene`、gene 输出是 **embedding cos 谱无方向语义**（方向证据只剩 GEARS）；
  ⑤ISP cls_and_gene 显存大（batch 32 OOM 于 P40+并行训练时，batch 8 通过）。

## L-2026-0918-01｜方案 §6 五候选分流编码为确定性路由契约，不接入 formal lineage

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0918-01 |
| 时间戳 | 2026-09-18 11:30 |
| 条目版本 | v2.1.0（模块级：候选分流/证据隔离） |
| 决策级别 | **模块级**（把 observed-gate 方案第 6 章编成可测试代码，不改变 formal gate） |
| 决策来源 | 用户指令执行 `docs/guides/observed_gate_davf_no_perturbation_execution_plan.md` §6 |
| 状态 | 已落地代码与定向测试；真实分流跑通；**formal biology PASS 仍为 0** |

### 1. 决策内容

- 新增 `src/analysis/ad_candidate_routing.py` + `scripts/route_ad_candidates.py` 作为
  五候选 × KO/KD 的唯一分流入口。复用已有
  `davf_axis_coverage_audit.tsv` 与 d2 centered DEG 表，不重算覆盖、不改 FDR。
- 硬约束写入代码而非文档：词表命中不是训练轴；本地 scPerturb 扰动列表不是 KD
  机制；KO 不得取反成 KD；本 CLI `formal_invocation_allowed` 恒为 false；锚点
  pass 或 fail 都不得把外部资产写入 formal lineage（方案 §6.4 明确不为当前
  GEARS/Geneformer 新增 formal consumer）。
- 外部 evidence payload 增加 `lineage_boundary=supplementary_only` 与
  `may_enter_lineage=false`；`network_counterfactual` 的矩阵值只进
  `influence_score`，`predicted_direction` 保持 null。

### 2. 当前真实跑通事实

输入：`outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv`、
`outputs/ptm_activity/20260916_d2/ad_deg_centered_aggregate.tsv`、
`outputs/external_evidence/apoe_anchor_backtest.json`（verdict=fail）。
产物：`outputs/ptm_activity/20260918_candidate_routing/`。

- observed 显著行 = 0，min FDR = 0.52538（与 d2 EX 记录一致）
- APOE KO = `engineering_verification`；APP/PSEN1/BACE1/MAPT KO = `B2` /
  `direction_only`；五者 KD = `unavailable`
- `n_formal_invocations = 0`，`biology_pass = false`

### 3. 明确不做

- 不重跑 K562 GEARS / Geneformer 以期待锚点自动通过
- 不创建虚拟 KD 轴或伪 z0/z1
- 不把本次 sidecar 送入 `run_davf_perturbgen_e2e.py --run-perturbgen`
- 公共 Perturb-seq inventory 仍为空，B1/B3 只在提供机制可追溯 TSV 后才会启用

## L-2026-0918-02｜observed gate 选分支 3；KD 并入 KO；不再检索公共 Perturb-seq

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0918-02 |
| 时间戳 | 2026-09-18 13:00 |
| 条目版本 | v2.2.0（模块级：准入语义/干预范围） |
| 决策级别 | **模块级**（修订 AD observed 准入语义与候选范围，不宣称 biology PASS） |
| 决策来源 | 用户正式指令（observed gate 选 3；KD 不再单独做；不再找公共 Perturb-seq） |
| 状态 | 已冻结为 `ptm2cellnet.ad-research-decision/v1`；**formal biology PASS 仍为 0** |

### 1. 决策内容

1. **observed gate = 分支 3（修订语义）**，不是加 donor、不是预注册小 panel、
   也不是放宽 FDR/raw p/universe。donor-level Welch+BH 与 `deg_max_fdr=0.05`
   继续作为报告/显著性标签；准入与三方 gate 的 observed 臂改为 signed
   disease−normal 方向一致。FDR>0.05 **不得**改标成显著。
2. **KD 并入 KO**：本 AD 五候选只做 KO loss-of-function。DAVF 干预标签 0/1
   仍分离；不得从 KO 取反、不得把 KO checkpoint 当 KD 用。
3. **不再检索公共 Perturb-seq**。B1/B3 transfer 为 out of scope；分流 CLI
   遇到非空 inventory 硬失败。

### 2. 落地

- schema / 默认冻结：`src/analysis/ad_research_decision.py`、
  `configs/research/ptm_research_config.yaml`
- 消费方：`ad_candidate_routing`、`direction_gate`（`observed_significance_required`）、
  `build_celltype_candidate_specs`、`ptm_gene_score` 交集 `I_c`、E2E candidate spec
- 未完成：真实 PTM/KSTAR 仍缺，E2E 仍不能宣称 biology PASS；分流 CLI 仍不发
  formal invocation

### 3. 明确不做

- 不把 FDR 0.52 写成显著
- 不恢复 KD 独立工作流
- 不开始公共 Perturb-seq 盘点或下载
- 不把本次决策写成 formal biology PASS

## L-2026-0918-03｜PTM smoke 契约测试；KSTAR 保持独立环境方案

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0918-03 |
| 时间戳 | 2026-09-18 13:20 |
| 条目版本 | v2.3.0（模块级：阶段 2 前置契约 / 外部工具边界） |
| 决策级别 | **模块级**（用确定性 PTM fixture 测阶段 1/3/4/5；KSTAR 只出方案不进 core） |
| 决策来源 | 用户正式指令（PTM 由助手生成 smoke 数据先测试；KSTAR 给出方案） |
| 状态 | smoke 已落地并跑通；KSTAR 方案冻结、**未执行**；**formal biology PASS 仍为 0** |

### 1. 决策内容

1. **PTM 先用 smoke**。`src/analysis/ptm_smoke.py` + `scripts/generate_ptm_smoke.py`
   写出 §4.1 十三列位点表（3+3 donor、1 个 replicate、1 个未映射蛋白、1 行无 donor）、
   二元 KSTAR evidence 形状、activity 占位和一张 kinase→TF→gene 小网
   （`release=smoke-2026-09-18`）。`--run-pipeline` 调用已有阶段 1/3/4/5 CLI，
   **不调用 KSTAR**。
2. **activity 占位的 `method=KSTAR` 只满足 `primary_activity_method` 过滤**；
   `method_version=smoke-stub-20260918` 与 `kstar_ran=false` 禁止把它当成激酶推断。
3. **KSTAR 给出方案、不实现 runner**。合同为 `docs/guides/kstar_activity_plan.md`：
   独立 conda、PhosphoSitePlus freeze、standardized PTM → 二元 evidence →
   `ptm_activity.tsv` adapter；kinase–substrate 传播网与 2026-09-16 TF-only
   OmniPath freeze 分离。不把 `kstar` 写入 `requirements-core.txt`。

### 2. 当前跑通事实

产物：`outputs/ptm_activity/20260918_ptm_smoke/`。

- 阶段 1：51 行输入、1 个 replicate 组、1 个未映射蛋白、6 donor
- 阶段 3：7 条 (source, target) gene score，全部 `prediction_status=direction_only`
- 阶段 4：EX `n_formal_concordant=4`（signed admission，FDR 全 >0.05）
- 阶段 5：EX 3 个候选（GSK3B/CDK5/AKT1），`observed_significant=false`；INH 0 候选，
  source 无本 cell type DEG 进 exploratory
- `biology_pass=false`，`kstar_ran=false`

测试：`tests/unit/analysis/test_ptm_smoke.py`、
`tests/unit/scripts/test_generate_ptm_smoke.py`。

### 3. 明确不做

- 不把 smoke 位点表或 stub activity 写成真实 phosphoproteome / KSTAR
- 不在 core 环境 `import kstar` 或新增 PhosR R 依赖
- 不把 2026-09-16 OmniPath TF-only 网改成激酶传播网
- 不把本次 EX 三候选送入 `run_davf_perturbgen_e2e.py --run-perturbgen`
- 不宣称 biology PASS

## L-2026-0920-01｜TD-20-01：KSTAR 环境与映射工程已验证，网络未完成不得冒充正式分析

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0920-01 |
| 时间戳 | 2026-09-20 |
| 条目版本 | v2.4.0（模块级：KSTAR 独立环境 / 入口 / lineage 边界） |
| 决策级别 | **模块级**（同步当前真实源码、测试和代理核验结果；不宣称生物学完成） |
| 决策来源 | TD-20-01 文档同步指令与当前仓库实现 |
| 状态 | Python 3.12.14 + KSTAR 1.2.0 环境、adapter/CLI/setup、network release assembler 已落地；Figshare network installer 仍 HTTP 202 且无 `Location`/`Retry-After`，正式 KSTAR activity、kinase→TF propagation 与 biology PASS 仍未完成 |

### 1. 可复用决策与证据

1. 独立解释器 `/home/scu/anaconda3/envs/kstar/bin/python` 的 `sys.prefix`、
   `sys.executable`、Python 版本和 `kstar==1.2.0` 校验通过；官方
   `kstar.mapping.ExperimentMapper` smoke 得到 7 mapped rows。
2. `scripts/setup_kstar_env.sh` 对缺失的官方 ST/Y network 调用 installer；本次官方
   返回 HTTP 202，且响应没有 `Location`/`Retry-After`，所需 network assets 未完成，因此
   setup 硬失败，不伪造 network。
3. `kstar_adapter` 的 signed score 必须来自成对 increased/decreased p-value 比较；
   q-value 是两路 p-value 合并后统一 BH，不是全局取反。正式输出的
   `method_version` 使用真实 KSTAR 1.2.x（当前 `1.2.0`），不使用 smoke-stub。
4. `scripts/run_kstar_activity.py` 在两次官方 `run_kstar_analysis` 后严格检查
   `activities_mann_whitney`/`fpr_mann_whitney` 的索引（含两张结果索引一致性）、directional data 列以及空/重复 kinase；
   空索引名只规范为 `KSTAR_KINASE`，再经官方 `save_kstar` 重写 TSV 并由 `from_kstar` 重载。
   根因是 KSTAR 1.2.0 新 `run_kstar` 路径返回未命名索引，导致 `save_kstar` 写出空首列表头，而
   `from_kstar` 硬编码 `index_col=KSTAR_KINASE`。
5. post-fix 临时 synthetic smoke 的两路结果首行已分别含 `KSTAR_KINASE` 与 directional data 列，
   CLI 内部两次 `from_kstar` 均成功；但 test-only synthetic 网络两路 p 值没有方向性证据，CLI
   以 `paired KSTAR outputs contain no directional kinase evidence` 退出 1，未生成
   `ptm_activity.tsv`。这是预期硬失败契约，不是完整 CLI 十三列表 smoke，也不是正式 biology PASS；
   不添加 fallback。

### 2. 边界

- smoke fixture、ExperimentMapper mapping 和 HTTP 202 都只能证明工程边界，不能当作
  KSTAR activity、formal lineage、正式 kinase→TF propagation 或 biology PASS。
- 本轮不改变既有 shared prepare/reuse、独立环境、no-hash 和正式验收约束；不把上述
  handover smoke 或 mapping 结果写成正式结果。
- adapter manifest 保持 `biology_pass=false`、`may_enter_lineage=false`；正式验收仍需
  真实 PTM、可用且冻结的 kinase/TF network、预注册 kinase benchmark，以及真实 AD
  队列前置条件。

## L-2026-0920-02｜官方 KSTAR network file 60883384 入口诊断

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0920-02 |
| 时间戳 | 2026-09-20 |
| 条目版本 | v1.0.0（模块级：官方 network 下载入口诊断） |
| 决策级别 | **模块级**（记录入口响应，不构成 network 或生物学结果） |
| 状态 | setup 使用的 canonical `figshare.com` 请求此前安装诊断曾返回 HTTP 202；本轮受控 `curl` 在同一 URL 得到 403 HTML（520 bytes），`ndownloader.figshare.com` 同 file ID 得到 HTTP 202 空响应（`Content-Length: 0`）；两者均无 `Location`，且都不是 tar.gz |

### 可复用事实

- 官方 KSTAR network file `60883384` 的两个入口响应都不能提供可用 network 资产；这只是网络入口诊断，不是 biology PASS。
- 未修改安装脚本，也未安装伪造资产；正式 network 仍需用户或外部可用资产。

## L-2026-0921-01｜KSTAR 钉包与 PhosphoSitePlus 衍生资源 hash；adapter Ensembl 交接；另冻 kinase+TF 网

| 属性 | 值 |
|---|---|
| 记录编号 | L-2026-0921-01 |
| 时间戳 | 2026-09-21 09:30 |
| 条目版本 | v2.5.0（模块级：KSTAR 环境 pin / adapter Ensembl / kinase+TF freeze） |
| 决策级别 | **模块级**（工程资产与交接契约；不宣称生物学完成） |
| 决策来源 | 用户指令执行 2026-09-18 建议任务 1–3，按当前源码独立落地 |
| 状态 | 钉包与 RESOURCE_FILES hash 已冻结；adapter 可从 standardized_ptm 写出 KSTAR 二元 evidence 并在成对 directional 结果上生成十三列 `ptm_activity.tsv`（Ensembl `regulator_id`）；`omnipath-kinase+tf-2026-09-21` 已另冻；官方 ST/Y network 仍须匹配 `unique_reference_id`；**formal biology PASS 仍为 0** |

### 1. 决策与证据

1. **独立环境钉包**。正式环境名仍为 `kstar`，与 `perturbgen` 分离。冻结
   `environments/kstar/environment.yml`（`python=3.12.14`、`kstar==1.2.0` 及直接依赖）
   与 `environments/kstar/requirements-lock.txt`。不写入 `requirements-core.txt`。
2. **PhosphoSitePlus 衍生资源 hash**。KSTAR 1.2.0 随包 `RESOURCE_FILES` 来自
   ProteomeScout 2020-02-26 / KinPred 人类蛋白组，不是 live PSP dump。
   `unique_reference_id=a7dfa119afa4f833375dfaf0c503ee1bca28c9fe7d868e4249907442d7821c64`。
   `HumanPhosphoProteome.csv` sha256
   `c2ebbcac778940558b2af94bfcc14715c71f0bde8d734f9cc1cc0fe319828cdd`。
   setup 与 `run_kstar_activity.py` 均校验；HTTP 202 / HTML / 过小响应不得当作
   ST/Y network。
3. **Adapter**。`build_kstar_input` 只用 `has_donor` 且有限 `ptm_value_normalized`，
   τ=1.2 与 smoke evidence 口径一致，列名为 `data:{contrast}:increased/decreased`。
   `convert_kstar_outputs` 成对比较 unsigned p 值后写十三列；`--mode analysis` 必须
   `--ensembl-mapping`，symbol 缺失或一对多硬失败。`method_version` 必须是真实
   `1.2.x`，禁止 `smoke-stub-*`。
4. **kinase+TF 新 release**。`scripts/build_kinase_signed_edges.py` 从 OmniPath
   `datasets=omnipath,kinaseextra`（102,397 行，sha256 `618c166c…`）生成
   15,133 条 `kinase_substrate:signaling` 边（+10,643 / −4,490），release
   `omnipath-kinase-2026-09-21`。再与只读的 `omnipath-2026-09-16` TF 表
   （12,878 边，sha256 `ba500f61…`，组装前后不变）合成
   `omnipath-kinase+tf-2026-09-21`（28,011 行，sha256 `9483f613…`）。
   GSK3B `ENSG00000082701` 在该网上不再是 `seeds_without_node`（2,281 个 gene
   score / 4,806 条路径，工程连通）。`biology_pass=false`。
5. 定向测试 21 passed；主环境 mypy 目标文件 0 errors；kstar 环境 RESOURCE_FILES
   hash 实装校验通过。

### 2. 边界

- 不把本次 freeze、mapping smoke 或 hash pin 写成 KSTAR activity 或 biology PASS。
- 不改 2026-09-16 TF-only 文件；正式传播须显式切换 `network_release`。
- 官方 KSTAR ST/Y network 仍须与冻结 `unique_reference_id` 一致；缺文件继续硬失败。
- 真实 PTM、kinase benchmark 与 AD 队列前置条件不变。


## L-2026-0921-02 技术债修复轮（TD-01 方向 metrics + 严重/高/中债）

日期：2026-09-21。依据 `project_analysis_20260921.md` §6/§7，本地修复并验证。

1. **TD-01 KSTAR 双方向 metrics（严重）**。两方向 evidence 集不同 →
   `n_substrates`/`network_coverage` 合理不同，全表相等断言在真实数据上必失败。
   修复：`convert_kstar_outputs` 改收 `increased_metrics`/`decreased_metrics`
   （必须成对），每个 kinase 按胜出方向（较小 p 值）绑定该方向 metrics；
   runner 落盘两方向 metrics TSV 并写入 manifest（schema
   `ptm2cellnet.kstar-adapter-manifest/v2`）。单 `metrics=` 参数已删除。
2. **TD-02 activity 准入（严重）**。`activities_for_propagation`（method/contrast
   之外全选）已删除；唯一入口 `src/analysis/ptm_activity_admission.py::
   select_activities_for_propagation`，消费 config `activity_admission` 冻结
   `ActivityAdmissionPolicy`（q/substrates/coverage 三阈值 + policy hash）；
   rejected 行带原因进 gene-score manifest。无独立 benchmark 时
   `benchmark_gate.available=false`，admitted 仅 exploratory。config 默认登记
   宽松阈值（1.0/0/0.0），收紧属研究决策，须显式修订 config。
3. **TD-04 formal/exploratory mode（高）**。`PTMResearchConfig.mode` 默认
   `exploratory`；`formal` 拒绝 `ptm_cohort`/`cohort_h5ad` 的 `PENDING*`。KSTAR
   `_require_manifest` 要求结构化 stage-1 manifest（schema_version 匹配
   `ptm2cellnet.ptm-input-manifest/v1` + source.sha256 + standardization），
   空 `{}` 硬失败；`build_kstar_input(min_donors_per_state=)` donor 下限
   （runner `--min-donors-per-state`，默认 1，正式运行显式提高）。
4. **TD-05 network verifier（高）**。`verify_kstar_network_dir` 返回
   `KSTARNetworkAudit`；ST/Y 各须恰 50 个非空文件，Unique Network ID 双双 pin
   （`0c85777e…`/`23ce4b6c…`）；缺文件/空文件/ID 漂移硬失败；audit 入 runner
   manifest。
5. **TD-07 release 绑定（高）**。config `network_release` →
   `omnipath-kinase+tf-2026-09-21` + `network_release_manifest`；
   `verify_network_release_binding` 三向校验（config release 名 / manifest
   combined sha256+rows / 实际 --network-tsv），传播 CLI 启动执行；
   `gene_edge_types` 增加 `kinase_substrate:signaling`。
6. **TD-12 marker 分层（中）**。2,959 项测试此前 0 项带 slow/gpu marker（分层
   形同虚设）。17 项重型 integration（train_resume/davf 重型/lightning/
   manifest CLI/scperturb 注册）标记 slow；full-test.yml 改 `not gpu`（承接
   slow），CI fast job 语义不变。fast 2,868 项 275 秒完成（此前 188 项超时）。
7. **未动项**：TD-03 环境依赖冲突（`pip check` 3 组：numpy 2.4.3 vs <2、
   scvi-tools 1.4.3 vs <1.0、torchaudio 2.4.1 vs >=2.5；pip-audit 2026-09-21
   口径 26 包/96 条唯一记录）——修复须隔离环境重建+全量回归，不得在当前可用
   环境上盲动；TD-08 Workflow B 编排、TD-09 复杂度拆分、TD-10/11 异常与安全
   分流为专项工作。策略见 `project_repair_report_20260921.md`。
