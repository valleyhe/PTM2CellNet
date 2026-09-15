# PTM2CellNet

## What This Is

PTM2CellNet 是一个用于蛋白质翻译后修饰（PTM）分析和细胞状态预测的深度学习系统。v2.0 集成了 DAVF 方向感知速度场模型，通过级联融合架构实现 PTM→信号通路效应预测。

## Core Value

提供端到端的蛋白质 PTM 分析与细胞状态预测能力，集成 DAVF 实现从 PTM 修饰到细胞信号通路响应的完整预测链路。

## Context

### 技术栈
- **深度学习框架**: PyTorch 2.7.1+cu118 + PyTorch Lightning 2.5.1
- **序列编码**: CNN, Transformer, LSTM, Mamba, ESM-2 (LoRA), ProtBERT, ProtT5
- **DAVF**: LatentDAVF + BiPerturbEncoder + DeltaProjection + scVI
- **PTM处理**: 注意力机制融合的PTM嵌入模块 + PTMDirectionMapper
- **API服务**: FastAPI
- **数据处理**: pandas, NumPy, BioPython, anndata, scvi-tools

### 模型架构组件
1. **Sequence Encoder**: 支持多种编码器（CNN/Transformer/LSTM/Mamba/预训练模型）
2. **PTM Module**: PTM类型嵌入 + 位置感知注意力 + 序列融合
3. **DAVF Branch** (v2.0): PTMDirectionMapper → DAVFInferenceModule → [B, 128] features
4. **Cascade Fusion** (v2.0): Concat(seq_features, davf_features) → Predictor
5. **Pooling**: Mean / Attention / Multi-Head Attention / Weighted Mean
6. **Predictor**: 分类/回归/多任务预测头

### 当前状态
- v1.0 已发布: 6 phases, ~959 tests, 核心模块 76-100% 覆盖率
- v2.0 已发布: 5 phases, +91 tests, DAVF 集成完成
- ~21,302 LOC (src), ~19,627 LOC (tests)
- 配置: configs/davf_integration.yaml 为 DAVF 运行时配置源

## Requirements

### Validated

- ✓ 基础数据流: DataLoader → Preprocessor → FeatureExtractor → Dataset → Model
- ✓ 多编码器支持: CNN, Transformer, LSTM, Mamba
- ✓ 预训练编码器: ESM-2, ProtBERT, ProtT5
- ✓ 多种池化策略: Mean, Attention, Multi-Head Attention, Weighted Mean
- ✓ 多任务学习: MultiTaskPredictor, HierarchicalMultiTaskPredictor
- ✓ 分类与回归支持
- ✓ FastAPI服务框架
- ✓ 370+ 项测试覆盖 (v1.0 Phase 1-4)
- ✓ 训练核心模块测试: losses, optimizers, metrics (Phase 5)
- ✓ 回调与评估测试: callbacks, evaluators, visualization (Phase 5)
- ✓ Lightning训练模块测试: lightning_module, trainers, peft_config (Phase 5)
- ✓ GenKI集成合约测试: adapter, reports, contracts (Phase 5)
- ✓ API路由覆盖测试 (Phase 6)
- ✓ GenKI集成深度测试 (Phase 6)
- ✓ 信号网络/增强/变异效应测试 (Phase 6)
- ✓ 数据/模型零覆盖模块测试 (Phase 6)
- ✓ 959+ 测试覆盖 (v1.0 complete)
- ✓ ESM2长序列处理集成 (Phase 7, TRAIN-03)
- ✓ 评估指标集成: MCC + per-PTM-type (Phase 7, EVAL-01)
- ✓ DAVF 模块导入与检查点管理 — v2.0 (DAVF-IMPORT, DAVF-CHECKPOINT, DAVF-DEPS)
- ✓ PTMDirectionMapper PTM→扰动映射 — v2.0 (DAVF-MAPPER)
- ✓ DAVFInferenceModule + DeltaProjection 特征提取 — v2.0 (DAVF-INFERENCE)
- ✓ 级联融合架构 + use_davf 向后兼容 — v2.0 (DAVF-CASCADE, DAVF-CONFIG)
- ✓ 端到端 DAVF 流水线集成测试 — v2.0 (DAVF-TEST)

### Historical Debt Checklist

以下 FIX 条目属于 v2.1 技术债里程碑，已由
`.planning/REQUIREMENTS.md` 的 TEST/DEPS/CODE/CONF 需求和 Phase 15--17
验收关闭；保留在这里是为了历史追踪，不应再被当作未完成任务：

- [x] FIX-01: 修复失败测试（peft_config mock、Lightning API、pathway_integration）
- [x] FIX-02: 补齐可选依赖（anndata、sspa、lion-pytorch）
- [x] FIX-03: 修复 `evaluation/__init__.py` 静默吞 `ImportError`
- [x] FIX-04: 整合根目录孤立脚本到 `src/scripts` 体系
- [x] FIX-05: 修复 `.gitignore` 白名单遗漏关键文件
- [x] FIX-06: Lightning Trainer 默认 Logger 配置
- [x] FIX-07: 根目录重复文件归属决策与清理
- [x] FIX-08: KEGG/Reactome 加载 stub 明确化

### Active Follow-up

- [ ] PTM activity 主线外部资产接入与正式运行：方案 §10 六项外部输入
  （PTM 定量、KSTAR/PhosR activity 表、OmniPath signed 网络、network id map、
  activity benchmark、AD donor-level DEG 表）+ downstream-target lineage
  接线；契约层已落地，当前状态见 `docs/guides/ptm_activity_pipeline.md`
  与 `docs/CURRENT_STATUS.md`。
- [ ] DAVF 端到端微调：属于 2026-08-21 PerturbGen 后续方案 M4，当前状态见
  `project_analysis_20260914.md` 的未解决清单；不属于已完成的 v2.2 工程契约验收。
- [ ] 真实数据/权重/图的科学验收：属于后续 Gate-0、Gate-E、Gate-4、Gate-5，
  不用 synthetic fixture 代替，当前状态见 `docs/CURRENT_STATUS.md` 和
  `project_analysis_20260914.md`。
- [ ] 文档持续维护：以 `docs/CURRENT_STATUS.md`、
  `project_analysis_20260914.md` 和 `docs/guides/ptm_activity_pipeline.md`
  为活动入口；历史报告只作追溯。

### Current research mainline contract (2026-09-14)

当前研究主线是 PTM activity → AD 交集上游接入既有 DAVF × PerturbGen 验收路径
（`docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` v1.0、
`docs/guides/ptm_activity_pipeline.md`），分四个终点记录：

0. **上游 PTM activity 主线（2026-09-14 契约层落地）**：全局 PTM 定量 →
   KSTAR/PhosR activity（独立环境，主环境只消费标准表）→ signed network
   有符号简单路径传播 → global gene score → AD per-cell-type donor-level
   DEG 同方向交集 → gene-level intervention candidate spec。方向字段分字段
   记录、source/target 角色分离、无独立 null 不写 PTM 侧 q 值
   （`direction_only`）。真实 PTM/网络/benchmark 资产是外部输入（方案 §10）。
1. 候选准入：外部 `PTM proposal/candidate_spec` → scVI/PTM 映射 → DAVF
   方向与置信证据 → 独立 donor-level 表达方向三方 gate → evidence/invocation。
   classifier 只预测 site presence；候选方向来自外部假设或逐 site override，不能写成 rawsite 自动因果表达推断。阶段 5 CLI 产出的 spec 走同一入口。
2. PerturbGen 准备与运行：六阶段为
   `tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report`；
   E2E 默认只导出报告，只有显式 `--run-perturbgen` 才执行这条准备/执行链，不能把它写成另一条 PTM 推理入口。
   `source_intervention=[src]` 与 `within_state=[tgt]+pert_tps` 是两个实验场景；正式候选要求两场景 AND。自 2026-09-13 第四轮起每 route 的准备三阶段在 `_prepare/` 公共执行一次，候选循环只跑 perturb/export/report。
3. 统计验收：matched-null、候选 empirical-p 聚合、formal 输入隔离、未扰动质量和 donor split 接口均已存在，E2E 经 `--assemble-statistical-evidence` 自动接续并写回 lineage；matched-null 批跑仍由 `run_matched_null_stages.py` 独立执行。正式结果必须绑定真实 normal/disease raw counts、显式 donor、≥3 可评估 donor（pairing 显式声明）、canonical Ensembl、scVI/embedding manifest、真实 null/质量/双场景统计，不得将 smoke、synthetic、mock、bridge 或四队列规划写成生物学 PASS。

方向记录必须包含 `context`、`intervention`、比较基准、研究目标（关联、复现或逆转）、来源及训练/held-out 划分。观测 donor-level disease−normal、DAVF 干预后 decode−当前 context decode 和 PerturbGen 效用预测保持不同语义；不采用全局同号/取反，不把病程签名或 normal/disease 对比作为 KO/KD ground truth，也不把结果升级为治疗因果或临床疗效。

Workflow A 是 gate 后效用评估；Workflow B 是固定基础 encoder → 冻结 embedding asset → LatentDAVF 重训 → Gate-E 的独立资产生命周期。基础 export 不消费候选结果、不训练本次新 checkpoint、不回灌本次 DAVF。正式 CLI 选择 `perturb`/`--path` 时继续要求绑定通过的 E2E gate report；invocation 拒绝 non-pass，但低层 runner 当前只执行 StagePlan。

### Cancelled

- [x] 实时质谱流 / 实时质谱数据处理 — 2026-07-05 取消，不再规划流式摄取、队列消费或在线质谱预测
- [x] 自定义 PTM 数据库 — 2026-07-05 取消，不再规划用户自带 PTM 数据库/catalog/API
- [x] GUI — 2026-07-05 取消，不再规划 Streamlit、Gradio 或桌面界面
- [x] API key 功能扩展 — 2026-07-05 取消；现有兼容性中间件可保留，但不作为新增需求

### Out of Scope

- 分布式训练支持 — 当前单节点足够
- 实时大规模推理服务 — 当前批处理模式满足需求
- 实时质谱流处理 — 已取消，当前批处理架构满足需求
- 自定义 PTM 数据库 — 已取消，使用公共 PTM 数据源和标准文件导入
- GUI — 已取消，使用 CLI / Python API / FastAPI
- API key 功能扩展 — 已取消，不再作为 roadmap 项
- 通用蛋白质结构预测 — AlphaFold2 已解决
- ESM-3 集成 — 等待稳定性和 HuggingFace 支持

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 多种编码器支持 | 不同场景需要不同编码器（速度vs精度） | ✓ 已实现 |
| Attention Pooling | 比Mean Pooling更灵活，可学习重要位置 | ✓ 性能测试通过 |
| FastAPI for API | 现代、快速、易用 | ✓ 已实现 |
| Phase 5-6 测试覆盖完成 | 为训练/评估/集成/API模块补齐单元测试 | ✓ 959+ tests |
| Phase 7 训练与指标集成 | 长序列处理和MCC/per-PTM指标接入生产路径 | ✓ TRAIN-03, EVAL-01 |
| rename attention.py→davf_attention.py | 避免与现有attention模块冲突 | ✓ Good |
| Default latent_davf_ibd_norman | 10-dim latent, scVI compatible, best general performance | ✓ Good |
| Parallel DAVF branch | DAVF与ESM-2并行运行 | ✓ Good |
| Feature concatenation before predictor | 拼接ESM-2+DAVF特征再预测 | ✓ Good |
| use_davf=False default | 向后兼容v1.0行为 | ✓ Good |
| Runtime YAML config | configs/davf_integration.yaml为配置源 | ✓ Good |
| Multitask unsupported w/ DAVF | 显式拒绝，约束明确 | ✓ Good |
| Freeze DAVF first | 训练DeltaProjection+Predictor后再fine-tune DAVF | ✓ Good |

## Current Milestone: v2.2 Cross-Scale Scientific Closure & Reproducible Baseline

**Goal:** Close the PTM→signal graph→cell-state scientific scale gap and establish a versioned data inventory plus a reproducible PMADS Ridge baseline.

**Target features:**
- Versioned data manifest covering the required PTM, perturbation, single-cell and graph sources, formats, licenses/access status and quality gates;
- deterministic PMADS normalization, fixed split, Ridge baseline, metrics and artifact provenance;
- opt-in MultiPLMEncoder, CIGNNSignalBridge with sensitivity matrix, and CellGraphCompass-style delta-expression/state decoder;
- integration, gradient/checkpoint, data-quality and CPU performance verification with a dated technical report.

## Evolution

This document evolves at phase transitions and milestone boundaries.

**v2.2 boundary (2026-08-08):** the new model is opt-in and does not change the
default ``PTM2CellNetBase`` prediction contract. Missing pretrained weights,
graph files or controlled datasets must be explicit in provenance; no synthetic
fallback is accepted as evidence of biological validity.

---
*Last updated: 2026-09-14 after the PTM activity → AD intersection mainline contract layer landed; current evidence is in `docs/CURRENT_STATUS.md` and `project_analysis_20260914.md`*
