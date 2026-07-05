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

### Active

- [ ] FIX-01: 修复 14 个失败测试（peft_config mock、Lightning API、pathway_integration）
- [ ] FIX-02: 补齐 requirements.txt 缺失依赖（anndata, sspa）
- [ ] FIX-03: 修复 evaluation/__init__.py 静默吞 ImportError
- [ ] FIX-04: 整合根目录孤立脚本到 src/scripts 体系
- [ ] FIX-05: 修复 .gitignore 白名单遗漏关键文件
- [ ] FIX-06: Lightning Trainer 默认 Logger 配置
- [ ] FIX-07: 根目录重复文件归属决策与清理
- [ ] FIX-08: KEGG/Reactome 加载 stub 明确化
- [ ] DAVF 端到端微调 (v2.2)
- [ ] scVI decode 集成 (v2.2)
- [ ] 通路知识库上下文相关映射 (v2.2)
- [ ] 文档完善
- [ ] 更多PTM类型支持

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

## Current Milestone: v2.1 Technical Debt & Test Stabilization

**Goal:** Fix all failing tests,补齐缺失依赖, clean up orphaned code, improve error handling — return codebase to healthy state.

**Target features:**
- Fix 14 failing tests (peft_config mock targets, Lightning API, pathway_integration)
- Add missing dependencies to requirements.txt (anndata, sspa)
- Fix evaluation/__init__.py silent ImportError swallowing
- Consolidate 7 orphaned root-level scripts into src/scripts
- Fix .gitignore whitelist missing critical files
- Add default Logger config for Lightning Trainer
- Decide fate of duplicate root files (signaling_network.py, pathway_knowledge_base.py)
- Clarify KEGG/Reactome loading stubs

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-07-05 after cancelling real-time mass-spec streaming, custom PTM database, GUI, and API-key expansion*
