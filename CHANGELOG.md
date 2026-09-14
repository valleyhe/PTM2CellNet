# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Gate-0 cohort pairing semantics: `within_donor` (default, ≥3 donors shared across states) and `between_donor` (case-control, disjoint donor groups with ≥3 donors each); E2E flag `--perturbgen-cohort-pairing` (2026-09-14)
- AD case-control cohort tooling: `scripts/audit_ad_cohort_gate0.py` (read-only Gate-0 audit of data/AD GEO cohorts) and `scripts/standardize_gse174367_ad_cohort.py` (donor-evidence-validated standardization; 7/7 cell-type between_donor preflight PASS)
- E2E statistical assembly: `--assemble-statistical-evidence` with `--deg-table`/`--null-distribution-manifest` chains unperturbed quality → formal eval input → empirical-p/BH-FDR → dual-path AND into report lineage; core extracted to `src/integration/perturbgen/replay_evaluation.py` (2026-09-13)
- Cross-candidate shared prepare: `orchestrator.build_shared_prepare_plans` runs tokenise/train_mask/train_decoder once per route under `<root>/<route>/_prepare/`; candidates reuse artifacts via `resolve_prepare_artifact_references` (2026-09-13)
- Donor-row binding for DAVF pair generation: `build_scperturb_latent_pairs --donor-obs-column/--train-donors/--held-out-donors` writes `target_donors` and `dataset.donor_rows` (`ptm2cellnet.donor_split/v1`) (2026-09-13)
- `SemanticContext` seven-field contract on candidates/invocations (context, intervention, baseline, objective, cohort, sources, reference axis) with fail-fast validation (2026-09-13)

### Changed
- `evaluate_perturbgen_dual_path.py` is now a thin shell over `replay_evaluation.py` (single implementation)
- Formal Gate-0 acceptance wording now covers both pairing modes (≥3 shared donors within_donor; ≥3 per disjoint group between_donor)

## [2.1.0] - 2026-08-24

> 2026-05-04「milestone v2.1: Technical Debt & Test Stabilization」启动的开发周期收口发布
>（TD-N-31，2026-08-24 补记；条目依据 git 历史与 `project_analysis_20260824.md`）。

### Added
- 完整人类蛋白质组序列下载 (UniProt 204,729条)
- 数据整合脚本 v2 (80,508条整合数据 + 112,012条已标记数据)
- Mamba编码器支持 (Selective State Space Models)
- PEFT/LoRA微调集成
- GENKI3可解释性框架集成
- 氨基酸字符标准化 (U→C, X→A, J→L, B→D, Z→E, O→K)
- Docker生产部署配置与docker-compose编排
- ESM-3 编码器与 PTM 虚拟扰动管线（2026-07-04/05）
- DAVF / Mamba / PTM-site 三类 E2E 测试与 README Quick Start（2026-07-05）
- 跨尺度 (cross-scale) 模型、E2E 管线与可复现数据基线（2026-08-08）
- API 在线服务端点 initialize/predict/batch_predict（F-01，2026-08-16）
- scPerturb h5ad 端到端解析器与 BioPlex/RegNetwork/STRING 图导入器（F-05/F-03，2026-08-17/18）
- 数据 manifest 契约补全：GSE90546 目录、kinase-substrate/scPerturb/EPSD 注册（2026-08-16）
- 预训练 checkpoint 完整状态持久化与精确 resume（N20，2026-08-16）
- DAVF × PerturbGen 双路径整合：M1-M3 工程主干（2026-08-22）+ M4 embedding 注入链路 schema v2（2026-08-24）
- PerturbGen 运维脚本：环境搭建/离线 wheel/证据采集/M0 smoke/cohort 审计（2026-08-24）
- CI 依赖一致性门禁（pip check + `scripts/check_requirements_consistency.py`，TD-N-32/34，2026-08-24）

### Changed
- PTM融合机制扩展为 attention + gated 两种策略
- 池化方式支持 mean / attention / weighted_mean / multi-head
- Python版本要求从3.9+更新为3.10+
- mypy配置从3.9更新到3.10
- API 阻塞推理卸载至线程池并默认单 worker（N01/TD-M08，2026-08-16）
- UniProt ID-mapping 轮询改为指数退避 + 显式 num_workers（N08/N09，2026-08-16）
- PTM 窗口常量集中单源，消除 31/15 魔法数（N05，2026-08-16）
- `predict_variant` 拆分为内聚辅助函数（TD-LONG-01/N12，2026-08-18）
- CI analysis job 覆盖扩展至整个 perturbgen 单测目录（U-15，2026-08-24）
- 移除空壳 `uv.lock`，钉扎源统一为 requirements-lock.txt（TD-N-27，2026-08-24）

### Fixed
- dbPTM数据格式解析兼容性修复 (5列 vs 6列格式)
- 预训练编码器测试网络依赖问题
- 日志f-string格式警告 (W1203)
- 混合行尾符问题 (CRLF → LF)
- evaluation/__init__.py 类型注解问题
- API auto-init 发现 checkpoint 同目录 config（P1-1，2026-08-04）
- 数据管线输出 provenance 列并契约门禁真实数据（P2-1，2026-08-04）
- DAVF collate fn 与 gene mapper 错误处理（2026-07-06）
- requirements-lock.txt 移除互斥的 scgpt 钉扎、补钉 datasketch/uvicorn（TD-N-34，2026-08-24）
- GeneMapper 外部调用挂死：daemon 线程 + 硬超时（TD-N-24，随提交 63ebf75 闭环）
- 同名双份 test_gene_mapper 测试合并至 tests/unit/analysis/（TD-N-33，2026-08-24）
- pretrain_masked_ptm 286 行超长函数拆分为 5 个内聚辅助函数（TD-N-26-B，2026-08-24）

### Known Issues
- PTM位点-氨基酸匹配率 87.13% (多源数据整合预期范围内)
- cell_state标签仅2种 (Quiescent/Activated)，需扩展
- PerturbGen Gate-0 M0⑥ donor cohort 外部数据缺口（0 合规候选）阻断 M4 重训 / M6 / Gate-4 科学验收链（project_analysis_20260824.md §3 U-01）
- Geneformer「未知基因 hash 映射 + 随机 fallback」子项维持登记，待 Gate-E 通过后随 M7 删除

## [1.0.0] - 2026-02-24

### Added
- 基础模型架构 (CNN, Transformer, LSTM编码器)
- PTM处理模块 (注意力融合)
- 分类/回归预测器
- 数据处理模块 (加载、预处理、特征提取)
- 训练器模块 (原生 + Lightning)
- 评估模块 (指标、可视化)
- API模块 (FastAPI)
- 单元测试套件
