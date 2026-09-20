# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Versioning note
- Package metadata (`setup.py` / `src/__version__` / API `version` fields, all `1.0.0`) intentionally trails this changelog (`2.1.0` released 2026-08-24): they will be unified at the next real release rather than bumped just to match (analysis 2026-09-14 §6.2 TD-14-01).

### Added
- 2026-09-18 PTM smoke 契约 + KSTAR 独立环境方案：`src/analysis/ptm_smoke.py` 与
  `scripts/generate_ptm_smoke.py` 生成可重放位点/evidence/activity-stub/smoke 网络并跑通
  阶段 1/3/4/5（跳过 KSTAR）。`method=KSTAR` 仅满足 `primary_activity_method`，
  `method_version=smoke-stub-*` 禁止进 formal lineage。方案见
  `docs/guides/kstar_activity_plan.md`（不把 kstar 写入 `requirements-core`）。
  测试：`tests/unit/analysis/test_ptm_smoke.py`、`tests/unit/scripts/test_generate_ptm_smoke.py`。
  不宣称 biology PASS
- 2026-09-18 AD 研究决策冻结：`src/analysis/ad_research_decision.py`（`ptm2cellnet.ad-research-decision/v1`）把 observed gate 分支 3、KD 并入 KO、公共 Perturb-seq out of scope 写成可审计 schema。分流/方向 gate/候选 spec/交集/E2E 读取该规则；报告 FDR 仍为 0.05。测试：`tests/unit/analysis/test_ad_research_decision.py` 及现有 routing/gate/spec 回归。不宣称 biology PASS
- 2026-09-18 方案 §6 候选分流：`src/analysis/ad_candidate_routing.py` 冻结五候选 canonical Ensembl 并按 KO 路由（B1–B6 + APOE KO engineering；KD 行默认不输出）；`scripts/route_ad_candidates.py` 写出 exploratory sidecar 且永不生成 formal invocation。外部 evidence payload 增加 `lineage_boundary=supplementary_only` / `may_enter_lineage=false`；Geneformer `network_counterfactual` 只保留 `influence_score`，不得填 `predicted_direction`。测试：`tests/unit/analysis/test_ad_candidate_routing.py`、`tests/unit/scripts/test_route_ad_candidates.py`。真实跑通 formal invocation=0，不宣称 biology PASS
- 2026-09-15 批次 A：新增 AD donor-level DEG 聚合与 CLI（`src/analysis/ad_deg_table.py`、`scripts/build_ad_deg_table.py`），产出 aggregate 八列 + donor-level 长表，复用 donor log2 normalization/Welch/BH；manifest 显式记录 normal reference、effect scale、output contracts；新增 A 核心/CLI/双消费者测试
- PTM-activity → AD intersection mainline, contract layer of `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md` §6.2 (2026-09-14): `src/analysis/ptm_research_config.py` (stage-0 frozen research design, seven-field semantic-context template), `src/analysis/ptm_activity.py` (§4.1 input contract + §5.1 preprocessing + §4.2 activity-table parsing), `src/analysis/signed_network.py` (§4.3 edge contract + signed simple-path propagation with decay/coverage/degree normalization), `src/analysis/ptm_gene_score.py` (§4.4 score table + §5.4 per-cell-type Ensembl intersection with membership/evidence tiers), `src/integration/perturbgen/downstream_target_evaluation.py` (target-set sidecar + per-target delta concordance), and the four stage CLIs `run_ptm_activity.py` / `build_ptm_global_gene_scores.py` / `build_ptm_ad_intersections.py` / `build_celltype_candidate_specs.py` (E2E-compatible candidate specs with real per-cell-type context binding and hard token verification); guide at `docs/guides/ptm_activity_pipeline.md` (synthetic contract pass only — real PTM assets remain external inputs, 方案 §10)
- M6 between_donor frozen assets for GSE174367 (EX cell type, `outputs/perturbgen/frozen/20260914_gse174367_ex/`): frozen manifest (sha256-bound, train 12 / held-out 6 stratified seed=2 split with sex balance), 90-run acceptance plan, 5×99 matched-null selections, and provenance; `_validate_frozen_cohort_asset` PASS on the real cohort (2026-09-14, round 8; data-contract freeze, not a biology PASS)
- Real-assets drift guard `tests/real_assets/test_real_frozen_cohort.py`: reloads the frozen manifest (sha256 re-verified) and replays the between_donor asset validation behind `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1`
- GPU execution runbook for the six-stage + matched-null path in `docs/guides/perturbgen_bridge.md` (frozen-manifest-bound donor lists, Python-API null batch, candidate-spec inputs)
- Frozen-cohort pairing propagation (F-10): `FrozenCohortManifest.pairing` (default `within_donor`), `run_frozen_acceptance.py --freeze --cohort-pairing`, pairing-aware donor/state coverage in `_validate_frozen_cohort_asset`, and an E2E hard check that `--perturbgen-cohort-pairing` matches the frozen manifest (2026-09-14)
- Statistical lineage pairing (F-11): eval input `cohort_pairing` field + `statistical_evidence.pairing`, verified against the frozen manifest by `verify_eval_input_against_manifest`
- Held-out state coverage guard (F-12): `build_scperturb_latent_pairs --state-obs-column/--require-state-coverage` fails when the held-out donor pool covers fewer than two states and records the observed coverage in `pair_manifest.json`
- `--deg-donor-column/--deg-gene-column/--deg-effect-column/--deg-fdr-column` on the E2E CLI (F-16); statistical-argument validation extracted and run before the six-stage execution (TD-14-02)
- Synthetic unit tests for `standardize_gse174367_ad_cohort.py` (donor evidence, per-sample Diagnosis uniqueness, PAR_Y collapse, barcode alignment) and the audit verdict logic (F-13/F-15/TD-14-06)

### Changed
- 2026-09-15 批次 B：`scripts/run_davf_perturbgen_e2e.py` 显式接入 `--downstream-target-sidecar`；对 gated candidate 的 `result.h5ad` 以 `pred_counts` 为 baseline、`X` 为 perturbed 计算 donor-level delta，写入 payload/lineage；`matches_predicted` 与 `matches_observed` 分开记录，source 三方 gate 行为不变，下游 lineage 已接线
- 2026-09-15 批次 C：`PropagationConfig.max_paths_per_seed` 增加正整数校验；超限 hard fail 且不截断，写入 per-seed diagnostics/manifest
- Repository-wide documentation sync to the PTM-activity mainline guide (2026-09-14): README research-path/module tables, `.planning/{task_plan,STATE,ROADMAP,PROJECT,REQUIREMENTS,progress,findings}` (task plan rewritten to the mainline execution order; new P-01..P-06 mainline requirement rows), `docs/index.rst` toctree, E2E/bridge guide cross-references to `build_celltype_candidate_specs.py`, the 2026-08-21 proposal upstream note, `TEST_COVERAGE.md` stale-cohort wording plus a repaired `task_plan.md` link, and CURRENT_STATUS doc entries/boundaries
- `null_selection._token_values` now skips only the exact formal special tokens (`<cls>`, `<eos>`, `<mask>`, `<pad>`) in embedding vocabularies; unknown/invalid gene keys still hard-fail, and ranked genes keep original vocabulary positions (round 8, exposed by the first real `embedding_asset_20260822/vocabulary.json` consumption)
- `audit_ad_cohort_gate0.py` dual-pairing wording, per-cohort `gate0_status`, and derived verdict — rerun on real data yields `GATE0_UNLOCKED_FOR_BETWEEN_DONOR_GSE174367_ONLY` instead of the stale blanket BLOCKED (F-14, dead `_first_matrix_group` removed, empty-glob now raises, TD-14-04)
- `standardize_gse174367_ad_cohort.py` per-sample cardinality check now includes `Diagnosis` (F-13) and exits 0/2/1 for PASS/PARTIAL/NO_CELL_TYPE_PASSED (TD-14-05)
- JSON serialization converged into `reports.to_plain_object` (replaces four near-duplicate implementations, TD-14-03)
- `evaluate_perturbgen_dual_path.py` is now a thin shell over `replay_evaluation.py` (single implementation)
- Gate-0 cohort pairing semantics: `within_donor` (default, ≥3 donors shared across states) and `between_donor` (case-control, disjoint donor groups with ≥3 donors each); E2E flag `--perturbgen-cohort-pairing` (2026-09-14)
- AD case-control cohort tooling: `scripts/audit_ad_cohort_gate0.py` (read-only Gate-0 audit of data/AD GEO cohorts) and `scripts/standardize_gse174367_ad_cohort.py` (donor-evidence-validated standardization; 7/7 cell-type between_donor preflight PASS)
- E2E statistical assembly: `--assemble-statistical-evidence` with `--deg-table`/`--null-distribution-manifest` chains unperturbed quality → formal eval input → empirical-p/BH-FDR → dual-path AND into report lineage; core extracted to `src/integration/perturbgen/replay_evaluation.py` (2026-09-13)
- Cross-candidate shared prepare: `orchestrator.build_shared_prepare_plans` runs tokenise/train_mask/train_decoder once per route under `<root>/<route>/_prepare/`; candidates reuse artifacts via `resolve_prepare_artifact_references` (2026-09-13)
- Donor-row binding for DAVF pair generation: `build_scperturb_latent_pairs --donor-obs-column/--train-donors/--held-out-donors` writes `target_donors` and `dataset.donor_rows` (`ptm2cellnet.donor_split/v1`) (2026-09-13)
- `SemanticContext` seven-field contract on candidates/invocations (context, intervention, baseline, objective, cohort, sources, reference axis) with fail-fast validation (2026-09-13)

### Verification (2026-09-15)
- 相关定向测试 **81 passed、8 warnings**；ruff check、目标文件 format check、mypy src（172 files、0 errors）、requirements consistency、compileall、git diff --check 均通过；全量非 slow/gpu 为 **2780 passed / 22 skipped**，另有 1 个已知既有 real DAVF checkpoint failure；全仓 format check 仍有 **325 个既有待格式化文件**；真实冻结资产测试 **1 passed**，GPU probe 为 Tesla P40 / CUDA 11.8 / torch CUDA available
- 当前无冻结 `ptm_research_config`，且 KSTAR/PhosR/activity benchmark 等外部资产缺失，不能执行真实 AD CLI/D1 六阶段；不宣称 biology PASS

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
