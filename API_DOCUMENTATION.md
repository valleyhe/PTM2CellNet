# PTM2CellNet API 文档入口

该文件的旧版 API 快照已归档，原文保留于
[`archive/20260808/docs/API_DOCUMENTATION.md`](archive/20260808/docs/API_DOCUMENTATION.md)。

当前维护入口：

- 部署与探针：[`docs/guides/deployment.md`](docs/guides/deployment.md)
- 当前状态与审计：[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
- 运行服务后生成的 OpenAPI：`/docs` 或 `/redoc`
- 当前代码、E2E 与风险复核：[`project_analysis_20260922.md`](project_analysis_20260922.md)（上一轮 2026-09-21 分析/修复报告已归档至 [`archive/20260922/`](archive/20260922/MANIFEST.md)）

## PerturbGen 集成合同（`src/integration/perturbgen`）

外部调用方构造正式候选时必须遵守以下公开合同（均 fail-fast，缺字段/非法值硬失败）：

- `SemanticContext`（`contracts.py`，七字段必填）：`context`、`intervention`、
  `comparison_baseline`、`reference_axis`、`research_objective`、
  `evidence_source`、`cohort`。`research_objective` 只接受
  `association`/`replication`/`reversal`；`intervention` 必须与 invocation 的
  KO/KD route 一致。构造入口 `SemanticContext.from_mapping(mapping)` 或
  `normalize_semantic_context`。
- `PerturbGenInvocation`（`orchestrator.py`）：只接受 direction gate `pass` 的
  `CandidateEvidence`，且候选与 invocation 的 `semantic_context`、gene、Ensembl、
  DAVF evidence 与 `davf_score == confidence` 必须完全一致。
- 共享准备与候选计划（`orchestrator.py`）：`build_shared_prepare_plans(config,
  output_root=...)` 为一条 route 公共计划 `tokenise → train_mask →
  train_decoder`；`build_candidate_stage_plans(..., skip_prepare_stages=True,
  prepare_artifact_paths=...)` 产出只含 perturb/export/report 的候选计划并把
  `@artifact:tokenise/train_*` 引用解析到共享准备产物。
- `PerturbGenRunner`（`runner.py`）：工程执行层，只执行 `StagePlan`，不重查 DAVF
  gate；正式执行的唯一公开入口是 `run_perturbgen_pipeline.py
  --e2e-gate-report` 与 orchestrator invocation。
- 统计证据组装（`scripts/run_davf_perturbgen_e2e.py
  --assemble-statistical-evidence`）：显式提供 donor 级 DEG 表与
  `perturbgen_null_distribution/v1` manifest 后，自动完成未扰动质量提取、formal
  评估输入、候选 empirical-p（`conservative_max_required_runs`）、BH-FDR 与
  双路径 AND；结果与 lineage 写回 E2E report 的 `statistical_evidence` 节。

旧版内容中的端点清单、环境变量和响应示例可能与当前代码不一致，不应作为
接口契约使用。
