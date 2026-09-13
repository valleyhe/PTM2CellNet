# 2026-08-27 综合审计发现记录

## 基线事实

1. 根目录已有 `project_analysis_20260816.md` 至 `project_analysis_20260824.md`，以及多份 `project_repair_report_*.md`。
2. 仓库已经存在按日期组织的 `archive/20260808`、`archive/20260816`、`archive/20260820`、`archive/20260821`、`archive/20260822`、`archive/20260824`。
3. 当前主分支没有未提交修改；远程主分支没有本地缺失提交。

## 约束

- 仅归档能够由内容差异或报告状态证明过时的活动文件。
- 不把已经位于 `archive/` 或 `docs/archive/` 的历史材料重复迁移。
- 不把 README、API 文档或现行状态文档仅因日期较旧就判为过时，必须有代码/需求矛盾证据。
- 未取得真实 PerturbGen 权重、合规 normal/disease donor cohort 或 Gate-0 证据时，报告不得把运行状态写成已完成。

## 2026-08-27 复核新增事实

1. 正式 v2.1/v2.2 需求共 24 项，需求层面完成；后续 M0-M7 是 supplemental proposal，不能混入正式完成数。
2. 真实链路的关键断点已由只读调用链复核确认：API DAVF dict/PTMSite 不兼容；`use_davf` 不会动态切换模型；PerturbGen `gene_to_token` 没有进入 DAVF；runner 不自动接 evaluator/formal evidence；GenKI 默认 `soft_ko` 不符合 adapter 合约。
3. 新技术债从 TD-N-35 编到 TD-N-56；无严重级别。高风险包括 API 状态竞态、format/CI 门禁、wheel CLI、DAVF payload/asset、PerturbGen 闭环、GenKI 默认模式和假绿色 Gate 证据。
4. 代码质量实测以 SSH_unit Python 3.12 为准：全量测试 2329/16、compileall 和 Ruff check 通过；mypy 23 错误、pip check 3 冲突、format check 252 文件待格式化均如实写入报告。
5. 活动文档可修正的问题优先修正；历史报告正文只迁移已被当前基线取代的文件，既有 archive 不重复迁移。

## 最终版本控制记录

- 审计内容提交 `f0be1abbceca7105d5605bb9d8c989c619a71ea7`，元数据提交 `fce11aebb30e8d339d8c94aca43cdc73a68b2552` 和 `3656921c401e5000b74446d080e7779cb35a8b44`。
- final `git merge --ff-only origin/main` 输出 `Already up to date.`；`main...origin/main = 47 0`；最终 HEAD `b1a315572ecd2bdd2b6dfe013e117ec8f81d20df`；最终工作树干净。

## 2026-09-01 续作发现

1. `direction_gate.py` 严格要求 PTM proposal、DAVF expression delta 和 observed normal/disease direction 三方一致；缺证据为 inconclusive，完整冲突为 fail，fallback 不得通过。
2. `DAVFInferenceModule.predict_expression_direction()` 只接受真实 checkpoint、scVI decoder 和 verified embedding asset；`LegacyLatentDAVF` 没有 `predict()` 时现在显式报错。
3. API 生成的 DAVF 列表已保留为 list-of-lists；有效 PTM 缺少 `gene_symbol` 时返回 400；variant 派生 PTM 暂无 gene symbol，因此 cell-state 子结果保持不可用并写入 warning。
4. `gene_to_token` 已从 embedding asset 注入 DAVF 和 strict mapper，但 `scripts/run_perturbgen_pipeline.py` 没有调用新的 mainline 入口；真实闭环仍是开放项。
5. 全量离线验证 2373 项收集，2357 passed、16 skipped、47 warnings；compileall/Ruff check 通过，mypy 5 errors、pip check 3 conflicts、format check 298 files 待格式化。

## 续作版本控制

- 代码变更前 `b1a3155`，代码提交 `19024a1`；远程 `origin/main=c76fff8`，快进合并无冲突。
- 当前权威报告：`project_analysis_20260901.md`；旧报告归档于 `archive/20260901/reports/`。
