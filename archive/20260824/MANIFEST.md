# 归档批次 20260824

**归档日期**：2026-08-24
**执行轮次**：项目代码与文档综合处理及技术分析（第五轮周期性复核，PerturbGen M4 代码前置落地后的首次全量复核）
**取代者**：仓库根目录 [`project_analysis_20260824.md`](../../project_analysis_20260824.md)

## 1. 归档清单

| 归档文件 | 原路径 | 版本标签 | 归档依据 |
|---|---|---|---|
| `reports/project_analysis_20260822.md` | 仓库根 | `1710e43`（2026-08-22，`git mv` 保留历史） | 完成度基线（models 94.4%、PerturbGen 全口径 49.0%）已被 M4 代码前置与 M0⑤/M2 真实 evidence 推进至 98.6% / 71.4%；TD-N-10/TD-N-24 已在 `63ebf75` 闭环；「CI analysis job 缺失」表述过时。其 TD-N-10~25 登记表仍为 TD-N-26+ 排除基准 |
| `reports/project_analysis_20260823.md` | 仓库根 | `1d6e30d`（2026-08-23 入库） | 其修复依据 `[R3]` 两项时间窗口任务已全部执行完毕；测试基线 2321 被 0824 实测 2328/16 取代；「CI analysis job 缺失」表述过时 |
| `reports/project_repair_report_20260816.md` | 仓库根 | `1d6e30d` | 点时修复记录（D1/N08/N09/N05+D3），修复早已合入主干；K/F/N 编号上游出处随归档可回溯 |
| `reports/project_repair_report_20260817.md` | 仓库根 | `1d6e30d` | 点时修复记录 v5.0（N04/GSE90546 manifest/N07/N08），内容被后续多轮分析吸收 |
| `reports/project_repair_report_20260817_k02_f08_f05.md` | 仓库根 | `1d6e30d` | 点时专项记录（Mamba fused、GSE90546 解析、scperturb h5ad loader），已合入主干 |
| `reports/project_repair_report_20260822.md` | 仓库根 | `1d6e30d` | 其 §5 排期 2026-08-23 的两项阻塞任务已全部完成；未解决项状态由权威分析接管 |
| `reports/project_repair_report_20260823.md` | 仓库根 | `1d6e30d` | 全部修复项已入库（`63ebf75`）并被权威分析复核吸收 |
| `planning/task_plan.md`、`planning/findings.md`、`planning/progress.md` | 仓库根（原为 gitignore 本地工作文件） | 内容日期 2026-08-16~08-22 | planning-with-files 工作流遗留状态，对应修复周期已收口；本轮起纳入版本控制归档（沿用 `archive/20260820/planning/` 惯例） |

## 2. 判定标准

沿用 `archive/20260816/MANIFEST.md` 以降规则：

- **过时** = 内容与当前代码实现或需求规范存在实质性差异。
- **过期** = 生成时间超过 30 天，或报告内容已无法反映当前项目状态。

本批次的修复报告均不满足"超 30 天"，判定依据为第二标准：其待办/未解决
清单所指向的工作已执行完毕或被后续文档取代，继续留存根目录会误导
"仍有未收口工作"的判断。

## 3. 判定为无需归档的边界项

| 文件/类别 | 处置 | 理由 |
|---|---|---|
| 根目录 `project_analysis_20260816.md`、`project_analysis_20260820.md`、`project_analysis_20260821.md` | 保留 | 均已是重定向短页（13~16 行），无误导正文 |
| `docs/` 下 10 份入口短页 | 保留并刷新权威指向（TD-N-28 清偿） | 正文已在 `archive/20260808/`、`archive/20260816/`；本轮将指向链终点统一刷新为 `project_analysis_20260824.md` |
| `docs/CURRENT_STATUS.md` | 内容刷新（本批次提交内完成） | 活动状态文档；按惯例刷新而非归档 |
| `docs/TEST_COVERAGE.md` | 保留 | 历史基线记录 + ratchet 门禁政策；基线滞后登记为 TD-N-30（受 TD-N-09 阻塞无法本地复测） |
| `docs/DATA_UPDATE_WORKFLOW.md` | 保留 | 数据源注册流程说明，与 datasets.yaml 契约仍一致，未受 PerturbGen 改动影响 |
| `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | 保留 | PerturbGen 现行需求基线（v2.0，"已批准实施"状态） |
| `.planning/{REQUIREMENTS,ROADMAP,PROJECT,STATE,MILESTONES}.md` | 保留 | 正式需求规范（v2.2）与 roadmap，任务 1 对照基准 |
| `docs/guides/*.md` | 保留 | 六篇指南与 CLI/API 一致；PerturbGen 指南缺失维持 TD-N-19 登记 |
| `AGENTS.md` | 头部现状注记刷新（TD-N-29 清偿）；蓝图正文保留 | 蓝图类名经抽查与 src 一致（SequenceEncoder/PTMModule/PTM2CellNetBase/PTM2CellNet 均存在）；仅测试基线与技术债注记行过时 |
| `lessons.md`、`codex_mcp_prompt.md` | 保留 | 活动记忆文件与工具提示词，非项目文档 |

## 4. 版本信息

- 归档操作基于本地 `main` = `6864821`（docs: refresh CURRENT_STATUS for PerturbGen M4 landing and env closure）
- 本轮功能提交 = `63ebf75`（feat(integration): PerturbGen M4 embedding injection chain + env evidence tooling，24 files，+1912/-44）
- 昨日报告入库提交 = `1d6e30d`（docs: add 20260823 comprehensive analysis and point-in-time repair reports）
- 远程 `origin/main` = `c76fff8`（fetch 后无新提交；本地领先 35 / 落后 0）
- 合并前基线 = `1710e43`（2026-08-22 12:13:21 +0800，本地领先远程 32 / 落后 0，无分叉可直接快进语义合并，无冲突）
- 本批次归档以 `git mv` 执行，保留完整重命名历史；原路径留重定向短页（沿用既往惯例）
- 最终批次哈希在权威报告落地后回填（沿用 `115de00` 惯例）：归档提交与报告提交哈希见权威报告 §1
