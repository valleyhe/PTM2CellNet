# 归档清单 2026-09-14

- **归档执行时间**：2026-09-14（第六轮综合处理：VCS 收口、文档时效审计、对抗性分析）
- **归档时基线提交**：`55e65ed`（main，`docs: sync living docs with landed round-4/5 behavior`）
- **归档方式**：`git mv`（保留完整提交历史，可用 `git log --follow` 追溯）
- **归档判定标准**（沿用 `archive/20260913/ARCHIVE_MANIFEST.md`）：
  1. 过时 DOCUMENT：内容与当前代码或现行需求（`docs/CURRENT_STATUS.md`、`lessons.md` 最新条目、测试/CI）存在实质性差异；
  2. 过期 REPORT：生成于 2026-08-15 之前，或内容已无法反映当前项目状态。
- **审计方式**：4 个并行子代理（需求对照 / 技术债 / 文档时效 / diff 对抗审查）+ 主上下文逐条证据核验；文档时效子代理对 docs/ 与根目录全部被跟踪 .md 的结论为"除下列两项外无新增 git 归档候选"。

## 归档文件

| 文件 | 归档前路径 | 归档前 blob（`55e65ed`） | 最后修改 | 类别 | 说明 |
|---|---|---|---|---|---|
| `project_analysis_20260913.md` | 仓库根目录 | `e0c07cbf8de6` | 2026-09-13 23:14 +0800 | 过期 REPORT + 过时 DOCUMENT | 被 `project_analysis_20260914.md` 取代。§4.5a/§6.2 的"F-01/F-02/F-03 未闭合"表述已被第四轮（`d10e96c`/`5e71c69`）证伪；§4.3 A-01"0 合规 cohort"已被第五轮 between_donor + GSE174367 preflight（`57e72d6`）更新。 |
| `task_plan.md` | 仓库根目录 | `4144e6e3a73b` | 2026-09-13 18:55 +0800 | 过时 DOCUMENT | 2026-08-27 综合审计任务计划，全部执行项已完成，基线信息（HEAD `a8c480c` 时代）早已过时；保留会误导为现行计划。 |

## 未归档（含理由）

- `project_repair_report_20260913.md`：第三/四轮修复的逐项证据记录，其 F 系列闭合声明与当前代码一致（本轮子代理 D 逐项核验通过），无实质性差异，保留根目录作为修复细节参考。
- `README.md`、`docs/CURRENT_STATUS.md`、`lessons.md`、各 guides、`API_DOCUMENTATION.md`：本轮已同步或本就准确。
- `docs/_build/`：2026-08-09 的本地 Sphinx 构建产物，未被 git 跟踪（`.gitignore` 已排除），属本地工作区卫生问题，可本地删除或重新生成，不属 git 归档范围。

## 本轮配套的文档更新（未归档、原地修订）

反向漂移（代码已落地、文档仍写"未实现"）已在提交 `55e65ed` 修复：`README.md` 研究路径第 2/3 点与格式损坏、`docs/guides/perturbgen_bridge.md` §1 六阶段表、方案 v2.1 §11 复审注记、`CHANGELOG.md` [Unreleased]、`.planning/REQUIREMENTS.md` A-01/A-05/A-06 状态。未跟踪的本地 `AGENTS.md` 约束条款与三个 `docs/PTM2CellNet_*.md` 入口存根同步更新（存根随本归档批次提交，指向新报告）。
