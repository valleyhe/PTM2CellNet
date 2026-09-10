# 归档批次 20260827

**归档日期**：2026-08-27
**审计前分支**：`main`
**审计前版本**：`a8c480ca97a9902e7f34871358102b18bfebdd3f`
**远程同步快照**：`origin/main = c76fff881f268f9bd0b39d68db8ba547115ec4cf`；执行 `git fetch origin main` 后 `main...origin/main = 43 0`
**后续取代者**（2026-09-10 已再归档）：[`project_analysis_20260901.md`](../../archive/20260910/reports/project_analysis_20260901.md)

## 1. 归档文件

| 归档文件 | 原路径 | 原始版本标签 | 最后修改时间 | 判定与处理 |
|---|---|---|---|---|
| `reports/project_analysis_20260824.md` | `project_analysis_20260824.md` | `7e4b66d` | 2026-08-24 02:34:05 +08:00 | 过时：测试、审计基线和未完成项已由 2026-08-27 复核刷新；原路径保留入口短页 |
| `reports/project_repair_report_20260824.md` | `project_repair_report_20260824.md` | `a8c480c` | 2026-08-24 20:22:40 +08:00 | 过时：点时修复记录已被本轮综合报告取代；原路径保留入口短页 |
| `reports/TEST_COVERAGE_20260809.md` | `docs/TEST_COVERAGE.md` | `366e691` | 2026-08-09 18:49:29 +08:00 | 过期：生成时间超过 30 天且覆盖率数字不能代表 2026-08-27；活动路径重建为当前治理说明 |

## 2. 判定标准

- **过时**：内容与当前代码实现或当前需求规范存在实质差异。
- **过期**：生成时间超过 30 天，或内容已无法反映当前项目状态。
- 只含历史跳转信息的入口短页不重复移动；它们保留原路径，避免断链并保留审计入口。
- `.planning/REQUIREMENTS.md`、PerturbGen v2.0 方案和活动状态文档属于当前规范/活动记录，
  不作为历史报告移动；其中过时状态已直接修订并保留 Git 历史。

## 3. 版本与审计记录

- 归档动作使用 `git mv`，因此原文件的完整历史由 Git rename detection 和
  `git log --follow -- <path>` 保留。
- 归档前已确认本地 `main` 不落后 `origin/main`，没有新 worktree，也没有覆盖或删除既有归档。
- 本批次审计内容提交：`f0be1abbceca7105d5605bb9d8c989c619a71ea7`。
- 提交后执行 `git merge --ff-only origin/main`，实际输出为 `Already up to date.`，
  `main...origin/main = 44 0`；报告同时记录归档前版本和远程同步关系。
