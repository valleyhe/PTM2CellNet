# 归档批次 20260901

**归档日期**：2026-09-01<br>
**归档前代码基线**：`19024a1b5255d07bc04d5a4ee178ee111869c1d3`<br>
**远程同步快照**：`origin/main = c76fff881f268f9bd0b39d68db8ba547115ec4cf`；`git merge --ff-only origin/main` 输出 `Already up to date.`<br>
**当前权威报告**：[`project_analysis_20260901.md`](../project_analysis_20260901.md)

## 归档文件

| 归档文件 | 原路径 | 原始版本标签 | 归档原因 |
|---|---|---|---|
| `reports/project_analysis_20260827.md` | `project_analysis_20260827.md` | `b1a3155` | 本轮 API→DAVF、strict vocabulary 和方向 gate 已改变原报告的 F-01/F-04/F-07 等结论；旧正文不再代表当前代码 |

## 判定与审计规则

- 文档与当前代码/需求存在实质差异即判为过时；检测报告超过 30 天或基线被后续状态取代即判为过期。
- 已归档的 2026-08-24 及更早材料不重复移动；仍承担当前约束的需求、方案和状态文档保留并更新链接。
- 归档使用 `git mv`，没有删除正文；原始修改记录可用 `git log --follow` 追溯。
- 本批次新增报告和清单在代码提交 `19024a1` 后生成，最终文档提交 hash 以 Git 历史为准。
