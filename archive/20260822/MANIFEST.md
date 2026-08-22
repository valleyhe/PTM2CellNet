# 归档批次 20260822

**归档日期**：2026-08-22
**执行轮次**：项目代码与文档综合处理及技术分析（第四轮周期性复核，PerturbGen 双路径集成落地后的首次全量复核）
**取代者**：仓库根目录 [`project_analysis_20260822.md`](../../project_analysis_20260822.md)

## 1. 归档清单

| 归档文件 | 原路径 | 版本标签 | 归档依据 |
|---|---|---|---|
| `reports/project_analysis_20260821.md` | 仓库根 | `115de00`（2026-08-22，`git mv` 保留历史） | 其测试基线（单元 1957+6 / 集成 107+1）未含 PerturbGen 新增 112 例（2026-08-22 全量实测 2310 passed / 15 skipped / 1 deselected）；技术债 TD-N-01~09 被本轮 TD-N-10~25 扩充（新增 CI 门禁 anndata 缺失、GeneMapper 外部包无超时致测试/生产挂死、mypy +18 等）；验证矩阵未含网络隔离口径 |

## 2. 判定标准

沿用 `archive/20260816/MANIFEST.md` 以降规则：

- **过时** = 内容与当前代码实现或需求规范存在实质性差异。
- **过期** = 生成时间超过 30 天，或报告内容已无法反映当前项目状态。

## 3. 判定为无需归档的边界项

| 文件/类别 | 处置 | 理由 |
|---|---|---|
| `docs/CURRENT_STATUS.md` | 内容刷新（本批次提交内完成） | 活动状态文档；0 次 PerturbGen 提及已过时，按惯例刷新而非归档 |
| `docs/TEST_COVERAGE.md` | 保留 | 带日期的历史基线记录（2026-08-09 基线段）+ ratchet 现行门禁政策，非误导性过时 |
| `docs/` 下 8 份入口短页 | 保留 | 正文已在 `archive/20260808/`、`archive/20260816/`，现存仅重定向入口 |
| `.planning/{REQUIREMENTS,ROADMAP,PROJECT,STATE,MILESTONES}.md` | 保留 | 正式需求规范与 roadmap，是任务 1 对照基准；PerturbGen 属超出 REQUIREMENTS v2.2 的批准增量（由设计方案文档承载） |
| `docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` | 保留（随 `ddbf571` 入库） | PerturbGen 现行需求基线（v2.0，"已批准实施"状态） |
| `docs/guides/*.md` | 保留 | 指南与 CLI/API 一致；PerturbGen 指南缺失登记为技术债 TD-N-19（文档债）而非归档项 |
| `project_analysis_20260816.md`、`project_analysis_20260820.md`（根目录） | 保留 | 均已是重定向短页 |

## 4. 版本信息

- 归档操作基于本地 `main` = `ddbf571`（feat(integration): PerturbGen dual-path bridge engineering mainline）
- 远程 `origin/main` = `c76fff8`（fetch 后无新提交，本地领先 / 落后 0+领先数见权威报告 §1.1）
- 本批次归档以 `git mv` 执行，保留完整重命名历史；原路径留重定向短页（沿用既往惯例）
- 最终批次哈希在权威报告落地后回填（沿用 `115de00` 惯例）：报告提交 = <见 project_analysis_20260822.md §1.1 版本表>
