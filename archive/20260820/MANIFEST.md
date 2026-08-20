# 2026-08-20 文档与报告归档清单

## 1. 清单元数据

| 字段 | 值 |
|---|---|
| 审计日期 | 2026-08-20 |
| 远程基线 `origin/main` | `c76fff881f268f9bd0b39d68db8ba547115ec4cf`（`git fetch` 后确认，远程自 2026-08-16 发布后无新提交） |
| 本轮起点本地 `main` | `cea16f3`（领先远程 22 个提交、落后 0 个，本地为远程超集；工作树干净，无待合并分支） |
| 本轮归档同步提交 | `d0a3b0c`（`git log -1 -- archive/20260820/`；R100 纯 rename 保历史），前导 `d703430`（tests lint 修复），后续 `a5ab6b2`（原路径重定向短页） |
| 验证状态 | `python -m compileall src scripts` 通过（exit 0）；全量单测 **1957 passed / 6 skipped**（98.69s，6 个 skip 均为环境守卫） |
| 审计范围 | 根目录活动报告、`docs/` 直下文档与报告、`.planning/` 审计快照；沿用 `archive/20260816/MANIFEST.md` 归档规则 |
| 过时判定 | 内容与当前代码实现或需求规范存在实质性差异 |
| 报告过期判定 | 生成超过 30 天，或内容不能反映当前项目状态 |
| 审计方法 | 主代理执行版本控制与盘点；3 个只读分析子代理并行执行（需求对照 / 完成度评估 / 技术债），结论见 `project_analysis_20260820.md` |
| 本批次实际移动文件 | 2 个（reports/ 1 + planning/ 1），另在原路径留 1 个重定向短页 |

## 2. 归档文件与版本标签

| 归档路径 | 原路径 | 最后修改提交（版本标签） | 类型 | 归档依据（关键证据） |
|---|---|---|---|---|
| `reports/project_analysis_20260816.md` | `./project_analysis_20260816.md` | `4e5e234`（2026-08-17，git mv 保留历史） | 综合分析报告 v5.0 | 测试基线 L15 记 **1716 passed / 4 skipped**，2026-08-20 实测 **1957 passed / 6 skipped**；报告发布后 `main` 新增 5 个实质性提交（`565e29f` F-03 收口、`3da79c8` K02 锁定、`a0cfb61` TD-LONG-01、`85998f7` Geneformer 契约套件、`cea16f3` GSE90546 解耦），其技术债与缺口清单不再反映现状；被 `project_analysis_20260820.md` 取代为权威报告 |
| `planning/v1.0-MILESTONE-AUDIT.md` | `./.planning/v1.0-MILESTONE-AUDIT.md` | 未追踪（front matter 自记 `audited: 2026-04-05`，距今 >30 天） | 里程碑审计快照 | 快照性质（`status: gaps_found` 为当时状态）；其后 2026-07-03/08-08/08-09/08-16/08-17 多轮系统性复核与修复已改变其记录的全部 gap 状态；活动里程碑记录保留在 `.planning/MILESTONES.md` |

时间戳与版本标签说明：文件名自带日期（YYYYMMDD / ISO 日期）；上表"最后修改
提交"即归档前的 Git 版本标签；tracked 文件经 `git mv` 保留全部历史
（`git log --follow -- archive/20260820/<file>`）。

## 3. 本轮判定为"无需归档"的边界项（避免误删）

| 文件 | 判定 | 依据 |
|---|---|---|
| `docs/` 下 7 个 2026-08-04/08-08 报告短页 | 已是重定向入口，非过时正文 | 正文已在 `archive/20260808/`、`archive/20260816/` 批次归档（见各批次 MANIFEST） |
| `project_repair_report_20260816/17*.md`（3 份） | 保留 | 修复轮审计记录，内容为已完成事实，未被取代 |
| `.planning/MILESTONES.md` | 保留 | 累积里程碑记录（同 CHANGELOG 性质），非快照 |
| `data/README.md` | 保留 | 所列 UniProt/PhosphoSitePlus/dbPTM 数据源与 `data/` 现有布局（uniprot/dbptm/ptmdb/cptac/external）仍一致，无实质性差异 |
| `CLAUDE.md` / `codex_mcp_prompt.md` | 保留 | 前者为工具链活动配置，后者为本轮任务输入快照（均未追踪） |

## 4. 保留与追溯规则（沿用 20260816 批次）

- 活动入口页不因"历史"字样删除：`./project_analysis_20260816.md` 原路径
  保留重定向短页，指向本目录与 `project_analysis_20260820.md`。
- 归档文件保留原路径对应的 Git 版本历史：`git log --follow`。
- 本批次不覆盖、删除或改写既往 `archive/` 批次中的历史正文。
- `docs/CURRENT_STATUS.md` 为活动状态文档，不归档；其过时测试基线行
  （L15，1716/4）由本轮直接更新为 2026-08-20 实测值。
