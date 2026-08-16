# 2026-08-16 文档与报告归档清单

## 1. 清单元数据

| 字段 | 值 |
|---|---|
| 审计日期 | 2026-08-16 |
| 合并前 `main`（= 远程基线） | `c76fff881f268f9bd0b39d68db8ba547115ec4cf`（与 `origin/main` 一致，fetch 已确认） |
| 本轮代码 `main` 提交 | `32ded55`（feat(data): register Norman/Adamson import pipeline and OmniPath kinase-substrate snapshot） |
| 本轮归档同步提交 | 见 `git log -1 -- archive/20260816/`（本文件所在提交） |
| 验证状态 | `python -m compileall src scripts` 通过；全量单测 **1716 passed / 4 skipped**（79.4s） |
| 审计范围 | 根目录活动文档、`docs/` 直下文档与报告、入口短页、归档惯例（沿用 `archive/20260809/MANIFEST.md` 规则） |
| 过时判定 | 与当前代码/需求存在实质差异 |
| 报告过期判定 | 生成超过 30 天，或内容不能反映当前状态 |
| 审计方法 | 4 个只读分析子代理并行执行（需求对照 / 完成度评估 / 技术债 / 文档台账），判定均附 file:line 证据 |
| 本批次实际移动文件 | 8 个（reports/ 3 + docs/ 3 + detection_artifacts/ 2），另在原路径留 3 个重定向短页 |

## 2. 归档文件与版本标签

| 归档路径 | 原路径 | 最后修改提交（版本标签） | 类型 | 归档依据（关键证据） |
|---|---|---|---|---|
| `reports/project_analysis_20260809.md` | `./project_analysis_20260809.md` | `c76fff8`（2026-08-09，git mv 保留历史） | 综合审计报告 | L34"9 个数据集只有 PMADS 本地文件"、L196"kinase_substrate loader 仍为 null"、L28"1920 passed/13 skipped"——5 个数据集已翻转 implemented、测试基线已变为 1716/4；被 `project_analysis_20260816.md` 取代为权威报告 |
| `reports/project_repair_report_20260809.md` | `./project_repair_report_20260809.md` | 未追踪（本批次归档后首次入库，mtime 2026-08-09） | 修复执行报告 | L20"1918 passed/13 skipped"、L236 TD-C01"8 个真实 cross-scale 输入无授权快照"——今日 5 项已登记；全文以 project_analysis_20260809.md 为依据，随其一并被取代 |
| `docs/E2E训练与推理现状分析_2026-08-08.md` | `docs/` 同名 | `366e691`（2026-08-09，git mv 保留历史） | E2E 审计报告 | L38 T-01"cross_scale_training 缺少 8 个 required 输入"（现仅 3 个 controlled）；§4.1 P0-01 图类快照缺失断言失效；L64"1897 passed"过时；文件头部已自声明历史快照 |
| `docs/E2E训练与推理代码修复报告_2026-08-08_v2.md` | `docs/` 同名 | `d0c78a1`（2026-08-09，git mv 保留历史） | 修复报告 v2 | L161 将 Norman/Adamson、kinase-substrate、STRING、BioPlex、RegNetwork 列为"数据所有者职责"——`32ded55` 已登记其中 5 项 implemented；L128-129 测试计数过时 |
| `docs/r01_r03_systematic_repair_report_20260808.md` | `docs/` 同名 | `366e691`（2026-08-09，git mv 保留历史） | 组件修复报告 | L25 R-03"真实 STRING/BioPlex/激酶-底物图未提供"——三类图快照（OmniPath enzsub 41,506 条等）已登记；L77/L93"1897 passed/13 skipped"过时 |
| `detection_artifacts/coverage.xml` | `./coverage.xml` | 未追踪（生成于 2026-08-09 18:35，SHA-256 `1876bd77…b6fe96`） | 覆盖率检测报告（Cobertura XML） | 生成于 `32ded55` 与 2026-08-16 修复轮之前，不能反映当前代码；已被 2026-08-16 13:50 的根目录 `.coverage` 取代；沿用 `archive/20260808/detection_artifacts/` 旧快照归档惯例 |
| `detection_artifacts/coverage.json` | `./coverage.json` | 未追踪（生成于 2026-08-08 22:30，SHA-256 `30ad399e…cd4736f`） | 覆盖率检测报告（JSON） | 同上（早于 coverage.xml 一天的更旧快照）；归档后原路径删除，不留重定向页（检测产物非链接入口） |
| `reports/project_analysis_20260816_v3_repair_iterations.md` | `./project_analysis_20260816.md`（v3.0 中间版，从未提交） | 未追踪（2026-08-16 14:54 前后定稿，本轮 21:05 归档） | 修复迭代中间版报告 | v3.0 为"代码修复迭代任务"产物（N20/TD-M08/N03-N04 三轮迭代），结构为修复报告而非综合分析；其权威内容已由 `project_repair_report_20260816.md`（根目录，当日）承载，综合分析结构由 v4.0（同日）恢复，本中间版归档留痕 |

时间戳与版本标签说明：文件名自带日期后缀（YYYYMMDD / YYYY-MM-DD）；上表"最后
修改提交"即归档前的 Git 版本标签；`git mv` 保留全部历史（`git log --follow`）。

## 3. 保留与追溯规则（沿用 20260809 批次）

- 活动入口页不因"历史"字样删除：8 个纯重定向短页保留原路径；本次归档的 3 个
  docs 报告在原路径新留重定向短页，指向本目录与当前权威报告。
- 归档文件保留原路径对应的 Git 版本历史：`git log --follow -- archive/20260816/<file>`。
- 本批次不覆盖、删除或改写 `archive/20260808/`、`archive/20260809/` 与
  `docs/archive/` 中的历史正文。
- 当前综合报告为仓库根目录的 `project_analysis_20260816.md`；本轮代码已直接
  提交到 `main`（提交前已确认 `main` = `origin/main` = `c76fff8`，无合并冲突）。

## 4. 保留清单要点（30 个，完整台账见 project_analysis_20260816.md 第 8 章）

| 类别 | 文件 | 处理 |
|---|---|---|
| 活动状态页 | `docs/CURRENT_STATUS.md` | 保留原路径，本批次已刷新（权威指向 20260816 报告、测试基线 1716/4、数据集差距 3 项 controlled） |
| 覆盖率治理 | `docs/TEST_COVERAGE.md` | 保留，基线数字待下一轮刷新（登记为报告 F-14/N06 跟进项） |
| 活日志 | `CHANGELOG.md` | 保留，缺 8 月条目（登记为跟进项） |
| 入口短页 ×8 | `API_DOCUMENTATION.md`、`docs/PTM2CellNet_{项目,技术,文件说明}文档.md`、`docs/td01_td02_*`、`docs/E2E*2026-08-04*`、`docs/E2E*2026-08-08.md`（首版） | 保留原路径（纯重定向，避免破坏链接） |
| 有效指南/手册 | `docs/DATA_UPDATE_WORKFLOW.md`、`docs/guides/*` ×6、`README.md`、`CONTRIBUTING.md` | 保留（命令/路径经核实有效）；`guides/quickstart.md` 的 `/predict` 示例缺 `/api/v1` 前缀，登记为跟进修正 |
| 通用规范 | `CLAUDE.md`、`coding-guide.txt`、`codex_mcp_prompt.md` | 保留（与项目状态无耦合） |
| 其他 | `AGENTS.md`（L912 数据源表建议更新 PhosphoSitePress→OmniPath 表述）、`paper/first_draft_zh.md` | 保留 |
| 构建产物 | `docs/_build/`（gitignored） | 不归档；内嵌 pre-32ded55 datasets.yaml 快照，建议重建 |
