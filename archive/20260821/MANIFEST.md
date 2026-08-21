# 归档批次 20260821

**归档日期**：2026-08-21
**执行轮次**：项目代码与文档综合处理及技术分析（第三轮周期性复核）
**取代者**：仓库根目录 [`project_analysis_20260821.md`](../../project_analysis_20260821.md)

## 1. 归档清单

| 归档文件 | 原路径 | 版本标签 | 归档依据 |
|---|---|---|---|
| `reports/project_analysis_20260820.md` | 仓库根 | `919f4ed`（2026-08-21 00:25，`git mv` 保留历史） | 其 mypy "25 errors / 12 files" 结论被 2026-08-21 实测 **5 errors / 2 files** 推翻（高估）；pip-audit 首审发现 64 个漏洞条目未收录；测试基线数字虽复现（1957/6、107/1）但技术债与缺口清单已被新报告取代 |

## 2. 判定标准

沿用 `archive/20260816/MANIFEST.md` 与 `archive/20260820/MANIFEST.md` 规则：

- **过时** = 内容与当前代码实现或需求规范存在实质性差异。
- **过期** = 生成时间超过 30 天，或报告内容已无法反映当前项目状态。

## 3. 判定为无需归档的边界项

| 文件/类别 | 理由 |
|---|---|
| `docs/` 下 8 份入口短页（E2E×3、r01_r03、td01_td02、问题修复、技术文档、项目文档、文件说明） | 正文已在 `archive/20260808/`、`archive/20260816/`，现存仅重定向入口；其中 3 份短页"当前权威"指向 `project_analysis_20260816.md` 的过时链接由本轮刷新为指向新报告（属内容修正而非归档） |
| `.planning/{REQUIREMENTS,ROADMAP,PROJECT,STATE,MILESTONES}.md` | 正式需求规范与 roadmap（v2.2 全 Complete），是任务 1 对照基准，非过时文档；最后更新 2026-08-08 与需求状态一致 |
| `docs/CURRENT_STATUS.md` | 活动状态文档，本轮按实测刷新（mypy 实况、pip-audit 结果、权威报告指向） |
| `docs/TEST_COVERAGE.md` | 覆盖率治理记录（75.27% 基线 + ratchet 策略），仍为现行门禁依据 |
| `docs/guides/*.md`（6 份） | 安装/数据/训练/部署/真实资产指南与当前 CLI/API 一致（跨尺度端点已覆盖）；`/predict/variant` 缺口登记为 TD-N-03 继续跟踪 |
| `project_analysis_20260816.md`（根目录现存文件） | 已是重定向短页（正文在 `archive/20260820/reports/`），保留入口职责 |
| 根目录散落会话产物（`omp-session-*.html`、`x.pt`、`*.zip` 等） | 未入库（`.gitignore` 白名单外），属工作区卫生问题，登记 TD-N-07 由清理动作处理，不适用文档归档流程 |

## 4. 版本信息

- 归档前本地 `main`：`919f4ed`（docs: record batch commit hashes in 20260820 report and MANIFEST）
- 远程 `origin/main`：`c76fff8`（fetch 后无新提交，本地领先 26 / 落后 0）
- 本批次归档以 `git mv` 执行，保留完整重命名历史；原路径留重定向短页（沿用既往惯例）。
