# 归档清单与来源记录

归档日期：2026-08-08
仓库基线：`main` @ `6c64117537b6e4921b9bdd1ee66a32641336d628`（归档前）
归档方式：Git 跟踪文件先使用 `git mv` 移入归档，再在原路径创建当前入口页；忽略文件使用显式移动。
本次提交分别记录归档原件和入口页，原始提交、版本、路径和 SHA-256 由本清单保留；忽略文件无 Git
历史，保留原始修改时间和 SHA-256。

| 原始路径 | 归档路径 | 原始版本/日期 | Git 最后记录或文件记录 | SHA-256 | 归档原因 |
|---|---|---|---|---|---|
| `API_DOCUMENTATION.md` | `archive/20260808/docs/API_DOCUMENTATION.md` | API v1；原文无独立发布日期 | `720df38`，2026-07-02；`chore: commit all code changes across src, scripts, tests, and configs` | `c6d2815ea8c99334ad85fed972b3c8318867d711d68c1c108f793dd139d0409a` | 端点快照缺少当前 `/live`、`/ready`、`/metrics`、`/initialize` 等契约，环境变量和探针语义已演进。 |
| `docs/PTM2CellNet_项目文档.md` | `archive/20260808/docs/PTM2CellNet_项目文档.md` | v2.0；2026-07-11 | `40d87aa`，2026-08-04；`docs: archive outdated reports and review snapshots, refresh doc pointers` | `19fd1250bcaf708e2df4cee491ae953e1380f3b5f4b2119022102a1b85d34dc4` | 正文仍写 93+ 文件、1531+ 测试和 10 个 `mypy: ignore-errors`，与当前 124 个源文件、最新实测和类型标记不符。 |
| `docs/PTM2CellNet_技术文档.md` | `archive/20260808/docs/PTM2CellNet_技术文档.md` | v1.0；2026-07-06 | `40d87aa`，2026-08-04；同上 | `d02536c52c9f24c24d281b9fd91245c7e662e100249237df79cc220d0685a456` | 技术/测试计数和部分配置说明属于旧快照，已由代码、指南和本次分析报告取代。 |
| `docs/PTM2CellNet_文件说明.md` | `archive/20260808/docs/PTM2CellNet_文件说明.md` | v1.0；2026-07-06 | `40d87aa`，2026-08-04；同上 | `e3e985398e488d29a4b2d92041094a105fdbf68c5962a906621b4272e08ca15d` | 输入输出契约缺少本轮 DAVF/API 字段演进，且引用旧 checkpoint/测试基线。 |
| `docs/问题修复与系统性复核报告_2026-08-04.md` | `archive/20260808/reports/问题修复与系统性复核报告_2026-08-04.md` | 报告日期 2026-08-04；目标 `main @ f4e6d40` | `9f10e31`，2026-08-04；`docs: add fix & systematic review report (3 iterations, P0-P2 closed)` | `3fb5d75b1a949d75a05ed077c425a202cc23660bea0cc684ecbe2474fcfff200` | 当前工作树已在该基线之后增加 DAVF 兼容、多任务和 PTM 模块开关改动，本报告不再是当前基线。 |
| `docs/E2E训练和推理能力评估报告_2026-08-04_v2.md` | `archive/20260808/reports/E2E训练和推理能力评估报告_2026-08-04_v2.md` | 报告日期 2026-08-04；评估 `main @ 9f10e31` | `6c64117`，2026-08-04；`docs: archive pre-fix E2E assessment, refresh status docs, add post-fix v2 report` | `0db80c690ce8ef4de3b6eabbd3cbdd8e55576d922d3bdc82e3422f63da081d57` | 已被本次 2026-08-08 复核报告取代；其中的测试数量、技术债和分支指针只代表历史快照。 |
| `蛋白质PTM与细胞状态预测项目需求.docx` | `archive/20260808/requirements/蛋白质PTM与细胞状态预测项目需求.docx` | 文档属性创建/修改 2026-02-24；摘要要求 Python 3.9.16、Torch 1.13.0 | 未纳入 Git；原始 mtime `2026-02-24 16:18:14 +0800` | `1255cc0a2c29c6e48555e1392cd6daa69f57629ec4df67205f4a379c849a9e45` | 需求基线版本和运行时版本已过期；内容仍作为本报告的需求追溯证据保留。 |
| `.coverage` | `archive/20260808/detection_artifacts/.coverage` | 无版本字段；旧 coverage 数据 | 未纳入 Git；原始 mtime `2026-04-11 21:09:58 +0800` | `e75dc03c8964da50d12133e7381d1ff0098046f4446664a1601f9b93991a98ab` | 旧测试运行生成的机器相关检测产物，不代表当前测试覆盖率。 |
| `coverage.json` | `archive/20260808/detection_artifacts/coverage.json` | 无版本字段；旧 coverage 数据 | 未纳入 Git；原始 mtime `2026-04-04 22:40:12 +0800` | `19c37176ae1f40c41d62d1ea4337504720255d7badb9505d0fc77fbb22431c64` | 旧测试运行生成的机器相关检测报告，不代表当前测试覆盖率。 |
| `project_analysis_20260808.md` | `archive/20260808/reports/project_analysis_20260808_pre_cross_scale.md` | 报告日期 2026-08-08；跨尺度实现前基线 | `7d50d6c`，2026-08-08；`audit: archive stale docs and document project gaps` | `ad1fa47ab8d9fdb9c56b1333633f0590b03e1502d335766f7f9ca7cb0beb3f50` | 报告仍把 R-01/R-03/R-07 记为未实现，已被跨尺度组件、PMADS 基线和当前 E2E 复核取代。 |
| `docs/E2E训练与推理代码修复报告_2026-08-08.md` | `archive/20260808/reports/E2E训练与推理代码修复报告_2026-08-08.md` | 报告日期 2026-08-08；同日首版 | 未纳入 Git；原始 mtime `2026-08-08 19:45:57 +0800` | `2e0a8c3304dc10a1808b2031160db50c3e064f0448482cb6cb09961d28ef730e` | 同日 v2 报告已补齐三轮修复结果；首版仍把跨尺度训练、推理和 manifest 门禁记为未实现。 |
| `docs/td01_td02_technical_summary_20260808.md` | `archive/20260808/reports/td01_td02_technical_summary_20260808_pre_final.md` | 报告日期 2026-08-08；最终回归前技术快照 | 未纳入 Git；原始 mtime `2026-08-08 18:52:35 +0800` | `f20bbd8ccc2c8f06641ccc5fbee7022ffe9d0043ad50dc51be9fb07e72fa9515` | “最终回归”数字为 1794 collected / 1781 passed，早于续训边界和最佳 artifact 修复；当前数字由 E2E 现状报告与 CURRENT_STATUS 提供。 |
| `docs/guides/data_integration.md`（旧快照副本） | `archive/20260808/docs/data_integration_guide_pre_manifest.md` | 未标版本；记录 24,195 条旧整合规模 | `720df38`，2026-07-02；`chore: commit all code changes across src, scripts, tests, and configs` | `62baca89a5c76f09285a0cf04f9be7fe2097073098a678f47356db67a3cfbc13` | 数据规模和来源说明已过时；活跃指南改为引用版本化 manifest，并保留旧快照供追溯。 |
| `data/raw/DOWNLOAD_REPORT.md` | `archive/20260808/detection_artifacts/data_reports/DOWNLOAD_REPORT.md` | 报告日期 2026-04-11 | 未纳入 Git；原始 mtime `2026-04-11 17:08:12 +0800` | `1497a4e8e07356142014a56a2becb6dec8c60b17e3a0850abd654664ac99176b` | 下载状态、数量和源可用性是旧运行快照，不能代表当前数据资产状态。 |
| `data/processed/INTEGRATION_REPORT.md` | `archive/20260808/detection_artifacts/data_reports/INTEGRATION_REPORT.md` | v3；报告日期 2026-04-11 | 未纳入 Git；原始 mtime `2026-04-11 19:21:20 +0800` | `8175d2bad184a9e0572efcbb827d0c4db77060da38fc92fbd3aee4018ae4acc2` | 旧整合命令、数据规模和质量结论已被版本化 manifest 与当前工作流取代。 |
| `data/processed/VERIFICATION_REPORT.md` | `archive/20260808/detection_artifacts/data_reports/VERIFICATION_REPORT.md` | 报告日期 2026-04-11 | 未纳入 Git；原始 mtime `2026-04-11 20:28:11 +0800` | `8a44e160ac2de76099910f39eb84ed4a9ed884cfc012f779735db907c01a21a0` | 验证结论只对应 2026-04-11 数据/代码快照，不再是当前质量验收证据。 |

## 保留在原路径的治理/当前入口

`AGENTS.md`、`CLAUDE.md`、`coding-guide.txt` 和 `docs/CURRENT_STATUS.md` 未移动：前两类是
工作区治理规则，后者是当前状态入口。本批次为被移动的活跃文件创建了同路径入口页，避免
外部链接直接失效；入口页明确指向归档原文。

## 2026-08-08 覆盖率与跨尺度复核增量

本增量以 `7d50d6c` 为跟踪文件基线，归档跨尺度实现前综合分析、数据接入指南旧快照和三份
未纳入 Git 的 2026-04-11 数据检测报告。当前权威状态由
[`docs/E2E训练与推理现状分析_2026-08-08.md`](../../docs/E2E训练与推理现状分析_2026-08-08.md)
与 [`docs/CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md) 提供；本表中的历史文件不得用于
发布、测试覆盖率或科学效果声明。
