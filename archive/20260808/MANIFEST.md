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

## 保留在原路径的治理/当前入口

`AGENTS.md`、`CLAUDE.md`、`coding-guide.txt` 和 `docs/CURRENT_STATUS.md` 未移动：前两类是
工作区治理规则，后者是当前状态入口。本批次为被移动的活跃文件创建了同路径入口页，避免
外部链接直接失效；入口页明确指向归档原文。
