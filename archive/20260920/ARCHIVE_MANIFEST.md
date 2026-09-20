# 2026-09-20 归档清单

## 归档元数据

| 项目 | 值 |
|---|---|
| 归档日期 | 2026-09-20 |
| 归档方式 | `cp --preserve=timestamps`，仅复制，不删除原文件 |
| 原文件是否保留 | 是 |
| Git 历史是否修改 | 否 |
| 生成报告 | `project_analysis_20260920.md` |

## 文件映射

| 原路径 | 归档副本 | 原始大小 | 原始 mtime（Asia/Shanghai） | 最近 Git 提交 | 归档理由与证据 |
|---|---|---:|---|---|---|
| `project_analysis_20260917.md` | `archive/20260920/project_analysis_20260917.md` | 27,928 bytes | 2026-09-18 00:26:26.852377883 +0800 | `780d683df73b4a19fa5de66ab33f1c200e4969db`（2026-09-18，`docs: finalize 20260917 report checkpoints`） | 新报告取代旧报告；旧报告不能反映当前工作区 24 个已跟踪修改、12 个未跟踪文件、2026-09-18 AD/KO-only 决策以及本次审计结果。 |
| `docs/guides/real_assets_acceptance.md` | `archive/20260920/real_assets_acceptance.md` | 8,509 bytes | 2026-09-13 17:59:35.054679084 +0800 | `520e99249e41ba837965ccd4cbb33f0eac1e2cdf`（2026-09-13，`feat: align PTM-DAVF-PerturbGen evidence pipeline`） | 原文第 4 节第 53–61 行声称每个候选重复 prepare 且无复用 CLI；当前 `docs/guides/davf_perturbgen_e2e.md:170-179` 与 `src/integration/perturbgen/orchestrator.py` 已有共享 prepare 方案，属于实质性过时。 |

## 未归档材料

其他历史报告和数据记录未因“日期较早”自动归档：`project_analysis_20260914.md`、
`project_analysis_20260915.md`、`project_repair_report_20260913.md` 仍被当前状态或
历史追溯引用，且本次没有足够的新增证据证明其作为历史证据不可用；现有 `archive/`
下材料也不重复复制。
