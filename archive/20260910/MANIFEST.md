# 归档清单 2026-09-10

- **归档执行时间**：2026-09-10
- **归档时基线提交**：`25edec2`（main，`chore: update dataset manifests, analysis deps and status docs`）
- **归档方式**：`git mv`（保留完整提交历史，可用 `git log --follow` 追溯）
- **归档判定标准**：
  1. 报告生成时间超过 30 天（均产生于 2026-08-04～2026-08-08）；
  2. 报告结论已被 `project_analysis_20260901.md` 及本轮 `project_analysis_20260910.md` 重新核验并取代，内容无法反映当前 DAVF→PerturbGen 主线实现状态；
  3. 属点时检测/修复报告（test report / repair report 性质），非现行需求或指南。

## 归档文件（reports/）

| 文件 | 归档前路径 | 归档前 blob（HEAD） | 说明 |
|---|---|---|---|
| E2E训练与推理代码修复报告_2026-08-08.md | docs/ | `4e1c775cf3b2` | TD01/TD02 修复点时报告 |
| E2E训练与推理代码修复报告_2026-08-08_v2.md | docs/ | `0f56171bdfd4` | 同上第二版 |
| E2E训练与推理现状分析_2026-08-08.md | docs/ | `0c5d22239edb` | E2E 能力现状分析 |
| E2E训练和推理能力评估报告_2026-08-04_v2.md | docs/ | `f3cef0df0780` | E2E 能力评估 v2 |
| r01_r03_systematic_repair_report_20260808.md | docs/ | `314d4510c5db` | R-01～R-03 系统性修复报告 |
| td01_td02_technical_summary_20260808.md | docs/ | `c162c5fdc815` | TD01/TD02 技术总结 |
| 问题修复与系统性复核报告_2026-08-04.md | docs/ | `b6e702d58820` | 问题修复与复核报告 |

## 同步修改

- `docs/index.rst`：Status and Reports toctree 移除上述 7 个条目，避免 Sphinx 死链。

## 后续追加

- 2026-09-10（同批收尾提交）：`project_analysis_20260901.md`（原根目录，归档前 blob `e20087a0fed7`）追加归档至本目录。原因：其"U-03 gate 未接线 / TD-N-48 / TD-N-57"等结论已被 `orchestrator.py` + `run_davf_perturbgen_e2e.py`（提交 `d3a2f55`）解决，测试基线由 2357 passed 更新为 2475 passed；被 `project_analysis_20260910.md` 取代。相关活动文档（CURRENT_STATUS、TEST_COVERAGE、perturbgen_bridge、API_DOCUMENTATION、根目录两个 stub）的引用链接已同步更新。
