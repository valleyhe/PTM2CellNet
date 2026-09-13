# 归档清单 2026-09-13

- **归档执行时间**：2026-09-13
- **归档时基线提交**：`6a194a1`（main，`Harden DAVF-PerturbGen gate binding and dual-path evaluation contracts.`）
- **归档方式**：`git mv`（保留完整提交历史，可用 `git log --follow` 追溯）
- **归档判定标准**：
  1. 过时 DOCUMENT：内容与当前代码或现行需求（`docs/CURRENT_STATUS.md`、`lessons.md` 最新条目、测试/CI）存在实质性差异；
  2. 过期 REPORT：生成于 2026-08-14 之前，或内容已无法反映当前主线（DAVF 三方 gate 绑定、route-aware 双路径、tokenise/h5ad lineage）。

未归档：`README.md`、`AGENTS.md`、`CLAUDE.md`、`lessons.md`、`docs/CURRENT_STATUS.md`、仍匹配代码的安装/数据/训练/部署/真实资产/E2E/bridge/KO-KD 指南、`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`（现行需求基线）、`docs/guides/gse_normal_disease_davf_plan_20260904.md`（仍是 GSE214695 操作文档）、源码与测试、`project_repair_report_20260913.md`（本轮修复证据）。

## 归档文件

### reports/

| 文件 | 归档前路径 | 归档前 blob（`6a194a1`） | 最后修改 | 类别 | 说明 |
|---|---|---|---|---|---|
| `project_analysis_20260910.md` | 仓库根目录 | `75864fa872ae` | 2026-09-10 17:36 +0800 | 过期 REPORT + 过时 DOCUMENT | 被 `project_analysis_20260913.md` 取代。§4/§9 对 G-3 完整闭环、唯一外部阻塞的表述已被 2026-09-13 源码证伪。 |
| `project_repair_report_20260910.md` | 仓库根目录 | `bd0b77603bf9` | 2026-09-10 12:34 +0800 | 过期 REPORT | 9/10 点时修复日志，已被 `project_repair_report_20260913.md` 覆盖。 |
| `davf_perturbgen_e2e_execution_report_20260902.md` | `docs/guides/` | `03e84e654360` | 2026-09-02 22:00 +0800 | 过期 REPORT | 9/02 首次 E2E 接线点时报告；不能反映 9/13 YAML identity binding 与 route-aware 模式。 |

### guides/

| 文件 | 归档前路径 | 归档前 blob | 最后修改 | 类别 | 说明 |
|---|---|---|---|---|---|
| `davf_perturbgen_retraining_plan_20260902.md` | `docs/guides/` | `2bda1cb8fed0` | 2026-09-02 21:52 +0800 | 过时 DOCUMENT | 文内已声明被 `davf_ko_kd_training.md` 取代；仍把 Norman scVI 写成唯一坐标系。 |
| `davf_cell_baseline_training_plan_20260904.md` | `docs/guides/` | `3c1af7165257` | 2026-09-05 21:28 +0800 | 过期 REPORT | 点时训练战役；可复现命令已在 `davf_ko_kd_training.md`。 |

### stubs/

根目录 2026-08 分析/修复入口短页（正文已在更早批次 `git mv` 到 `archive/20260820`–`archive/20260827`）。短页仍把过时报告称为“当前权威”，会误导。本次整链移入 `stubs/`。

| 文件 | 归档前 blob | 说明 |
|---|---|---|
| `project_analysis_20260816.md` | `e7d43b639263` | 指向已归档 v5.0 正文 |
| `project_analysis_20260820.md` | `66c9ba680b6e` | 指向已归档 0820 正文 |
| `project_analysis_20260821.md` | `2c2f338292ea` | 指向已归档 0821 正文 |
| `project_analysis_20260822.md` | `f095d9a0aff5` | 指向已归档 0822 正文 |
| `project_analysis_20260823.md` | `1e264bff77b9` | 指向已归档 0823 正文 |
| `project_analysis_20260824.md` | `210cabf4a324` | 曾指向 20260910 权威报告 |
| `project_repair_report_20260816.md` | `d5f9a2d84557` | 修复 stub |
| `project_repair_report_20260817.md` | `69d48939af7f` | 修复 stub |
| `project_repair_report_20260817_k02_f08_f05.md` | `d4d23bbfbee0` | 修复 stub |
| `project_repair_report_20260822.md` | `0c09f57d9875` | 修复 stub |
| `project_repair_report_20260823.md` | `1f31c6e50e99` | 修复 stub |
| `project_repair_report_20260824.md` | `a62b3b82dfc2` | 曾指向 20260910 |

### 本轮追加：无 Git 历史文件

以下文件在归档前均被 `.gitignore:6` 忽略，未进入 Git index，因此没有可保留的
Git 历史或 blob；本轮仅移动文件，未计算文件哈希。归档前 HEAD：
`f3374d223a11749edd76a3ee6ad83e9b84b8ebdf`。

| 文件 | 原路径 | 目标路径 | Git 历史 | 原因 |
|---|---|---|---|---|
| `coding-guide.txt` | `coding-guide.txt` | `archive/20260913/guides/coding-guide.txt` | 无（ignored/untracked） | 2026-03 通用旧流程，无活动入口引用；与当前项目规范不再同步。 |
| `codex_mcp_prompt.md` | `codex_mcp_prompt.md` | `archive/20260913/guides/codex_mcp_prompt.md` | 无（ignored/untracked） | 2026-08-09 旧任务模板，无仓内引用；保留作历史追溯。 |
| `findings.md` | `findings.md` | `archive/20260913/planning/findings.md` | 无（ignored/untracked） | 2026-09-01 历史审计追踪，仍指向已归档的旧权威分析。 |
| `progress.md` | `progress.md` | `archive/20260913/planning/progress.md` | 无（ignored/untracked） | 2026-09-10 历史收口记录，仍指向已归档的旧权威分析。 |

## 同步修改

- 权威分析改为仓库根目录 `project_analysis_20260913.md`。
- `docs/CURRENT_STATUS.md`、`AGENTS.md`、`docs/TEST_COVERAGE.md`、`docs/guides/perturbgen_bridge.md`、`API_DOCUMENTATION.md`、`docs/PTM2CellNet_*.md` 指针已刷新。
- `perturbgen_bridge.md` 的“详细执行方案”改为 `davf_ko_kd_training.md`。
