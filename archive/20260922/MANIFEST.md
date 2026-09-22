# 归档清单 20260922

- **归档操作日期**：2026-09-22
- **归档操作基线**：`main` @ `25c0ab3`（2026-09-21 修复轮三语义提交之后）
- **操作方式**：tracked 文件 `git mv`/`git rm`（保留全部 git 历史）；untracked 笔记 `mv` 后首次纳入版本控制
- **新权威报告**：仓库根 `project_analysis_20260922.md`

## A 档：被取代的分析/修复报告（git mv，历史保留）

| 文件 | 最后内容提交 | 归档原因 |
|---|---|---|
| `project_analysis_20260921.md` | `25c0ab3`（2026-09-22 补交） | 权威报告被 `project_analysis_20260922.md` 取代（报告链 0921→0922） |
| `project_repair_report_20260921.md` | `25c0ab3`（2026-09-22 补交） | 同上；修复内容已由代码、测试与 CURRENT_STATUS 引用 |
| `project_analysis_20260920.md` | `25c0ab3`（含 0921 续审附注） | 已被 0921 报告取代，0921 又被 0922 取代 |
| `project_analysis_20260915.md` | 2026-09-16 | 滞留根目录的历史报告（后被 0917/0920/0921/0922 逐代取代） |
| `project_analysis_20260914.md` | 2026-09-16 | 滞留根目录的历史报告；其引用的 0913 正本已在 `archive/20260914/` |
| `project_repair_report_20260913.md` | 2026-09-14 | `docs/CURRENT_STATUS.md` 早已标注"已归档"但文件滞留根目录，本次补齐归档事实 |

## B 档：无引用的根目录过时笔记（原 untracked，首次入库）

| 文件 | 文件时间 | 归档原因 |
|---|---|---|
| `Codex_prompt_GSE147528_GSE157827_GSE174367_GSE188545.md` | 2026-09-14 | 一次性外部执行 prompt，已消费；全仓无引用 |
| `IBD_dataset.md` | 2026-09-13 | 非 AD 主线的早期数据探索笔记，自带"非正式 cohort"边界；全仓无引用 |
| `PerturbGen_分析与预训练模型使用指南.md` | 2026-09-13 | 对话整理稿；正式指南为 `docs/guides/perturbgen_bridge.md`，本文全仓无引用 |

## 特殊处理

| 文件 | 处理 | 说明 |
|---|---|---|
| `project_analysis_20260917.md` | `git rm`（根目录删除） | 与 `archive/20260920/project_analysis_20260917.md` 逐字节相同（sha256 `31a8ca97…` 双向核对），根目录为冗余副本；正本保留在 20260920 批次 |

## 保留在根目录的活文档

`README.md`、`CONTRIBUTING.md`、`CHANGELOG.md`、`API_DOCUMENTATION.md`、`lessons.md`、
`AGENTS.md`/`CLAUDE.md`（本仓协作配置）、`project_analysis_20260922.md`（新权威报告）。

## 追加归档（2026-09-22 环境隔离轮）

| 文件 | 处理 | 归档原因 |
|---|---|---|
| `project_analysis_20260922.md`（旧本，修复轮依据） | `git mv` | 被环境隔离轮新写的同名权威报告取代（报告链 0922a→0922b） |
| `project_repair_report_20260922.md` | `mv`（原 untracked） | 修复轮语义已由代码、测试、lessons L-2026-0922-02 与新报告引用；本轮已消费其 §4 未解决项清单 |

## 追加归档（2026-09-22 综合处理轮第二轮）

| 文件 | 处理 | 归档原因 |
|---|---|---|
| `project_analysis_20260922_env_isolation.md`（原名根目录 `project_analysis_20260922.md`，环境隔离轮权威报告 0922b） | `mv`（当时未入库，随本批次首次入库） | 被本轮综合处理报告（0922c）取代为根目录权威报告；lessons L-2026-0922-03/04 引用的"本轮报告"即此本，改名映射在此登记 |
