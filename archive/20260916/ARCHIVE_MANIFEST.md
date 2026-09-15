# 归档清单（2026-09-16）

## 操作记录

- 归档日期：2026-09-16（Asia/Shanghai）。
- 操作范围：仅归档已核实过时的三个快照；使用 `mv` 将文件可恢复地移入归档目录，未删除文件、未修改源码、未提交 Git。
- 目标归档目录：`archive/20260916/`。
- 未计算 SHA、hash 或其他摘要值。

## 归档前 Git 状态

- 工作目录：`/home/scu/PTM2CellNet`。
- 分支状态：`## main`；工作区在归档前已有 tracked/untracked 改动。
- 归档前 HEAD：`fb6edcba0ede4d080392b9763da27d22695376df`。
- `git log -1 --oneline`：`fb6edcb docs: publish 2026-09-14 comprehensive analysis as authoritative report`。
- 对三个源文件执行 `git status --short --untracked-files=all -- <paths>` 的结果均为 `??`（untracked）：
  - `?? project_repair_report_20260915.md`
  - `?? project_repair_report_20260914.md`
  - `?? docs/PTM_activity_代码现状分析与后续执行方案_20260914.md`
- 因此实际状态是“三个目标均未跟踪”，但仓库本身存在 Git history；以上记录以命令输出为准。
- 移动前 `archive/20260916/`、三个归档目标名和本清单均不存在。

## 已移动文件

| 原始相对路径 | 原始绝对路径 | 归档相对路径 | 归档绝对路径 | 原始 mtime（移动后保持） | 移动前大小/行数 | 过时判定 | 依据章节 / 当前产物 |
|---|---|---|---|---|---:|---|---|
| `project_repair_report_20260915.md` | `/home/scu/PTM2CellNet/project_repair_report_20260915.md` | `archive/20260916/project_repair_report_20260915.md` | `/home/scu/PTM2CellNet/archive/20260916/project_repair_report_20260915.md` | `2026-09-15 20:05:31.714867625 +0800` | 18378 bytes / 121 lines | A 内容漂移；C 当前状态不能反映 | 原文 §1.1–§1.3、§2、§3–§6、§9；后续 `project_analysis_20260915.md`、`docs/CURRENT_STATUS.md`、PTM 现行方案/指南及 `outputs/ptm_activity/20260915_d0/` 产物已更新当前事实。 |
| `project_repair_report_20260914.md` | `/home/scu/PTM2CellNet/project_repair_report_20260914.md` | `archive/20260916/project_repair_report_20260914.md` | `/home/scu/PTM2CellNet/archive/20260916/project_repair_report_20260914.md` | `2026-09-14 17:01:17.396364280 +0800` | 18571 bytes / 276 lines | A 内容漂移；C 当前状态不能反映 | 原文 §1–§6，尤其 §3.3、§4、§5、§6；其验证快照已被 2026-09-15 分析/修复报告和后续工程产物 supersede。 |
| `docs/PTM_activity_代码现状分析与后续执行方案_20260914.md` | `/home/scu/PTM2CellNet/docs/PTM_activity_代码现状分析与后续执行方案_20260914.md` | `archive/20260916/PTM_activity_代码现状分析与后续执行方案_20260914.md` | `/home/scu/PTM2CellNet/archive/20260916/PTM_activity_代码现状分析与后续执行方案_20260914.md` | `2026-09-15 11:38:32.496162338 +0800` | 25462 bytes / 373 lines | A 内容漂移；C 当前状态不能反映 | 原文 §1–§6，尤其 §2.3、§3、§4；阶段 A/B 已有实现和 `outputs/ptm_activity/20260915_d0/` 证据，当前权威依据改为现行方案与指南。 |

## 移动后验证

- 三个归档路径均通过 `test -f`。
- 三个原始路径均通过“不存在”检查，未保留同名副本。
- 对三个归档文件执行 `rg -n '^(#|##|###)'` 均找到原始标题/章节。
- `stat` / `wc` 对比移动前后：
  - `project_repair_report_20260915.md`：18378 bytes、121 lines、mtime 不变。
  - `project_repair_report_20260914.md`：18571 bytes、276 lines、mtime 不变。
  - `PTM_activity_代码现状分析与后续执行方案_20260914.md`：25462 bytes、373 lines、mtime 不变。
- 移动后目标相关 Git 状态仅显示三个 `archive/20260916/` 文件为 `??`；原始路径不再显示。

## 明确保留在原位、未归档的资料

以下文件和证据不在本次归档范围内，继续作为当前状态、方案、指南或运行证据使用：

- `docs/CURRENT_STATUS.md`
- `docs/TEST_COVERAGE.md`
- `project_analysis_20260914.md`
- `project_analysis_20260915.md`
- `docs/PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md`
- `docs/guides/ptm_activity_pipeline.md`
- `docs/guides/davf_perturbgen_e2e.md`
- `docs/guides/perturbgen_bridge.md`
- data 证据：`data/manifests/README.md`、`data/manifests/datasets.yaml`
- outputs 证据：
  - `outputs/ptm_activity/20260915_d0/ad_deg_manifest.json`
  - `outputs/ptm_activity/20260915_d0/ad_deg_aggregate.tsv`
  - `outputs/ptm_activity/20260915_d0/ad_deg_donor_level.tsv`
  - `outputs/ptm_activity/20260915_d0/e2e_dryrun_candidate_spec.json`
  - `outputs/ptm_activity/20260915_d0/e2e_dryrun/e2e_report.json`
  - `outputs/ptm_activity/20260915_d0/e2e_dryrun/engineering_context_ex.h5ad`

