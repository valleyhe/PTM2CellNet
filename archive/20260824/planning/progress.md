# 执行进度

## 2026-08-16
- 已确认活动目标为：依据 `project_analysis_20260816.md` 完成三轮代码修复迭代、E2E 系统复核和最终报告。
- 已定位最新分析文档；读取 §3.3、§7.2–§7.4、§8，选定三个代码侧迭代主题：N04 公共 ensemble 聚合、GSE90546 产物 manifest、N07 multitask 热路径。
- 基线测试：ensemble 单元+cross-scale 集成 **11 passed / 8 warnings**；GSE/manifest 单元+集成 **32 passed**；multitask 单元+cross-scale 集成 **10 passed / 8 warnings**。
- 基线首条命令误引用不存在的 `tests/unit/models/test_ensemble.py`（0 collected，exit 4），随后改用实际测试路径通过；错误已记录并未重复。
- 下一步：实施迭代一 N04 公共聚合复用。

## 测试记录
| 轮次 | 单元测试 | 集成测试 | 结果 | 未解决问题 |
|---|---|---|---|---|
| 基线 | ensemble 10；multitask 9；GSE 20 | cross-scale 2；GSE/manifest 12 | 通过；另有 8 warnings | N04 双实现、GSE manifest catalog、multitask iterrows |
| 迭代一 | 待运行 | 待运行 | 待定 | 待定 |
| 迭代二 | 待运行 | 待运行 | 待定 | 待定 |
| 迭代三 | 待运行 | 待运行 | 待定 | 待定 |

## 2026-08-22 DAVF × PerturbGen

- M0 核查：源码 `a9a9375` 与 Tesla P40 可用；encoder 权重、独立 Python 3.11 环境、合规 AnnData 缺失，Gate-0 明确 BLOCKED。
- 已新增 M1–M3 主进程代码、严格 embedding asset 协议、六阶段 runner、双路径 CLI 与 19 个相关测试文件。
- 独立 Review 发现并修复上游 CLI 参数名、目录 fingerprint、脏输出 resume、官方 h5ad 字段、flat dotted key、asset stage 弱校验和执行层单路径等问题。
- 聚焦验证：ruff 全绿；`146 passed, 8 warnings`。
- 全仓验证：收集 2289 项，600 秒超时（exit 124，运行约 3%），不计为通过。
- 补齐上游真实产物契约：tokenise 六个具名产物按公式绑定，checkpoint/h5ad 严格唯一发现，manifest artifact 支持下游 argv/YAML/fingerprint 引用。
- 新增 M5 benchmark 和 real-assets smoke 入口；聚焦终验 `86 passed, 1 skipped`，skip 为未提供真实资产时的预期行为。
- 补齐 M3 版本化候选评估 CLI：严格绑定成功 stage manifest 与 h5ad SHA-256，按 donor backed 切片聚合 `pred_counts`/`X`，输出 BH-FDR 双路径报告。
- 补齐 M5 独立 GPU workflow 与正式 evidence 硬门；benchmark 增加进程树峰值显存、多轮产物隔离和资源 P50/P95。最新 PerturbGen 聚焦回归 `111 passed, 1 skipped`。
- 第三轮 Gate-0 外部状态复核：本地无 encoder 权重和独立 Python 3.11；30 个 scPerturb h5ad 中 4 个截断，26 个可读文件仍无满足 normal/disease 配对与 ≥3 显式 donor 的 cohort。阻断证据已刷新。
