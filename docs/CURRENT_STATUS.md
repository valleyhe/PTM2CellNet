# PTM2CellNet Current Status

**Last updated: 2026-08-04（P0–P2 修复完成后）**

The authoritative sources for current project status are:

- **[问题修复与系统性复核报告_2026-08-04](问题修复与系统性复核报告_2026-08-04.md)** — P0–P2 修复闭环与三轮迭代记录
- **[E2E训练和推理能力评估报告_2026-08-04_v2](E2E训练和推理能力评估报告_2026-08-04_v2.md)** — 修复后 E2E 能力评估
- **[项目代码现状系统性复核报告 v19](archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)** — 上一轮系统性复核基线

## Quick Reference

- **Test baseline (2026-08-04, main @ 9f10e31)**: 1547 unit tests pass (4 skipped), 67 integration tests pass (1 skipped), 42 E2E tests pass; ruff clean; mypy 0 errors on changed modules
- **P0–P2 修复闭环**: P1-1（API 自动初始化 sibling config 发现）、P2-1（数据管线 provenance 列 + 发布门禁真实生效）、P2-2（配置引用，并入 P1-1）全部关闭；无 P0 阻塞项
- **Type checking**: `python -m mypy src --show-error-codes` 0 errors in 123 files（v19 基线）；avoid bare `mypy` because this workstation has a user-level mypy on a different Python environment
- **Production readiness**: Core training/inference/API operational; API auto-init now works with `PTM2CELLNET_CHECKPOINT` alone (sibling config discovery); real-data release gate enforced via provenance columns (deployable flag)
- **Cancelled features**: Real-time mass cytometry, custom PTM DB, GUI, API key expansion (see v14 section 3.1)

## Archived Reports

All historical snapshots live in [`docs/archive/`](archive/), including the
pre-fix `E2E训练和推理能力评估报告_2026-08-04.md`
(`archive/status_and_audit_reports/`). Always refer to the latest documents
listed above for current status.
