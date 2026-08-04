# PTM2CellNet Current Status

**Last updated: 2026-08-04**

The authoritative source for current project status is:

**[项目代码现状系统性复核报告 v19](archive/systematic_review_reports/项目代码现状系统性复核报告_2026-07-06_v19.md)**

## Quick Reference

- **Test baseline**: v19 records 1521 unit tests pass (5 skipped), 67 integration tests pass (1 skipped), 42 E2E tests pass, ruff clean
- **Type checking**: `python -m mypy src --show-error-codes` reports **0 errors in 123 files** (v19); mypy module-level overrides remain at 2; avoid bare `mypy` because this workstation has a user-level mypy on a different Python environment (see v14 report section 5.1)
- **Code duplication**: 3 处 (amp_compat 合并 + PTMSiteDict 统一，v19)；测试随机种子全部覆盖 (conftest autouse)
- **Production readiness**: Core training/inference/API operational; production fail-fast security guards (TD-M3/M4); FastAPI lifespan modernized (F-08); real-assets acceptance layer covers ESM-3 / DAVF E2E / DDP / external services / **CPTAC-PDC real download + predictor wiring** (opt-in via `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1`)
- **Planning sync**: All 13 v2.1 requirements marked Complete; Phase 15/16/17 closed; TEST-04 criteria revised from "830/830" to current scope
- **Cancelled features**: Real-time mass cytometry, custom PTM DB, GUI, API key expansion (see v14 section 3.1)

## Archived Reports

All older `项目代码现状系统性复核报告_v*` files are historical snapshots and
live in [`docs/archive/systematic_review_reports/`](archive/systematic_review_reports/).
Always refer to the latest version (highest v number) for current status.
