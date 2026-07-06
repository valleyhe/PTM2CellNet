# PTM2CellNet Current Status

**Last updated: 2026-07-06**

The authoritative source for current project status is:

**[项目代码现状系统性复核报告 v18](项目代码现状系统性复核报告_2026-07-06_v18.md)**

## Quick Reference

- **Test baseline**: v18 records 1521 unit tests pass (5 skipped), 67 integration tests pass (1 skipped), ruff clean, `python -m mypy src` reports 31 errors in 16 files (see v18 §5.1)
- **Type checking**: `python -m mypy src --show-error-codes` reports 31 errors (v17 baseline was 0 — see v18 §5.1 for analysis and resolution strategy); mypy module-level overrides remain at 2; avoid bare `mypy` because this workstation has a user-level mypy on a different Python environment (see v14 report section 5.1)
- **Production readiness**: Core training/inference/API operational; production fail-fast security guards (TD-M3/M4); FastAPI lifespan modernized (F-08); real-assets acceptance layer covers ESM-3 / DAVF E2E / DDP / external services / **CPTAC-PDC real download + predictor wiring** (opt-in via `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1`)
- **Planning sync**: All 13 v2.1 requirements marked Complete; Phase 15/16/17 closed; TEST-04 criteria revised from "830/830" to current scope
- **Cancelled features**: Real-time mass cytometry, custom PTM DB, GUI, API key expansion (see v14 section 3.1)

## Archived Reports

All older `项目代码现状系统性复核报告_v*` files are historical snapshots.
Always refer to the latest version (highest v number) for current status.
