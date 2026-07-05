# PTM2CellNet Current Status

**Last updated: 2026-07-06**

The authoritative source for current project status is:

**[项目代码现状系统性复核报告 v16](项目代码现状系统性复核报告_2026-07-06_v16.md)**

## Quick Reference

- **Test baseline**: v16 records 1496 unit tests pass (5 skipped), 6 real-assets tests gated (skipped by default), ruff clean, `python -m mypy src` clean (122 source files, 0 errors, 0 `type: ignore`)
- **Type checking**: `python -m mypy src --show-error-codes --no-error-summary` passes; avoid bare `mypy` because this workstation has a user-level mypy on a different Python environment (see v14 report section 5.1)
- **Production readiness**: Core training/inference/API operational; production fail-fast security guards added (TD-M3/M4); real-assets acceptance layer added for ESM-3 / DAVF E2E / DDP / external services (opt-in via `PTM2CELLNET_RUN_REAL_ASSET_TESTS=1`)
- **Planning sync**: All 13 v2.1 requirements marked Complete; Phase 15/16/17 closed; TEST-04 criteria revised from "830/830" to current scope
- **Cancelled features**: Real-time mass cytometry, custom PTM DB, GUI, API key expansion (see v14 section 3.1)

## Archived Reports

All older `项目代码现状系统性复核报告_v*` files are historical snapshots.
Always refer to the latest version (highest v number) for current status.
