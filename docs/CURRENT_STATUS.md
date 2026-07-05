# PTM2CellNet Current Status

**Last updated: 2026-07-05**

The authoritative source for current project status is:

**[项目代码现状系统性复核报告 v15](项目代码现状系统性复核报告_2026-07-05_v15.md)**

## Quick Reference

- **Test baseline**: v14 recorded 1452 unit tests pass, 183 integration/E2E tests pass, ruff clean; v15 adds current governance and implementation-gap audit
- **Type checking**: `python -m mypy src --show-error-codes --no-error-summary` passes; avoid bare `mypy` because this workstation has a user-level mypy on a different Python environment (see v14 report section 5.1)
- **Production readiness**: Core training/inference/API operational; ESM-3 production assets, DAVF real-checkpoint E2E, multi-GPU DDP, and CPTAC/PDC validation need closure
- **Cancelled features**: Real-time mass cytometry, custom PTM DB, GUI, API key expansion (see v14 section 3.1)

## Archived Reports

All older `项目代码现状系统性复核报告_v*` files are historical snapshots.
Always refer to the latest version (highest v number) for current status.
