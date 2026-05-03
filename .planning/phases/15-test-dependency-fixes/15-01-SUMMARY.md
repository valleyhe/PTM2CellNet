---
phase: 15-test-dependency-fixes
plan: 01
subsystem: testing
tags: [pytest, mock, lightning, peft, sspa, fixtures]

# Dependency graph
requires:
  - phase: none
    provides: n/a
provides:
  - "14 previously failing tests now passing across 3 test files"
  - "Fixture-scoped sspa mock pattern in conftest.py"
affects: [15-02, 17-test-gate]

# Tech tracking
tech-stack:
  added: []
  patterns: [session-scoped sys.modules mock in conftest.py, stub module injection for lightning loggers]

key-files:
  created:
    - tests/unit/analysis/conftest.py
  modified:
    - tests/unit/training/test_peft_config.py
    - tests/unit/test_train_pretrained_script.py
    - tests/unit/analysis/test_pathway_integration.py

key-decisions:
  - "Patched module-level SSPA_AVAILABLE and sspa vars in pathway_integration via conftest session fixture since module was already imported before fixture runs"
  - "Injected stub loggers module on lightning.L object to fix Lightning 2.x missing top-level loggers attribute"
  - "Explicitly clear side_effect and return_value on child mocks in reset fixture since MagicMock.reset_mock() does not clear them"

patterns-established:
  - "Session-scoped sys.modules mock injection: set mock in sys.modules AND patch module-level flags in conftest.py"
  - "Explicit mock state cleanup: clear side_effect/return_value on child method mocks between tests"

requirements-completed: [TEST-01, TEST-02, TEST-03]

# Metrics
duration: 4min
completed: 2026-05-03
---

# Phase 15 Plan 01: Fix Failing Tests Summary

**Fixed 14 failing tests: peft_config mock target correction, Lightning 2.x loggers attribute injection, and pathway_integration state leakage via fixture-scoped sspa mock**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-03T16:46:32Z
- **Completed:** 2026-05-03T16:50:51Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Fixed 3 peft_config tests by correcting mock patch targets from `get_peft_model` to `_get_peft_model_fn`
- Fixed 1 Lightning test by injecting stub `loggers` module onto `lightning.L` since Lightning 2.x does not expose `loggers` as a top-level attribute
- Fixed 10 pathway_integration tests by replacing module-level `sys.modules` sspa injection with session-scoped conftest fixture and per-test reset fixture

## Task Commits

1. **Task 1: Fix peft_config mock targets** - `8732f9b` (fix)
2. **Task 2: Verify Lightning test and fix pathway_integration state leakage** - `95826d1` (fix)

## Files Created/Modified
- `tests/unit/training/test_peft_config.py` - Updated 3 mock patch targets to `_get_peft_model_fn`
- `tests/unit/test_train_pretrained_script.py` - Fixed Lightning `L.loggers` monkeypatch with stub module injection
- `tests/unit/analysis/conftest.py` - New file: session-scoped sspa mock fixture with SSPA_AVAILABLE patching
- `tests/unit/analysis/test_pathway_integration.py` - Replaced module-level sys.modules injection with fixture-based approach; removed `_mock_call_count` resets

## Decisions Made
- Patched `SSPA_AVAILABLE` and `sspa` module-level variables in `pathway_integration` via conftest session fixture, since the module is already imported at collection time and `sys.modules` injection alone does not retroactively change the module's local bindings
- Used stub `types.ModuleType("lightning.loggers")` approach for Lightning test rather than trying to import `lightning.pytorch.loggers` which would require tensorboard installed
- Explicitly clear `side_effect` and `return_value` on child mock methods (`process_kegg`, `process_reactome`) in the autouse reset fixture since `MagicMock.reset_mock()` only resets call history, not behavior configuration on child mocks

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Lightning test needed L.loggers stub injection**
- **Found during:** Task 2 (Lightning test verification)
- **Issue:** `L.loggers` does not exist as attribute on `lightning` module in Lightning 2.x. The test monkeypatched `train_pretrained.L.loggers` but `setattr` on a non-existent nested attribute fails with `AttributeError`
- **Fix:** Created a stub `types.ModuleType("lightning.loggers")` with `TensorBoardLogger = DummyTBLogger` and used `monkeypatch.setattr(L, "loggers", stub, raising=False)` to inject it
- **Files modified:** `tests/unit/test_train_pretrained_script.py`
- **Verification:** Test passes
- **Committed in:** `95826d1`

**2. [Rule 3 - Blocking] sspa mock needed module-level variable patching**
- **Found during:** Task 2 (pathway_integration tests)
- **Issue:** Session-scoped conftest fixture injects `sys.modules['sspa']` but `pathway_integration` module was already imported with `SSPA_AVAILABLE = False` and `sspa = None`. Mock injection in `sys.modules` alone does not retroactively change the module's local bindings
- **Fix:** In conftest session fixture, also patch `src.analysis.pathway_integration.SSPA_AVAILABLE = True` and `src.analysis.pathway_integration.sspa = mock_sspa`
- **Files modified:** `tests/unit/analysis/conftest.py`
- **Verification:** All 10 pathway tests pass
- **Committed in:** `95826d1`

**3. [Rule 1 - Bug] MagicMock.reset_mock() does not clear side_effect on child mocks**
- **Found during:** Task 2 (pathway_integration tests)
- **Issue:** `test_kegg_network_error_handling` sets `process_kegg.side_effect = Exception("Network error")`. The autouse reset fixture called `reset_mock()` but this does NOT clear `side_effect` on child method mocks, causing `test_load_kegg_pathways_function` to fail with the stale error
- **Fix:** Added explicit `side_effect = None` and `return_value = MagicMock()` resets for `process_kegg` and `process_reactome` in the reset fixture
- **Files modified:** `tests/unit/analysis/test_pathway_integration.py`
- **Verification:** All tests pass with no cross-test leakage
- **Committed in:** `95826d1`

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes were necessary for correctness. The plan's fixture approach was sound but required these implementation details to handle real mock/Lightning behavior.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Self-Check: PASSED

All files exist. All commits verified in git log.

## Next Phase Readiness
- All 14 previously failing tests now pass
- Plan 15-02 (dependency declarations) can proceed independently
- Phase 17 TEST-04 (830/830 gate) depends on both 15-01 and 15-02 completing

---
*Phase: 15-test-dependency-fixes*
*Completed: 2026-05-03*
