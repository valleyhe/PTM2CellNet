# Phase 15: Test & Dependency Fixes - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix all failing tests and declare missing dependencies so the test suite runs clean. This is the first phase of v2.1 — pure stabilization, no new features.

**In scope:**
- Fix 3 peft_config tests — update mock targets to match current `src/training/peft_config.py` internal function alias
- Fix 1 Lightning API test — update to Lightning 2.x `lightning.pytorch.loggers.TensorBoardLogger` import path
- Fix 10 pathway_integration tests — resolve state leakage between tests from module-level `sys.modules` mock injection
- Add `anndata` to requirements.txt — imported by `src/integration/genki/perturbation.py` and `reference_data.py`
- Add `sspa` to requirements.txt as optional dependency — used by `src/analysis/pathway_integration.py` with try/except guard
- Audit `lion-pytorch` — confirm whether to keep or remove from requirements.txt

**Out of scope:**
- Fix evaluation/__init__.py silent ImportError (Phase 16, CODE-01)
- Consolidate orphaned root scripts (Phase 16, CODE-02)
- Fix .gitignore whitelist (Phase 17, CONF-01)
- Lightning default Logger config (Phase 17, CONF-02)
- 830/830 pass rate target (Phase 17, TEST-04 — requires all Phase 15+16 fixes)

</domain>

<decisions>
## Implementation Decisions

### peft_config Mock Strategy (TEST-01)
- **D-01:** Update test mock targets from `get_peft_model` to `_get_peft_model_fn` — the source file aliases `peft.get_peft_model` as `_get_peft_model_fn` internally. Tests currently patch the wrong name, causing mock assertions to fail.
- **D-02:** Only update mock targets in `tests/unit/training/test_peft_config.py` — do not change `src/training/peft_config.py` which correctly uses the aliased internal name.

### Lightning API Import Path (TEST-02)
- **D-03:** Update to Lightning 2.x canonical import: `lightning.pytorch.loggers.TensorBoardLogger`. The codebase runs Lightning 2.5.1 so this is the correct namespace.

### pathway_integration Test Isolation (TEST-03)
- **D-04:** Replace module-level `sys.modules['sspa'] = mock_sspa` with fixture-based mock scoping that sets up and tears down per test — prevents state leakage where one test's mock configuration bleeds into the next.
- **D-05:** Reset mock call counts and return values in a `@pytest.fixture(autouse=True)` or per-test setup — current code sets `_mock_call_count = 0` manually but not consistently.

### Dependency Classification (DEPS-01, DEPS-02, DEPS-03)
- **D-06:** Add `anndata>=0.8.0` to requirements.txt — it is a direct import (not try/except guarded) in two files: `src/integration/genki/perturbation.py` and `src/integration/genki/reference_data.py`. Missing this causes ImportError at runtime.
- **D-07:** Add `sspa>=0.2.0` as optional dependency — code already has `try: import sspa` guard. List in requirements.txt with a comment marking it optional, or use extras syntax.
- **D-08:** Keep `lion-pytorch>=0.0.7` in requirements.txt — confirmed conditionally used in `src/training/lightning_module.py:416` (`from lion_pytorch import Lion` with try/except fallback to AdamW). It IS a real dependency.

### Claude's Discretion
- Exact fixture structure for pathway_integration tests (autouse vs explicit)
- Whether to use pip extras syntax for sspa or a comment annotation
- Test assertion style (keep existing patterns)
- Error messages in test docstrings

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Definition
- `.planning/ROADMAP.md` — Phase 15 definition, success criteria, plan breakdown (15-01, 15-02)
- `.planning/REQUIREMENTS.md` — TEST-01 through TEST-03, DEPS-01 through DEPS-03

### Test Files (must fix)
- `tests/unit/training/test_peft_config.py` — 3 tests needing mock target update (TEST-01)
- `tests/unit/test_train_pretrained_script.py` — 1 test needing Lightning import fix (TEST-02)
- `tests/unit/analysis/test_pathway_integration.py` — 10 tests with state leakage (TEST-03)

### Source Files (reference only, minimal changes)
- `src/training/peft_config.py` — Contains `_get_peft_model_fn` internal alias (mock target)
- `src/analysis/pathway_integration.py` — Contains `try: import sspa` pattern
- `src/training/lightning_module.py` — Contains `from lion_pytorch import Lion` (DEPS-03 evidence)
- `src/integration/genki/perturbation.py` — Contains `import anndata` (DEPS-01 evidence)
- `src/integration/genki/reference_data.py` — Contains `import anndata` (DEPS-01 evidence)

### Project Context
- `.planning/STATE.md` — Current project state, v2.1 milestone focus
- `.planning/PROJECT.md` — Tech stack, current status, key decisions

### Codebase Analysis
- `.planning/codebase/TESTING.md` — Test framework, patterns, mock conventions
- `.planning/codebase/CONCERNS.md` — Known issues including duplicate signaling_network, silent imports

### Prior Phase Context
- `.planning/phases/13-architecture-cascade-fusion/13-CONTEXT.md` — Cascade fusion decisions (DAVF test patterns)
- `.planning/phases/14-integration-testing-verification/14-REVIEW.md` — DAVF pipeline review findings

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PEFT_AVAILABLE` flag in `peft_config.py` — already used by tests for skipif markers; keep using this pattern
- `MockLoraModel` / `MockPeftModel` test helpers — well-structured mock classes; keep as-is
- Module-level sspa mock in `test_pathway_integration.py` — structure exists but needs fixture conversion

### Established Patterns
- Tests use `unittest.mock.patch` with full module path for mocking (e.g., `"src.training.peft_config.get_peft_model"`)
- Tests use `@pytest.mark.skipif(not PEFT_AVAILABLE, reason="...")` for optional deps
- Lightning 2.x namespace convention: `lightning.pytorch.*` (not `pytorch_lightning.*`)
- Requirements.txt uses comment sections for grouping (Core data, Deep learning, Bioinformatics, etc.)

### Integration Points
- `requirements.txt` — the single source of truth for pip dependencies
- `tests/unit/` — unit test directory structure mirrors `src/` module structure
- `src/training/peft_config.py:131` — the actual call site `_get_peft_model_fn(base_model, lora_config)` that tests must mock
- `src/training/lightning_module.py:416` — lion-pytorch import site (confirms DEPS-03)

### Key Dimensions
- 14 tests to fix (3 peft_config + 1 Lightning + 10 pathway_integration)
- 2 dependencies to add (anndata required, sspa optional)
- 1 dependency to confirm (lion-pytorch: keep)

</code_context>

<specifics>
## Specific Requirements

### Success Criteria (from ROADMAP.md)
1. `pytest tests/training/test_peft_config.py` — all 3 peft_config tests pass with mock targets matching the current `src/training/peft_config.py` API
2. `pytest tests/` Lightning API test passes using correct `lightning.pytorch.loggers.TensorBoardLogger` import path
3. `pytest tests/` pathway_integration tests pass consistently with no state leakage between test runs
4. `pip install -r requirements.txt` installs `anndata` as a declared dependency; `sspa` listed as optional
5. `lion-pytorch` is either confirmed used by training configs or removed from requirements.txt

### Test Fix Details
- TEST-01 mock target: change `patch("src.training.peft_config.get_peft_model")` → `patch("src.training.peft_config._get_peft_model_fn")`
- TEST-02 import: change `pytorch_lightning.loggers.TensorBoardLogger` → `lightning.pytorch.loggers.TensorBoardLogger`
- TEST-03 isolation: convert module-level `sys.modules['sspa'] = mock_sspa` to fixture-scoped mock with cleanup

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 15-test-dependency-fixes*
*Context gathered: 2026-05-04*
