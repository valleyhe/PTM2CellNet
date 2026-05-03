# PTM2CellNet Roadmap

## Milestones

- ✅ **v1.0 Core Architecture & Testing** — Phases 1-6 (shipped 2026-04-04)
- ✅ **v2.0 DAVF Integration** — Phases 10-14 (shipped 2026-05-03)
- 🚧 **v2.1 Technical Debt & Test Stabilization** — Phases 15-17 (in progress)

## Phases

<details>
<summary>✅ v1.0 Core Architecture & Testing (Phases 1-6) — SHIPPED 2026-04-04</summary>

- [x] Phase 1: Stack Upgrade & Data Integrity (1/1 plans)
- [x] Phase 2: Training Pipeline Optimization (1/1 plans)
- [x] Phase 3: Analysis Features & API Enhancement (3/3 plans)
- [x] Phase 4: Testing & Validation (1/1 plans)
- [x] Phase 5: Test Coverage Completion (4/4 plans)
- [x] Phase 6: Coverage Gap Closure (4/4 plans)

</details>

<details>
<summary>✅ v2.0 DAVF Integration (Phases 10-14) — SHIPPED 2026-05-03</summary>

- [x] Phase 10: DAVF Module Import & Foundation (1/1 plans)
- [x] Phase 11: PTM Direction Mapping (1/1 plans)
- [x] Phase 12: DAVF Inference Wrapper (1/1 plans)
- [x] Phase 13: Architecture Cascade Fusion (1/1 plans)
- [x] Phase 14: Integration Testing & Verification (1/1 plans)

</details>

### 🚧 v2.1 Technical Debt & Test Stabilization (In Progress)

- [ ] **Phase 15: Test & Dependency Fixes** — Fix failing tests and declare missing dependencies
- [ ] **Phase 16: Code Health & Consolidation** — Fix silent errors, consolidate orphaned scripts, resolve duplicates
- [ ] **Phase 17: Configuration & Final Verification** — Fix gitignore, add logger config, achieve 830/830 pass rate

## Phase Details

### Phase 15: Test & Dependency Fixes
**Goal**: All tests pass with correct dependencies declared
**Depends on**: Phase 14 (v2.0 shipped)
**Requirements**: TEST-01, TEST-02, TEST-03, DEPS-01, DEPS-02, DEPS-03
**Success Criteria** (what must be TRUE):
  1. `pytest tests/training/test_peft_config.py` — all 3 peft_config tests pass with mock targets matching the current `src/training/peft_config.py` API
  2. `pytest tests/` Lightning API test passes using correct `lightning.pytorch.loggers.TensorBoardLogger` import path
  3. `pytest tests/` pathway_integration tests pass consistently with no state leakage between test runs
  4. `pip install -r requirements.txt` installs `anndata` as a declared dependency; `sspa` listed as optional
  5. `lion-pytorch` is either confirmed used by training configs or removed from requirements.txt
**Plans**: TBD

Plans:
- [ ] 15-01: Fix failing tests (TEST-01, TEST-02, TEST-03)
- [ ] 15-02: Fix dependencies (DEPS-01, DEPS-02, DEPS-03)

### Phase 16: Code Health & Consolidation
**Goal**: No silent error swallowing, no orphaned root scripts, no duplicate source files
**Depends on**: Phase 15
**Requirements**: CODE-01, CODE-02, CODE-03, CONF-03
**Success Criteria** (what must be TRUE):
  1. `python -c "from src.evaluation import *"` logs a visible warning when optional dependencies are missing instead of silently exporting `None`
  2. Root directory contains zero orphaned `.py` scripts — each has been moved to `scripts/`, integrated into `src/`, or deleted
  3. Only one canonical `signaling_network.py` exists in the codebase; the duplicate is removed or archived
  4. Calling `_load_pathway_db()` in signaling network modules prints a clear "not yet implemented" message with guidance instead of silently returning empty data
**Plans**: TBD

Plans:
- [ ] 16-01: Fix evaluation silent ImportError (CODE-01)
- [ ] 16-02: Consolidate orphaned root scripts (CODE-02)
- [ ] 16-03: Resolve duplicate signaling_network.py (CODE-03, CONF-03)

### Phase 17: Configuration & Final Verification
**Goal**: Repository configuration is correct and all 830 tests pass at 100%
**Depends on**: Phase 16
**Requirements**: CONF-01, CONF-02, TEST-04
**Success Criteria** (what must be TRUE):
  1. `git status` tracks `configs/`, `docs/`, `Dockerfile`, `docker-compose.yml`, `setup.py`, `setup.cfg`, `pytest.ini`, `.coveragerc`, `.pylintrc`, `API_DOCUMENTATION.md`, `CHANGELOG.md`, and `LICENSE`
  2. Running any Lightning Trainer without explicit logger config produces no "no logger configured" warning
  3. `pytest tests/ --tb=short` reports 830 passed, 0 failed, 0 skipped, 0 errors
**Plans**: TBD

Plans:
- [ ] 17-01: Fix gitignore whitelist and add Lightning logger config (CONF-01, CONF-02)
- [ ] 17-02: Final verification — achieve 830/830 (TEST-04)

## Progress

**Execution Order:**
Phases execute in numeric order: 15 → 16 → 17

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Stack Upgrade & Data Integrity | v1.0 | 1/1 | Complete | 2026-03-30 |
| 2. Training Pipeline Optimization | v1.0 | 1/1 | Complete | 2026-03-30 |
| 3. Analysis Features & API Enhancement | v1.0 | 3/3 | Complete | 2026-03-31 |
| 4. Testing & Validation | v1.0 | 1/1 | Complete | 2026-03-31 |
| 5. Test Coverage Completion | v1.0 | 4/4 | Complete | 2026-04-02 |
| 6. Coverage Gap Closure | v1.0 | 4/4 | Complete | 2026-04-04 |
| 10. DAVF Module Import & Foundation | v2.0 | 1/1 | Complete | 2026-05-03 |
| 11. PTM Direction Mapping | v2.0 | 1/1 | Complete | 2026-05-03 |
| 12. DAVF Inference Wrapper | v2.0 | 1/1 | Complete | 2026-05-03 |
| 13. Architecture Cascade Fusion | v2.0 | 1/1 | Complete | 2026-05-03 |
| 14. Integration Testing & Verification | v2.0 | 1/1 | Complete | 2026-05-03 |
| 15. Test & Dependency Fixes | v2.1 | 0/2 | Not started | - |
| 16. Code Health & Consolidation | v2.1 | 0/3 | Not started | - |
| 17. Configuration & Final Verification | v2.1 | 0/2 | Not started | - |

---
*Created: 2026-03-30*
*Updated: 2026-05-04 — v2.1 Technical Debt & Test Stabilization roadmap created (Phases 15-17)*
