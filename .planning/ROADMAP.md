# PTM2CellNet Roadmap

## Milestones

- ✅ **v1.0 Core Architecture & Testing** — Phases 1-6 (shipped 2026-04-04)
- ✅ **v2.0 DAVF Integration** — Phases 10-14 (shipped 2026-05-03)
- ✅ **v2.1 Technical Debt & Test Stabilization** — Phases 15-17 (complete 2026-07-06)
- ✅ **v2.2 Cross-Scale Scientific Closure & Reproducible Baseline** — Phases 18-20 (complete 2026-08-08)

## Scope Update (2026-07-05)

The following items are no longer project requirements and must not appear as future implementation phases: real-time mass-spec streaming, custom PTM database support, GUI, and new API-key feature work. Existing compatibility code may remain, but roadmap and plan documents should treat these as cancelled/out-of-scope.

## Post-v2.2 acceptance boundary (2026-09-01)

The v2.2 engineering milestone is complete. The separate DAVF × PerturbGen
follow-up proposal remains an acceptance track rather than a completed roadmap
phase: Gate-0 donor eligibility, Gate-E real DAVF retraining, Gate-4 release
evidence and Gate-5 scientific validation are still open. See
`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md` and the current
root audit report for evidence. The direction-gate library is implemented, but
the production pipeline still does not invoke it automatically. Synthetic
fixtures and opt-in code paths must
not be described as real biological validation.

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

### ✅ v2.1 Technical Debt & Test Stabilization (Complete)

- [x] **Phase 15: Test & Dependency Fixes** — Fix failing tests and declare missing dependencies (Complete 2026-05-04)
- [x] **Phase 16: Code Health & Consolidation** — Fix silent errors, consolidate orphaned scripts, resolve duplicates (Complete; implementation chosen over original wording, see Traceability notes in REQUIREMENTS.md)
- [x] **Phase 17: Configuration & Final Verification** — Fix gitignore, add logger config, full test pass (Complete; TEST-04 acceptance criteria revised from "830/830" to current scope)

### Cancelled / Out of Scope

- [x] V2-02: Real-time mass-spec streaming — cancelled 2026-07-05
- [x] V2-03: Custom PTM database support — cancelled 2026-07-05
- [x] V2-05: GUI — cancelled 2026-07-05
- [x] SEC-AUTH-APIKEY: API key feature expansion — cancelled 2026-07-05

## Phase Details

### Phase 15: Test & Dependency Fixes
**Goal**: All tests pass with correct dependencies declared
**Depends on**: Phase 14 (v2.0 shipped)
**Requirements**: TEST-01, TEST-02, TEST-03, DEPS-01, DEPS-02, DEPS-03
**Status**: ✅ Complete (2026-05-04)
**Success Criteria** (what must be TRUE):
  1. `pytest tests/training/test_peft_config.py` — all 3 peft_config tests pass with mock targets matching the current `src/training/peft_config.py` API ✅ (15-01)
  2. `pytest tests/` Lightning API test passes using correct `lightning.pytorch.loggers.TensorBoardLogger` import path ✅ (15-01)
  3. `pytest tests/` pathway_integration tests pass consistently with no state leakage between test runs ✅ (15-01)
  4. `anndata` declared in requirements-analysis.txt (GenKI/scVI capability tier); `sspa` declared as optional in requirements-analysis.txt ✅ (15-02)
  5. `lion-pytorch` confirmed used by `src/training/lightning_module.py` lion optimizer path; declared in requirements-mamba.txt ✅ (15-02)
**Plans**: 2 plans (both complete)

Plans:
- [x] 15-01-PLAN.md — Fix failing tests: peft_config mock targets, Lightning import, pathway_integration state leakage (TEST-01, TEST-02, TEST-03) — committed 8732f9b, 9586d1
- [x] 15-02-PLAN.md — Fix dependencies: declare anndata, declare sspa optional, confirm lion-pytorch (DEPS-01, DEPS-02, DEPS-03) — committed de41473

### Phase 16: Code Health & Consolidation
**Goal**: No silent error swallowing, no orphaned root scripts, no duplicate source files
**Depends on**: Phase 15
**Requirements**: CODE-01, CODE-02, CODE-03, CONF-03
**Status**: ✅ Complete (criterion revisions recorded in REQUIREMENTS.md Traceability)
**Success Criteria** (what must be TRUE):
  1. `python -c "from src.evaluation import *"` does not silently export `None`; optional submodules use `LazyImport` that re-raises the underlying `ImportError` on first access. ✅ (implementation stricter than original "log warning" wording)
  2. Root directory contains zero orphaned `.py` scripts — each has been moved to `scripts/`, integrated into `src/`, or deleted. ✅ (only `setup.py` + `signaling_network.py` shim remain)
  3. One canonical `src/models/signaling_network.py` implementation; root `signaling_network.py` exists only as a re-export shim. ✅ (criterion revised: "shim that only re-exports" is permitted; old 632-line duplicate deleted)
  4. `_load_pathway_db()` does not raise `NotImplementedError` — it logs a warning and falls back to a built-in pathway set. ✅ (criterion revised: graceful fallback instead of explicit "not implemented" message)
**Plans**: complete (no separate plan files; delivered incrementally)

Plans:
- [x] 16-01: Fix evaluation silent ImportError (CODE-01) — LazyImport adopted
- [x] 16-02: Consolidate orphaned root scripts (CODE-02) — 7 orphans removed
- [x] 16-03: Resolve duplicate signaling_network.py (CODE-03, CONF-03) — shim + canonical

### Phase 17: Configuration & Final Verification
**Goal**: Repository configuration is correct and the full test suite passes
**Depends on**: Phase 16
**Requirements**: CONF-01, CONF-02, TEST-04
**Status**: ✅ Complete
**Success Criteria** (what must be TRUE):
  1. `git ls-files` tracks `configs/`, `docs/`, `Dockerfile`, `docker-compose.yml`, `setup.py`, `setup.cfg`, `pytest.ini`, `.coveragerc`, `.pylintrc`, `API_DOCUMENTATION.md`, `CHANGELOG.md`, and `LICENSE` ✅
  2. Running any Lightning Trainer via `PTM2CellNetLightning.create_trainer()` without explicit logger config auto-creates a `TensorBoardLogger`; no "no logger configured" warning. ✅
  3. `pytest tests/unit --tb=short` reports 1452 passed, 5 skipped (env-guard skips); `pytest tests/e2e tests/integration tests/test_*.py` reports 183 passed, 1 skipped. ✅ (TEST-04 criterion revised from "830/830" to current scope — see REQUIREMENTS.md Traceability)
**Plans**: complete

Plans:
- [x] 17-01: Fix gitignore whitelist and add Lightning logger config (CONF-01, CONF-02)
- [x] 17-02: Final verification — full test pass (TEST-04, criteria revised)

### ✅ v2.2 Cross-Scale Scientific Closure & Reproducible Baseline (Complete 2026-08-08)

**Goal**: close the PTM→signal graph→cell-state scale gap and establish a
reproducible data inventory plus PMADS Ridge reference pipeline.

- [x] **Phase 18: Data Manifest & PMADS Ridge Baseline** — versioned source inventory, schema/quality checks, deterministic Ridge artifacts (DATA-01, DATA-02, BASE-01, BASE-02)
- [x] **Phase 19: Cross-Scale Scientific Model** — MultiPLM fusion, CIGNN sensitivity bridge, cell-state decoder and opt-in composition (MODEL-01..MODEL-04)
- [x] **Phase 20: Verification & Technical Summary** — integration/performance gates, maintenance workflow and dated report (BASE-03, VERIFY-01, VERIFY-02)

### Phase 18: Data Manifest & PMADS Ridge Baseline
**Goal**: Freeze a versioned data inventory and produce an offline-reproducible PMADS Ridge reference artifact.
**Depends on**: Phase 17
**Requirements**: DATA-01, DATA-02, BASE-01, BASE-02
**Status**: ✅ Complete (2026-08-08)
**Success Criteria** (what must be TRUE):
  1. `data/manifests/datasets.yaml` lists all required PTM, perturbation and signaling graph sources with source/access/license/format/schema/quality metadata.
  2. Manifest validation catches malformed entries, missing required snapshots and SHA-256 drift without treating controlled null paths as fake data.
  3. `scripts/baseline_pmads_ridge.py` runs on a real or user-supplied PMADS-compatible file with deterministic split, leakage-safe features, metrics and a self-describing artifact.
  4. Unit and CLI tests prove the manifest and baseline contracts offline.
**Plans**: 1 plan (complete)

Plans:
- [x] 18-01-PLAN.md — Data manifest, PMADS Ridge baseline, CLI and tests

### Phase 19: Cross-Scale Scientific Model
**Goal**: Implement an opt-in differentiable protein/PTM → typed signal graph → perturbed cell-state model.
**Depends on**: Phase 18
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04
**Status**: ✅ Complete (2026-08-08)
**Success Criteria** (what must be TRUE):
  1. MultiPLMEncoder fuses Ankh39/ESM-2/ProtT5 contracts and reports missing weights/fallbacks explicitly.
  2. CIGNNSignalBridge consumes explicit edges/types and computes the exact sensitivity matrix with differentiable message passing.
  3. CellGraphCompassHead returns masked delta-expression and cell-state outputs with a stable combined loss.
  4. CrossScalePTM2CellNet supports config/checkpoint round-trip and emits model/data/graph provenance without altering PTM2CellNetBase defaults.
**Plans**: 1 plan (complete)

Plans:
- [x] 19-01-PLAN.md — Cross-scale scientific model components and integration tests

### Phase 20: Verification & Technical Summary
**Goal**: Execute the TD-01/TD-02 verification gates, measure resource use and publish the traceable technical summary.
**Depends on**: Phase 19
**Requirements**: BASE-03, VERIFY-01, VERIFY-02
**Status**: ✅ Complete (2026-08-08)
**Success Criteria** (what must be TRUE):
  1. Data refresh and baseline maintenance steps are documented and executable.
  2. Targeted tests, lint/type/compile checks and a CPU benchmark produce captured quantitative evidence.
  3. The dated technical summary reports environment, parameters, flow, metrics, anomalies, scientific interpretation, limitations and reproduction commands.
**Plans**: 1 plan (complete)

Plans:
- [x] 20-01-PLAN.md — Verification, performance benchmark and technical summary

## Progress

**Execution Order:**
Phases execute in numeric order: 15 → 16 → 17 → 18 → 19 → 20

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
| 15. Test & Dependency Fixes | v2.1 | 2/2 | Complete | 2026-05-04 |
| 16. Code Health & Consolidation | v2.1 | 3/3 | Complete | 2026-07-06 |
| 17. Configuration & Final Verification | v2.1 | 2/2 | Complete | 2026-07-06 |
| 18. Data Manifest & PMADS Ridge Baseline | v2.2 | 1/1 | Complete | 2026-08-08 |
| 19. Cross-Scale Scientific Model | v2.2 | 1/1 | Complete | 2026-08-08 |
| 20. Verification & Technical Summary | v2.2 | 1/1 | Complete | 2026-08-08 |

---
*Created: 2026-03-30*
*Updated: 2026-08-08 — v2.2 Phase 18/19/20 completed for TD-01 and TD-02*
