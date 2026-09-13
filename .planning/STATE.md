---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Cross-Scale Scientific Closure & Reproducible Baseline
status: complete
# `status` records the completed v2.2 milestone; the active follow-up is tracked below.
active_track: DAVF × PerturbGen acceptance
active_track_status: pending
active_next_action: freeze direction semantics, evidence sources and donor split
stopped_at: none
last_updated: "2026-09-13T00:00:00+08:00"
last_activity: 2026-09-13 — research contract review synchronized direction semantics, evidence independence, invocation boundary and formal cohort status
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# PTM2CellNet State

## Project Reference

See: .planning/PROJECT.md (active contract updated 2026-09-13)

**Core Value**: 提供端到端的蛋白质 PTM 分析与细胞状态预测能力，集成DAVF方向感知模型实现PTM→信号通路效应预测。

**Current Focus**: v2.2 complete；post-v2.2 DAVF × PerturbGen acceptance track is active. Engineering bridge and statistical interfaces exist, while direction/source semantics, public prepare/reuse wiring, E2E statistical continuation and formal donor-cohort gates remain open.

**Key Constraints**:

- Verification environment: Python 3.12.13 / PyTorch 2.4.1+cu118 / Lightning 2.6.5
- v1.0 + v2.0 tests must continue passing
- Default DAVF model: latent_davf_ibd_norman (10-dim latent, scVI) remains the historical/default serving path; formal KO/KD route assets use separate schema-v2 `LatentDAVF` checkpoints with 64-dimensional, 4018-gene route-specific scVI coordinates.
- v2.2 cross-scale model remains opt-in until real assets and biological acceptance tests exist
- Formal PerturbGen evidence requires a real `normal/disease` cohort with raw counts, explicit donor identity and at least 3 shared donors; local scPerturb files cannot be reinterpreted to satisfy this contract.
- Formal direction records must bind `context`, `intervention`, comparison baseline, research objective, source/cohort and train-only/held-out donor provenance. Observed donor-level disease−normal, DAVF intervention delta and PerturbGen utility are separate evidence types.

---

## Shipped Milestones

### v2.1 Technical Debt & Test Stabilization — SHIPPED 2026-07-06

**Phases**: 15-17 | **Plans**: 7 (15-01, 15-02, 16-01..03, 17-01, 17-02)
**Outcome**: All 13 v2.1 requirements complete. TEST-04 acceptance criteria
revised from "830/830" to the current scope (1452 unit + 183 non-unit passing;
6 intentional env-guard skips). Several Phase 16/17 acceptance clauses were
reconciled with the implementation actually delivered (LazyImport re-raise,
shim+canonical signaling_network, graceful pathway fallback). See
REQUIREMENTS.md Traceability for the criterion revisions.

### v2.0 DAVF Integration — SHIPPED 2026-05-03

**Phases**: 10-14 | **Plans**: 5 | **Tests added**: ~91

### v1.0 Core Architecture & Testing — SHIPPED 2026-04-04

**Phases**: 1-6 | **Plans**: 16 | **Tests**: ~959

---

## Current Position

Phase: 20 of 20 (historical v2.2 complete)
Active track: DAVF × PerturbGen acceptance (pending; Gate-0 compliant cohorts: 0)
Status: v2.2 implementation complete；active follow-up remains pending direction/source contracts, public prepare/reuse, E2E statistics and real Gate-E/Gate-4/Gate-5 evidence.
Progress: v2.2 [██████████] 100%；active track pending

---

## Accumulated Context

### Decisions

Cross-phase decisions affecting future work:

- Phase 15 before 16: tests and deps must be fixed before code consolidation changes risk breaking more things
- Phase 17 last: TEST-04 (full suite pass) is the final gate — all fixes must land first
- Patched SSPA_AVAILABLE/sspa module vars via conftest since module is pre-imported at collection time
- Stub module injection for lightning.L.loggers since Lightning 2.x lacks top-level loggers attribute
- Explicit side_effect/return_value cleanup needed on child mocks since reset_mock() does not clear them
- CODE-01 implemented via LazyImport (re-raise) rather than logging.warning — stricter and accepted
- CODE-03 implemented as shim re-export rather than outright deletion — preserves back-compat imports
- CONF-03 implemented as graceful fallback to built-in pathway set rather than an explicit "not implemented" message
- Dependencies split into requirements-{core,pretrained,mamba,analysis,dev,docs}.txt; anndata/sspa live in analysis, lion-pytorch in mamba

### Pending Todos

- Keep restricted/remote data as manifest entries and local-import contracts only.
- Real-asset scientific acceptance remains a follow-up requiring owner-mounted weights/graphs and independent validation.

### Blockers/Concerns

No blocking issue for the completed v2.2 engineering-contract scope. Promotion to
real-asset/scientific E2E is blocked by the items in
`docs/CURRENT_STATUS.md` and `project_analysis_20260913.md`; they remain outside
the current automated acceptance boundary:

- Real-asset / real-hardware / real-network / real-data acceptance layer (ESM-3, DAVF E2E, distributed, CPTAC) — these require resources not available in CI and are tracked as future opt-in jobs.
- New API-key expansion remains cancelled; existing compatibility code is not a pending research task.

### Post-v2.2 DAVF × PerturbGen acceptance status (2026-09-13)

- `run_davf_perturbgen_e2e.py` connects candidate spec → scVI context → route-specific DAVF → direction gate → isolated PerturbGen stage plans. E2E defaults to report export; only explicit `--run-perturbgen` runs the six-stage preparation/execution chain. The formal CLI requires an E2E report bound to a passing invocation; the lower-level runner currently executes StagePlan without independently re-checking the gate.
- The local 2026-09-13 Gate-0 audit covered 30 scPerturb H5AD files; 26 were readable and 0 satisfied the formal donor/state/Ensembl contract. Evidence: `outputs/perturbgen/spike/20260913_donor_audit/evidence.json`.
- DatlingerBock2021 real-data preflight was rejected because `state` and `donor` are absent. It remains an engineering smoke dataset, not formal utility evidence.
- Existing matched-null generation, candidate empirical-p aggregation, formal input isolation, unperturbed-quality extraction, donor split and dual-path AND interfaces are code-complete; E2E does not yet auto-continue them from stage reports.
- Next order is: freeze direction semantics and independent source/split design → specify public prepare/reuse and invocation boundary → connect statistical continuation → run real cohort/Gate-E/Gate-4/Gate-5 evidence. Retraining or rerunning on existing cell-line files would not close the scientific gate.

---

## Session Continuity

Last session: 2026-09-13
Stopped at: synchronized the active DAVF × PerturbGen contract; local Gate-0 remains blocked with 0 compliant cohorts.
Resume file: None

---
*Last updated: 2026-09-13 — post-v2.2 DAVF × PerturbGen acceptance contract and evidence status synchronized*
