---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: Cross-Scale Scientific Closure & Reproducible Baseline
status: complete
stopped_at: none
last_updated: "2026-08-08T00:00:00Z"
last_activity: 2026-08-08 — v2.2 TD-01/TD-02 implementation, verification and technical summary completed
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# PTM2CellNet State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core Value**: 提供端到端的蛋白质 PTM 分析与细胞状态预测能力，集成DAVF方向感知模型实现PTM→信号通路效应预测。

**Current Focus**: v2.2 complete — TD-01/TD-02 implementation and verification handed off.

**Key Constraints**:

- Verification environment: Python 3.12.13 / PyTorch 2.4.1+cu118 / Lightning 2.6.5
- v1.0 + v2.0 tests must continue passing
- Default DAVF model: latent_davf_ibd_norman (10-dim latent, scVI)
- v2.2 cross-scale model remains opt-in until real assets and biological acceptance tests exist

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

Phase: 20 of 20 (complete)
Status: v2.2 implementation and verification complete.
Progress: [██████████] 100%

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
`docs/E2E训练与推理现状分析_2026-08-08.md`; they remain outside the current
automated acceptance boundary:

- Real-asset / real-hardware / real-network / real-data acceptance layer (ESM-3, DAVF E2E, distributed, CPTAC) — these require resources not available in CI and are tracked as future opt-in jobs.
- Production fail-fast for missing API key / empty download allowlist — handled as separate hardening work.

---

## Session Continuity

Last session: 2026-08-08
Stopped at: v2.2 Phase 20 verification and technical summary complete.
Resume file: None

---
*Last updated: 2026-08-08 — v2.2 Phase 18/19/20 completed for TD-01 and TD-02*
