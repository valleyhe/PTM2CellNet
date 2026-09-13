# Requirements: PTM2CellNet v2.1

**Defined:** 2026-05-04
**Core Value:** Fix all failing tests,补齐缺失依赖, clean up orphaned code, improve error handling — return codebase to healthy state.

## v2.1 Requirements

### Test Stabilization (TEST)

- [x] **TEST-01**: Fix 3 peft_config tests — update mock targets to match current `src/training/peft_config.py` API (replace `get_peft_model` with actual function name). **Completed 2026-05-03 (15-01-SUMMARY).**
- [x] **TEST-02**: Fix 1 Lightning API test — update `train_pretrained_script.py` test to use correct `lightning.pytorch.loggers.TensorBoardLogger` import path. **Completed 2026-05-03 (15-01-SUMMARY).**
- [x] **TEST-03**: Fix 10 pathway_integration tests — resolve state leakage between tests, align mock interfaces with actual KEGG/Reactome stub behavior. **Completed 2026-05-03 (15-01-SUMMARY).**
- [x] **TEST-04**: Achieve full test pass with documented skip policy. **Completed.** Updated acceptance criteria: as of v15 the suite is 1452 unit passed / 5 skipped + 183 non-unit passed / 1 skipped (skips are intentional env guards, not failures). The legacy "830/830" number is superseded by the current scope; see Traceability below.

### Dependency Management (DEPS)

- [x] **DEPS-01**: Declare `anndata`. **Completed 2026-05-04 (15-02-SUMMARY)** — declared in `requirements-analysis.txt` (`anndata>=0.10,<0.12`) so it ships with the GenKI/scVI capability tier that imports it; intentionally not in core to keep the lean image lean.
- [x] **DEPS-02**: Declare `sspa` as optional. **Completed 2026-05-04 (15-02-SUMMARY)** — declared in `requirements-analysis.txt` (`sspa>=0.2.0`); the import in `src/analysis/pathway_integration.py` is guarded so absence is non-fatal.
- [x] **DEPS-03**: Audit `lion-pytorch`. **Completed 2026-05-04 (15-02-SUMMARY)** — confirmed used by `src/training/lightning_module.py:441` (`from lion_pytorch import Lion`) under the `lion` optimizer path. Declared in `requirements-mamba.txt` (`lion-pytorch>=0.0.7`) since it is paired with the Mamba training capability.

### Code Health (CODE)

- [x] **CODE-01**: Fix `evaluation/__init__.py`. **Completed.** No bare `except ImportError: pass` remains; optional submodules (`Evaluator`, `explainers`, `visualization`) are exposed via `LazyImport` (`src/utils/lazy_import.py`) that re-raises the original `ImportError` on access instead of silently exporting `None`. Acceptance wording updated from "logs a warning" to "lazy import re-raises the underlying error".
- [x] **CODE-02**: Consolidate 7 orphaned root-level Python scripts. **Completed.** All seven orphans removed; root directory now contains only `setup.py` and the compatibility `signaling_network.py` shim (see CODE-03).
- [x] **CODE-03**: Resolve duplicate `signaling_network.py`. **Completed.** Root `signaling_network.py` is now a thin re-export shim of the canonical `src/models/signaling_network.py`; the old 632-line duplicate implementation is deleted. Acceptance wording updated to "one canonical implementation + a compatibility shim that only re-exports".

### Configuration & Infrastructure (CONF)

- [x] **CONF-01**: Fix `.gitignore` whitelist. **Completed.** `.gitignore` whitelists `configs/`, `docs/`, `Dockerfile`, `docker-compose.yml`, `setup.py`, `setup.cfg`, `pytest.ini`, `.coveragerc`, `.pylintrc`, `API_DOCUMENTATION.md`, `CHANGELOG.md`, `LICENSE`; `git ls-files` confirms tracking.
- [x] **CONF-02**: Add default Logger configuration for Lightning Trainer. **Completed.** `src/training/logging_config.py` provides `configure_default_logger()`; `PTM2CellNetLightning.create_trainer()` (`src/training/lightning_module.py:495`) auto-instantiates a `TensorBoardLogger` when none is supplied.
- [x] **CONF-03**: Clarify KEGG/Reactome loading stubs. **Completed.** `src/models/signaling_network.py:271-283` no longer raises `NotImplementedError`; on failure it logs a warning and falls back to a built-in pathway set.

## v2.2 Requirements (Current)

### Reproducible data and baseline (DATA/BASE)

- [x] **DATA-01**: The repository provides a versioned YAML data manifest that enumerates every required PTM, single-cell perturbation and signaling-graph source, its access/license status, source reference, expected format, canonical fields and quality gates; unavailable or controlled assets are represented explicitly rather than fabricated. **Completed 2026-08-08.**
- [x] **DATA-02**: A manifest loader/validator and update/check workflow validates identifiers, schema, local file presence and optional SHA-256 snapshots, and returns actionable failures for stale or incomplete entries. **Completed 2026-08-08.**
- [x] **BASE-01**: A PMADS-compatible normalizer and Ridge regression baseline use deterministic feature extraction, a fixed seed/split with optional protein grouping, leakage-safe preprocessing, and classification/regression metrics. **Completed 2026-08-08.**
- [x] **BASE-02**: `scripts/baseline_pmads_ridge.py` produces a self-describing model artifact, metrics/prediction tables and provenance linking the input dataset, manifest, split policy, code revision and runtime parameters; a smoke fixture runs offline. **Completed 2026-08-08.**
- [x] **BASE-03**: The project documents and tests a standard data refresh and baseline maintenance workflow, including checksum refresh, quality review, split regeneration and artifact comparison. **Completed 2026-08-08.**

### Cross-scale scientific model (MODEL)

- [x] **MODEL-01**: An opt-in `MultiPLMEncoder` provides a common contract for Ankh39, ESM-2 and ProtT5 representations, projection/fusion, precomputed embeddings and explicit missing-weight/fallback provenance without changing the default single-scale model. **Completed 2026-08-08.**
- [x] **MODEL-02**: `CIGNNSignalBridge` consumes an explicit typed PPI/kinase-substrate graph, performs differentiable message passing and exposes the sensitivity matrix `S=(1-alpha)(I-alpha G')^-1`; missing graph edges must fail loudly unless an explicit research-only approximation flag is supplied. **Completed 2026-08-08.**
- [x] **MODEL-03**: `CellGraphCompassHead` decodes signal features into a delta-expression spectrum and perturbed cell-state logits, supports gene masks and graph edges, and exposes a numerically stable multi-task loss and metrics. **Completed 2026-08-08.**
- [x] **MODEL-04**: An opt-in cross-scale model composes protein/PTM conditioning, graph propagation and cell decoding, supports checkpoint round-trip/configuration, and emits model/data/graph/fallback provenance. **Completed 2026-08-08.**

### Verification and reporting (VERIFY)

- [x] **VERIFY-01**: Unit, integration and performance checks cover dimensions, gradients, deterministic splitting, quality failures, checkpoint round-trip, sensitivity-matrix invariants and CPU latency/resource measurements. **Completed 2026-08-08.**
- [x] **VERIFY-02**: A dated Markdown technical summary reports environment, parameters, execution flow, resource use, quantitative metrics, anomalies/resolutions, scientific interpretation, limitations and exact reproduction commands. **Completed 2026-08-08.**

## v2.2 Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 18 | Complete |
| DATA-02 | Phase 18 | Complete |
| BASE-01 | Phase 18 | Complete |
| BASE-02 | Phase 18 | Complete |
| BASE-03 | Phase 20 | Complete |
| MODEL-01 | Phase 19 | Complete |
| MODEL-02 | Phase 19 | Complete |
| MODEL-03 | Phase 19 | Complete |
| MODEL-04 | Phase 19 | Complete |
| VERIFY-01 | Phase 20 | Complete |
| VERIFY-02 | Phase 20 | Complete |

## v2.2 Out of Scope

| Feature | Reason |
|---------|--------|
| Downloading or redistributing restricted databases | Access and licensing must be confirmed by the data owner; the repository records manifests and local-import contracts only. |
| Claiming the original Ankh39/Cell-Graph-Compass papers were exactly reproduced | The repository implements an explicitly versioned, testable equivalent contract; scientific equivalence requires the original weights/data and separate validation. |
| Replacing the default PTM2CellNet/DAVF serving path | TD-01 is opt-in until real assets and biological acceptance tests exist. |
| Real-time mass-spec streaming, custom PTM database, GUI, API-key expansion | These remain cancelled project scope. |

## v2.2 Completion Evidence

Completion evidence is recorded in the phase verification files, executed test
results, baseline smoke output and dated technical summary; real-asset skips and
the boundary between engineering fixtures and biological evidence are explicit.

## Supplemental active proposal

The document [`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](../docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)
is a follow-up implementation proposal, not an additional completed requirement
in the 24-item v2.1/v2.2 acceptance set above. Its M0--M7 gates remain separately
tracked because Gate-0 needs owner-supplied donor data and Gate-E/Gate-4/Gate-5
need real assets and scientific acceptance. The dated status and evidence are
maintained in `docs/CURRENT_STATUS.md` and the repository-root
`project_analysis_20260913.md`; the older `project_analysis_20260901.md` reference
is historical and is not an active execution source.

## Active DAVF × PerturbGen acceptance track (2026-09-13)

This track updates the pending research contract without reopening the completed
24-item v2.1/v2.2 engineering requirements. Code facts and scientific evidence
remain separate: an existing interface or smoke run is not a biological PASS.

| ID | Active requirement | Status / evidence |
|---|---|---|
| A-01 | Gate-0 must use a real `normal/disease` raw-count cohort with explicit donor identity, canonical Ensembl IDs, frozen scVI gene order/embedding manifest and at least 3 shared donors. | **Open / blocked**: the 2026-09-13 audit found 0 compliant cohorts among 30 local scPerturb H5AD files (`outputs/perturbgen/spike/20260913_donor_audit/evidence.json`). The four-cohort IBD plan is a source design, not proof of eligibility. |
| A-02 | Training-only and held-out donor lists must be explicit, disjoint, provenance-bound and sufficient for the frozen M6 acceptance. | **Interface present, evidence open**: `train_latent_davf.py`, E2E and `donor_split.py` accept/bind donor lists; real lists, historical checkpoint provenance and held-out metrics are not verified. |
| A-03 | Direction evidence must record `context`, `intervention`, comparison baseline, research objective (`association`/`replication`/`reversal`) and source/cohort split. Classifier site presence, external candidate direction, observed donor-level disease−normal and DAVF intervention delta are separate fields. | **Open / contract update**: current bridge is `candidate_spec → scVI/PTMDirectionMapper → DAVF direction → three-way gate → evidence/invocation`; E2E defaults to report export and only explicit `--run-perturbgen` runs the six-stage preparation/execution chain; no global same-sign/flip rule and no claim that disease signature is intervention ground truth. |
| A-04 | Formal PerturbGen evaluation must keep `source_intervention=[src]` and `within_state=[tgt]+pert_tps` as two scenes and require both scenes for the existing AND verdict. | **Engineering interface present; formal evidence open**: one scene is only scene-specific exploration/utility, not universal treatment efficacy. |
| A-05 | E2E must automatically continue existing matched-null generation, candidate empirical-p aggregation, formal input isolation, unperturbed quality extraction and dual-path statistics, with lineage in the report. | **Interfaces present; orchestration open**: current report aggregates `stage_manifest` only and does not yet auto-run the statistical chain. |
| A-06 | Public prepare/reuse must be specified around a fixed cohort, vocabulary, training configuration and asset versions; candidate loops then run only the two perturbation/utility scenes. | **Open**: current `orchestrator.py` repeats tokenise/training/perturb/export/report per candidate and E2E uses independent candidate directories; same-directory resume is not cross-candidate reuse. Do not add a new hash/scheduler/compatibility framework to imply completion. |
| A-07 | Workflow A (gate → utility) and Workflow B (fixed encoder → frozen embedding asset → LatentDAVF retraining → Gate-E) must remain separate asset lifecycles. | **Open**: current base encoder export is fixed, does not consume candidate results or retrain a candidate checkpoint, and is not fed back into the current DAVF run. |
| A-08 | Formal release evidence requires real donor-level null, quality, dual-scene statistics plus Gate-E/Gate-4/Gate-5 evidence; smoke, synthetic, mock and bridge results remain engineering evidence. | **Open**: Gate-E ≥200 benchmark and formal real evidence have not been executed. |

### Active dependency order

1. Freeze direction semantics, research objective, evidence sources and
   train-only/held-out donor design.
2. Implement the public prepare/reuse and formal invocation boundary described in
   the proposal; the runner continues to execute a `StagePlan`, while the formal
   CLI requires an E2E report bound to a passing invocation.
3. Connect the existing null, quality, empirical-p/q and dual-scene interfaces to
   E2E reporting with complete lineage.
4. With owner-supplied assets, run Gate-0, donor-held-out DAVF/M6, Gate-E and
   Gate-4/Gate-5 real acceptance. Until then, report only engineering or
   scene-specific exploratory utility.

## Cancelled Requirements

These items are explicitly removed from project scope as of 2026-07-05 and must not be reintroduced into roadmap, phase plans, or agent task prompts unless the project owner reverses this decision.

- **V2-02**: 实时质谱流 / 实时质谱数据处理
- **V2-03**: 自定义 PTM 数据库 / 用户自带 PTM database catalog
- **V2-05**: GUI / Streamlit / Gradio / desktop interface
- **SEC-AUTH-APIKEY**: 新增或扩展 API key 功能；现有兼容性中间件可保留，但不再作为待实现需求规划

## Out of Scope

| Feature | Reason |
|---------|--------|
| KEGG/Reactome full implementation | Requires external service integration and operational maintenance — too complex for debt milestone |
| DAVF gene mode implementation | New feature, not debt — defer to v2.2 |
| New model architectures | Out of scope — this is stabilization only |
| Data directory population | Data files are runtime artifacts, not code debt |
| Real-time mass-spec streaming | Cancelled scope item; batch/offline workflows are sufficient |
| Custom PTM database | Cancelled scope item; use supported public sources and standard file imports |
| GUI | Cancelled scope item; supported interfaces are CLI, Python API, and FastAPI |
| API key feature expansion | Cancelled scope item; do not add new API-key auth work to plans |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TEST-01 | Phase 15 | Complete (2026-05-03, 15-01) |
| TEST-02 | Phase 15 | Complete (2026-05-03, 15-01) |
| TEST-03 | Phase 15 | Complete (2026-05-03, 15-01) |
| TEST-04 | Phase 17 | Complete (criteria revised to current scope: 1452 unit + 183 non-unit passed; 5+1 intentional env-guard skips; see v15 review §3.1) |
| DEPS-01 | Phase 15 | Complete (2026-05-04, 15-02; in requirements-analysis.txt) |
| DEPS-02 | Phase 15 | Complete (2026-05-04, 15-02; in requirements-analysis.txt) |
| DEPS-03 | Phase 15 | Complete (2026-05-04, 15-02; lion-pytorch confirmed used, declared in requirements-mamba.txt) |
| CODE-01 | Phase 16 | Complete (LazyImport re-raises; acceptance wording revised) |
| CODE-02 | Phase 16 | Complete (7 orphans deleted; only setup.py + shim remain) |
| CODE-03 | Phase 16 | Complete (root signaling_network.py is re-export shim; canonical impl in src/models/) |
| CONF-01 | Phase 17 | Complete (.gitignore whitelist + git ls-files confirms tracking) |
| CONF-02 | Phase 17 | Complete (logging_config.configure_default_logger + Lightning auto-logger) |
| CONF-03 | Phase 16 | Complete (_load_pathway_db warns + fallback instead of NotImplementedError) |

**Coverage:**
- v2.1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0
- All Complete as of v15 systematic review (2026-07-05).

**Note on acceptance-criteria drift:** Several requirements' success criteria
were written against an early codebase and were later superseded. The
implementation that satisfied each requirement matches the intent but not
always the literal original wording:
- CODE-01: implementation chose `LazyImport` (re-raise on access) over
  `logging.warning`. This is stricter — failures surface at first use rather
  than at import time — and is accepted as the canonical interpretation.
- CODE-03: implementation kept a re-export shim rather than deleting the
  root file entirely, to preserve backwards-compatible imports.
- CONF-03: implementation chose graceful fallback over an explicit
  "not yet implemented" message, since the fallback is itself a usable
  built-in pathway set.

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-09-13 — v2.2 formal requirements remain complete; active DAVF × PerturbGen acceptance track and its open evidence gates are documented above*
