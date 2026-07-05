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

## v2.2 Requirements (Deferred)

- **V22-01**: DAVF 端到端微调
- **V22-02**: scVI decode 集成
- **V22-03**: 通路知识库上下文相关映射
- **V22-04**: 文档完善
- **V22-05**: 更多PTM类型支持

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
*Last updated: 2026-07-06 — Phase 15/16/17 marked Complete after v15 systematic review; TEST-04 criteria revised to current test scope*
