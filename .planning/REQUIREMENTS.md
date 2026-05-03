# Requirements: PTM2CellNet v2.1

**Defined:** 2026-05-04
**Core Value:** Fix all failing tests,补齐缺失依赖, clean up orphaned code, improve error handling — return codebase to healthy state.

## v2.1 Requirements

### Test Stabilization (TEST)

- [ ] **TEST-01**: Fix 3 peft_config tests — update mock targets to match current `src/training/peft_config.py` API (replace `get_peft_model` with actual function name)
- [ ] **TEST-02**: Fix 1 Lightning API test — update `train_pretrained_script.py` test to use correct `lightning.pytorch.loggers.TensorBoardLogger` import path
- [ ] **TEST-03**: Fix 10 pathway_integration tests — resolve state leakage between tests, align mock interfaces with actual KEGG/Reactome stub behavior
- [ ] **TEST-04**: Achieve 830/830 tests passing (100% pass rate) with no skipped failures

### Dependency Management (DEPS)

- [ ] **DEPS-01**: Add `anndata` to requirements.txt — imported by `src/integration/genki/perturbation.py` and `reference_data.py` but not declared
- [ ] **DEPS-02**: Add `sspa` to requirements.txt as optional dependency — used by `src/analysis/pathway_integration.py` with try/except guard
- [ ] **DEPS-03**: Audit `lion-pytorch` — listed in requirements.txt but never imported in src/; verify if used by training configs or remove

### Code Health (CODE)

- [ ] **CODE-01**: Fix `evaluation/__init__.py` — replace bare `except ImportError: pass` with `logging.warning()` so missing dependencies are visible at runtime instead of silently exporting `None`
- [ ] **CODE-02**: Consolidate 7 orphaned root-level Python scripts — decide fate for each (move to `scripts/`, integrate into `src/`, or delete):
  - `esm2_encoder.py` (370 lines)
  - `train_esm2.py` (388 lines)
  - `verify_esm_fix.py` (98 lines)
  - `search_tips.py` (5 lines)
  - `test_uniprot_fetch.py` (146 lines)
  - `create_tech_doc.py` (1442 lines)
  - `pathway_knowledge_base.py` (608 lines, duplicate of `src/data/extended_pathway_kb.py`)
- [ ] **CODE-03**: Resolve duplicate root-level `signaling_network.py` (632 lines) vs `src/models/signaling_network.py` (507 lines) — decide which version is canonical, remove or archive the other

### Configuration & Infrastructure (CONF)

- [ ] **CONF-01**: Fix `.gitignore` whitelist — add `configs/`, `docs/`, `Dockerfile`, `docker-compose.yml`, `setup.py`, `setup.cfg`, `pytest.ini`, `.coveragerc`, `.pylintrc`, `API_DOCUMENTATION.md`, `CHANGELOG.md`, `LICENSE` to git tracking
- [ ] **CONF-02**: Add default Logger configuration for Lightning Trainer — suppress "no logger configured" warnings in training scripts and configs
- [ ] **CONF-03**: Clarify KEGG/Reactome loading stubs — improve `_load_pathway_db()` error messages in both `signaling_network.py` and `src/models/signaling_network.py` to indicate feature is not yet implemented and provide guidance

## v2.2 Requirements (Deferred)

- **V22-01**: DAVF 端到端微调
- **V22-02**: scVI decode 集成
- **V22-03**: 通路知识库上下文相关映射
- **V22-04**: 文档完善
- **V22-05**: 更多PTM类型支持

## Out of Scope

| Feature | Reason |
|---------|--------|
| KEGG/Reactome full implementation | Requires API keys and external service integration — too complex for debt milestone |
| DAVF gene mode implementation | New feature, not debt — defer to v2.2 |
| New model architectures | Out of scope — this is stabilization only |
| Data directory population | Data files are runtime artifacts, not code debt |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TEST-01 | Phase 15 | Pending |
| TEST-02 | Phase 15 | Pending |
| TEST-03 | Phase 15 | Pending |
| TEST-04 | Phase 17 | Pending |
| DEPS-01 | Phase 15 | Pending |
| DEPS-02 | Phase 15 | Pending |
| DEPS-03 | Phase 15 | Pending |
| CODE-01 | Phase 16 | Pending |
| CODE-02 | Phase 16 | Pending |
| CODE-03 | Phase 16 | Pending |
| CONF-01 | Phase 17 | Pending |
| CONF-02 | Phase 17 | Pending |
| CONF-03 | Phase 16 | Pending |

**Coverage:**
- v2.1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 — traceability mapped to Phases 15-17*
