---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: Technical Debt & Test Stabilization
status: planning
last_updated: "2026-05-04T00:00:00.000Z"
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# PTM2CellNet State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core Value**: 提供端到端的蛋白质 PTM 分析与细胞状态预测能力，集成DAVF方向感知模型实现PTM→信号通路效应预测。

**Current Focus**: v2.1 Technical Debt & Test Stabilization

**Key Constraints**:
- PyTorch 2.7.1+cu118 / Lightning 2.5.1
- DAVF source: source_code.zip at project root
- DAVF checkpoints: model_checkpoints.zip at project root
- v1.0 + v2.0 tests continue passing with backward compatibility
- Default DAVF model: latent_davf_ibd_norman (10-dim latent, scVI)

---

## Shipped Milestones

### v2.0 DAVF Integration — SHIPPED 2026-05-03

**Phases**: 10-14 | **Plans**: 5 | **Tests added**: ~91
**Key**: DAVF module import, PTM direction mapping, inference wrapper, cascade fusion, E2E testing
**Archive**: `.planning/milestones/v2.0-ROADMAP.md`, `.planning/milestones/v2.0-REQUIREMENTS.md`

### v1.0 Core Architecture & Testing — SHIPPED 2026-04-04

**Phases**: 1-6 | **Plans**: 16 | **Tests**: ~959
**Key**: Stack upgrade, training optimization, analysis features, testing, coverage
**Archive**: `.planning/milestones/v1.0-ROADMAP.md`, `.planning/milestones/v1.0-REQUIREMENTS.md`

---

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-05-04 — Milestone v2.1 started

---

## Technical Debt

- 14 test failures (peft_config mock, Lightning API, pathway_integration)
- requirements.txt missing anndata, sspa
- evaluation/__init__.py silently swallows ImportError
- 7 orphaned root-level Python scripts not in src/scripts
- .gitignore whitelist missing configs/, docs/, Dockerfile, etc.
- Lightning Trainer has no logger configured
- Root-level duplicate files (signaling_network.py, pathway_knowledge_base.py)
- KEGG/Reactome loading stub unimplemented

---
*Last updated: 2026-05-04 — v2.1 Technical Debt milestone started*
