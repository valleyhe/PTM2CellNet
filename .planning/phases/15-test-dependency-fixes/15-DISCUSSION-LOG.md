# Phase 15: Test & Dependency Fixes - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 15-test-dependency-fixes
**Mode:** --auto (fully autonomous)
**Areas discussed:** peft_config mock targets, Lightning API import path, pathway_integration test isolation, sspa dependency classification, lion-pytorch audit

---

## peft_config Mock Strategy (TEST-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Patch `_get_peft_model_fn` | Update mock targets to match the actual internal alias used in peft_config.py | ✓ |
| Patch at peft package level | Mock `peft.get_peft_model` before import | |

**Auto-selected:** Patch `_get_peft_model_fn`
**Rationale:** Source file at `src/training/peft_config.py:131` uses `_get_peft_model_fn(base_model, lora_config)` internally. Tests should mock at the point of use, not at the external package.

---

## Lightning API Import Path (TEST-02)

| Option | Description | Selected |
|--------|-------------|----------|
| `lightning.pytorch.loggers.TensorBoardLogger` | Lightning 2.x canonical namespace | ✓ |
| Keep `pytorch_lightning.loggers` | Legacy namespace | |

**Auto-selected:** `lightning.pytorch.loggers.TensorBoardLogger`
**Rationale:** Codebase uses Lightning 2.5.1. The `pytorch_lightning` namespace is deprecated since 2.0.

---

## pathway_integration Test Isolation (TEST-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Fixture-based mock scoping with per-test cleanup | Convert module-level sys.modules injection to pytest fixtures | ✓ |
| Class-level setup/teardown | Use setUp/tearDown within test classes | |

**Auto-selected:** Fixture-based mock scoping
**Rationale:** Module-level `sys.modules['sspa'] = mock_sspa` persists across tests causing state leakage. Fixtures provide proper scoping and cleanup. Also need to reset mock call counts/return values between tests.

---

## sspa Dependency Classification (DEPS-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Optional dependency with comment | Add to requirements.txt with `# optional` annotation | ✓ |
| Required dependency | Add as normal requirement | |
| pip extras syntax | Use `[pathway]` extras | |

**Auto-selected:** Optional dependency with comment
**Rationale:** Code already has `try: import sspa` guard in `src/analysis/pathway_integration.py`. Core functionality works without it. Keep as optional to avoid blocking install for users who don't need pathway integration.

---

## lion-pytorch Audit (DEPS-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep in requirements.txt | Confirmed conditionally used | ✓ |
| Remove from requirements.txt | Not used | |

**Auto-selected:** Keep in requirements.txt
**Rationale:** Found at `src/training/lightning_module.py:416`: `from lion_pytorch import Lion` with try/except fallback to AdamW. It IS a real dependency used for the Lion optimizer.

---

## Claude's Discretion

- Exact fixture implementation for pathway_integration tests
- Whether sspa uses extras syntax or comment annotation
- Test assertion style consistency
- Error message formatting in tests

## Deferred Ideas

None — discussion stayed within phase scope.
