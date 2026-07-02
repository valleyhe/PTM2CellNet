# PTM2CellNet Fusion Type Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ptm_fusion_type` to `PTM2CellNet.__init__` and pass it to `PTMModule`.

**Architecture:** `PTM2CellNet` will accept `ptm_fusion_type` with default `"attention"` and pass it as `fusion_type` when constructing `PTMModule`.

**Tech Stack:** Python, PyTorch.

---

## Chunk 1: PTM2CellNet Parameter Plumb

### File Structure
- Modify: `src/models/architectures.py`
- Spec: `docs/superpowers/specs/2026-03-13-ptm2cellnet-fusion-type-design.md`

### Task 1: Add `ptm_fusion_type` parameter and pass-through

**Files:**
- Modify: `src/models/architectures.py`

- [ ] **Step 1: Update `PTM2CellNet.__init__` signature**

```python
    def __init__(
        self,
        ...,
        ptm_fusion_type: str = "attention",
        ...,
    ):
```

- [ ] **Step 2: Pass `fusion_type` into `PTMModule`**

```python
        self.ptm_module = PTMModule(
            ...,
            fusion_type=ptm_fusion_type,
        )
```

- [ ] **Step 3: Run inline validation checks**

Run:
```bash
python - <<'PY'
import torch
from src.models.architectures import PTM2CellNet

batch_size = 2
seq_len = 4
embed_dim = 8
num_classes = 3

# Default (attention)
model = PTM2CellNet(encoder_type="mamba")

# Explicit gated
model = PTM2CellNet(encoder_type="mamba", ptm_fusion_type="gated")

# Forward pass (use minimal dummy batch)
# NOTE: adjust keys to match PTM2CellNet expected input format
batch = {
    "sequence": torch.randint(0, 20, (batch_size, seq_len)),
    "ptm_types": torch.randint(0, 6, (batch_size, seq_len)),
    "ptm_positions": torch.randint(0, 1000, (batch_size, seq_len)),
    "ptm_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
}
output = model(batch)
assert "logits" in output
assert output["logits"].shape[0] == batch_size
print("PTM2CellNet fusion_type smoke test passed")
PY
```
Expected: prints `PTM2CellNet fusion_type smoke test passed`.

- [ ] **Step 4: Commit**

```bash
git add src/models/architectures.py
git commit -m "feat: add ptm_fusion_type to PTM2CellNet"
```

## Plan Review Loop
- Dispatch the plan-document-reviewer subagent for Chunk 1 with the spec at `docs/superpowers/specs/2026-03-13-ptm2cellnet-fusion-type-design.md`.
- If issues are found, fix and re-dispatch until approved.
