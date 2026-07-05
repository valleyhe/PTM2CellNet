# PTM Module Fusion Type Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `fusion_type` parameter to `PTMModule` to select between attention and gated fusion with validation.

**Architecture:** `PTMModule` will validate `fusion_type`, then create either a stack of `PTMAttention` layers or `GatedPTMFusion` layers. The forward path remains unchanged.

**Tech Stack:** Python, PyTorch (`torch`, `torch.nn`), typing (`Optional`, `cast`).

---

## Chunk 1: Configurable Fusion in PTMModule

### File Structure
- Modify: `src/models/ptm_modules.py`
- Spec: `docs/superpowers/specs/2026-03-13-ptm-fusion-type-design.md`

### Task 1: Add `fusion_type` parameter and validation

**Files:**
- Modify: `src/models/ptm_modules.py`

- [ ] **Step 1: Update `PTMModule.__init__` signature and validate `fusion_type`**

```python
    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int,
        max_position: int = 1000,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "attention",
    ):
        super().__init__()
        allowed = {"attention", "gated"}
        if fusion_type not in allowed:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}. Expected one of {sorted(allowed)}")
```

- [ ] **Step 2: Instantiate layers based on `fusion_type`**

```python
        self.embedding = PTMEmbedding(num_ptm_types, embed_dim, max_position=max_position, dropout=dropout)
        if fusion_type == "attention":
            self.layers = nn.ModuleList(
                [PTMAttention(embed_dim, num_attention_heads, dropout=dropout) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList([GatedPTMFusion(embed_dim, dropout=dropout) for _ in range(num_layers)])
```

- [ ] **Step 3: Run inline validation checks**

Run:
```bash
python - <<'PY'
import torch
from src.models.ptm_modules import PTMModule

seq_emb = torch.randn(2, 4, 8)
ptm_types = torch.randint(0, 6, (2, 4))
ptm_pos = torch.randint(0, 1000, (2, 4))

# Default (attention)
module_attn = PTMModule(num_ptm_types=5, embed_dim=8)
out_attn = module_attn(seq_emb, ptm_types, ptm_pos)
assert out_attn.shape == (2, 4, 8)

# Gated
module_gated = PTMModule(num_ptm_types=5, embed_dim=8, fusion_type="gated")
out_gated = module_gated(seq_emb, ptm_types, ptm_pos)
assert out_gated.shape == (2, 4, 8)

# Invalid
try:
    PTMModule(num_ptm_types=5, embed_dim=8, fusion_type="invalid")
except ValueError:
    pass
else:
    raise AssertionError("Expected ValueError for invalid fusion_type")

print("PTMModule fusion_type smoke test passed")
PY
```
Expected: prints `PTMModule fusion_type smoke test passed`.

- [ ] **Step 4: Commit**

```bash
git add src/models/ptm_modules.py
git commit -m "feat: add fusion_type to PTMModule"
```

## Plan Review Loop
- Dispatch the plan-document-reviewer subagent for Chunk 1 with the spec at `docs/superpowers/specs/2026-03-13-ptm-fusion-type-design.md`.
- If issues are found, fix and re-dispatch until approved.
