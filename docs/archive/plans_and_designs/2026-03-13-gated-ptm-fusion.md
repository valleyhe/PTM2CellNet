# Gated PTM Fusion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `GatedPTMFusion` module to `src/models/ptm_modules.py` without modifying existing classes.

**Architecture:** Implement a new `nn.Module` that gates between sequence and PTM embeddings using a learned sigmoid gate, optional mask, dropout, residual, and LayerNorm. Keep it self-contained in `ptm_modules.py` and consistent with existing module style.

**Tech Stack:** Python, PyTorch (`torch`, `torch.nn`), typing (`Optional`, `cast`).

---

## Chunk 1: Standalone Module

### File Structure
- Modify: `src/models/ptm_modules.py`
- Spec: `docs/superpowers/specs/2026-03-13-gated-ptm-fusion-design.md`

### Task 1: Add `GatedPTMFusion` class

**Files:**
- Modify: `src/models/ptm_modules.py`

- [ ] **Step 1: Add the new class after existing classes**

```python
class GatedPTMFusion(nn.Module):
    """Gated PTM fusion module.

    This module uses a learnable gate to dynamically fuse sequence and PTM
    embeddings, enabling fine-grained feature interaction.

    Args:
        embed_dim: Feature dimension.
        dropout: Dropout probability.
        use_residual: Whether to use residual connection.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, use_residual: bool = True) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.use_residual = use_residual

    def forward(
        self,
        sequence_emb: torch.Tensor,
        ptm_emb: torch.Tensor,
        ptm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        combined = torch.cat([sequence_emb, ptm_emb], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        if ptm_mask is not None:
            gate = gate * ptm_mask.unsqueeze(-1)
        fused = gate * sequence_emb + (1.0 - gate) * ptm_emb
        fused = self.dropout(fused)
        if self.use_residual:
            fused = fused + sequence_emb
        output = self.layer_norm(fused)
        return output
```

- [ ] **Step 2: Run a quick shape check (no file changes)**

Run:
```bash
python - <<'PY'
import torch
from src.models.ptm_modules import GatedPTMFusion

module = GatedPTMFusion(embed_dim=128, dropout=0.1)
seq_emb = torch.randn(4, 50, 128)
ptm_emb = torch.randn(4, 50, 128)
output = module(seq_emb, ptm_emb)
assert output.shape == (4, 50, 128)
print("GatedPTMFusion smoke test passed")
PY
```
Expected: prints `GatedPTMFusion smoke test passed`.

- [ ] **Step 3: Commit**

```bash
git add src/models/ptm_modules.py
git commit -m "feat: add gated ptm fusion module"
```

## Plan Review Loop
- Dispatch the plan-document-reviewer subagent for Chunk 1 with the spec at `docs/superpowers/specs/2026-03-13-gated-ptm-fusion-design.md`.
- If issues are found, fix and re-dispatch until approved.
