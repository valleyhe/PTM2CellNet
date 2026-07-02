# Gated PTM Fusion (Step 1) Design

Date: 2026-03-13

## Goal
Add a standalone `GatedPTMFusion` class to `src/models/ptm_modules.py` without modifying existing classes. This enables gated fusion of sequence and PTM embeddings with an optional mask and residual normalization.

## Scope
- In scope: Implement `GatedPTMFusion` as a new `nn.Module` class.
- Out of scope: Integrating the module into `PTMModule` or adding configuration switches (planned for Step 2).

## Architecture
Single module in `ptm_modules.py` that mirrors existing module style:
- Linear gate projection over concatenated embeddings
- Elementwise gate-based fusion
- Dropout, optional residual, LayerNorm

## API
```
class GatedPTMFusion(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, use_residual: bool = True) -> None
    def forward(
        self,
        sequence_emb: torch.Tensor,  # [B, L, D]
        ptm_emb: torch.Tensor,        # [B, L, D]
        ptm_mask: Optional[torch.Tensor] = None  # [B, L]
    ) -> torch.Tensor:  # [B, L, D]
```

## Data Flow
1. Concatenate `sequence_emb` and `ptm_emb` along the feature dimension.
2. Project to gate logits and apply sigmoid.
3. If `ptm_mask` is provided, set gate to zero where mask is 0.
4. Fuse: `gate * sequence_emb + (1 - gate) * ptm_emb`.
5. Apply dropout.
6. If `use_residual`, add `sequence_emb`.
7. Apply LayerNorm and return.

## Error Handling
- No explicit error handling beyond tensor shape expectations; consistent with existing module patterns.
- Mask is optional and only applied when provided.

## Testing
- Manual shape check as provided in the task description.
- No new automated tests in Step 1 (Step 2 may add coverage when integration occurs).

## Compatibility
- Uses existing imports and typing style: `Optional`, `cast`, `torch`, `nn`.
- Preserves file style and does not modify existing classes.
