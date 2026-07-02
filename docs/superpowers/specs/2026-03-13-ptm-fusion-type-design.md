# PTM Module Fusion Type (Step 2) Design

Date: 2026-03-13

## Goal
Extend `PTMModule` to support a configurable fusion mechanism with `fusion_type` selecting between attention-based fusion and gated fusion, while preserving the default behavior.

## Scope
- In scope: Add `fusion_type` parameter to `PTMModule`, validate it, and instantiate either `PTMAttention` or `GatedPTMFusion` layers.
- Out of scope: Any other architectural changes or new modules.

## API
```
class PTMModule(nn.Module):
    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int,
        max_position: int = 1000,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "attention",
    )
```

## Behavior
- `fusion_type="attention"`: current behavior using `PTMAttention` layers.
- `fusion_type="gated"`: use `GatedPTMFusion` layers.
- Invalid values raise `ValueError`.
- Forward flow unchanged; each layer is called with `(x, ptm_emb, ptm_mask)`.

## Data Flow
1. Embed PTM types and positions into `ptm_emb`.
2. Iterate over `self.layers` (attention or gated) and fuse with `sequence_emb`.
3. Return final fused embedding.

## Error Handling
- Validate `fusion_type` against allowed values and raise `ValueError` on mismatch.

## Testing
- Inline smoke check:
  - Default `fusion_type` (attention) works.
  - `fusion_type="gated"` works.
  - Invalid `fusion_type` raises `ValueError`.

## Compatibility
- Default is `"attention"`, preserving existing behavior and positional argument compatibility.
