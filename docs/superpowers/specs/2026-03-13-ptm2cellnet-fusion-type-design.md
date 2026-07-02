# PTM2CellNet Fusion Type (Step 3) Design

Date: 2026-03-13

## Goal
Expose PTM fusion selection at the architecture level by adding a `ptm_fusion_type` parameter to `PTM2CellNet.__init__` and passing it to `PTMModule`.

## Scope
- In scope: Modify `PTM2CellNet.__init__` in `src/models/architectures.py` to accept `ptm_fusion_type` and pass it to `PTMModule`.
- Out of scope: Any changes to other methods, files, or new test files.

## API
```
class PTM2CellNet(nn.Module):
    def __init__(..., ptm_fusion_type: str = "attention", ...)
```

## Behavior
- Default `ptm_fusion_type="attention"` preserves backward compatibility.
- Passes `fusion_type=ptm_fusion_type` into `PTMModule` constructor.

## Validation
Inline smoke check:
1. Default constructor works.
2. Explicit `ptm_fusion_type="gated"` works.
3. Forward pass produces `output["logits"]` with expected shape.

## Compatibility
- No change to other methods or modules.
- Defaults preserve current behavior.
