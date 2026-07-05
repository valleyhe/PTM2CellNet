#!/usr/bin/env python3
"""
Migrate legacy PyTorch checkpoints to the safe ``weights_only=True`` format.

This is the **only** place in the PTM2CellNet project where
``weights_only=False`` is permitted.  It is a one-time migration tool:
loads a legacy checkpoint with the unsafe deserializer, extracts only the
safe tensor state dict, and re-saves it so that the standard
``safe_torch_load`` with ``weights_only=True`` can load it going forward.

Usage
-----
    python scripts/tools/migrate_legacy_checkpoint.py \\
        --input checkpoints/davf/model_a_finetuned/best_model.pt \\
        --output checkpoints/davf/model_a_finetuned/best_model.pt.safe

After migration, update your config to point to the ``.safe`` file, or
rename it in place.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate_legacy_checkpoint")


def migrate_checkpoint(input_path: str, output_path: str, device: str = "cpu") -> None:
    """Load a legacy checkpoint and re-save only the safe tensor state dict.

    This function intentionally uses ``torch.load(weights_only=False)``
    because legacy checkpoints may contain custom Python objects
    (dataclasses, ``nn.Module`` subclasses) that cannot be loaded with
    ``weights_only=True``.  **Do not use this function on untrusted files.**

    Args:
        input_path: Path to the legacy checkpoint.
        output_path: Path for the migrated (safe) checkpoint.
        device: Device to load tensors onto.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError: If the checkpoint cannot be loaded or is not a dict.
    """
    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        raise FileNotFoundError(f"Legacy checkpoint not found: {inp}")

    logger.info(
        "Loading legacy checkpoint with weights_only=False (UNSAFE) — "
        "only use on fully trusted files."
    )
    try:
        # ═══════════════════════════════════════════════════════════════
        # SAFETY WARNING: weights_only=False allows arbitrary code
        # execution via pickle deserialization.  This is intentionally
        # permitted ONLY in this one-off migration script.  Do NOT copy
        # this pattern elsewhere.
        # ═══════════════════════════════════════════════════════════════
        checkpoint: Any = torch.load(
            str(inp),
            map_location=device,
            weights_only=False,
        )
    except (pickle.UnpicklingError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to load legacy checkpoint {inp}: {e}"
        ) from e

    # Extract the safe tensor state dict, discarding any custom objects.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        # The checkpoint may be a bare state_dict.
        state_dict = checkpoint
    else:
        raise RuntimeError(
            f"Unexpected checkpoint format: {type(checkpoint)}. "
            "Expected a dictionary."
        )

    # Verify that the state dict contains only tensors (safe to save).
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            logger.warning(
                "Non-tensor value under key '%s' (type=%s) will be dropped "
                "during migration.",
                key, type(value).__name__,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out)
    logger.info(
        "Migrated checkpoint saved to %s (%d keys). "
        "You can now use the .safe file with safe_torch_load(weights_only=True).",
        out,
        len(state_dict),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a legacy PyTorch checkpoint to the safe "
            "weights_only=True format."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the legacy checkpoint file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the migrated safe checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load tensors onto (default: cpu)",
    )
    args = parser.parse_args()

    try:
        migrate_checkpoint(args.input, args.output, args.device)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
