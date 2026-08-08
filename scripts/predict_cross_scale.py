#!/usr/bin/env python3
"""Run offline cross-scale batch inference from a self-describing artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cross_scale_dataset import CrossScaleNPZDataset  # noqa: E402
from src.inference.cross_scale_predictor import (  # noqa: E402
    CrossScaleArtifactError,
    CrossScalePredictor,
    load_cross_scale_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model, manifest, _ = load_cross_scale_artifact(args.artifact, device=args.device)
        dataset = CrossScaleNPZDataset(
            args.data,
            backbone_names=model.protein_encoder.backbone_names,
            require_targets=False,
        )
        predictor = CrossScalePredictor(model, manifest, device=args.device)
        result = predictor.predict_dataset(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            sample_id=np.asarray(result["sample_id"]),
            delta_expression=result["delta_expression"].numpy(),
            cell_state_logits=result["cell_state_logits"].numpy(),
            cell_state_probabilities=result["cell_state_probabilities"].numpy(),
            cell_state_index=result["cell_state_index"].numpy(),
            cell_state_label=np.asarray(result["cell_state_label"]),
        )
        provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
        provenance_path.write_text(
            json.dumps(result["provenance"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "ok": True,
            "sample_count": len(result["sample_id"]),
            "output": str(args.output),
            "provenance": str(provenance_path),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (CrossScaleArtifactError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
