#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Legacy wrapper for PTM site prediction.

This compatibility entry point preserves existing ``python scripts/batch_predict.py``
usage, but the canonical script name is ``scripts/predict_ptm_sites.py`` because
the tool predicts PTM sites rather than cell states.

Scope boundary (avoid confusion with the API ``/batch_predict`` endpoint):
    - This CLI predicts **PTM sites** (which residues are modified) and is a
      thin re-export of ``scripts/predict_ptm_sites.py``.
    - The HTTP API ``/batch_predict`` endpoint (``src/api/routes/predictions.py``)
      performs **cell-state** batch prediction from sequences + PTM sites.
    They are distinct workflows sharing only a name; do not use this script
      expecting cell-state batch inference -- use ``scripts/predict.py --input``
      or the API instead.

All CLI subcommands (``fasta``, ``single``, ``variant``) and classes
(``BatchPTMPredictor``, ``MultiTaskPTMPredictor``) are re-exported from
``scripts/predict_ptm_sites.py`` so that existing invocations of
``python scripts/batch_predict.py fasta -i ...`` continue to work unchanged.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict_ptm_sites import (  # noqa: F401
    BatchPTMPredictor,
    MultiTaskPTMPredictor,
    _build_parser,
    main,
    parse_args,
)


def batch_process_directory(
    input_dir: str,
    output_dir: str,
    model_path: str,
    model_type: str = "cnn_multitask",
    ptm_types: list[str] | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, dict]:
    """Run PTM site prediction on every FASTA file in a directory.

    Convenience function that discovers all ``*.fasta`` / ``*.fa`` files
    under *input_dir*, predicts PTM sites for each, and writes results
    to corresponding CSV files in *output_dir*.

    Args:
        input_dir: Directory containing input FASTA files.
        output_dir: Directory where per-file CSV results are written.
        model_path: Path to the trained model checkpoint.
        model_type: Model architecture (``"cnn_multitask"``,
            ``"esm2_t12"``, or ``"esm2_t6"``).
        ptm_types: List of PTM types to predict. Defaults to the model's
            built-in list.
        threshold: Probability threshold for site calling.
        device: Compute device (``"cpu"`` or ``"cuda"``).
        batch_size: Inference batch size.

    Returns:
        Dict mapping each input filename to the stats dict returned by
        :meth:`BatchPTMPredictor.predict_fasta`.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictor = BatchPTMPredictor(
        model_path=model_path,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
    )

    all_stats: dict[str, dict] = {}
    for fasta_file in sorted(input_path.glob("*.fasta")):
        out_file = output_path / f"{fasta_file.stem}_ptm_predictions.csv"
        stats = predictor.predict_fasta(
            fasta_path=str(fasta_file),
            output_path=str(out_file),
            ptm_types=ptm_types,
            threshold=threshold,
        )
        all_stats[fasta_file.name] = stats

    # Also pick up .fa files that weren't already matched as .fasta
    for fasta_file in sorted(input_path.glob("*.fa")):
        if fasta_file.name in all_stats:
            continue
        out_file = output_path / f"{fasta_file.stem}_ptm_predictions.csv"
        stats = predictor.predict_fasta(
            fasta_path=str(fasta_file),
            output_path=str(out_file),
            ptm_types=ptm_types,
            threshold=threshold,
        )
        all_stats[fasta_file.name] = stats

    return all_stats


if __name__ == "__main__":
    main()
