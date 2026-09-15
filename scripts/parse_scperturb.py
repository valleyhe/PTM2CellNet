#!/usr/bin/env python3
"""End-to-end parser for scPerturb h5ad studies (F-05, 2026-08-17).

Closes the residual F-05 gap recorded in the 2026-08-16 analysis: the
registration script ``import_scperturb.py`` only probes h5ad structure,
it does not consume the expression matrix.  This script consumes each
registered study h5ad and produces standardized research artifacts
aligned with the GSE133344/GSE90546 bundle shape:

    <output>/<study>_expression.npz         (cells x genes sparse matrix + obs_index)
    <output>/<study>_delta_expression.npz  (per-perturbation Δ vs control mean)
    <output>/<study>_perturbations.tsv     (perturbation_id, n_cells, perturbation_type)

Design notes
------------
* ``anndata`` non-backed read — the 30 study files are individually
  small enough to load; we keep the implementation simple rather than
  streaming chunk-by-chunk across the file boundary.
* Per-study control identification: ``perturbation`` column values
  ``"control"`` / ``"control_"`` / empty / NaN → control pool; all other
  values become candidate perturbation IDs (also consults
  ``perturbation_type`` to skip non-targeting controls).
* Skips studies without a ``perturbation`` column or with < 2 unique
  perturbation labels (cannot compute Δ); reports the reason in the
  summary.
* Fail-soft per study: a corrupt / unparseable h5ad is reported with
  its error and the remaining studies still parse.
* Pure read-only on ``data/raw/``; writes only under ``--output``.

Usage::

    python scripts/parse_scperturb.py --study DatlingerBock2017 \\
        --output data/processed/scperturb/
    python scripts/parse_scperturb.py  # all 30 studies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = "data/raw/scperturb"
DEFAULT_OUTPUT = "data/processed/scperturb"
STUDY_MANIFEST = "scperturb_study_manifest.json"

# Values in ``perturbation`` obs column that mark control cells.
_CONTROL_LABELS = frozenset({"control", "control_", "non-targeting", "non_targeting", "nt", ""})

# ``perturbation_type`` values that mark non-targeting / control cells.
_CONTROL_TYPE_LABELS = frozenset({"control", "control_", "non-targeting", "nt", "unedited"})


class ScPerturbParseError(RuntimeError):
    """Raised when a single study cannot be parsed (fail-soft recoverable)."""


def _is_control_cell(perturbation: str, perturbation_type: str) -> bool:
    """Return True iff a cell should be treated as control."""
    p = "" if perturbation is None else str(perturbation).strip().lower()
    pt = "" if perturbation_type is None else str(perturbation_type).strip().lower()
    if p in _CONTROL_LABELS:
        return True
    if pt in _CONTROL_TYPE_LABELS:
        return True
    return False


def _normalize_id(value: Any) -> str:
    """Stable perturbation ID — strip + NaN-safe."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _ensure_dense_or_csr(matrix) -> sp.csr_matrix:
    """Return the X matrix as CSR (load sparse as-is, densify otherwise)."""
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(np.asarray(matrix))


def _delta_expression(
    matrix: sp.csr_matrix,
    obs: pd.DataFrame,
    gene_symbols: Sequence[str],
) -> Tuple[sp.csr_matrix, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-perturbation Δ = mean(X[target]) - mean(X[control]).

    Returns (delta, perturbation_ids, n_cells, control_mean, control_mask_count).
    ``delta`` has rows = perturbation IDs (only non-control), columns = genes.
    Genes with all-zero expression across the study are dropped (no signal).
    """
    if "perturbation" not in obs.columns:
        raise ScPerturbParseError("obs missing 'perturbation' column")
    perturb = obs["perturbation"].to_numpy()
    ptype = obs["perturbation_type"].to_numpy() if "perturbation_type" in obs.columns else np.array([""] * len(obs))

    control_mask = np.fromiter(
        (_is_control_cell(p, t) for p, t in zip(perturb, ptype, strict=True)),
        dtype=bool,
        count=len(perturb),
    )
    if not control_mask.any():
        raise ScPerturbParseError("no control cells identified")
    control_mean = np.asarray(matrix[control_mask].mean(axis=0)).flatten()

    # Group cells by (normalized) perturbation ID, drop empties.
    labels = [_normalize_id(p) for p in perturb]
    unique_ids: List[str] = []
    for label in labels:
        if label and not _is_control_cell(label, ""):
            if label not in unique_ids:
                unique_ids.append(label)
    if len(unique_ids) < 1:
        raise ScPerturbParseError("no target perturbations found (only control pool)")

    rows: List[np.ndarray] = []
    cell_counts: List[int] = []
    for pid in unique_ids:
        target_mask = np.fromiter(
            ((not _is_control_cell(l, t)) and l == pid for l, t in zip(labels, ptype, strict=True)),
            dtype=bool,
            count=len(labels),
        )
        n_cells = int(target_mask.sum())
        if n_cells < 1:
            continue
        target_mean = np.asarray(matrix[target_mask].mean(axis=0)).flatten()
        rows.append(target_mean - control_mean)
        cell_counts.append(n_cells)

    if not rows:
        raise ScPerturbParseError("all perturbations had zero cells after filtering")
    delta = sp.csr_matrix(np.vstack(rows))
    return (
        delta,
        unique_ids[: len(cell_counts)],
        np.asarray(cell_counts),
        control_mean,
        np.array([int(control_mask.sum())]),
    )


def parse_study(h5ad_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Parse one h5ad study; write NPZ/TSV artifacts; return a per-study summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not h5ad_path.is_file():
        raise ScPerturbParseError(f"missing h5ad: {h5ad_path}")
    study_id = h5ad_path.stem
    try:
        adata = ad.read_h5ad(h5ad_path)
    except (OSError, ValueError, KeyError) as exc:
        raise ScPerturbParseError(f"failed to read h5ad: {exc}") from exc

    if adata.n_obs < 2:
        raise ScPerturbParseError(f"too few cells ({adata.n_obs})")
    matrix = _ensure_dense_or_csr(adata.X)
    gene_symbols = list(adata.var.index.astype(str))

    try:
        delta, perturbation_ids, n_cells, control_mean, control_count = _delta_expression(
            matrix, adata.obs.copy(), gene_symbols
        )
    except ScPerturbParseError:
        raise
    except (ValueError, np.AxisError) as exc:
        raise ScPerturbParseError(f"delta computation failed: {exc}") from exc

    expression_path = output_dir / f"{study_id}_expression.npz"
    delta_path = output_dir / f"{study_id}_delta_expression.npz"
    perturbations_path = output_dir / f"{study_id}_perturbations.tsv"

    sp.save_npz(
        expression_path,
        matrix,
    )
    np.savez_compressed(
        delta_path,
        delta=delta.toarray(),
        perturbation_ids=np.asarray(perturbation_ids, dtype="U"),
        gene_symbols=np.asarray(gene_symbols, dtype="U"),
        control_mean=control_mean,
        n_cells=n_cells,
        control_cell_count=control_count,
    )
    perturbations_path.write_text(
        "perturbation_id\tn_cells\n"
        + "".join(f"{pid}\t{cnt}\n" for pid, cnt in zip(perturbation_ids, n_cells, strict=True)),
        encoding="utf-8",
    )

    return {
        "study_id": study_id,
        "status": "parsed",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_nonzero": int(matrix.nnz),
        "n_perturbations": len(perturbation_ids),
        "n_control_cells": int(control_count[0]) if control_count.size else 0,
        "expression_npz": str(expression_path),
        "delta_expression_npz": str(delta_path),
        "perturbations_tsv": str(perturbations_path),
    }


def parse_studies(
    raw_root: Path,
    output_dir: Path,
    studies: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Parse all (or selected) studies under raw_root; return aggregate report.

    Fail-soft per study — a study failure is recorded under ``errors`` while
    the remaining studies continue.  Returns a dict suitable for JSON dump.
    """
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    # Optional: prefer the manifest's ``studies`` list when no filter given
    # and the manifest exists, so we honour the registration order.
    if studies:
        targets = [s if s.endswith(".h5ad") else f"{s}.h5ad" for s in studies]
    else:
        manifest_path = output_dir / STUDY_MANIFEST
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                # Take filenames from manifest studies; fall back to raw glob.
                targets = []
                for entry in manifest.get("studies", []):
                    name = entry.get("file")
                    if name and (raw_root / name).is_file():
                        targets.append(name)
                if not targets:
                    targets = sorted(p.name for p in raw_root.glob("*.h5ad"))
            except (OSError, ValueError, KeyError):
                targets = sorted(p.name for p in raw_root.glob("*.h5ad"))
        else:
            targets = sorted(p.name for p in raw_root.glob("*.h5ad"))

    parsed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for name in targets:
        try:
            summary = parse_study(raw_root / name, output_dir)
            parsed.append(summary)
        except ScPerturbParseError as exc:
            errors.append({"file": name, "status": "skipped", "reason": str(exc)})
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"file": name, "status": "error", "reason": str(exc)})
    return {
        "schema_version": "ptm2cellnet.scperturb.parse.v1",
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "n_targets": len(targets),
        "n_parsed": len(parsed),
        "n_errors": len(errors),
        "parsed": parsed,
        "errors": errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="raw h5ad directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output directory for parsed artifacts")
    parser.add_argument(
        "--study",
        action="append",
        default=[],
        help="limit to specific study filename(s); repeatable; default: all registered",
    )
    parser.add_argument("--manifest", default="", help="write JSON summary to this path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output)
    studies = args.study or None

    try:
        report = parse_studies(raw_root, output_dir, studies=studies)
    except FileNotFoundError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    if args.manifest:
        Path(args.manifest).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "n_targets": report["n_targets"],
                "n_parsed": report["n_parsed"],
                "n_errors": report["n_errors"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["n_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
