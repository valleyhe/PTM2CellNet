#!/usr/bin/env python3
"""APOE anchor backtest for external perturbation evidence (先证后用, 2026-09-17).

Preregistered criteria (fixed before running; pass requires ALL of 1-3):

1. GEARS direction (primary): among the shared readouts, the top-100 genes by
   |predicted delta| must agree in sign with the Frangieh measured APOE-KO
   log-fold-change in >= 0.59 of cases (binomial p < 0.05 at n=100).
2. Whole-shared-readout Spearman rho > 0 with p < 0.05 (supporting).
3. Mean-shift baseline control: the same top-100 criterion computed for a
   fixed "average training perturbation" prediction must be strictly lower
   than the GEARS criterion (GEARS must beat the naive baseline).

Geneformer influence-spectrum overlap is reported as exploratory (no pass
line): hypergeometric fold-enrichment of top-100 overlap with measured
|log2FC| top-100.

Ground truth: FrangiehIzar2021_RNA (APOE CRISPR KO, 1,127 cells vs 57,605
controls), donor-free co-culture — a same-cell-line (K562-coculture) sanity
anchor, NOT a disease-context validation. Failing the backtest means the
evidence source must not enter lineage consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, hypergeom, spearmanr

TOP_N = 100
PASS_FRACTION = 0.59  # binomial p<0.05 at n=100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gears-manifest", type=Path, required=True)
    parser.add_argument("--geneformer-manifest", type=Path, default=None)
    parser.add_argument(
        "--truth-h5ad",
        type=Path,
        default=Path("data/raw/scperturb/FrangiehIzar2021_RNA.h5ad"),
        help="Frangieh RNA h5ad with perturbation obs and ensembl_id var",
    )
    parser.add_argument("--truth-gene", default="APOE")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_asset_matrix(manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    import anndata as ad

    manifest = json.loads(manifest_path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    h5ad_path = Path(manifest["predictions_h5ad"])
    if not h5ad_path.is_absolute():
        h5ad_path = manifest_path.parent / h5ad_path
    adata = ad.read_h5ad(h5ad_path)
    symbols = adata.obs["gene_symbol"].astype(str).tolist()
    frame = pd.DataFrame(np.asarray(adata.X), index=symbols, columns=[str(v) for v in adata.var_names])
    return frame, manifest


def _measured_log2fc(truth_h5ad: Path, gene: str) -> pd.Series:
    import anndata as ad

    adata = ad.read_h5ad(truth_h5ad.expanduser().resolve(strict=True))
    perturbation = adata.obs["perturbation"].astype(str).to_numpy()
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    counts = counts.tocsr()
    library = np.asarray(counts.sum(axis=1)).ravel()
    # per-cell library scaling then log2(x + 1); zeros stay zero so the sparse
    # structure is preserved
    scaled = counts.multiply((1e4 / library)[:, None]).tocsr()
    scaled.data = np.log2(scaled.data + 1.0)
    ko_mask = perturbation == gene
    ctrl_mask = perturbation == "control"
    ko = np.asarray(scaled[ko_mask].mean(axis=0)).ravel()
    ctrl = np.asarray(scaled[ctrl_mask].mean(axis=0)).ravel()
    delta = ko - ctrl
    return pd.Series(delta, index=pd.Index(adata.var["ensembl_id"].astype(str).to_numpy()))


def _mean_shift_baseline(norman_h5ad: Path) -> pd.Series:
    """Average measured perturbation effect across all Norman training conditions."""

    import anndata as ad

    adata = ad.read_h5ad(norman_h5ad.expanduser().resolve(strict=True))
    condition = adata.obs["condition"].astype(str).to_numpy()
    ctrl_mean = np.asarray(adata.X[condition == "ctrl"].mean(axis=0)).ravel()
    pert_mask = condition != "ctrl"
    pert_mean = np.asarray(adata.X[pert_mask].mean(axis=0)).ravel()
    return pd.Series(pert_mean - ctrl_mean, index=[str(v) for v in adata.var_names])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gears_matrix, gears_manifest = _load_asset_matrix(args.gears_manifest)
    predicted = gears_matrix.loc[args.truth_gene] if args.truth_gene in gears_matrix.index else None
    if predicted is None:
        print(f"[backtest] gene {args.truth_gene} absent from the GEARS asset", file=sys.stderr)
        return 1

    measured = _measured_log2fc(args.truth_h5ad, args.truth_gene)
    shared = measured.index.intersection(predicted.index)
    shared = shared[(measured.loc[shared] != 0) | (predicted.loc[shared] != 0)]
    if len(shared) < TOP_N:
        print(f"[backtest] only {len(shared)} shared readouts; insufficient for top-{TOP_N}", file=sys.stderr)
        return 1

    pred = predicted.loc[shared].to_numpy(dtype=float)
    truth = measured.loc[shared].to_numpy(dtype=float)
    top_idx = np.argsort(-np.abs(pred))[:TOP_N]
    agree = float(np.mean(np.sign(pred[top_idx]) == np.sign(truth[top_idx])))
    binom_p = float(binomtest(int(agree * TOP_N), TOP_N, 0.5).pvalue)
    rho, rho_p = spearmanr(pred, truth)

    baseline = _mean_shift_baseline(Path("data/raw/gears/norman/perturb_processed.h5ad"))
    baseline_shared = baseline.reindex(shared).to_numpy(dtype=float)
    base_top = np.argsort(-np.abs(baseline_shared))[:TOP_N]
    base_agree = float(np.mean(np.sign(baseline_shared[base_top]) == np.sign(truth[base_top])))

    geneformer_overlap = None
    if args.geneformer_manifest is not None:
        gf_matrix, _ = _load_asset_matrix(args.geneformer_manifest)
        if args.truth_gene in gf_matrix.index:
            influence = gf_matrix.loc[args.truth_gene]
            influence_shared = influence.reindex(shared).to_numpy(dtype=float)
            # cos -> 1 means unchanged; influence strength = 1 - cos
            strength = 1.0 - influence_shared
            pred_top = set(np.argsort(-np.abs(pred))[:TOP_N])
            strength_top = set(np.argsort(-np.abs(strength))[:TOP_N])
            truth_top = set(np.argsort(-np.abs(truth))[:TOP_N])
            overlap = pred_top & strength_top & truth_top
            expected = hypergeom.mean(len(shared), len(pred_top & truth_top), len(strength_top))
            geneformer_overlap = {
                "triple_overlap": len(overlap),
                "expected_under_random": float(expected),
                "fold_enrichment": (len(overlap) / expected) if expected > 0 else None,
            }

    criteria = {
        "c1_gears_top100_sign_agreement": {
            "value": agree,
            "threshold": PASS_FRACTION,
            "pass": agree >= PASS_FRACTION,
            "binom_p": binom_p,
        },
        "c2_spearman_all_shared": {
            "rho": float(rho),
            "p": float(rho_p),
            "pass": rho > 0 and rho_p < 0.05,
        },
        "c3_beats_mean_shift_baseline": {
            "gears_agreement": agree,
            "baseline_agreement": base_agree,
            "pass": agree > base_agree,
        },
    }
    for item in criteria.values():
        item["pass"] = bool(item["pass"])
    verdict = "pass" if all(item["pass"] for item in criteria.values()) else "fail"

    payload = {
        "schema_version": "ptm2cellnet.apoe-anchor-backtest/v1",
        "gene": args.truth_gene,
        "evidence_source": gears_manifest.get("model", {}).get("source", ""),
        "n_shared_readouts": int(len(shared)),
        "criteria": criteria,
        "geneformer_exploratory": geneformer_overlap,
        "verdict": verdict,
        "note": (
            "sanity anchor on same-cell-line KO data (Frangieh co-culture), not a disease-context "
            "validation; failing verdict blocks lineage consumption of this evidence source"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "agreement": agree, "baseline": base_agree, "rho": float(rho)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
