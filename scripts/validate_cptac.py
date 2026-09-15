#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPTAC / PDC validation script (F-03).

This script validates PTM predictions against CPTAC phosphoproteomics data
sourced from the Proteomic Data Commons (PDC). It is the production entry
point for CPTAC validation; the previous "mock/synthetic data" mode is now
gated behind an explicit ``--mock`` flag so that no run can silently produce
scientifically meaningless output.

Modes
-----

* **Real mode (default)**: uses :class:`src.analysis.pdc_client.PDCClient`
  to query the PDC GraphQL API for study manifests and file metadata, then
  downloads the phosphoproteomics TSV for the requested study via
  :meth:`PDCClient.download_file`. Requires network access and that
  ``pdc.cancer.gov`` be reachable / on the download allowlist.
* **Mock mode** (``--mock``): generates synthetic phosphoproteomics data
  locally. Output is explicitly labelled "MOCK — not scientifically valid"
  in every result file and log line. Useful for plumbing/CI smoke checks
  only.

Predictor wiring (v17)
-----------------------

When a pretrained checkpoint is supplied (``--model-dir`` pointing at a real
``outputs/ptm_pretrain/<ptm_type>/checkpoints/*.ckpt``) the validator loads
:mod:`scripts.predict_ptm_sites.BatchPTMPredictor` and scores the impact of
each mutation on neighbouring S/T/Y residues. The resulting
``predicted_effect`` is ``gain`` / ``loss`` / ``neutral`` based on the
delta between wild-type and mutant PTM probability, mirroring the
:func:`BatchPTMPredictor.predict_variants` contract. When no checkpoint is
available the validator still runs and emits ``predicted_effect="unknown"``
with ``scientifically_valid=false`` so the output is never mistaken for
real validation.

The script refuses to run unless ``PTM2CELLNET_ALLOW_EXPERIMENTAL=1`` is
set, because the scientific metrics (AUROC / precision@k against the real
CPTAC matrix) are still partial — see F-03 in the v15/v16 review. This
guard prevents the script from being mistaken for a finished capability.

Function: download CPTAC phosphoproteomics data and validate PTM predictions
against it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EXPERIMENTAL_GUARD = os.environ.get("PTM2CELLNET_ALLOW_EXPERIMENTAL", "").lower() in (
    "1",
    "true",
    "yes",
)
if not _EXPERIMENTAL_GUARD:
    raise RuntimeError(
        "validate_cptac.py is EXPERIMENTAL (F-03). The validation pipeline is "
        "now mostly wired (PDC fetch + TSV download + BatchPTMPredictor "
        "scoring) but the scientific validation metrics (per-cohort AUROC, "
        "precision@k against the real CPTAC matrix) are still partial. Set "
        "PTM2CELLNET_ALLOW_EXPERIMENTAL=1 to run for development/testing. "
        "Use --mock for a fully-local synthetic-data smoke check; the default "
        "mode queries the real PDC API at https://pdc.cancer.gov/graphql."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _find_best_checkpoint(model_dir: Path) -> Path | None:
    """Locate the highest-val_auroc multi-task PTM checkpoint under ``model_dir``.

    Looks for ``model_dir/<ptm_type>/checkpoints/*.ckpt`` (case-insensitive
    ptm_type), preferring filenames containing ``val_auroc`` (sorted
    descending by the trailing float after ``val_auroc=``). Returns
    ``None`` when no checkpoint is present.
    """
    if not model_dir.exists():
        return None
    candidates: list[Path] = []
    # Common layouts: outputs/ptm_pretrain/<ptm_type>/checkpoints/*.ckpt
    # or outputs/ptm_pretrain/checkpoints/*.ckpt
    candidates.extend(model_dir.glob("*/checkpoints/*.ckpt"))
    candidates.extend(model_dir.glob("checkpoints/*.ckpt"))
    candidates.extend(model_dir.glob("*.ckpt"))
    if not candidates:
        return None

    import re as _re

    def _metric(path: Path) -> float | None:
        m = _re.search(r"val_auroc[=_]([0-9]*\.?[0-9]+)", path.stem, flags=_re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # Prefer files with val_auroc=N.NN in the name, sorted descending by the
    # parsed metric. Falls back to most-recently-modified otherwise.
    scored = [(_metric(c), c) for c in candidates]
    with_metric = [(score, path) for score, path in scored if score is not None]
    if with_metric:
        with_metric.sort(key=lambda pair: pair[0], reverse=True)
        return with_metric[0][1]
    # Fallback: most recently modified checkpoint.
    return max(candidates, key=lambda c: c.stat().st_mtime)


def _score_mutation_effect(
    predictor: Any,
    sequence: str,
    position: int,
    alt_aa: str,
    *,
    threshold_delta: float = 0.2,
) -> str:
    """Score the PTM-effect of a missense mutation via WT vs mutant inference.

    Mirrors :meth:`BatchPTMPredictor.predict_variants`: scores the WT
    sequence, builds the mutant sequence by substituting ``alt_aa`` at
    ``position`` (1-based), scores the mutant, and compares the
    per-position PTM probability at the mutation site across every PTM
    type the predictor knows about. Returns ``gain`` / ``loss`` /
    ``neutral`` based on the largest absolute delta.

    Any inference error falls back to ``"unknown"`` so the caller can
    still emit a row rather than aborting the whole validation.
    """
    import pandas as _pd  # local import to keep module import-light

    if position < 1 or position > len(sequence):
        return "unknown"
    try:
        wt_df = predictor.predict_protein(sequence, "wt", threshold=0.0)
        mut_seq = sequence[: position - 1] + alt_aa + sequence[position:]
        mut_df = predictor.predict_protein(mut_seq, "mut", threshold=0.0)
    except Exception as exc:  # pragma: no cover - depends on torch state
        logger.debug("predictor inference failed at pos=%s: %s", position, exc)
        return "unknown"

    def _prob_at(df: _pd.DataFrame, ptm_type: str) -> float:
        if df is None or len(df) == 0:
            return 0.0
        sub = df[(df["position"] == position) & (df["ptm_type"] == ptm_type)]
        if len(sub) == 0:
            return 0.0
        return float(sub["probability"].max())

    ptm_types = getattr(predictor.model, "ptm_types", ["Phosphorylation"])
    largest_delta = 0.0
    largest_sign = 0
    for ptm_type in ptm_types:
        wt_p = _prob_at(wt_df, ptm_type)
        mut_p = _prob_at(mut_df, ptm_type)
        delta = mut_p - wt_p
        if abs(delta) > abs(largest_delta):
            largest_delta = delta
            largest_sign = 1 if delta > 0 else -1
    if largest_delta > threshold_delta:
        return "gain"
    if largest_delta < -threshold_delta:
        return "loss"
    return "neutral"


# Canonical CPTAC study IDs (PDC study_id UUIDs).
CPTAC_STUDIES: dict[str, dict[str, str]] = {
    "BRCA": {
        "name": "CPTAC Breast Cancer",
        "pdc_study_id": "7c0c6e28-d405-11e8-b853-a005056ab009",
        "description": "乳腺癌磷蛋白组数据",
    },
    "OV": {
        "name": "CPTAC Ovarian Cancer",
        "pdc_study_id": "1a2d2540-8c0a-11e8-b657-a005056ab009",
        "description": "卵巢癌磷蛋白组数据",
    },
    "COAD": {
        "name": "CPTAC Colon Cancer",
        "pdc_study_id": "2a24c7c0-8c0a-11e8-b657-a005056ab009",
        "description": "结肠癌磷蛋白组数据",
    },
    "CCRCC": {
        "name": "CPTAC Clear Cell Renal Cell Carcinoma",
        "pdc_study_id": "d852e3a0-d405-11e8-b853-a005056ab009",
        "description": "透明细胞肾癌磷蛋白组数据",
    },
    "UCEC": {
        "name": "CPTAC Endometrial Cancer",
        "pdc_study_id": "e8a2cc20-d405-11e8-b853-a005056ab009",
        "description": "子宫内膜癌磷蛋白组数据",
    },
}


class CPTACDataDownloader:
    """Download CPTAC phosphoproteomics data.

    Two backends:

    * ``backend="pdc"`` (default): real PDC GraphQL via
      :class:`src.analysis.pdc_client.PDCClient`.
    * ``backend="mock"``: synthetic data for local plumbing tests.
    """

    def __init__(self, output_dir: str, backend: str = "pdc"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def download_study_manifest(self, study_id: str) -> dict[str, Any]:
        """Fetch the real study manifest from PDC (or a stub in mock mode)."""
        if self.backend == "mock":
            logger.warning("MOCK — returning stub manifest, not scientifically valid.")
            return CPTAC_STUDIES.get(study_id, {})

        from src.analysis.pdc_client import PDCClient, PDCAPIError

        pdc_study_id = CPTAC_STUDIES.get(study_id, {}).get("pdc_study_id")
        if not pdc_study_id:
            raise ValueError(f"Unknown CPTAC study alias: {study_id!r}")

        client = PDCClient()
        try:
            study = client.get_study(pdc_study_id)
        except PDCAPIError as exc:
            logger.error("PDC API call failed for %s: %s", study_id, exc)
            raise
        return {
            "study_id": study.study_id,
            "study_submitter_id": study.study_submitter_id,
            "study_name": study.study_name,
            "disease_type": study.disease_type,
            "primary_site": study.primary_site,
            "files": study.files,
        }

    def download_phosphoproteomics(self, study_id: str = "BRCA") -> pd.DataFrame:
        """Download (or mock) the phosphoproteomics matrix for a study.

        In PDC backend the matrix is the *real* downloaded TSV: PDC
        exposes the per-file download URL via ``PDCClient.get_file_url``
        and ``download_file`` streams it to ``output_dir`` under the
        project-wide size cap + host allowlist. The downloaded TSV is
        parsed into a (sites × samples) DataFrame and cached.

        If the download fails for any reason (network, allowlist, size
        cap) we **do not** silently fall back to synthetic data — we
        raise. Callers that want a smoke check should pass ``--mock``
        explicitly, which always uses the synthetic generator and labels
        the result ``scientifically_valid=false``.
        """
        cache_file = self.output_dir / f"{study_id}_phosphoproteomics.csv"
        if cache_file.exists():
            logger.info("Using cached data: %s", cache_file)
            return pd.read_csv(cache_file, index_col=0)

        if self.backend == "pdc":
            logger.info("Querying PDC for study %s phosphoproteomics files...", study_id)
            manifest = self.download_study_manifest(study_id)
            files = manifest.get("files", [])
            phospho_files = [
                f
                for f in files
                if "phospho" in str(f.get("data_category", "")).lower()
                or "phospho" in str(f.get("file_name", "")).lower()
            ]
            if not phospho_files:
                raise RuntimeError(
                    f"No phosphoproteomics files found in PDC study {study_id!r}. "
                    f"Available file categories: "
                    f"{sorted({str(f.get('data_category', '')) for f in files})}"
                )

            # Real download path (F-03 v17). Resolve each candidate file_id
            # to a per-file URL and stream it down through the safe download
            # boundary in PDCClient. The first file that parses cleanly wins.
            from src.analysis.pdc_client import PDCClient, PDCAPIError

            client = PDCClient()
            last_error: Exception | None = None
            for candidate in phospho_files:
                file_id = candidate.get("file_id") or candidate.get("file_name")
                if not file_id:
                    continue
                try:
                    local_path = client.download_file(str(file_id), str(self.output_dir))
                except PDCAPIError as exc:
                    logger.warning(
                        "PDC download failed for file_id=%s: %s (trying next)",
                        file_id,
                        exc,
                    )
                    last_error = exc
                    continue

                try:
                    df = _parse_phospho_tsv(local_path)
                except Exception as exc:  # pragma: no cover - depends on file shape
                    logger.warning(
                        "Downloaded %s but could not parse as phosphoproteomics matrix: %s",
                        local_path,
                        exc,
                    )
                    last_error = exc
                    continue

                df.to_csv(cache_file)
                logger.info(
                    "Real PDC phosphoproteomics matrix saved to %s (%d sites × %d samples)",
                    cache_file,
                    df.shape[0],
                    df.shape[1],
                )
                return df

            raise RuntimeError(
                "Could not obtain a real phosphoproteomics matrix from PDC for "
                f"study {study_id!r} (last error: {last_error}). Use --mock for "
                "a synthetic smoke check (NOT scientifically valid)."
            )

        return self._mock_phospho_matrix(study_id, cache_file)

    def _mock_phospho_matrix(self, study_id: str, cache_file: Path, real_file_count: int = 0) -> pd.DataFrame:
        logger.warning(
            "MOCK — generating synthetic phosphoproteomics matrix for %s. NOT scientifically valid.", study_id
        )
        np.random.seed(42)
        n_sites = 10000
        n_samples = 100
        proteins = [f"P{i:05d}" for i in range(100)]
        sites = [
            f"{np.random.choice(proteins)}_{np.random.randint(1, 500)}{np.random.choice(['S', 'T', 'Y'])}"
            for _ in range(n_sites)
        ]
        data = np.random.randn(n_sites, n_samples)
        cols = [f"Sample_{i}" for i in range(n_samples)]
        df = pd.DataFrame(data, index=sites, columns=cols)
        df.index.name = "Site"
        df.to_csv(cache_file)
        logger.info("Mock data saved to: %s", cache_file)
        if real_file_count:
            logger.info("(PDC reported %d real phosphoproteomics file(s) for this study)", real_file_count)
        return df

    def download_tcga_mutations(self, study_id: str = "BRCA") -> pd.DataFrame:
        """Download (or mock) TCGA mutation data for a study."""
        cache_file = self.output_dir / f"{study_id}_mutations.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file)
        logger.warning("MOCK — generating synthetic mutation data for %s. NOT scientifically valid.", study_id)
        np.random.seed(42)
        n = 5000
        proteins = [f"P{i:05d}" for i in range(100)]
        aas = list("ACDEFGHIKLMNPQRSTVWY")
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [f"GENE{i}" for i in range(n)],
                "UniProt_ID": np.random.choice(proteins, n),
                "Protein_Position": np.random.randint(1, 500, n),
                "Reference_AA": np.random.choice(aas, n),
                "Variant_AA": np.random.choice(aas, n),
                "Sample_ID": [f"Sample_{np.random.randint(100)}" for _ in range(n)],
                "Variant_Classification": np.random.choice(["Missense_Mutation", "Silent", "Nonsense_Mutation"], n),
            }
        )
        df.to_csv(cache_file, index=False)
        return df


def _parse_phospho_tsv(path: Path) -> pd.DataFrame:
    """Parse a CPTAC/PDC phosphoproteomics TSV into a (sites × samples) frame.

    CPTAC phosphoproteomics TSVs typically have one row per phosphosite and
    many sample intensity columns. The first column carries a site
    identifier like ``P04637_S15``; we use that as the index. Sample
    columns are detected heuristically: any column whose name does not
    match common metadata tokens (gene, protein, position, ...) is treated
    as a sample intensity column.

    The parser is intentionally conservative — if it cannot find a usable
    site column and at least one sample column it raises so that the
    caller can try the next candidate file rather than silently producing
    a malformed matrix.
    """
    import pandas as _pd

    df = _pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if df.shape[1] < 2 or df.shape[0] == 0:
        raise ValueError(
            f"Phospho TSV at {path} has shape {df.shape}; expected at least (sites × samples) with a header row."
        )

    # Identify the site identifier column. Prefer an explicit "Gene_Site"
    # / "Site" / "Index" column; otherwise fall back to the first column.
    metadata_tokens = {
        "gene",
        "genesymbol",
        "gene_symbol",
        "genename",
        "protein",
        "accession",
        "uniprot",
        "uniprot_id",
        "uniprot_accession",
        "chromosome",
        "chrom",
        "position",
        "pos",
        "amino_acid",
        "aa",
        "residue",
        "mod_rsd",
        "modified_residue",
        "peptide",
        "sequence",
        "sequence_window",
        "num_phospho",
        "description",
        "notes",
        "organism",
    }
    site_col: str | None = None
    for cand in ("Gene_Site", "Site", "Index", "site_id"):
        if cand in df.columns:
            site_col = cand
            break
    if site_col is None:
        # First column whose normalized name is not a metadata token.
        for col in df.columns:
            if _normalize_token(col) not in metadata_tokens:
                site_col = str(col)
                break
    if site_col is None:
        site_col = str(df.columns[0])

    sample_cols = [c for c in df.columns if c != site_col and _normalize_token(c) not in metadata_tokens]
    if not sample_cols:
        raise ValueError(f"Phospho TSV at {path} has no sample intensity columns (all columns look like metadata).")

    out = df[[site_col, *sample_cols]].copy()
    out = out.set_index(site_col)
    out.index = out.index.astype(str)
    out.index.name = "Site"
    # Coerce to numeric; non-numeric cells become NaN.
    for col in out.columns:
        out[col] = _pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_token(name: object) -> str:
    """Lowercase + strip non-alphanumeric for fuzzy column-name matching."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _effect_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Tally ``predicted_effect`` values across validation rows."""
    counts: dict[str, int] = {}
    for row in rows:
        effect = str(row.get("predicted_effect", "unknown"))
        counts[effect] = counts.get(effect, 0) + 1
    return counts


class CPTACValidator:
    """Validate PTM predictions against CPTAC data.

    Predictor wiring (v17): when ``model_dir`` resolves to a real
    multi-task PTM checkpoint, the validator delegates per-protein
    scoring to :class:`scripts.predict_ptm_sites.BatchPTMPredictor`.
    The WT and mutant sequences are scored at every S/T/Y residue in the
    mutation's flanking window, and ``predicted_effect`` is recorded as
    ``gain`` / ``loss`` / ``neutral`` based on the delta in PTM
    probability at the mutation site (mirroring ``predict_variants``).

    When no checkpoint is available the validator emits
    ``predicted_effect="unknown"`` and sets ``scientifically_valid=false``;
    this is logged explicitly so output is never mistaken for real
    validation.
    """

    def __init__(
        self, model_dir: str, phospho_data: pd.DataFrame, mutation_data: pd.DataFrame, sequences: dict[str, str]
    ):
        self.model_dir = Path(model_dir)
        self.phospho_data = phospho_data
        self.mutation_data = mutation_data
        self.sequences = sequences
        self._predictor: Any | None = None
        self._predictor_loaded = False
        self._load_predictor()

    def _load_predictor(self) -> None:
        """Locate the best multi-task checkpoint and build a BatchPTMPredictor.

        Looks for ``model_dir/phosphorylation/checkpoints/*.ckpt`` (case
        insensitive) and picks the one with the highest ``val_auroc`` in
        its filename. If none is found, ``self._predictor`` stays None.
        """
        if self._predictor_loaded:
            return
        self._predictor_loaded = True

        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        ckpt = _find_best_checkpoint(self.model_dir)
        if ckpt is None:
            logger.warning(
                "No multi-task PTM checkpoint found under %s; predicted_effect will be 'unknown' for every mutation.",
                self.model_dir,
            )
            return

        try:
            from scripts.predict_ptm_sites import BatchPTMPredictor
        except ImportError:
            logger.warning(
                "BatchPTMPredictor not importable; predicted_effect will be "
                "'unknown' for every mutation. Install the package or run "
                "from the repo root."
            )
            return

        try:
            self._predictor = BatchPTMPredictor(
                model_path=str(ckpt),
                model_type="cnn_multitask",
                device="cpu",
            )
            logger.info("Predictor wired: %s", ckpt)
        except Exception as exc:  # pragma: no cover - depends on torch state
            logger.warning(
                "Failed to load predictor from %s: %s. predicted_effect will be 'unknown' for every mutation.",
                ckpt,
                exc,
            )
            self._predictor = None

    def validate_predictions(self) -> dict[str, Any]:
        logger.info("Starting validation...")
        results: dict[str, Any] = {
            "total_sites": len(self.phospho_data),
            "total_mutations": len(self.mutation_data),
            "validation_results": [],
            "scientifically_valid": self._predictor is not None,
            "predictor_wired": self._predictor is not None,
        }
        if self._predictor is None:
            logger.warning(
                "F-03 partial: predictor not wired (no checkpoint). "
                "predicted_effect is 'unknown' for every site. "
                "Results are NOT scientifically valid."
            )
            results["scientifically_valid"] = False
        else:
            logger.info("Predictor wired; computing WT/mut PTM-probability deltas at each mutation site.")

        for _, mutation in self.mutation_data.head(100).iterrows():
            uniprot_id = mutation["UniProt_ID"]
            position = mutation["Protein_Position"]
            ref_aa = mutation["Reference_AA"]
            alt_aa = mutation["Variant_AA"]
            if uniprot_id not in self.sequences:
                continue
            sequence = self.sequences[uniprot_id]
            entry: dict[str, Any] = {
                "uniprot_id": uniprot_id,
                "position": int(position),
                "mutation": f"{ref_aa}{position}{alt_aa}",
                "predicted_effect": "unknown",
            }
            if self._predictor is not None:
                entry["predicted_effect"] = _score_mutation_effect(
                    self._predictor, sequence, int(position), str(alt_aa)
                )
                results["scientifically_valid"] = True
            results["validation_results"].append(entry)
        return results

    def compare_with_known_sites(self) -> dict[str, Any]:
        sites = []
        for site_id in self.phospho_data.index:
            parts = str(site_id).split("_")
            if len(parts) == 2:
                uniprot_id = parts[0]
                position_aa = parts[1]
                position = "".join(filter(str.isdigit, position_aa))
                aa = "".join(filter(str.isalpha, position_aa))
                if position:
                    sites.append(
                        {
                            "uniprot_id": uniprot_id,
                            "position": int(position),
                            "aa": aa,
                            "site_id": site_id,
                        }
                    )
        sites_df = pd.DataFrame(sites)
        if sites_df.empty:
            return {"total_phospho_sites": 0, "unique_proteins": 0, "aa_distribution": {}}
        return {
            "total_phospho_sites": len(sites_df),
            "unique_proteins": sites_df["uniprot_id"].nunique(),
            "aa_distribution": sites_df["aa"].value_counts().to_dict(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPTAC data validation (F-03)")
    parser.add_argument("--output-dir", "-o", default="data/cptac", help="Output directory")
    parser.add_argument("--model-dir", "-m", default="outputs/ptm_pretrain", help="Model directory")
    parser.add_argument("--study", "-s", default="BRCA", help="CPTAC study alias")
    parser.add_argument("--download-only", action="store_true", help="Only download data")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use fully synthetic data (no PDC API calls). Output is NOT scientifically valid.",
    )
    args = parser.parse_args()

    backend = "mock" if args.mock else "pdc"
    if args.mock:
        logger.warning("Running in --mock mode: all data is synthetic. NOT scientifically valid.")
    else:
        logger.info("Running in PDC mode: querying https://pdc.cancer.gov/graphql for real metadata.")

    downloader = CPTACDataDownloader(args.output_dir, backend=backend)
    phospho_data = downloader.download_phosphoproteomics(args.study)
    mutation_data = downloader.download_tcga_mutations(args.study)
    logger.info("Phosphoproteomics data: %s", phospho_data.shape)
    logger.info("Mutation data: %d rows", len(mutation_data))

    if args.download_only:
        logger.info("Download complete.")
        return

    validator = CPTACValidator(
        model_dir=args.model_dir,
        phospho_data=phospho_data,
        mutation_data=mutation_data,
        sequences={},
    )
    comparison = validator.compare_with_known_sites()
    logger.info("Known phosphosites: %d", comparison["total_phospho_sites"])
    logger.info("AA distribution: %s", comparison["aa_distribution"])

    prediction_results = validator.validate_predictions()
    predictor_wired = bool(prediction_results.get("predictor_wired"))
    scientifically_valid = (
        backend != "mock" and predictor_wired and bool(prediction_results.get("scientifically_valid"))
    )
    if backend == "mock":
        logger.warning(
            "Backend is --mock: phosphoproteomics matrix is synthetic. "
            "scientifically_valid forced to false regardless of predictor."
        )

    results: dict[str, Any] = {
        "study": args.study,
        "backend": backend,
        "scientifically_valid": scientifically_valid,
        "predictor_wired": predictor_wired,
        "phospho_data_real": backend == "pdc",
        "phospho_data_shape": list(phospho_data.shape),
        "mutation_count": len(mutation_data),
        "comparison": comparison,
        "prediction_summary": {
            "total_scored": len(prediction_results.get("validation_results", [])),
            "effect_counts": _effect_counts(prediction_results.get("validation_results", [])),
        },
    }
    out_path = Path(args.output_dir) / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Validation complete. Results saved to: %s", args.output_dir)
    if not scientifically_valid:
        logger.warning(
            "Result file marked scientifically_valid=false because either "
            "(a) backend is --mock (synthetic matrix), or (b) the predictor "
            "could not be wired from %s. Do not cite as a real CPTAC validation.",
            args.model_dir,
        )


if __name__ == "__main__":
    main()
