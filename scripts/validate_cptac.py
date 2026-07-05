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
  downloads the phosphoproteomics TSV for the requested study. Requires
  network access and that ``pdc.cancer.gov`` be in the download allowlist.
* **Mock mode** (``--mock``): generates synthetic phosphoproteomics data
  locally. Output is explicitly labelled "MOCK — not scientifically valid"
  in every result file and log line. Useful for plumbing/CI smoke checks
  only.

The script refuses to run unless ``PTM2CELLNET_ALLOW_EXPERIMENTAL=1`` is
set, because the validation pipeline (predictor wiring, scientific metrics)
is still partial — see F-03 in the v15 review. This guard prevents the
script from being mistaken for a finished capability.

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
        "partial: PDC fetching is implemented but the predictor wiring and "
        "scientific metrics are not yet complete. Set "
        "PTM2CELLNET_ALLOW_EXPERIMENTAL=1 to run for development/testing. "
        "Use --mock for a fully-local synthetic-data smoke check; the default "
        "mode queries the real PDC API at https://pdc.cancer.gov/graphql."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
        """Download (or mock) the phosphoproteomics matrix for a study."""
        cache_file = self.output_dir / f"{study_id}_phosphoproteomics.csv"
        if cache_file.exists():
            logger.info("Using cached data: %s", cache_file)
            return pd.read_csv(cache_file, index_col=0)

        if self.backend == "pdc":
            logger.info("Querying PDC for study %s phosphoproteomics files...", study_id)
            manifest = self.download_study_manifest(study_id)
            files = manifest.get("files", [])
            phospho_files = [
                f for f in files
                if "phospho" in str(f.get("data_category", "")).lower()
                or "phospho" in str(f.get("file_name", "")).lower()
            ]
            if not phospho_files:
                raise RuntimeError(
                    f"No phosphoproteomics files found in PDC study {study_id!r}. "
                    f"Available file categories: "
                    f"{sorted({str(f.get('data_category','')) for f in files})}"
                )
            logger.warning(
                "PDC returns file metadata only; automated TSV download is not "
                "yet wired in. Falling back to mock data for the matrix. "
                "See F-03 in the v15 review."
            )
            # File metadata is real; matrix is still mock until the TSV
            # download path is added. This is explicit so callers cannot
            # mistake the matrix for real data.
            return self._mock_phospho_matrix(study_id, cache_file, real_file_count=len(phospho_files))

        return self._mock_phospho_matrix(study_id, cache_file)

    def _mock_phospho_matrix(
        self, study_id: str, cache_file: Path, real_file_count: int = 0
    ) -> pd.DataFrame:
        logger.warning(
            "MOCK — generating synthetic phosphoproteomics matrix for %s. "
            "NOT scientifically valid.", study_id
        )
        np.random.seed(42)
        n_sites = 10000
        n_samples = 100
        proteins = [f"P{i:05d}" for i in range(100)]
        sites = [
            f"{np.random.choice(proteins)}_{np.random.randint(1, 500)}"
            f"{np.random.choice(['S', 'T', 'Y'])}"
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
        df = pd.DataFrame({
            "Hugo_Symbol": [f"GENE{i}" for i in range(n)],
            "UniProt_ID": np.random.choice(proteins, n),
            "Protein_position": np.random.randint(1, 500, n),
            "Reference_AA": np.random.choice(aas, n),
            "Variant_AA": np.random.choice(aas, n),
            "Sample_ID": [f"Sample_{np.random.randint(100)}" for _ in range(n)],
            "Variant_Classification": np.random.choice(
                ["Missense_Mutation", "Silent", "Nonsense_Mutation"], n
            ),
        })
        df.to_csv(cache_file, index=False)
        return df


class CPTACValidator:
    """Validate PTM predictions against CPTAC data.

    NOTE: the predictor wiring is partial (F-03). ``validate_predictions``
    returns ``predicted_effect='unknown'`` for every site until the model
    integration lands; this is explicit and logged so the output is never
    mistaken for real validation.
    """

    def __init__(self, model_dir: str, phospho_data: pd.DataFrame,
                 mutation_data: pd.DataFrame, sequences: dict[str, str]):
        self.model_dir = Path(model_dir)
        self.phospho_data = phospho_data
        self.mutation_data = mutation_data
        self.sequences = sequences
        self.models = self._load_models()

    def _load_models(self) -> dict[str, Any]:
        from src.utils.io import safe_torch_load

        sys.path.insert(0, str(self.model_dir.parent.parent))
        try:
            from src.models.ptm_site_predictor import PTMSitePredictor
        except ImportError:
            logger.warning("PTMSitePredictor not importable; running without models.")
            return {}

        models: dict[str, Any] = {}
        for ptm_type in ["Phosphorylation"]:
            model_path = self.model_dir / ptm_type.lower() / "checkpoints"
            ckpt_files = list(model_path.glob("*.ckpt")) if model_path.exists() else []
            if not ckpt_files:
                continue
            best = sorted(
                [f for f in ckpt_files if "val_auroc" in f.stem],
                key=lambda x: x.stem,
                reverse=True,
            )[0]
            checkpoint = safe_torch_load(best, map_location="cpu")
            model = PTMSitePredictor(
                vocab_size=21, embed_dim=64, hidden_dim=128, encoder_type="cnn"
            )
            if "state_dict" in checkpoint:
                sd = {
                    (k[6:] if k.startswith("model.") else k): v
                    for k, v in checkpoint["state_dict"].items()
                }
                model.load_state_dict(sd)
            model.eval()
            models[ptm_type] = model
        return models

    def validate_predictions(self) -> dict[str, Any]:
        logger.info("Starting validation...")
        logger.warning(
            "F-03 partial: predicted_effect is 'unknown' for every site — "
            "predictor wiring is not yet complete. Results are NOT scientifically valid."
        )
        results: dict[str, Any] = {
            "total_sites": len(self.phospho_data),
            "total_mutations": len(self.mutation_data),
            "validation_results": [],
            "scientifically_valid": False,
        }
        for _, mutation in self.mutation_data.head(100).iterrows():
            uniprot_id = mutation["UniProt_ID"]
            position = mutation["Protein_position"]
            ref_aa = mutation["Reference_AA"]
            alt_aa = mutation["Variant_AA"]
            if uniprot_id not in self.sequences:
                continue
            results["validation_results"].append({
                "uniprot_id": uniprot_id,
                "position": position,
                "mutation": f"{ref_aa}{position}{alt_aa}",
                "predicted_effect": "unknown",
            })
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
                    sites.append({
                        "uniprot_id": uniprot_id,
                        "position": int(position),
                        "aa": aa,
                        "site_id": site_id,
                    })
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

    results: dict[str, Any] = {
        "study": args.study,
        "backend": backend,
        "scientifically_valid": False,
        "phospho_data_shape": list(phospho_data.shape),
        "mutation_count": len(mutation_data),
        "comparison": comparison,
    }
    out_path = Path(args.output_dir) / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Validation complete. Results saved to: %s", args.output_dir)
    logger.warning(
        "Result file marked scientifically_valid=false because the F-03 "
        "predictor wiring is partial. Do not cite as a real CPTAC validation."
    )


if __name__ == "__main__":
    main()
