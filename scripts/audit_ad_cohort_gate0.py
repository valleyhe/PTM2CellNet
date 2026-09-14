#!/usr/bin/env python
"""Audit data/AD GEO cohorts against the Gate-0 donor cohort contract.

Contract source: docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md
§4.6 rule 1, implemented by ``prepare_perturbgen_anndata``: raw integer
counts, canonical version-less ENSG, explicit donor-like annotation, and
>= 3 donors shared across the normal/disease states within one cohort.

Unlike scripts/audit_perturbgen_cohort.py (scPerturb h5ad files), this audit
targets the raw GEO downloads under data/AD and never modifies them. Policy
unchanged: ``sample``/``batch``/``replicate``/title labels are recorded as
raw labels and are NOT reinterpreted as donors or conditions.

Usage:
    python scripts/audit_ad_cohort_gate0.py

Writes outputs/perturbgen/spike/<date>_ad_cohort_audit/evidence.json.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
from pathlib import Path

import h5py
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "AD"
OUTPUT_DIR = REPO_ROOT / "outputs" / "perturbgen" / "spike" / (_dt.date.today().strftime("%Y%m%d") + "_ad_cohort_audit")

GATE0_REQUIREMENTS = (
    "real normal/disease cohort",
    "raw integer counts",
    "canonical version-less ENSG",
    "explicit donor annotation",
    ">= 3 donors shared across normal/disease states",
)


def _first_matrix_group(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {name: list(group.keys()) for name, group in handle.items()}


def audit_gse147528() -> dict:
    manifest = pd.read_csv(DATA_DIR / "metadata" / "GSE147528_sample_manifest.tsv", sep="\t")
    donors = sorted(manifest["donor_original"].astype(str).unique())
    braak_per_donor = manifest.groupby("donor_original")["braak_stage_original"].nunique()
    h5 = next((DATA_DIR / "extracted" / "GSE147528").glob("*.h5"))
    with h5py.File(h5, "r") as handle:
        group = next(iter(handle.values()))
        genes = [value.decode() for value in group["genes"][:5]]
        shape = tuple(int(v) for v in group["shape"][:])
    return {
        "tissue": "post-mortem human brain (EC + SFG), 10x snRNA-seq v2",
        "n_libraries": int(len(manifest)),
        "counts_level": "raw unfiltered droplets; no cell calling/EmptyDrops run",
        "matrix_shape_genes_x_barcodes": shape,
        "ensembl": {"column": "genes", "example": genes, "versioned": any("." in g for g in genes)},
        "donor": {
            "source": "explicit 'donor id' in GEO characteristics_ch1",
            "n_donors": len(donors),
            "regions_per_donor": int(manifest.groupby("donor_original")["region_original"].nunique().max()),
        },
        "condition": {
            "source": "absent in GEO metadata; only Braak stage per donor",
            "braak_by_donor": sorted(braak_per_donor.astype(str).unique()),
            "braak_constant_within_donor": bool((braak_per_donor == 1).all()),
            "braak_distribution": {
                str(k): int(v)
                for k, v in manifest.drop_duplicates("donor_original")["braak_stage_original"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "gate0_gaps": [
            "no diagnosis/condition field (Braak-only control definition would be a derivation, not source metadata)",
            "raw droplets need cell calling before any cohort h5ad can be built",
        ],
    }


def audit_gse157827() -> dict:
    manifest = pd.read_csv(DATA_DIR / "metadata" / "GSE157827_sample_manifest.tsv", sep="\t")
    diagnosis = manifest["diagnosis_or_condition_original"].value_counts().to_dict()
    feature_files = sorted((DATA_DIR / "extracted" / "GSE157827").glob("*features.tsv.gz"))
    ids = set()
    for path in feature_files:
        with gzip.open(path, "rt") as handle:
            ids.update(line.split("\t")[0] for line in handle)
    return {
        "tissue": "post-mortem human prefrontal cortex, 10x snRNA-seq v3",
        "n_libraries": int(len(manifest)),
        "counts_level": "official filtered MTX; non-negative integer check passed 21/21 (reports/GSE157827_qc.tsv)",
        "ensembl": {
            "column": "features.tsv column 1",
            "n_unique": len(ids),
            "versioned": any("." in value for value in ids),
        },
        "donor": {
            "source": "absent: no donor/subject id in characteristics or SOFT (checked)",
            "library_titles_are_distinct_subject_labels_candidate": sorted(manifest["title"])[:3] + ["..."],
            "reinterpretation_done": False,
        },
        "condition": {
            "source": "explicit 'diagnosis' in GEO characteristics_ch1",
            "distribution": {str(k): int(v) for k, v in diagnosis.items()},
        },
        "gate0_gaps": [
            "no explicit donor annotation (one library per subject is plausible but unverified)",
        ],
    }


def audit_gse174367() -> dict:
    meta = pd.read_csv(DATA_DIR / "raw_downloads" / "GSE174367" / "GSE174367_snRNA-seq_cell_meta.csv.gz")
    per_sample = meta.groupby("SampleID")["Diagnosis"].agg(["nunique", "first"])
    h5 = DATA_DIR / "raw_downloads" / "GSE174367" / "GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"
    with h5py.File(h5, "r") as handle:
        ids = [value.decode() for value in handle["matrix/features/id"][:2000]]
    return {
        "tissue": "post-mortem human prefrontal cortex, pooled filtered matrix + cell metadata",
        "n_samples": int(meta["SampleID"].nunique()),
        "n_cells": int(len(meta)),
        "counts_level": "official filtered H5; integer counts (reports/qc_report_GSE174367.md)",
        "ensembl": {
            "column": "matrix/features/id",
            "example": ids[:3],
            "versioned": any("." in value for value in ids),
        },
        "donor": {
            "source": "absent: cell metadata has no donor column",
            "columns_present": list(meta.columns),
            "sample_diagnosis_unique": bool((per_sample["nunique"] == 1).all()),
            "reinterpretation_done": False,
        },
        "condition": {
            "source": "cell-level 'Diagnosis' column",
            "distribution": {str(k): int(v) for k, v in meta["Diagnosis"].value_counts().items()},
            "per_sample": {str(k): str(v) for k, v in per_sample["first"].items()},
            "cell_level_covariates": ["Age", "Sex", "PMI", "Tangle.Stage", "Plaque.Stage", "RIN", "Batch"],
        },
        "gate0_gaps": [
            "versioned ENSG ids (ENSG00000223972.5) rejected by Gate-0 canonical-Ensembl rule",
            "no explicit donor annotation (SampleID is a raw label, not reinterpreted)",
        ],
    }


def audit_gse188545() -> dict:
    manifest = pd.read_csv(DATA_DIR / "metadata" / "GSE188545_sample_manifest.tsv", sep="\t")
    titles = sorted(manifest["title"])
    gene_files = sorted((DATA_DIR / "extracted" / "GSE188545").glob("*genes.tsv.gz"))
    ids = set()
    for path in gene_files:
        with gzip.open(path, "rt") as handle:
            ids.update(line.split("\t")[0] for line in handle)
    return {
        "tissue": "post-mortem human middle temporal gyrus, 10x snRNA-seq v3",
        "n_libraries": int(len(manifest)),
        "counts_level": "official filtered MTX; non-negative integer check passed 12/12 (reports/GSE188545_qc.tsv)",
        "ensembl": {
            "column": "genes.tsv column 1",
            "n_unique": len(ids),
            "versioned": any("." in value for value in ids),
        },
        "donor": {
            "source": "absent: no donor/subject id in characteristics or SOFT (checked)",
            "reinterpretation_done": False,
        },
        "condition": {
            "source": "titles only (ADxxMTG / HCxxMTG); characteristics have no diagnosis field",
            "title_ad": int(sum(title.startswith("AD") for title in titles)),
            "title_hc": int(sum(title.startswith("HC") for title in titles)),
            "title_to_condition_done": False,
        },
        "gate0_gaps": [
            "no explicit donor annotation",
            "condition only recoverable from titles, which is a derivation, not source metadata",
        ],
    }


def structural_shared_donor_analysis() -> dict:
    """Gate-0 as implemented requires donors observed in BOTH states.

    All four cohorts are post-mortem case-control designs: one donor belongs
    to exactly one condition. Within-donor normal/disease pairing is not a
    data-quality gap that standardization can fix; it is a semantics
    mismatch between the perturbation-paired contract and case-control
    disease cohorts.
    """
    return {
        "contract_semantics": "prepare_perturbgen_anndata _validate_obs_contract: shared_donors = normal_donors ∩ disease_donors >= min_donors (3)",
        "ad_cohort_design": "between-donor case-control; a post-mortem donor is either AD or control, never both",
        "max_achievable_shared_donors": 0,
        "conclusion": (
            "Even with donor/condition labels fully resolved, no data/AD cohort can pass the "
            "current Gate-0 preflight. Closing this gap is a research-contract decision "
            "(either use a perturbation cohort as the PerturbGen tokenise input and keep the AD "
            "cohort on the donor-level disease-normal axis, or explicitly redesign Gate-0 for "
            "between-donor case-control cohorts), not additional data wrangling."
        ),
        "decision_owner": "user/research contract; must be recorded in lessons.md before any code change",
    }


def main() -> int:
    audits = {
        "GSE147528": audit_gse147528(),
        "GSE157827": audit_gse157827(),
        "GSE174367": audit_gse174367(),
        "GSE188545": audit_gse188545(),
    }
    evidence = {
        "run_id": _dt.date.today().strftime("%Y%m%d") + "_ad_cohort_audit",
        "collected_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Gate-0 audit of the real patient normal/disease cohorts saved under data/AD (A-01 asset gap follow-up)"
        ),
        "gate0_requirements": list(GATE0_REQUIREMENTS),
        "policy": {
            "never_reinterpreted_as_donor": ["sample", "SampleID", "batch", "replicate", "title", "cell_line"],
            "title_to_condition_conversion": "not performed; recorded as derivation candidate only",
            "audit_only": "no file under data/AD was modified",
        },
        "real_patient_primary_tissue": True,
        "audits": audits,
        "structural_shared_donor_analysis": structural_shared_donor_analysis(),
        "verdict": "GATE0_BLOCKED_SEMANTICS_AND_LABELS",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "evidence.json"
    out_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "evidence_file": str(out_path),
                "verdict": evidence["verdict"],
                "donor_explicit": {g: a["donor"]["source"] for g, a in audits.items()},
                "condition_explicit": {g: a["condition"]["source"] for g, a in audits.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
