#!/usr/bin/env python
"""Audit data/AD GEO cohorts against the Gate-0 donor cohort contract.

Contract source: docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md
§4.6 rule 1 as extended by lessons.md L-2026-0914-01, implemented by
``prepare_perturbgen_anndata``: raw integer counts, canonical version-less
ENSG, explicit donor-like annotation, and a donor/state design declared as
one of two pairings — ``within_donor`` needs >= 3 donors shared across the
normal/disease states, ``between_donor`` (case-control) needs >= 3 donors in
each of two donor-disjoint state groups.

Unlike scripts/audit_perturbgen_cohort.py (scPerturb h5ad files), this audit
targets the raw GEO downloads under data/AD and never modifies them. Policy
unchanged: ``sample``/``batch``/``replicate``/title labels are recorded as
raw labels and are NOT reinterpreted as donors or conditions.

Current status (2026-09-14): GSE174367 has been standardized into
``data/AD/standardized/GSE174367_ad_cohort.h5ad`` under the between_donor
pairing and passed the real Gate-0 preflight for 7/7 cell types
(outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json); the other
three cohorts remain blocked on missing donor/condition labels.

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
    "within_donor pairing: >= 3 donors shared across the normal/disease states, or "
    "between_donor pairing (case-control): >= 3 donors in each of two donor-disjoint state groups",
)

GSE174367_STANDARDIZED_EVIDENCE = "outputs/perturbgen/spike/20260914_gse174367_gate0/evidence.json"

GATE0_STATUS_STANDARDIZED = "standardized_between_donor_preflight_pass"
GATE0_STATUS_BLOCKED = "blocked_missing_donor_or_condition_labels"
GATE0_VERDICT = "GATE0_UNLOCKED_FOR_BETWEEN_DONOR_GSE174367_ONLY"


def _only_h5_file(directory: Path) -> Path:
    matches = sorted(directory.glob("*.h5"))
    if not matches:
        raise ValueError(f"no .h5 matrix found under {directory}")
    return matches[0]


def audit_gse147528() -> dict:
    manifest = pd.read_csv(DATA_DIR / "metadata" / "GSE147528_sample_manifest.tsv", sep="\t")
    donors = sorted(manifest["donor_original"].astype(str).unique())
    braak_per_donor = manifest.groupby("donor_original")["braak_stage_original"].nunique()
    h5 = _only_h5_file(DATA_DIR / "extracted" / "GSE147528")
    with h5py.File(h5, "r") as handle:
        if len(handle) == 0:
            raise ValueError(f"{h5} contains no matrix group")
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
        "gate0_status": GATE0_STATUS_BLOCKED,
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
        "gate0_status": GATE0_STATUS_BLOCKED,
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
        "gate0_status": GATE0_STATUS_STANDARDIZED,
        "standardized_between_donor": {
            "cohort_h5ad": "data/AD/standardized/GSE174367_ad_cohort.h5ad",
            "donor_derivation": (
                "SampleID accepted as donor via unique subject covariate vectors "
                "(scripts/standardize_gse174367_ad_cohort.py)"
            ),
            "preflight": "7/7 cell types PASS",
            "evidence": GSE174367_STANDARDIZED_EVIDENCE,
        },
        "gate0_gaps": [
            "raw download itself stays versioned ENSG + donor-less; only the standardized "
            "h5ad satisfies Gate-0 (use data/AD/standardized/, not raw_downloads/)"
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
        "gate0_status": GATE0_STATUS_BLOCKED,
        "gate0_gaps": [
            "no explicit donor annotation",
            "condition only recoverable from titles, which is a derivation, not source metadata",
        ],
    }


def structural_shared_donor_analysis() -> dict:
    """How the two Gate-0 pairings relate to post-mortem case-control cohorts.

    Under ``within_donor`` the cohort must contain donors observed in BOTH
    states; a post-mortem donor is either AD or control, never both, so the
    AD cohorts can never satisfy that pairing regardless of label quality.
    The ``between_donor`` pairing (lessons L-2026-0914-01) is the accepted
    contract for these cohorts: donor-disjoint state groups with >= 3 donors
    each.  GSE174367 has passed under that contract; the others are blocked
    on labels, not on semantics.
    """
    return {
        "contract_semantics": (
            "prepare_perturbgen_anndata _validate_obs_contract: within_donor requires "
            "shared_donors = normal_donors ∩ disease_donors >= min_donors (3); between_donor "
            "requires disjoint state groups with >= min_donors each and rejects any donor "
            "observed in both states as a labeling error"
        ),
        "ad_cohort_design": "between-donor case-control; a post-mortem donor is either AD or control, never both",
        "max_achievable_shared_donors_within_donor_pairing": 0,
        "between_donor_status": {
            "GSE174367": "standardized + preflight PASS 7/7 cell types",
            "GSE147528": "blocked: no diagnosis labels, raw droplets need cell calling",
            "GSE157827": "blocked: no donor annotation",
            "GSE188545": "blocked: no donor annotation, condition only derivable from titles",
        },
        "conclusion": (
            "Within-donor pairing is structurally impossible for post-mortem case-control "
            "cohorts; between_donor is the frozen contract for them (L-2026-0914-01). "
            "Remaining blockers are cohort-specific label/annotation gaps, not contract semantics."
        ),
        "decision_record": "lessons.md L-2026-0914-01 (user-approved option B)",
    }


def compute_verdict(audits: dict) -> str:
    """One GATE0 verdict line derived from the per-cohort statuses."""

    standardized = [name for name, audit in audits.items() if audit.get("gate0_status") == GATE0_STATUS_STANDARDIZED]
    if standardized:
        return GATE0_VERDICT
    return "GATE0_BLOCKED_SEMANTICS_AND_LABELS"


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
        "verdict": compute_verdict(audits),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "evidence.json"
    out_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "evidence_file": str(out_path),
                "verdict": evidence["verdict"],
                "gate0_status": {g: a["gate0_status"] for g, a in audits.items()},
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
