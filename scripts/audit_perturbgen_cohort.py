#!/usr/bin/env python
"""Audit local scPerturb h5ad files against the M0-6 donor cohort contract.

Contract source: docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md
§4.6 rule 1 — raw counts, version-less ENSG, unique obs_names, and
cell_type/state/donor annotations with >= 3 shared donors across
normal/disease states for the target cell type.

Policy (same doc, M1 risk table, and project_repair_report_20260822.md §5):
``sample``/``batch``/``replicate``/``cell_line`` columns are NEVER
reinterpreted as donor identifiers, and immortalized cell lines are not
donors. Only explicit donor-like columns (patient/donor/individual/...) count.

The script also executes the real M1 preflight
(``src.integration.perturbgen.data_prep.prepare_perturbgen_anndata``) against
DatlingerBock2021.h5ad — the raw-counts file validated by the M0-5 smoke — to
record the exact contract failure on real data. Fail-fast, no fabricated
column mappings: the preflight maps only columns that genuinely exist
(``celltype`` -> cell_type); state/donor have no real source in this file and
the contract must report them as missing.

Usage:
    python scripts/audit_perturbgen_cohort.py

Writes outputs/perturbgen/spike/<date>_donor_audit/evidence.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import warnings
from pathlib import Path

import anndata as ad

from src.integration.perturbgen.data_prep import prepare_perturbgen_anndata
from src.integration.perturbgen.contracts import PerturbGenDataSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "scperturb"
OUTPUT_DIR = REPO_ROOT / "outputs" / "perturbgen" / "spike" / (
    _dt.date.today().strftime("%Y%m%d") + "_donor_audit"
)

DONOR_LIKE_COLUMNS = ("patient", "donor", "patient_id", "donor_id", "individual")
# Only these values count as a normal-state label; missing/None/"" is absent
# annotation, not "normal".
NORMAL_STATE_VALUES = {"healthy", "normal"}
PREFLIGHT_FILE = "DatlingerBock2021.h5ad"
PREFLIGHT_CELL_TYPE = "T cells"
PREFLIGHT_MAX_CELLS = 5000


def audit_file(path: Path) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = ad.read_h5ad(path, backed="r")
    except Exception as exc:  # noqa: BLE001 - audit must record the failure
        return {"readable": False, "error": str(exc)[:200]}

    obs = adata.obs
    info: dict = {"readable": True, "n_cells": int(obs.shape[0])}

    info["obs_names_unique"] = bool(obs.index.is_unique)
    info["has_ensembl_id"] = "ensembl_id" in adata.var.columns

    donor_cols = {
        col: int(obs[col].nunique())
        for col in DONOR_LIKE_COLUMNS
        if col in obs.columns
    }
    if donor_cols:
        info["donor_like_columns"] = donor_cols
    if "cell_line" in obs.columns:
        info["cell_line_nunique"] = int(obs["cell_line"].nunique())
    if "tissue_type" in obs.columns:
        info["tissue_type_values"] = sorted(
            set(obs["tissue_type"].astype(str))
        )
    if "disease" in obs.columns:
        info["disease_values"] = {
            str(k): int(v)
            for k, v in obs["disease"].astype(str).value_counts().items()
        }

    tissue_types = set(info.get("tissue_type_values", []))
    disease_values = set(info.get("disease_values", {}))
    # State pairing must exist within ONE file: a healthy-class value and at
    # least one distinct disease value.
    has_normal = bool(disease_values & NORMAL_STATE_VALUES)
    has_disease = bool(disease_values - NORMAL_STATE_VALUES - {"None", ""})
    info["state_pair_possible"] = has_normal and has_disease
    max_donors = max(donor_cols.values()) if donor_cols else 0
    info["donor_count_ok"] = max_donors >= 3
    info["is_primary_tissue"] = bool(tissue_types) and tissue_types <= {
        "primary", "organoid", "primary_cells",
    }

    reasons: list[str] = []
    if not info["readable"]:
        reasons.append("unreadable file")
    if not info["obs_names_unique"]:
        reasons.append("obs_names not unique")
    if not info["has_ensembl_id"]:
        reasons.append("var lacks ensembl_id")
    if not info["is_primary_tissue"]:
        reasons.append(
            "tissue_type is not primary (cell lines are not donors per contract)"
        )
    if not donor_cols:
        reasons.append("no explicit donor-like column")
    elif not info["donor_count_ok"]:
        reasons.append(f"max donor-like column cardinality {max_donors} < 3")
    if not info["state_pair_possible"]:
        reasons.append(
            "no normal/disease state pair within the file "
            "(single disease state or no healthy-class label)"
        )
    info["compliant_candidate"] = not reasons
    info["rejection_reasons"] = reasons
    return info


def run_m1_preflight(path: Path) -> dict:
    """Execute the real M1 preflight on a real-data slice of one file.

    Slice, not synthesis: the first PREFLIGHT_MAX_CELLS cells keep all real
    obs columns and the full gene space. The contract failure mode (missing
    state/donor annotation) is independent of the row count.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # anndata 0.11 backed slicing is broken for CSR X (to_memory drops X,
        # direct slicing hits _validate_indices); load fully — the CSR matrix
        # for this file is ~700 MB in memory.
        full = ad.read_h5ad(path)
        end = min(full.n_obs, PREFLIGHT_MAX_CELLS)
        subset = full[:end].copy()
    # Real carrier move only: X holds raw integer counts (verified by the
    # M0-5 smoke, x_integer_ratio = 1.0); the contract requires layers[counts].
    subset.layers["counts"] = subset.X.copy()

    spec = PerturbGenDataSpec(cell_type_col="celltype")
    result: dict = {
        "file": path.name,
        "n_cells_sliced": int(end),
        "n_genes": int(subset.n_vars),
        "spec_cell_type_col": spec.cell_type_col,
        "spec_state_col": spec.state_col,
        "spec_donor_col": spec.donor_col,
        "state_column_exists": spec.state_col in subset.obs.columns,
        "donor_column_exists": spec.donor_col in subset.obs.columns,
    }
    try:
        report = prepare_perturbgen_anndata(
            subset, cell_type=PREFLIGHT_CELL_TYPE, spec=spec
        )
    except ValueError as exc:
        result["preflight"] = "REJECTED"
        result["contract_error"] = str(exc)
    else:  # pragma: no cover - real data is expected to fail the contract
        result["preflight"] = "ACCEPTED"
        result["report"] = {
            "evaluable_donors": list(report.evaluable_donors),
            "n_cells": report.n_cells,
        }
    return result


def main() -> int:
    files = sorted(DATA_DIR.glob("*.h5ad"))
    if not files:
        raise SystemExit(f"no h5ad files under {DATA_DIR}")

    audits = {path.name: audit_file(path) for path in files}
    readable = [name for name, info in audits.items() if info["readable"]]
    candidates = [
        name for name, info in audits.items() if info.get("compliant_candidate")
    ]

    preflight_target = DATA_DIR / PREFLIGHT_FILE
    preflight = run_m1_preflight(preflight_target)

    evidence = {
        "run_id": _dt.date.today().strftime("%Y%m%d") + "_donor_audit",
        "collected_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "M0-6 donor cohort audit per DAVF_PerturbGen plan section 4.6 "
            "rule 1 and M1 preflight on real data"
        ),
        "policy": {
            "donor_like_columns": list(DONOR_LIKE_COLUMNS),
            "normal_state_values": sorted(NORMAL_STATE_VALUES),
            "never_reinterpreted_as_donor": [
                "sample", "batch", "replicate", "cell_line",
                "CRISPR control", "plate well id",
            ],
            "state_pair_scope": "within a single file",
        },
        "files_audited": len(files),
        "files_readable": len(readable),
        "compliant_candidates": candidates,
        "audits": audits,
        "m1_preflight_datlinger": preflight,
        "verdict": "NO_LOCAL_COMPLIANT_COHORT" if not candidates else "CANDIDATES_FOUND",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "evidence.json"
    out_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))

    print(json.dumps({
        "evidence_file": str(out_path),
        "files_audited": len(files),
        "files_readable": len(readable),
        "compliant_candidates": candidates,
        "datlinger_preflight": preflight["preflight"],
        "datlinger_contract_error": preflight.get("contract_error"),
        "verdict": evidence["verdict"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
