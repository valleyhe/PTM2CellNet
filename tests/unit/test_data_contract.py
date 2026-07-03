"""Data contract + release-gate tests (P1-1).

Covers:
* ``src.data.data_contract.validate_data_contract`` / ``profile_dataset`` — the
  schema a real dataset must satisfy plus the provenance profile written into
  ``artifact_manifest.json``.
* ``src.training.artifacts.evaluate_release_gate`` — the configurable metric
  threshold gate that flips ``deployable``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.data_contract import (
    REQUIRED_COLUMNS,
    DataContractError,
    profile_dataset,
    validate_data_contract,
)
from src.training.artifacts import evaluate_release_gate


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

def _good_rows():
    return [
        {"sequence": "ACDEFGHIK", "ptm_sites": '[{"position":1,"type":"phosphorylation"}]', "cell_state": "a"},
        {"sequence": "ACDEFGHIK", "ptm_sites": "[]", "cell_state": "b"},
    ]


class TestValidateDataContract:
    def test_valid_dataset_passes(self):
        df = pd.DataFrame(_good_rows())
        report = validate_data_contract(df)
        assert report["ok"] is True
        assert not report["hard_failures"]

    def test_missing_required_column_is_hard_failure(self):
        df = pd.DataFrame(_good_rows()).drop(columns=["cell_state"])
        report = validate_data_contract(df)
        assert report["ok"] is False
        assert "cell_state" in " ".join(report["hard_failures"])

    def test_required_columns_constant(self):
        assert REQUIRED_COLUMNS == ("sequence", "ptm_sites", "cell_state")

    def test_out_of_range_ptm_position_warns_by_default(self):
        rows = _good_rows()
        rows[0]["ptm_sites"] = '[{"position":999,"type":"phosphorylation"}]'
        df = pd.DataFrame(rows)
        report = validate_data_contract(df)
        assert report["invalid_ptm_rows"] == 1
        # default mode => warning, not hard failure
        assert report["ok"] is True
        assert any("position" in w for w in report["warnings"])

    def test_out_of_range_ptm_position_hard_when_strict(self):
        rows = _good_rows()
        rows[0]["ptm_sites"] = '[{"position":0,"type":"phosphorylation"}]'
        df = pd.DataFrame(rows)
        report = validate_data_contract(df, require_all_rows_valid_ptm=True)
        assert report["ok"] is False
        assert report["invalid_ptm_rows"] == 1

    def test_nonstandard_aa_warns(self):
        rows = _good_rows()
        rows[0]["sequence"] = "ACXDEFGHIK"  # X is non-standard
        df = pd.DataFrame(rows)
        report = validate_data_contract(df)
        assert any("非标准氨基酸" in w for w in report["warnings"])

    def test_malformed_ptm_json_raises(self):
        rows = _good_rows()
        rows[0]["ptm_sites"] = "not-json"
        df = pd.DataFrame(rows)
        with pytest.raises(DataContractError):
            validate_data_contract(df)

    def test_recommended_columns_missing_warns(self):
        df = pd.DataFrame(_good_rows())
        report = validate_data_contract(df)
        assert report["missing_recommended"]  # none of the recommended cols present
        assert any("provenance" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Dataset profile
# ---------------------------------------------------------------------------

class TestProfileDataset:
    def test_profile_includes_label_distribution_and_length(self):
        df = pd.DataFrame(_good_rows())
        profile = profile_dataset(df)
        assert profile["row_count"] == 2
        assert profile["num_classes"] == 2
        assert "label_distribution" in profile
        assert profile["sequence_length"]["min"] == 9

    def test_profile_flags_homology_leakage(self):
        df = pd.DataFrame([
            {"sequence": "AAAA", "ptm_sites": "[]", "cell_state": "a",
             "protein_accession": "P00001", "split_group": "train"},
            {"sequence": "CCCC", "ptm_sites": "[]", "cell_state": "b",
             "protein_accession": "P00001", "split_group": "test"},
        ])
        profile = profile_dataset(df)
        assert profile["homology_leakage_accessions"] == 1
        assert "homology_leakage_warning" in profile

    def test_ptm_type_distribution_counted(self):
        df = pd.DataFrame([
            {"sequence": "AAA", "ptm_sites": '[{"position":1,"type":"phosphorylation"}]', "cell_state": "a"},
            {"sequence": "AAA", "ptm_sites": '[{"position":1,"type":"phosphorylation"},{"position":2,"type":"acetylation"}]', "cell_state": "b"},
        ])
        profile = profile_dataset(df)
        assert profile["ptm_type_distribution"]["phosphorylation"] == 2
        assert profile["ptm_type_distribution"]["acetylation"] == 1


# ---------------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------------

class TestEvaluateReleaseGate:
    def test_all_metrics_meet_thresholds(self):
        metrics = {"macro_f1": 0.7, "auc_roc": 0.85}
        gate = evaluate_release_gate(metrics, {"macro_f1": 0.6, "auc_roc": 0.8})
        assert gate["passed"] is True
        assert gate["missing"] == []

    def test_below_threshold_fails(self):
        metrics = {"macro_f1": 0.4}
        gate = evaluate_release_gate(metrics, {"macro_f1": 0.6})
        assert gate["passed"] is False
        assert gate["checked"]["macro_f1"]["passed"] is False

    def test_missing_metric_fails(self):
        gate = evaluate_release_gate({"macro_f1": 0.9}, {"macro_f1": 0.6, "auc_pr": 0.5})
        assert gate["passed"] is False
        assert "auc_pr" in gate["missing"]

    def test_threshold_met_at_boundary(self):
        gate = evaluate_release_gate({"macro_f1": 0.6}, {"macro_f1": 0.6})
        assert gate["passed"] is True
