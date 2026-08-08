"""Unit tests for the PMADS Ridge baseline."""

import json

import numpy as np
import pandas as pd
import pytest

from src.baselines.pmads_ridge import (
    PMADSRidgeBaseline,
    deterministic_split,
    prepare_pmads_frame,
    run_pmads_ridge,
    save_baseline_artifact,
)


def _frame(rows: int = 48) -> pd.DataFrame:
    records = []
    for index in range(rows):
        sequence = ("ACDEFGHIKLMNPQRSTVWY" * 3)[index : index + 12]
        label = index % 2
        records.append(
            {
                "sequence": sequence,
                "ptm_sites": json.dumps([{"position": 2 + (index % 4), "type": "phosphorylation"}]),
                "label": label,
                "effect": float(index) / rows,
                "protein_accession": f"P{index // 4:05d}",
                "intensity": float(index % 7),
            }
        )
    return pd.DataFrame(records)


def test_feature_extraction_and_split_are_deterministic():
    frame = prepare_pmads_frame(_frame())
    first = deterministic_split(frame, target_col="label", seed=7, group_col="protein_accession")
    second = deterministic_split(frame, target_col="label", seed=7, group_col="protein_accession")
    assert first["indices"] == second["indices"]
    train_groups = set(first["train"]["protein_accession"])
    test_groups = set(first["test"]["protein_accession"])
    assert train_groups.isdisjoint(test_groups)

    model = PMADSRidgeBaseline(alpha=0.5, task="classification")
    model.fit(first["train"], target_col="label")
    assert model._feature_matrix(first["train"]).shape[1] == len(model.feature_names_)
    assert any(name.startswith("aa_fraction_") for name in model.feature_names_)
    assert "label" not in model.numeric_columns_


def test_classification_baseline_runs_and_reports_metrics():
    result = run_pmads_ridge(
        _frame(),
        target_col="label",
        task="classification",
        seed=11,
        group_col="protein_accession",
    )
    assert set(result["metrics"]) == {"train", "validation", "test"}
    assert 0.0 <= result["metrics"]["test"]["accuracy"] <= 1.0
    assert len(result["predictions"]) == len(_frame())


def test_regression_baseline_runs():
    result = run_pmads_ridge(
        _frame(),
        target_col="effect",
        task="regression",
        seed=3,
        group_col="protein_accession",
    )
    assert "rmse" in result["metrics"]["test"]
    assert np.isfinite(result["metrics"]["test"]["rmse"])


def test_invalid_input_does_not_create_fallback_sequence():
    with pytest.raises(ValueError, match="sequence"):
        prepare_pmads_frame(pd.DataFrame({"label": [1], "ptm_position": [1]}))
    with pytest.raises(ValueError, match="target"):
        prepare_pmads_frame(_frame().drop(columns=["label"]), target_col="label")


def test_artifact_contains_metrics_predictions_and_provenance(tmp_path):
    result = run_pmads_ridge(_frame(), target_col="label", task="classification", seed=42)
    paths = save_baseline_artifact(result, tmp_path, parameters={"seed": 42}, is_demo_data=True)
    assert all((tmp_path / name).exists() for name in ("ridge_model.joblib", "metrics.json", "predictions.csv"))
    manifest = json.loads((tmp_path / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_class"] == "sklearn.linear_model.Ridge"
    assert manifest["not_for_biological_use"] is True
    assert manifest["split"]["seed"] == 42
    assert paths["manifest"].endswith("baseline_manifest.json")
