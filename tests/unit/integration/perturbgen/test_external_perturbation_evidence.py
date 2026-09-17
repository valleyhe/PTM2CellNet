"""Tests for the field-separated external perturbation-model evidence contract."""

from __future__ import annotations

import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.integration.perturbgen.external_perturbation_evidence import (
    EXTERNAL_PERTURBATION_EVIDENCE_SCHEMA_VERSION,
    ExternalPerturbationEvidenceError,
    evidence_missing_candidates,
    external_evidence_to_payload,
    load_external_prediction_asset,
)

_CANDIDATES = [
    ("ENSG00000130203", "APOE"),  # self readout present, positive delta
    ("ENSG00000142192", "APP"),  # self readout present, negative delta
    ("ENSG00000186868", "MAPT"),  # self readout present, zero delta -> indeterminate
    ("ENSG00000186318", "BACE1"),  # self readout absent -> explicit None
]
_READOUTS = ["ENSG00000130203", "ENSG00000142192", "ENSG00000186868", "ENSG00000099999", "ENSG00000100001"]


def _write_asset(tmp_path, *, delta_modifier=None, manifest_modifier=None, obs_modifier=None, var_names=None):
    delta = np.array(
        [
            [0.50, -0.10, 0.00, 0.20, -0.30],  # APOE: self row +0.50 -> up
            [-0.25, -0.60, 0.10, 0.00, 0.15],  # APP: self row -0.60 -> down
            [0.05, -0.02, 0.00, 0.30, 0.30],  # MAPT: self row 0.00 -> indeterminate
            [0.00, 0.40, -0.15, 0.10, 0.10],  # BACE1: no self row -> None
        ],
        dtype=float,
    )
    if delta_modifier is not None:
        delta = delta_modifier(delta)
    obs = pd.DataFrame(
        {
            "ensembl_id": [row[0] for row in _CANDIDATES],
            "gene_symbol": [row[1] for row in _CANDIDATES],
            "context": ["HumanCellLine_K562_10xChromium3-scRNA-seq_Replogle22"] * len(_CANDIDATES),
            "perturbation_semantics": ["crispri_kd"] * len(_CANDIDATES),
        },
        index=[row[1] for row in _CANDIDATES],
    )
    if obs_modifier is not None:
        obs = obs_modifier(obs)
    adata = ad.AnnData(X=delta, obs=obs)
    adata.var_names = [str(name) for name in (var_names or _READOUTS)]
    h5ad_path = tmp_path / "lpm_predictions.h5ad"
    adata.write_h5ad(h5ad_path)
    manifest = {
        "schema_version": "ptm2cellnet.external-perturbation-prediction/v1",
        "evidence_kind": "trained_response",
        "model": {
            "source": "gsk_lpm/perturblib",
            "perturblib_commit": "abc1234",
            "training_config": "replogle_k562_paper_lpm",
            "license": "Apache-2.0",
        },
        "context": "HumanCellLine_K562_10xChromium3-scRNA-seq_Replogle22",
        "perturbation_semantics": "crispri_kd",
        "seeds": [0, 1, 2, 3, 4],
        "aggregation": "mean over seed checkpoints of (perturbed - control) predicted readout",
        "predictions_h5ad": h5ad_path.name,
        "predictions_sha256": hashlib.sha256(h5ad_path.read_bytes()).hexdigest(),
    }
    if manifest_modifier is not None:
        manifest = manifest_modifier(manifest)
    manifest_path = tmp_path / "lpm_predictions.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class TestLoadLpmPredictionAsset:
    def test_valid_asset_loads_with_self_directions(self, tmp_path):
        evidence = load_external_prediction_asset(_write_asset(tmp_path))
        assert evidence.perturbation_semantics == "crispri_kd"
        assert evidence.seeds == (0, 1, 2, 3, 4)
        assert evidence.n_readout_genes == 5
        by_id = {prediction.ensembl_id: prediction for prediction in evidence.predictions}
        assert by_id["ENSG00000130203"].self_delta == pytest.approx(0.50)
        assert by_id["ENSG00000130203"].predicted_direction == "up"
        assert by_id["ENSG00000142192"].self_delta == pytest.approx(-0.60)
        assert by_id["ENSG00000142192"].predicted_direction == "down"
        assert by_id["ENSG00000186868"].predicted_direction == "indeterminate"
        # BACE1 has no self readout row: explicit None, never zero-filled.
        assert by_id["ENSG00000186318"].self_delta is None
        assert by_id["ENSG00000186318"].predicted_direction is None

    def test_wrong_schema_version_fails(self, tmp_path):
        manifest_path = _write_asset(tmp_path, manifest_modifier=lambda m: {**m, "schema_version": "other/v9"})
        with pytest.raises(ExternalPerturbationEvidenceError, match="schema_version"):
            load_external_prediction_asset(manifest_path)

    def test_hash_drift_fails(self, tmp_path):
        manifest_path = _write_asset(tmp_path, manifest_modifier=lambda m: {**m, "predictions_sha256": "0" * 64})
        with pytest.raises(ExternalPerturbationEvidenceError, match="hash drifted"):
            load_external_prediction_asset(manifest_path)

    def test_rejected_semantics_includes_mainline_token_mask(self, tmp_path):
        def switch_semantics(manifest):
            manifest = {**manifest, "perturbation_semantics": "token_mask_ko"}
            return manifest

        manifest_path = _write_asset(
            tmp_path,
            manifest_modifier=switch_semantics,
            obs_modifier=lambda obs: obs.assign(perturbation_semantics="token_mask_ko"),
        )
        with pytest.raises(ExternalPerturbationEvidenceError, match="perturbation_semantics must be one of"):
            load_external_prediction_asset(manifest_path)

    def test_non_ensembl_readout_fails(self, tmp_path):
        manifest_path = _write_asset(tmp_path, var_names=_READOUTS[:4] + ["GENE_SYMBOL_NOT_ENSEMBL"])
        with pytest.raises(ExternalPerturbationEvidenceError, match="canonical Ensembl"):
            load_external_prediction_asset(manifest_path)

    def test_non_finite_delta_fails(self, tmp_path):
        def poison(delta):
            delta = delta.copy()
            delta[0, 3] = np.nan
            return delta

        manifest_path = _write_asset(tmp_path, delta_modifier=poison)
        with pytest.raises(ExternalPerturbationEvidenceError, match="non-finite"):
            load_external_prediction_asset(manifest_path)

    def test_duplicate_candidate_fails(self, tmp_path):
        def duplicate(obs):
            obs = obs.copy()
            obs.iloc[1, obs.columns.get_loc("ensembl_id")] = obs.iloc[0, obs.columns.get_loc("ensembl_id")]
            return obs

        manifest_path = _write_asset(tmp_path, obs_modifier=duplicate)
        with pytest.raises(ExternalPerturbationEvidenceError, match="duplicate candidate"):
            load_external_prediction_asset(manifest_path)

    def test_invalid_evidence_kind_fails(self, tmp_path):
        manifest_path = _write_asset(tmp_path, manifest_modifier=lambda m: {**m, "evidence_kind": "causal_proof"})
        with pytest.raises(ExternalPerturbationEvidenceError, match="evidence_kind must be one of"):
            load_external_prediction_asset(manifest_path)

    def test_go_extrapolation_and_counterfactual_kinds_accepted(self, tmp_path):
        def to_gears(manifest):
            manifest = dict(manifest)
            manifest["evidence_kind"] = "go_extrapolation"
            manifest["perturbation_semantics"] = "unseen_perturbation_extrapolation"
            return manifest

        def obs_to_gears(obs):
            obs = obs.copy()
            obs["perturbation_semantics"] = "unseen_perturbation_extrapolation"
            return obs

        evidence = load_external_prediction_asset(
            _write_asset(tmp_path, manifest_modifier=to_gears, obs_modifier=obs_to_gears)
        )
        assert evidence.evidence_kind == "go_extrapolation"
        payload = external_evidence_to_payload(evidence)
        assert "GO-graph extrapolation" in payload["boundary"]

    def test_context_mismatch_between_manifest_and_h5ad_fails(self, tmp_path):
        def rename(obs):
            obs = obs.copy()
            obs["context"] = "SomeOtherContext"
            return obs

        manifest_path = _write_asset(tmp_path, obs_modifier=rename)
        with pytest.raises(ExternalPerturbationEvidenceError, match="do not match manifest context"):
            load_external_prediction_asset(manifest_path)


class TestPayloadAndCoverage:
    def test_payload_separates_provenance_design_and_boundary(self, tmp_path):
        evidence = load_external_prediction_asset(_write_asset(tmp_path))
        payload = external_evidence_to_payload(evidence)
        assert payload["schema_version"] == EXTERNAL_PERTURBATION_EVIDENCE_SCHEMA_VERSION
        assert payload["source"]["model_source"] == "gsk_lpm/perturblib"
        assert payload["source"]["predictions_sha256"] == evidence.predictions_sha256
        assert payload["design"]["context"].startswith("HumanCellLine_K562")
        assert payload["design"]["seeds"] == [0, 1, 2, 3, 4]
        assert payload["candidates"]["ENSG00000130203"]["predicted_direction"] == "up"
        assert payload["candidates"]["ENSG00000186318"]["predicted_direction"] is None
        assert "trained perturbation response" in payload["boundary"]
        assert "never a pass/fail decision" in payload["boundary"]

    def test_missing_candidates_are_explicit(self, tmp_path):
        evidence = load_external_prediction_asset(_write_asset(tmp_path))
        requested = {"ENSG00000130203": "APOE", "ENSG00000130204": "NEW_GENE"}
        assert evidence_missing_candidates(evidence, requested) == ("ENSG00000130204",)
        assert evidence_missing_candidates(evidence, {"ENSG00000130203": "APOE"}) == ()
