"""Unit tests for frozen AD candidate routing (方案 §6)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.ad_candidate_routing import (
    ADCandidateRoutingError,
    AxisCoverage,
    FROZEN_AD_CANDIDATES,
    LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY,
    exploratory_score,
    lineage_admission,
    load_axis_audit,
    load_grn_support,
    load_public_perturbation_inventory,
    parse_anchor_verdict,
    route_frozen_ad_candidates,
    summarize_observed_gate,
    write_routing_outputs,
)
from src.analysis.ad_research_decision import (
    FROZEN_AD_RESEARCH_DECISION,
    OBSERVED_ADMISSION_FDR_CUTOFF,
    OBSERVED_ADMISSION_SIGNED_DIRECTION,
    legacy_ad_research_decision,
)

_APOE = "ENSG00000130203"
_APP = "ENSG00000142192"
_BACE1 = "ENSG00000186318"


def _axis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gene_symbol": symbol,
                "ensembl_id": ensembl_id,
                "embedding_vocab": True,
                "ko_in_axis": symbol == "APOE",
                "kd_in_axis": False,
                "scperturb_perturbed_in": "FrangiehIzar2021_RNA" if symbol == "APOE" else "",
            }
            for symbol, ensembl_id in FROZEN_AD_CANDIDATES
        ]
    )


def _write_axis(path: Path) -> Path:
    _axis_frame().to_csv(path, sep="\t", index=False)
    return path


def _coverage_from_frame(frame: pd.DataFrame | None = None) -> dict[str, AxisCoverage]:
    table = frame if frame is not None else _axis_frame()
    coverage = {}
    for record in table.to_dict("records"):
        perturbed = tuple(part for part in str(record["scperturb_perturbed_in"]).split(";") if part)
        coverage[record["ensembl_id"]] = AxisCoverage(
            gene_symbol=record["gene_symbol"],
            ensembl_id=record["ensembl_id"],
            embedding_vocab=bool(record["embedding_vocab"]),
            ko_in_axis=bool(record["ko_in_axis"]),
            kd_in_axis=bool(record["kd_in_axis"]),
            scperturb_perturbed_in=perturbed,
        )
    return coverage


def _observed(*, max_fdr: float = 0.05, fdr: float = 0.52, admission_rule: str = OBSERVED_ADMISSION_SIGNED_DIRECTION):
    rows = []
    for _, ensembl_id in FROZEN_AD_CANDIDATES:
        rows.append(
            {
                "cell_type": "EX",
                "ensembl_id": ensembl_id,
                "gene_symbol": next(symbol for symbol, gene in FROZEN_AD_CANDIDATES if gene == ensembl_id),
                "log2fc": 0.2,
                "fdr": fdr,
                "observed_direction": "up",
                "n_normal_donors": 7,
                "n_disease_donors": 11,
            }
        )
    summary = summarize_observed_gate(pd.DataFrame(rows), max_fdr=max_fdr, admission_rule=admission_rule)
    return summary


def _route(**kwargs):
    defaults = {
        "axis_coverage": _coverage_from_frame(),
        "observed": _observed(),
        "anchor_verdict": "fail",
    }
    defaults.update(kwargs)
    return route_frozen_ad_candidates(**defaults)


def _by_key(report):
    return {(row.gene_symbol, row.intervention): row for row in report.rows}


class TestLineageAndScore:
    def test_lineage_never_admitted_even_on_anchor_pass(self):
        boundary, admitted = lineage_admission(anchor_verdict="pass")
        assert boundary == LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY
        assert admitted is False
        assert lineage_admission(anchor_verdict="fail") == (LINEAGE_BOUNDARY_SUPPLEMENTARY_ONLY, False)

    def test_exploratory_score_is_none_if_any_factor_missing(self):
        assert (
            exploratory_score(
                validation_calibration=0.8,
                context_similarity=None,
                model_agreement=1.0,
                network_support=1.0,
            )
            is None
        )
        assert exploratory_score(
            validation_calibration=0.5,
            context_similarity=0.5,
            model_agreement=0.5,
            network_support=0.5,
        ) == pytest.approx(0.0625)


class TestCurrentSplit:
    def test_apoe_ko_is_engineering_and_kd_is_not_emitted(self):
        report = _route()
        rows = _by_key(report)
        apoe_ko = rows[("APOE", "KO")]
        assert apoe_ko.selected_route == "APOE_KO_ENGINEERING"
        assert apoe_ko.status == "engineering_verification"
        assert apoe_ko.has_local_target_pair is True
        assert apoe_ko.formal_invocation_allowed is False
        assert apoe_ko.observed_gate_pass is True
        assert apoe_ko.observed_significant is False
        assert "observed_not_bh_significant" in apoe_ko.reasons
        assert ("APOE", "KD") not in rows
        assert all(row.intervention == "KO" for row in report.rows)
        assert report.coverage_claim == "five_candidates_ko_only"
        assert report.n_formal_invocations == 0
        assert report.may_enter_lineage is False
        assert report.to_payload()["biology_pass"] is False
        assert report.research_decision["kd_policy"] == FROZEN_AD_RESEARCH_DECISION.kd_policy

    def test_vocab_hit_is_not_training_axis_for_unseen_ko(self):
        rows = _by_key(_route())
        app_ko = rows[("APP", "KO")]
        assert app_ko.embedding_vocab is True
        assert app_ko.in_route_axis is False
        assert app_ko.selected_route == "B2"
        assert app_ko.status == "direction_only"
        assert "vocabulary_is_not_training_axis" in app_ko.reasons
        assert app_ko.predicted_direction_allowed is False
        assert app_ko.validation_status == "heldout_fail"

    def test_observed_and_intervention_axes_stay_separated(self):
        for row in _route().rows:
            assert row.semantic_context["reference_axis"] == "intervention_minus_control"
            assert row.observed_semantic_context["reference_axis"] == "disease_minus_normal"
            assert row.semantic_context["intervention"] == "KO"
            assert row.observed_semantic_context["intervention"] == "none"

    def test_signed_direction_admits_nonsignificant_fdr_without_relabeling(self):
        report = _route()
        assert report.n_observed_significant_rows == 0
        assert all(row.observed_gate_pass for row in report.rows)
        assert all(row.observed_significant is False for row in report.rows)
        assert all(row.formal_invocation_allowed is False for row in report.rows)
        assert report.n_formal_invocations == 0

    def test_fdr_below_threshold_still_does_not_create_invocation(self):
        observed = _observed(fdr=0.04)
        report = _route(observed=observed)
        assert report.n_observed_significant_rows == 5
        assert all(row.observed_gate_pass and row.observed_significant for row in report.rows)
        assert all(row.formal_invocation_allowed is False for row in report.rows)
        assert report.n_formal_invocations == 0

    def test_fdr_cutoff_admission_still_blocks_high_fdr(self):
        observed = _observed(fdr=0.52, admission_rule=OBSERVED_ADMISSION_FDR_CUTOFF)
        assert all(status.observed_gate_pass is False for status in observed.per_candidate.values())
        assert observed.n_significant_rows == 0


class TestInventoryAndGrn:
    def test_frozen_decision_rejects_public_inventory(self, tmp_path):
        inventory_path = tmp_path / "inventory.tsv"
        pd.DataFrame(
            [
                {
                    "dataset_id": "public_crispri",
                    "species": "human",
                    "cell_type": "iPSC_neuron",
                    "state": "differentiated",
                    "intervention": "KD",
                    "mechanism": "crispri",
                    "target_ensembl_id": _APP,
                    "target_gene_symbol": "APP",
                    "has_control": True,
                    "has_perturbed": True,
                }
            ]
        ).to_csv(inventory_path, sep="\t", index=False)
        inventory = load_public_perturbation_inventory(inventory_path)
        with pytest.raises(ADCandidateRoutingError, match="out of scope"):
            _route(inventory=inventory)

    def test_legacy_public_kd_inventory_does_not_use_ko_axis(self, tmp_path):
        inventory_path = tmp_path / "inventory.tsv"
        pd.DataFrame(
            [
                {
                    "dataset_id": "public_crispri",
                    "species": "human",
                    "cell_type": "iPSC_neuron",
                    "state": "differentiated",
                    "intervention": "KD",
                    "mechanism": "crispri",
                    "target_ensembl_id": _APP,
                    "target_gene_symbol": "APP",
                    "has_control": True,
                    "has_perturbed": True,
                }
            ]
        ).to_csv(inventory_path, sep="\t", index=False)
        inventory = load_public_perturbation_inventory(inventory_path)
        rows = _by_key(_route(inventory=inventory, decision=legacy_ad_research_decision()))
        assert rows[("APP", "KD")].selected_route == "B1"
        assert rows[("APP", "KD")].status == "exploratory_unvalidated"
        assert rows[("APOE", "KD")].status == "unavailable"
        assert rows[("APOE", "KO")].selected_route == "APOE_KO_ENGINEERING"

    def test_non_human_inventory_is_not_b1(self, tmp_path):
        inventory_path = tmp_path / "inventory.tsv"
        pd.DataFrame(
            [
                {
                    "dataset_id": "mouse_ko",
                    "species": "mouse",
                    "cell_type": "neuron",
                    "state": "adult",
                    "intervention": "KO",
                    "mechanism": "crispr_ko",
                    "target_ensembl_id": _APP,
                    "target_gene_symbol": "APP",
                    "has_control": True,
                    "has_perturbed": True,
                }
            ]
        ).to_csv(inventory_path, sep="\t", index=False)
        rows = _by_key(
            _route(
                inventory=load_public_perturbation_inventory(inventory_path),
                decision=legacy_ad_research_decision(),
            )
        )
        assert rows[("APP", "KO")].selected_route == "B2"
        assert rows[("APP", "KO")].public_target_specific is False

    def test_ko_row_with_crispri_mechanism_fails(self, tmp_path):
        inventory_path = tmp_path / "inventory.tsv"
        pd.DataFrame(
            [
                {
                    "dataset_id": "mixed",
                    "species": "human",
                    "cell_type": "neuron",
                    "state": "adult",
                    "intervention": "KO",
                    "mechanism": "crispri",
                    "target_ensembl_id": _APP,
                    "target_gene_symbol": "APP",
                    "has_control": True,
                    "has_perturbed": True,
                }
            ]
        ).to_csv(inventory_path, sep="\t", index=False)
        with pytest.raises(ADCandidateRoutingError, match="mixes KO intervention with KD mechanism"):
            load_public_perturbation_inventory(inventory_path)

    def test_mapping_conflict_fails(self, tmp_path):
        inventory_path = tmp_path / "inventory.tsv"
        pd.DataFrame(
            [
                {
                    "dataset_id": "bad_map",
                    "species": "human",
                    "cell_type": "neuron",
                    "state": "adult",
                    "intervention": "KO",
                    "mechanism": "ko",
                    "target_ensembl_id": "ENSG00000000001",
                    "target_gene_symbol": "APP",
                    "has_control": True,
                    "has_perturbed": True,
                }
            ]
        ).to_csv(inventory_path, sep="\t", index=False)
        with pytest.raises(ADCandidateRoutingError, match="mapping conflict"):
            load_public_perturbation_inventory(inventory_path)

    def test_bace1_without_grn_support_is_insufficient_not_zero_filled(self, tmp_path):
        grn_path = tmp_path / "grn.tsv"
        pd.DataFrame([{"target_ensembl_id": _BACE1, "has_tf_support": False, "has_direct_path": False}]).to_csv(
            grn_path, sep="\t", index=False
        )
        rows = _by_key(_route(grn_support=load_grn_support(grn_path)))
        assert rows[("BACE1", "KO")].grn_status == "insufficient"
        assert "grn_insufficient_no_zero_fill" in rows[("BACE1", "KO")].reasons
        assert rows[("BACE1", "KO")].status == "direction_only"

    def test_supported_grn_selects_b4(self, tmp_path):
        grn_path = tmp_path / "grn.tsv"
        pd.DataFrame([{"target_ensembl_id": _APP, "has_tf_support": True, "has_direct_path": True}]).to_csv(
            grn_path, sep="\t", index=False
        )
        rows = _by_key(_route(grn_support=load_grn_support(grn_path)))
        assert rows[("APP", "KO")].selected_route == "B4"
        assert rows[("APP", "KO")].evidence_kind == "network_counterfactual"


class TestHardStops:
    def test_counterfactual_predicted_direction_is_rejected(self):
        asset = {
            "evidence_kind": "network_counterfactual",
            "lineage_boundary": "supplementary_only",
            "candidates": {_APOE: {"predicted_direction": "up"}},
        }
        with pytest.raises(ADCandidateRoutingError, match="must not carry predicted_direction"):
            _route(external_assets=[asset])

    def test_non_supplementary_lineage_boundary_is_rejected(self):
        with pytest.raises(ADCandidateRoutingError, match="supplementary_only"):
            _route(external_assets=[{"evidence_kind": "go_extrapolation", "lineage_boundary": "formal"}])

    def test_routing_refuses_to_change_fdr_threshold(self):
        observed = _observed(max_fdr=0.2, fdr=0.1)
        with pytest.raises(ADCandidateRoutingError, match="FDR threshold"):
            _route(observed=observed)

    def test_axis_audit_rejects_unknown_gene(self, tmp_path):
        path = tmp_path / "axis.tsv"
        frame = _axis_frame()
        frame.loc[0, "gene_symbol"] = "TP53"
        frame.to_csv(path, sep="\t", index=False)
        with pytest.raises(ADCandidateRoutingError, match="outside the frozen AD catalog"):
            load_axis_audit(path)

    def test_b6_contraction_blocks_non_apoe(self):
        rows = _by_key(_route(contract_to_apoe_ko=True))
        assert rows[("APOE", "KO")].selected_route == "APOE_KO_ENGINEERING"
        assert rows[("APP", "KO")].selected_route == "B6"
        assert rows[("APP", "KO")].status == "blocked"
        assert ("APOE", "KD") not in rows


class TestOutputs:
    def test_write_outputs_keeps_formal_count_zero(self, tmp_path):
        report = _route()
        outputs = write_routing_outputs(report, tmp_path)
        payload = json.loads(outputs["report"].read_text(encoding="utf-8"))
        assert payload["n_formal_invocations"] == 0
        assert payload["may_enter_lineage"] is False
        candidates = pd.read_csv(outputs["candidates"], sep="\t")
        assert list(candidates["ensembl_id"]) == [ensembl_id for _, ensembl_id in FROZEN_AD_CANDIDATES]
        ranking = pd.read_csv(outputs["ranking"], sep="\t")
        assert list(ranking["intervention"].unique()) == ["KO"]
        decision = json.loads(outputs["observed_gate_decision"].read_text(encoding="utf-8"))
        assert decision["observed_admission_rule"] == OBSERVED_ADMISSION_SIGNED_DIRECTION
        assert decision["biology_pass"] is False
        assert parse_anchor_verdict({"verdict": "fail"}) == "fail"

    def test_real_axis_audit_reproduces_current_five_candidate_split(self):
        path = Path("outputs/ptm_activity/20260916_d1/davf_axis_coverage_audit.tsv")
        if not path.exists():
            pytest.skip("local axis audit artifact is not present")
        report = route_frozen_ad_candidates(
            axis_coverage=load_axis_audit(path),
            observed=summarize_observed_gate(None),
            anchor_verdict="fail",
        )
        rows = _by_key(report)
        assert rows[("APOE", "KO")].selected_route == "APOE_KO_ENGINEERING"
        assert ("APOE", "KD") not in rows
        for symbol in ("APP", "PSEN1", "BACE1", "MAPT"):
            assert rows[(symbol, "KO")].selected_route == "B2"
            assert rows[(symbol, "KO")].status == "direction_only"
            assert (symbol, "KD") not in rows
        assert report.n_formal_invocations == 0
        assert report.coverage_claim == "five_candidates_ko_only"
