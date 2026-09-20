import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from src.integration.perturbgen.contracts import CandidateEvidence, PerturbGenDataSpec
from src.integration.perturbgen.data_prep import (
    prepare_perturbgen_anndata,
    screen_candidate_for_perturbation,
)
from src.models.gene_vocabulary import GeneVocabularyEntry, GeneVocabularyResolver


def _make_adata(*, counts=None, donors=None, states=None, var=None):
    counts = np.asarray(
        counts
        if counts is not None
        else [
            [10, 5],
            [8, 4],
            [9, 0],
            [1, 7],
            [0, 6],
            [0, 8],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "cell_type": ["Mono"] * counts.shape[0],
            "state": states if states is not None else ["normal", "normal", "normal", "disease", "disease", "disease"],
            "donor": donors if donors is not None else ["d1", "d2", "d3", "d1", "d2", "d3"],
        },
        index=[f"cell_{idx}" for idx in range(counts.shape[0])],
    )
    var = (
        var.copy()
        if var is not None
        else pd.DataFrame(
            {"ensembl_id": ["ENSG00000141510.18", "ENSG00000146648"]},
            index=["TP53", "EGFR"],
        )
    )
    adata = anndata.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    return adata


def _candidate(**overrides):
    payload = {
        "gene_symbol": "TP53",
        "ensembl_id": "ENSG00000141510",
        "cell_type": "Mono",
        "ptm_context": "AKT1:S473",
        "observed_log2fc": 1.1,
        "observed_fdr": 0.02,
        "observed_direction": "up",
        "davf_action": "oe",
        "davf_score": 0.6,
        "davf_provenance": "davf-smoke",
    }
    payload.update(overrides)
    return CandidateEvidence(**payload)


def _resolver():
    return GeneVocabularyResolver(
        [
            GeneVocabularyEntry("ENSG00000141510", "TP53", token_id=1),
            GeneVocabularyEntry("ENSG00000146648", "EGFR", token_id=2),
        ]
    )


def test_prepare_perturbgen_anndata_validates_and_adds_n_counts():
    adata = _make_adata()
    prepared = prepare_perturbgen_anndata(adata, cell_type="Mono")

    assert prepared.report.evaluable_donor_count == 3
    assert tuple(prepared.report.evaluable_donors) == ("d1", "d2", "d3")
    assert "n_counts" in prepared.adata.obs.columns
    assert prepared.adata.obs["n_counts"].tolist() == [15.0, 12.0, 9.0, 8.0, 6.0, 8.0]
    assert prepared.adata.var["ensembl_id"].tolist()[0] == "ENSG00000141510"


def test_prepare_perturbgen_anndata_rejects_missing_counts_and_non_raw_values():
    adata = _make_adata()
    del adata.layers["counts"]
    with pytest.raises(ValueError, match="do not silently treat adata.X as raw counts"):
        prepare_perturbgen_anndata(adata, cell_type="Mono")

    bad = _make_adata(counts=[[0.1, 1.3], [2.2, 3.4], [4.1, 5.2], [6.1, 7.3], [8.4, 9.7], [10.2, 11.9]])
    with pytest.raises(ValueError, match="integer-like raw counts"):
        prepare_perturbgen_anndata(bad, cell_type="Mono")


def test_prepare_perturbgen_anndata_requires_three_shared_donors_and_unique_ensembl():
    few_donors = _make_adata(
        donors=["d1", "d2", "d1", "d1", "d2", "d4"],
        states=["normal", "normal", "normal", "disease", "disease", "disease"],
    )
    with pytest.raises(ValueError, match="requires at least 3 shared donors"):
        prepare_perturbgen_anndata(few_donors, cell_type="Mono")

    duplicate_var = pd.DataFrame(
        {"ensembl_id": ["ENSG00000141510.18", "ENSG00000141510.7"]},
        index=["TP53", "TP53_DUP"],
    )
    with pytest.raises(ValueError, match="duplicate ensembl ids"):
        prepare_perturbgen_anndata(_make_adata(var=duplicate_var), cell_type="Mono")


def test_between_donor_pairing_accepts_disjoint_case_control_donors():
    case_control = _make_adata(
        donors=["n1", "n2", "n3", "a1", "a2", "a3"],
        states=["normal", "normal", "normal", "disease", "disease", "disease"],
    )
    prepared = prepare_perturbgen_anndata(
        case_control, cell_type="Mono", spec=PerturbGenDataSpec(pairing="between_donor")
    )
    assert prepared.report.pairing == "between_donor"
    assert prepared.report.normal_donors == ("n1", "n2", "n3")
    assert prepared.report.disease_donors == ("a1", "a2", "a3")
    assert prepared.report.evaluable_donors == ("a1", "a2", "a3", "n1", "n2", "n3")
    assert prepared.report.normal_only_donors == ()
    assert prepared.report.disease_only_donors == ()


def test_between_donor_pairing_stays_opt_in_for_case_control_designs():
    case_control = _make_adata(
        donors=["n1", "n2", "n3", "a1", "a2", "a3"],
        states=["normal", "normal", "normal", "disease", "disease", "disease"],
    )
    with pytest.raises(ValueError, match="requires at least 3 shared donors"):
        prepare_perturbgen_anndata(case_control, cell_type="Mono")


def test_between_donor_pairing_requires_min_donors_per_group():
    thin_control_group = _make_adata(
        donors=["n1", "n2", "a1", "a2", "a3", "a4"],
        states=["normal", "normal", "disease", "disease", "disease", "disease"],
    )
    with pytest.raises(ValueError, match="at least 3 donors in each"):
        prepare_perturbgen_anndata(
            thin_control_group, cell_type="Mono", spec=PerturbGenDataSpec(pairing="between_donor")
        )


def test_between_donor_pairing_rejects_donor_in_both_states():
    with pytest.raises(ValueError, match="disjoint donor groups"):
        prepare_perturbgen_anndata(_make_adata(), cell_type="Mono", spec=PerturbGenDataSpec(pairing="between_donor"))


def test_screen_candidate_uses_observed_direction_not_davf_action():
    prepared = prepare_perturbgen_anndata(_make_adata(), cell_type="Mono")
    result = screen_candidate_for_perturbation(
        prepared, _candidate(observed_direction="up", davf_action="oe"), _resolver()
    )

    assert result.status == "evaluable"
    assert result.recommended_mode == "mask"
    assert result.candidate.davf_action == "oe"
    assert result.matched_ensembl_id == "ENSG00000141510"


def test_screen_candidate_marks_missing_expression_inconclusive():
    counts = [
        [0, 5],
        [0, 4],
        [0, 3],
        [0, 7],
        [0, 6],
        [0, 8],
    ]
    prepared = prepare_perturbgen_anndata(_make_adata(counts=counts), cell_type="Mono")
    result = screen_candidate_for_perturbation(prepared, _candidate(observed_direction="up"), _resolver())

    assert result.status == "inconclusive"
    assert result.reason_codes == ("gene_not_detected_in_target_cell_type",)


def test_source_ko_zero_expression_is_inconclusive_but_oe_is_not_filtered():
    counts = [
        [0, 5],
        [0, 4],
        [0, 3],
        [2, 7],
        [3, 6],
        [4, 8],
    ]
    prepared = prepare_perturbgen_anndata(_make_adata(counts=counts), cell_type="Mono")
    ko_result = screen_candidate_for_perturbation(prepared, _candidate(observed_direction="up"), _resolver())
    oe_result = screen_candidate_for_perturbation(
        prepared,
        _candidate(observed_direction="down", observed_log2fc=-1.0),
        _resolver(),
    )

    assert ko_result.status == "inconclusive"
    assert ko_result.reason_codes == ("source_gene_not_detected_in_normal_state",)
    assert oe_result.status == "evaluable"
    assert oe_result.recommended_mode == "overexpress"


def test_screen_candidate_fails_on_symbol_ensembl_mismatch():
    prepared = prepare_perturbgen_anndata(_make_adata(), cell_type="Mono", spec=PerturbGenDataSpec())
    bad_candidate = _candidate(gene_symbol="EGFR", ensembl_id="ENSG00000141510")
    result = screen_candidate_for_perturbation(prepared, bad_candidate, _resolver())

    assert result.status == "failed"
    assert result.reason_codes == ("candidate_symbol_ensembl_mismatch",)


def test_screen_candidate_requires_significant_observed_direction():
    prepared = prepare_perturbgen_anndata(_make_adata(), cell_type="Mono")
    result = screen_candidate_for_perturbation(prepared, _candidate(observed_fdr=0.2), _resolver())
    assert result.status == "inconclusive"
    assert result.reason_codes == ("observed_expression_not_significant",)


def test_screen_candidate_signed_admission_does_not_block_on_fdr():
    prepared = prepare_perturbgen_anndata(_make_adata(), cell_type="Mono")
    result = screen_candidate_for_perturbation(
        prepared,
        _candidate(observed_fdr=0.2),
        _resolver(),
        observed_significance_required=False,
    )
    assert "observed_expression_not_significant" not in result.reason_codes
