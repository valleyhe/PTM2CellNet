from src.integration.contracts import (
    BatchPerturbationRequest,
    BatchPerturbationResult,
    CandidateRecord,
    GenePerturbationRequest,
    PerturbationResult,
)


def test_candidate_record_creation_and_fields() -> None:
    record = CandidateRecord(
        sample_id="S1",
        protein_id="P04637",
        ptm_type="phosphorylation",
        ptm_position=15,
        baseline_label="activated",
        baseline_probability=0.91,
        perturbed_probability=0.34,
        delta_probability=0.57,
    )
    assert record.sample_id == "S1"
    assert record.protein_id == "P04637"
    assert record.ptm_type == "phosphorylation"
    assert record.ptm_position == 15
    assert record.baseline_label == "activated"
    assert record.baseline_probability == 0.91
    assert record.perturbed_probability == 0.34
    assert record.delta_probability == 0.57


def test_candidate_record_is_frozen() -> None:
    record = CandidateRecord(
        sample_id="S1",
        protein_id="P04637",
        ptm_type="phosphorylation",
        ptm_position=15,
        baseline_label="activated",
        baseline_probability=0.91,
        perturbed_probability=0.34,
        delta_probability=0.57,
    )
    try:
        record.delta_probability = 0.99
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower() or "has no setter" in str(exc).lower()
    else:
        raise AssertionError("Expected frozen dataclass mutation to fail")


def test_gene_perturbation_request_creation_and_fields() -> None:
    request = GenePerturbationRequest(
        gene_symbol="TP53",
        source_protein_id="P04637",
        source_ptm_type="phosphorylation",
        source_ptm_position=15,
        magnitude=0.6,
        mode="hard_ko",
    )
    assert request.gene_symbol == "TP53"
    assert request.source_protein_id == "P04637"
    assert request.source_ptm_type == "phosphorylation"
    assert request.source_ptm_position == 15
    assert request.magnitude == 0.6
    assert request.mode == "hard_ko"


def test_gene_perturbation_request_is_frozen() -> None:
    request = GenePerturbationRequest(
        gene_symbol="TP53",
        source_protein_id="P04637",
        source_ptm_type="phosphorylation",
        source_ptm_position=15,
        magnitude=0.6,
        mode="hard_ko",
    )
    try:
        request.mode = "soft_ptm"
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower() or "has no setter" in str(exc).lower()
    else:
        raise AssertionError("Expected frozen dataclass mutation to fail")


def test_perturbation_result_creation_fields_and_metadata_default() -> None:
    result = PerturbationResult(
        gene_symbol="TP53",
        mode="hard_ko",
        distance_score=1.23,
        ranked_genes=["BAX", "MDM2"],
    )
    assert result.gene_symbol == "TP53"
    assert result.mode == "hard_ko"
    assert result.distance_score == 1.23
    assert result.ranked_genes == ["BAX", "MDM2"]
    assert result.metadata == {}


def test_perturbation_result_preserves_custom_metadata() -> None:
    result = PerturbationResult(
        gene_symbol="TP53",
        mode="hard_ko",
        distance_score=1.23,
        ranked_genes=["BAX", "MDM2"],
        metadata={"pathways": ["apoptosis"]},
    )
    assert result.metadata["pathways"] == ["apoptosis"]


def test_perturbation_result_is_frozen() -> None:
    result = PerturbationResult(
        gene_symbol="TP53",
        mode="hard_ko",
        distance_score=1.23,
        ranked_genes=["BAX", "MDM2"],
    )
    try:
        result.distance_score = 2.0
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "cannot" in str(exc).lower() or "has no setter" in str(exc).lower()
    else:
        raise AssertionError("Expected frozen dataclass mutation to fail")


def test_contract_field_types_are_correct() -> None:
    record = CandidateRecord(
        sample_id="S1",
        protein_id="P1",
        ptm_type="p",
        ptm_position=1,
        baseline_label="a",
        baseline_probability=0.5,
        perturbed_probability=0.6,
        delta_probability=0.1,
    )
    assert isinstance(record.sample_id, str)
    assert isinstance(record.ptm_position, int)
    assert isinstance(record.delta_probability, float)

    request = GenePerturbationRequest(
        gene_symbol="G1", source_protein_id="P1", source_ptm_type="p",
        source_ptm_position=1, magnitude=0.5, mode="hard_ko",
    )
    assert isinstance(request.gene_symbol, str)
    assert isinstance(request.magnitude, float)

    result = PerturbationResult(
        gene_symbol="G1", mode="hard_ko", distance_score=1.0,
        ranked_genes=["A"], metadata={"k": "v"},
    )
    assert isinstance(result.ranked_genes, list)
    assert isinstance(result.metadata, dict)


def test_candidate_record_validate_reports_all_invalid_fields() -> None:
    record = CandidateRecord("", "", "p", 0, "state", -0.1, 1.1, 0.0)

    assert record.validate() == [
        "sample_id must not be empty",
        "protein_id must not be empty",
        "ptm_position must be >= 1",
        "baseline_probability must be in [0, 1]",
        "perturbed_probability must be in [0, 1]",
    ]


def test_gene_request_validate_accepts_modes_and_reports_invalid_values() -> None:
    valid = GenePerturbationRequest("TP53", "P1", "p", 1, 0.5, "soft_ptm")
    invalid = GenePerturbationRequest("", "P1", "p", 0, 0.0, "delete")

    assert valid.validate() == []
    assert invalid.validate() == [
        "gene_symbol must not be empty",
        "mode must be 'hard_ko' or 'soft_ptm', got 'delete'",
        "magnitude must be positive",
        "source_ptm_position must be >= 1",
    ]


def test_perturbation_result_validate_requires_gene_and_ranking() -> None:
    assert PerturbationResult("TP53", "hard_ko", 0.2, ["BAX"]).validate() == []
    assert PerturbationResult("", "hard_ko", 0.2, []).validate() == [
        "gene_symbol must not be empty",
        "ranked_genes must not be empty",
    ]


def test_batch_request_validation_handles_empty_nested_and_fail_fast() -> None:
    invalid = GenePerturbationRequest("", "P1", "p", 0, 0.0, "bad")
    another_invalid = GenePerturbationRequest("", "P2", "p", 1, 1.0, "hard_ko")

    assert BatchPerturbationRequest([]).validate() == ["requests must not be empty"]
    all_issues = BatchPerturbationRequest([invalid, another_invalid]).validate()
    assert any(issue.startswith("requests[0]:") for issue in all_issues)
    assert any(issue.startswith("requests[1]:") for issue in all_issues)

    fail_fast = BatchPerturbationRequest([invalid, another_invalid], fail_fast=True).validate()
    assert fail_fast
    assert all(issue.startswith("requests[0]:") for issue in fail_fast)


def test_batch_result_validation_prefixes_nested_result_index() -> None:
    valid = PerturbationResult("TP53", "hard_ko", 0.2, ["BAX"])
    invalid = PerturbationResult("", "hard_ko", 0.1, [])

    result = BatchPerturbationResult([valid, invalid], failed_requests=["G2"], total_time_ms=1.5)

    assert result.validate() == [
        "results[1]: gene_symbol must not be empty",
        "results[1]: ranked_genes must not be empty",
    ]
