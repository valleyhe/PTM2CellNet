from src.integration.contracts import CandidateRecord, GenePerturbationRequest, PerturbationResult


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
