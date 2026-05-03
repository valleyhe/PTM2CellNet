from src.integration.contracts import CandidateRecord
from src.integration.ptm_gene_mapper import ProteinGeneMapper


def test_gene_mapper_deduplicates_gene_symbols() -> None:
    mapper = ProteinGeneMapper({"P04637": "TP53", "TP53_HUMAN": "TP53"})

    mapped = mapper.map_many(["P04637", "TP53_HUMAN"])

    assert mapped == ["TP53"]


def test_map_candidate_uses_candidate_protein_id() -> None:
    mapper = ProteinGeneMapper({"P04637": "TP53"})
    candidate = CandidateRecord(
        sample_id="S1",
        protein_id="P04637",
        ptm_type="phosphorylation",
        ptm_position=15,
        baseline_label="activated",
        baseline_probability=0.91,
        perturbed_probability=0.34,
        delta_probability=0.57,
    )

    assert mapper.map_candidate(candidate) == "TP53"


def test_mapper_can_be_loaded_from_csv(tmp_path) -> None:
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("protein_id,gene_symbol\nP04637,TP53\nP00533,EGFR\n", encoding="utf-8")

    mapper = ProteinGeneMapper.from_csv(str(mapping_file))

    assert mapper.map_many(["P04637", "P00533"]) == ["TP53", "EGFR"]
