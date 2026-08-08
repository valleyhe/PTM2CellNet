import pytest

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


def _candidate(protein_id: str) -> CandidateRecord:
    return CandidateRecord("S1", protein_id, "phosphorylation", 1, "active", 0.8, 0.2, -0.6)


def test_mapper_normalizes_forward_reverse_and_container_lookups() -> None:
    mapper = ProteinGeneMapper({" p04637 ": " TP53 ", "P53_HUMAN": "tp53"})

    assert mapper.map_protein(" P04637 ") == "TP53"
    assert mapper.reverse_map("Tp53") == {"p04637", "P53_HUMAN"}
    assert mapper.reverse_map("missing") == set()
    assert "p04637" in mapper
    assert "missing" not in mapper
    assert mapper.size == len(mapper) == 2


def test_mapper_from_pairs_and_csv_schema_validation(tmp_path) -> None:
    assert ProteinGeneMapper.from_pairs([("P1", "G1")]).map_protein("p1") == "G1"

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("protein_id\nP1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gene_symbol"):
        ProteinGeneMapper.from_csv(str(invalid))


@pytest.mark.parametrize(
    "variant",
    ["9606.P04637", "P04637-2", "P04637.1"],
)
def test_map_candidate_fuzzy_matches_common_identifier_variants(variant) -> None:
    mapper = ProteinGeneMapper({"P04637": "TP53"})

    assert mapper.map_candidate(_candidate(variant)) == "TP53"


def test_map_candidate_rejects_unknown_identifier() -> None:
    mapper = ProteinGeneMapper({"P04637": "TP53"})

    with pytest.raises(KeyError, match="UNKNOWN"):
        mapper.map_candidate(_candidate("UNKNOWN"))


@pytest.mark.parametrize(
    "identifier",
    ["P04637", "UniProt:P04637", "sp|P04637|P53_HUMAN"],
)
def test_resolve_uniprot_accepts_common_formats(identifier) -> None:
    mapper = ProteinGeneMapper({"P04637": "TP53"})

    assert mapper.resolve_uniprot(identifier) == "TP53"


def test_map_many_skips_unknown_ids_without_reordering_known_genes() -> None:
    mapper = ProteinGeneMapper({"P1": "G1", "P2": "G2"})

    assert mapper.map_many(["missing", "P2", "P1", "P2"]) == ["G2", "G1"]
