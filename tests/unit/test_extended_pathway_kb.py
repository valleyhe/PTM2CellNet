import pytest

from src.data.extended_pathway_kb import (
    CELL_CYCLE_CHECKPOINTS,
    ExtendedPathwayKnowledgeBase,
    METABOLIC_PTM_REGULATION,
)


@pytest.fixture
def kb():
    return ExtendedPathwayKnowledgeBase()


class TestGetPathway:
    def test_returns_metabolic_pathway(self, kb):
        result = kb.get_pathway("glycolysis")
        assert result["category"] == "metabolic_ptm_regulation"
        assert "PFKFB3_S461ph" in result["sites"]

    def test_returns_cell_cycle_pathway(self, kb):
        result = kb.get_pathway("G2_M_checkpoint")
        assert result["category"] == "cell_cycle_checkpoint"
        assert "CDK1_T14ph" in result["sites"]

    def test_unknown_pathway_returns_empty_dict(self, kb):
        assert kb.get_pathway("not_a_pathway") == {}


class TestSearchPathways:
    def test_search_by_substring(self, kb):
        results = kb.search_pathways("checkpoint")
        names = {r["name"] for r in results}
        assert "G1_S_checkpoint" in names
        assert "G2_M_checkpoint" in names
        assert all("checkpoint" in r["name"].lower() for r in results)

    def test_search_is_case_insensitive(self, kb):
        lower = kb.search_pathways("glycolysis")
        upper = kb.search_pathways("GLYCOLYSIS")
        assert len(lower) == len(upper) > 0
        assert lower[0]["name"] == upper[0]["name"]

    def test_search_no_match_returns_empty_list(self, kb):
        assert kb.search_pathways("xyz_not_found") == []


class TestGetKinaseSubstrates:
    def test_ampk_substrates_in_metabolic_pathways(self, kb):
        results = kb.get_kinase_substrates("AMPK")
        site_ids = {r["site"] for r in results}
        assert "PFKFB3_S461ph" in site_ids
        assert "ACC1_S79ph" in site_ids
        assert all("AMPK" in (r.get("kinase") or "") for r in results)

    def test_cdk1_substrates_in_cell_cycle(self, kb):
        results = kb.get_kinase_substrates("CDK1")
        assert any(r["category"] == "cell_cycle_checkpoint" for r in results)

    def test_case_insensitive(self, kb):
        lower = kb.get_kinase_substrates("ampk")
        upper = kb.get_kinase_substrates("AMPK")
        assert len(lower) == len(upper) > 0

    def test_unknown_kinase_returns_empty_list(self, kb):
        assert kb.get_kinase_substrates("FAKE_KINASE") == []


class TestGetPtmAnnotations:
    def test_metabolic_site_annotation(self, kb):
        results = kb.get_ptm_annotations("PFKFB3")
        assert any(r["site"] == "PFKFB3_S461ph" for r in results)

    def test_conservation_annotation(self, kb):
        results = kb.get_ptm_annotations("AKT1")
        assert any(r["category"] == "ptm_conservation" for r in results)

    def test_disease_association_annotation(self, kb):
        results = kb.get_ptm_annotations("BRAF")
        assert any(r["category"] == "ptm_disease_association" for r in results)

    def test_case_insensitive(self, kb):
        lower = kb.get_ptm_annotations("akt1")
        upper = kb.get_ptm_annotations("AKT1")
        assert len(lower) == len(upper) > 0

    def test_unknown_accession_returns_empty_list(self, kb):
        assert kb.get_ptm_annotations("NOTREAL") == []


class TestDetectCrosstalk:
    def test_detects_shared_kinase(self, kb):
        result = kb.detect_crosstalk("glycolysis", "fatty_acid_metabolism")
        assert result["detected"] is True
        assert "AMPK" in result["shared_kinases"]
        assert result["crosstalk_score"] > 0.0

    def test_no_shared_elements(self, kb):
        result = kb.detect_crosstalk("glycolysis", "spindle_checkpoint")
        assert result["shared_kinases"] == []
        assert result["shared_sites"] == []
        assert result["detected"] is False

    def test_unknown_pathway_returns_zero_score(self, kb):
        result = kb.detect_crosstalk("glycolysis", "missing_pathway")
        assert result["detected"] is False
        assert result["crosstalk_score"] == 0.0

    def test_result_keys(self, kb):
        result = kb.detect_crosstalk("glycolysis", "tca_cycle")
        assert set(result.keys()) == {
            "pathway_a",
            "pathway_b",
            "shared_kinases",
            "shared_sites",
            "crosstalk_score",
            "detected",
        }


class TestModuleExports:
    def test_static_dicts_accessible(self):
        assert "PFKFB3_S461ph" in METABOLIC_PTM_REGULATION["glycolysis"]
        assert "G1_S_checkpoint" in CELL_CYCLE_CHECKPOINTS
