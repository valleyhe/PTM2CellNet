"""Tests for the frozen PTM research-design contract (方案 §7 阶段 0)."""

from __future__ import annotations

import pytest

from src.analysis.ptm_research_config import (
    PropagationConfig,
    PTMResearchConfigError,
    load_ptm_research_config,
    parse_ptm_research_config,
)


def _base_payload() -> dict:
    return {
        "schema_version": "ptm2cellnet.ptm-research-config/v1",
        "research_objective": "association",
        "reference_axis": "disease_minus_normal",
        "contrast": "disease-minus-normal",
        "primary_activity_method": "KSTAR",
        "network_release": "2026-08",
        "cell_types": ["EX", "IN"],
        "cohort_h5ad": "data/AD/standardized/GSE174367_ad_cohort.h5ad",
        "cohort_pairing": "between_donor",
        "species": "9606",
        "ptm_cohort": "CPTAC_TEST",
        "deg_max_fdr": 0.05,
        "min_donors_per_state": 3,
        "replicate_policy": "mean",
        "propagation": {"max_depth": 3, "decay": 0.5, "gene_edge_types": ["tf_regulation"]},
        "semantic_context": {
            "context": "GSE174367 {cell_type} cells",
            "intervention": "KO",
            "comparison_baseline": "donor-level disease vs normal",
            "reference_axis": "disease_minus_normal",
            "research_objective": "association",
            "evidence_source": "PTM activity network concordance + donor DEG",
            "cohort": "GSE174367",
        },
    }


class TestPTMResearchConfig:
    def test_valid_config_round_trips(self, tmp_path):
        path = tmp_path / "ptm_research_config.yaml"
        import yaml

        path.write_text(yaml.safe_dump(_base_payload()), encoding="utf-8")
        config = load_ptm_research_config(path)
        assert config.cell_types == ("EX", "IN")
        assert config.propagation.gene_edge_types == ("tf_regulation",)
        assert config.semantic_context_for_cell_type("EX")["context"] == "GSE174367 EX cells"

    def test_missing_key_fails(self):
        payload = _base_payload()
        del payload["deg_max_fdr"]
        with pytest.raises(PTMResearchConfigError, match="deg_max_fdr"):
            parse_ptm_research_config(payload)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("research_objective", "causal"),
            ("reference_axis", "flip_sign"),
            ("cohort_pairing", "across_species"),
            ("replicate_policy", "first"),
            ("deg_max_fdr", 0.0),
            ("min_donors_per_state", 0),
        ],
    )
    def test_invalid_values_fail(self, field, value):
        payload = _base_payload()
        payload[field] = value
        with pytest.raises(PTMResearchConfigError):
            parse_ptm_research_config(payload)

    def test_propagation_bounds(self):
        payload = _base_payload()
        payload["propagation"] = {"max_depth": 0, "decay": 0.5, "gene_edge_types": ["tf_regulation"]}
        with pytest.raises(PTMResearchConfigError, match="max_depth"):
            parse_ptm_research_config(payload)
        payload["propagation"] = {"max_depth": 2, "decay": 1.5, "gene_edge_types": ["tf_regulation"]}
        with pytest.raises(PTMResearchConfigError, match="decay"):
            parse_ptm_research_config(payload)
        payload["propagation"] = {"max_depth": 2, "decay": 0.5, "gene_edge_types": []}
        with pytest.raises(PTMResearchConfigError, match="gene_edge_types"):
            parse_ptm_research_config(payload)

    def test_max_paths_per_seed_defaults_to_none_and_accepts_positive_integer(self):
        config = parse_ptm_research_config(_base_payload())
        assert config.propagation.max_paths_per_seed is None

        payload = _base_payload()
        payload["propagation"]["max_paths_per_seed"] = 3
        config = parse_ptm_research_config(payload)
        assert config.propagation.max_paths_per_seed == 3

    @pytest.mark.parametrize("value", [0, -1, 1.5, True, "3"])
    def test_invalid_max_paths_per_seed_fails(self, value):
        payload = _base_payload()
        payload["propagation"]["max_paths_per_seed"] = value
        with pytest.raises(PTMResearchConfigError, match="max_paths_per_seed"):
            parse_ptm_research_config(payload)

    def test_direct_propagation_config_rejects_invalid_max_paths_per_seed(self):
        with pytest.raises(PTMResearchConfigError, match="max_paths_per_seed"):
            PropagationConfig(max_depth=2, decay=0.5, gene_edge_types=("tf_regulation",), max_paths_per_seed=False)

    def test_semantic_context_must_match_objective(self):
        payload = _base_payload()
        payload["semantic_context"]["research_objective"] = "reversal"
        with pytest.raises(PTMResearchConfigError, match="research_objective"):
            parse_ptm_research_config(payload)

    def test_semantic_context_missing_field(self):
        payload = _base_payload()
        del payload["semantic_context"]["reference_axis"]
        with pytest.raises(PTMResearchConfigError, match="reference_axis"):
            parse_ptm_research_config(payload)

    def test_cell_types_must_be_unique_and_present(self):
        payload = _base_payload()
        payload["cell_types"] = ["EX", "EX"]
        with pytest.raises(PTMResearchConfigError, match="duplicates"):
            parse_ptm_research_config(payload)
        payload["cell_types"] = []
        with pytest.raises(PTMResearchConfigError, match="cell_types"):
            parse_ptm_research_config(payload)

    def test_render_requires_template(self):
        payload = _base_payload()
        payload.pop("semantic_context")
        config = parse_ptm_research_config(payload)
        with pytest.raises(PTMResearchConfigError, match="semantic_context"):
            config.semantic_context_for_cell_type("EX")

    def test_deg_donor_aggregation_defaults_and_validation(self):
        payload = _base_payload()
        config = parse_ptm_research_config(payload)
        assert config.deg_donor_aggregation == "per_cell_log2_mean"
        assert config.observed_admission_rule == "fdr_cutoff"
        assert config.kd_policy == "separate_routes"
        assert config.public_perturbation_policy == "inventory_optional"

        payload["deg_donor_aggregation"] = "pseudobulk_counts"
        config = parse_ptm_research_config(payload)
        assert config.deg_donor_aggregation == "pseudobulk_counts"

        payload["deg_donor_aggregation"] = "median_of_medians"
        with pytest.raises(PTMResearchConfigError, match="deg_donor_aggregation"):
            parse_ptm_research_config(payload)

        payload = _base_payload()
        payload["observed_admission_rule"] = "signed_direction_without_fdr_cutoff"
        payload["kd_policy"] = "merged_into_ko_out_of_scope"
        payload["public_perturbation_policy"] = "out_of_scope"
        config = parse_ptm_research_config(payload)
        assert config.observed_admission_rule == "signed_direction_without_fdr_cutoff"
        payload["observed_admission_rule"] = "relax_fdr"
        with pytest.raises(PTMResearchConfigError, match="observed_admission_rule"):
            parse_ptm_research_config(payload)
