"""Tests for PTM virtual perturbation pipeline."""

import numpy as np
import pytest

from src.integration.ptm_virtual_perturbation import (
    PTMPerturbationPipelineConfig,
    PTMPerturbationProfile,
    PTMVirtualPerturbationEngine,
    apply_soft_perturbation,
    create_perturbation_pipeline,
)


def test_soft_perturbation_scales_node_and_incident_edges() -> None:
    counts = np.ones((4, 3), dtype=float)
    net = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ],
        dtype=float,
    )
    profile = PTMPerturbationProfile(target_gene_index=1, node_decay=0.4, edge_scale=0.5)

    counts_new, net_new = apply_soft_perturbation(counts, net, profile)

    assert counts_new[:, 1].mean() < counts[:, 1].mean()
    assert net_new[0, 1] == 0.45
    assert net_new[1, 2] == 0.35


def test_soft_profile_can_degenerate_to_hard_ko() -> None:
    profile = PTMPerturbationProfile(target_gene_index=0, node_decay=0.0, edge_scale=0.0)

    counts_new, net_new = apply_soft_perturbation(np.ones((2, 2)), np.ones((2, 2)), profile)

    assert counts_new[:, 0].sum() == 0.0
    assert net_new[0, :].sum() == 0.0
    assert net_new[:, 0].sum() == 0.0


class TestPTMPerturbationPipelineConfig:
    def test_valid_config(self) -> None:
        config = PTMPerturbationPipelineConfig(ref_root="/data/genki")
        assert config.ref_root == "/data/genki"
        assert config.significance_alpha == 0.05
        assert config.null_permutations == 32
        assert config.device == "cpu"

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="significance_alpha"):
            PTMPerturbationPipelineConfig(ref_root="/data", significance_alpha=0.0)

    def test_invalid_null_permutations_raises(self) -> None:
        with pytest.raises(ValueError, match="null_permutations"):
            PTMPerturbationPipelineConfig(ref_root="/data", null_permutations=0)


class TestPTMVirtualPerturbationEngine:
    def test_engine_initialization(self) -> None:
        """Engine can be initialized with a mock adapter."""
        import types

        mock_adapter = types.SimpleNamespace(
            load_reference_data=lambda: {},
            build_request=lambda *a, **kw: None,
            run=lambda req: None,
            run_batch=lambda reqs: [],
        )
        config = PTMPerturbationPipelineConfig(ref_root="/mock")
        engine = PTMVirtualPerturbationEngine(adapter=mock_adapter, config=config)
        assert engine.adapter is mock_adapter
        assert engine.config is config

    def test_run_ptm_perturbation_gene_not_found(self) -> None:
        """Engine raises ValueError when gene_symbol is not in reference data."""
        import types

        mock_adapter = types.SimpleNamespace(
            load_reference_data=lambda: {"gene_names": ["TP53", "EGFR"]},
            build_request=lambda *a, **kw: None,
            run=lambda req: None,
        )
        engine = PTMVirtualPerturbationEngine(adapter=mock_adapter)
        with pytest.raises(ValueError, match="Gene symbol 'INVALID_GENE' not found"):
            engine.run_ptm_perturbation(
                protein_id="P12345",
                ptm_type="phosphorylation",
                ptm_position=42,
                gene_symbol="INVALID_GENE",
            )

    def test_compare_perturbation_strategies(self) -> None:
        """compare_perturbation_strategies runs each strategy and returns results."""
        import types

        ref_data = {
            "gene_names": ["TP53", "EGFR", "KRAS"],
            "counts": np.ones((1, 3)),
            "network": np.eye(3),
        }
        mock_adapter = types.SimpleNamespace(
            load_reference_data=lambda: ref_data,
            build_request=lambda *a, **kw: types.SimpleNamespace(
                gene_symbol=kw.get("gene_symbol", ""),
                mode=kw.get("mode", "soft_ko"),
            ),
            run=lambda req: types.SimpleNamespace(
                gene_symbol=req.gene_symbol,
                mode=req.mode,
                distance_score=0.5,
                ranked_genes=[],
                metadata={},
            ),
        )
        engine = PTMVirtualPerturbationEngine(adapter=mock_adapter)
        strategies = [
            {"name": "soft", "mode": "soft_ko", "magnitude": 0.5},
            {"name": "hard", "mode": "hard_ko", "magnitude": 1.0},
        ]
        results = engine.compare_perturbation_strategies("TP53", strategies)
        assert "soft" in results
        assert "hard" in results


class TestCreatePerturbationPipeline:
    def test_factory_function_exists(self) -> None:
        """create_perturbation_pipeline is importable and returns the right type."""
        config = PTMPerturbationPipelineConfig(ref_root="/tmp")
        assert create_perturbation_pipeline is not None
        assert callable(create_perturbation_pipeline)
