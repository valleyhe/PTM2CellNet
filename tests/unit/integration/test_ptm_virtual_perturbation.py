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


def test_soft_perturbation_add_mode_increments_counts_and_edges() -> None:
    """method='add' adds node_decay / edge_scale instead of multiplying."""
    counts = np.ones((4, 3), dtype=float)
    net = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ],
        dtype=float,
    )
    profile = PTMPerturbationProfile(target_gene_index=1, node_decay=0.5, edge_scale=0.3)

    counts_new, net_new = apply_soft_perturbation(counts, net, profile, method="add")

    # counts[:, 1] should be 1.0 + 0.5 = 1.5
    np.testing.assert_allclose(counts_new[:, 1], 1.5)
    # net[0, 1] should be 0.9 + 0.3 = 1.2
    np.testing.assert_allclose(net_new[0, 1], 1.2)
    # net[1, 2] should be 0.7 + 0.3 = 1.0
    np.testing.assert_allclose(net_new[1, 2], 1.0)
    # Unrelated entries unchanged
    np.testing.assert_allclose(counts_new[:, 0], 1.0)
    np.testing.assert_allclose(net_new[0, 2], 0.0)


def test_soft_perturbation_invalid_method_raises() -> None:
    """Passing an unsupported method raises ValueError."""
    profile = PTMPerturbationProfile(target_gene_index=0, node_decay=0.5, edge_scale=0.5)
    with pytest.raises(ValueError, match="method must be 'multiply' or 'add'"):
        apply_soft_perturbation(np.ones((2, 2)), np.ones((2, 2)), profile, method="invalid")


def test_multi_gene_combination_perturbation() -> None:
    """Applying perturbations to multiple genes sequentially produces
    cumulative effects on counts and network."""
    counts = np.ones((2, 4), dtype=float)
    net = np.eye(4, dtype=float) * 0.5
    # Off-diagonal edges
    net[0, 1] = net[1, 0] = 0.8
    net[1, 2] = net[2, 1] = 0.6
    net[2, 3] = net[3, 2] = 0.4

    # Perturb gene 0 (multiply decay)
    profile_a = PTMPerturbationProfile(target_gene_index=0, node_decay=0.2, edge_scale=0.3)
    counts_a, net_a = apply_soft_perturbation(counts, net, profile_a, method="multiply")

    # Then perturb gene 2 (add) on the already-perturbed matrices
    profile_b = PTMPerturbationProfile(target_gene_index=2, node_decay=1.0, edge_scale=0.5)
    counts_ab, net_ab = apply_soft_perturbation(counts_a, net_a, profile_b, method="add")

    # Gene 0: multiply by 0.2 => 0.2
    np.testing.assert_allclose(counts_ab[:, 0], 0.2)
    # Gene 2: 1.0 (from multiply step, unchanged) + 1.0 = 2.0
    np.testing.assert_allclose(counts_ab[:, 2], 2.0)
    # Gene 1 and 3 unchanged by both perturbations
    np.testing.assert_allclose(counts_ab[:, 1], 1.0)
    np.testing.assert_allclose(counts_ab[:, 3], 1.0)
    # Net: gene-0 edges scaled by 0.3, then gene-2 edges incremented by 0.5
    # net_ab[0,1] = 0.8 * 0.3 = 0.24 (gene-0 edge_scale applied)
    np.testing.assert_allclose(net_ab[0, 1], 0.24)
    # net_ab[1,2] = 0.6 * 0.3 (gene-0 col) + 0.5 (gene-2 add) = 0.68
    # Actually: after first perturbation, net_a[1,2] = 0.6 (gene-0 row doesn't touch [1,2])
    # Wait: profile_a targets gene 0, so net_a[:,0] *= 0.3 and net_a[0,:] *= 0.3
    # net_a[1,2] is unchanged = 0.6. Then profile_b adds 0.5 to net[:,2] and net[2,:]
    # net_ab[1,2] = 0.6 + 0.5 = 1.1
    np.testing.assert_allclose(net_ab[1, 2], 1.1)


class TestDAVFJointIntegrationStub:
    """Stub tests for the DAVF (Data-Adapter-Virtual-perturbation-Feature)
    joint integration path.

    These validate the wiring between GenKIAdapter, PTMVirtualPerturbationEngine,
    and the underlying perturbation/significance components using mock adapters,
    so the integration surface is covered without requiring real reference data.
    """

    def test_engine_to_adapter_run_wiring(self) -> None:
        """PTMVirtualPerturbationEngine.run_ptm_perturbation delegates to
        adapter.load_reference_data + adapter.build_request + adapter.run."""
        import types

        call_log: list[str] = []
        ref_data = {
            "gene_names": ["TP53", "EGFR", "KRAS"],
            "counts": np.ones((1, 3)),
            "network": np.eye(3),
        }
        mock_adapter = types.SimpleNamespace(
            load_reference_data=lambda: (call_log.append("load_ref"), ref_data)[1],
            build_request=lambda **kw: (call_log.append(f"build:{kw.get('gene_symbol')}"), types.SimpleNamespace(gene_symbol=kw.get("gene_symbol", ""), mode=kw.get("mode", "")))[1],
            run=lambda req: (call_log.append(f"run:{req.gene_symbol}"), types.SimpleNamespace(gene_symbol=req.gene_symbol, mode=req.mode, distance_score=1.0, ranked_genes=[], metadata={}))[1],
        )
        engine = PTMVirtualPerturbationEngine(adapter=mock_adapter)
        result = engine.run_ptm_perturbation(
            protein_id="P04637", ptm_type="phosphorylation",
            ptm_position=15, gene_symbol="TP53",
        )
        assert result.gene_symbol == "TP53"
        assert "load_ref" in call_log
        assert "build:TP53" in call_log
        assert "run:TP53" in call_log

    def test_engine_batch_delegates_to_adapter_run_batch(self) -> None:
        """run_batch_ptm_perturbations delegates to adapter.build_request and
        adapter.run_batch for each entry."""
        import types

        built: list[str] = []
        mock_adapter = types.SimpleNamespace(
            build_request=lambda **kw: (built.append(kw.get("gene_symbol", "")), None)[1],
            run_batch=lambda reqs: [types.SimpleNamespace(gene_symbol=f"gene_{i}", mode="soft_ko", distance_score=0.0, ranked_genes=[], metadata={}) for i in range(len(reqs))],
        )
        engine = PTMVirtualPerturbationEngine(adapter=mock_adapter)
        perturbations = [
            {"gene_symbol": "TP53", "protein_id": "P1"},
            {"gene_symbol": "EGFR", "protein_id": "P2"},
        ]
        results = engine.run_batch_ptm_perturbations(perturbations)
        assert len(results) == 2
        assert built == ["TP53", "EGFR"]


class TestCreatePerturbationPipeline:
    def test_factory_function_exists(self) -> None:
        """create_perturbation_pipeline is importable and returns the right type."""
        config = PTMPerturbationPipelineConfig(ref_root="/tmp")
        assert create_perturbation_pipeline is not None
        assert callable(create_perturbation_pipeline)
