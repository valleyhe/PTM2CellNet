import numpy as np
import pytest

from src.integration.contracts import GenePerturbationRequest, PerturbationResult
from src.integration.genki_adapter import GenKIAdapter


class TestGenKIAdapterInit:
    def test_init_default(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        assert adapter.ref_root == "tests/fixtures/genki"
        assert adapter.gene_list_file is None
        assert adapter.network_file is None
        assert adapter.counts_file is None

    def test_init_with_files(self) -> None:
        adapter = GenKIAdapter(
            ref_root="root",
            gene_list_file="genes.txt",
            network_file="net.npy",
            counts_file="counts.npy",
        )
        assert adapter.ref_root == "root"
        assert adapter.gene_list_file == "genes.txt"
        assert adapter.network_file == "net.npy"
        assert adapter.counts_file == "counts.npy"

    def test_init_with_genki_source(self) -> None:
        adapter = GenKIAdapter(
            ref_root="ref/GenKI-master-src/GenKI-master",
            adata_file="data.h5ad",
            grn_file_dir="GRNs",
        )
        assert adapter.adata_file == "data.h5ad"
        assert adapter.grn_file_dir == "GRNs"

    def test_default_parameters(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        assert adapter.pcnet_name == "pcNet"
        assert adapter.cutoff == 85
        assert adapter.scoring_method == "shift"
        assert adapter.trainer_epochs == 10
        assert adapter.null_permutations >= 0
        assert adapter.significance_alpha == 0.05


class TestBackendDetection:
    def test_get_backend_info_explicit_files(self) -> None:
        adapter = GenKIAdapter(
            ref_root="unused",
            gene_list_file="g.txt",
            network_file="n.npy",
            counts_file="c.npy",
        )
        info = adapter.get_backend_info()
        assert info["backend"] == "array_files"
        assert info["uses_explicit_files"] is True

    def test_get_backend_info_fixture_files(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        info = adapter.get_backend_info()
        assert info["backend"] == "array_files"
        assert info["uses_explicit_files"] is False

    def test_get_backend_info_genki_source(self) -> None:
        adapter = GenKIAdapter(ref_root="ref/GenKI-master-src/GenKI-master")
        info = adapter.get_backend_info()
        assert info["backend"] == "genki_source"
        assert info["missing_dependencies"] == []

    def test_get_backend_info_unknown(self) -> None:
        adapter = GenKIAdapter(ref_root="/nonexistent/path/that/is/unknown")
        info = adapter.get_backend_info()
        assert info["backend"] == "unknown"

    def test_runtime_ready_true(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        assert adapter.get_backend_info()["runtime_ready"] is True

    def test_runtime_ready_false(self, monkeypatch) -> None:
        adapter = GenKIAdapter(ref_root="ref/GenKI-master-src/GenKI-master")
        monkeypatch.setattr(
            adapter,
            "get_backend_info",
            lambda: {
                "backend": "genki_source",
                "runtime_ready": False,
                "missing_dependencies": ["torch_geometric"],
            },
        )
        with pytest.raises(RuntimeError) as exc_info:
            adapter.validate_runtime_ready()
        assert "genki_source" in str(exc_info.value)
        assert "torch_geometric" in str(exc_info.value)


class TestRequestBuilding:
    def test_build_request_hard_ko(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        req = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        assert req.gene_symbol == "TP53"
        assert req.mode == "hard_ko"
        assert req.magnitude == 1.0

    def test_build_request_soft_ptm(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        req = adapter.build_request(gene_symbol="TP53", mode="soft_ptm", magnitude=0.5)
        assert req.mode == "soft_ptm"
        assert req.magnitude == 0.5

    def test_request_fields(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        req = adapter.build_request(gene_symbol="BAX", mode="hard_ko", magnitude=1.0)
        assert isinstance(req, GenePerturbationRequest)
        assert req.source_protein_id == ""
        assert req.source_ptm_type == ""
        assert req.source_ptm_position == -1


class TestReferenceDataLoading:
    def test_load_reference_data_mock_files(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        ref = adapter.load_reference_data()
        assert ref["gene_names"] == ["EGFR", "TP53", "BAX", "MDM2"]
        assert ref["network"].shape == (4, 4)
        assert ref["counts"].shape == (3, 4)

    def test_load_reference_data_caching(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        ref1 = adapter.load_reference_data()
        ref2 = adapter.load_reference_data()
        assert ref1 is ref2

    def test_gene_names_loaded(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        ref = adapter.load_reference_data()
        assert "TP53" in ref["gene_names"]

    def test_network_loaded(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        ref = adapter.load_reference_data()
        assert isinstance(ref["network"], np.ndarray)

    def test_counts_loaded(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        ref = adapter.load_reference_data()
        assert isinstance(ref["counts"], np.ndarray)


class TestRunHardKO:
    def test_run_hard_ko_returns_result(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        assert isinstance(result, PerturbationResult)
        assert result.gene_symbol == "TP53"
        assert result.mode == "hard_ko"

    def test_run_hard_ko_zeroes_target(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        ref = adapter.load_reference_data()
        idx = ref["gene_names"].index("TP53")
        result = adapter.run(request)
        assert result.distance_score > 0
        assert result.ranked_genes

    def test_run_hard_ko_ranked_genes(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        assert "TP53" not in result.ranked_genes
        assert len(result.ranked_genes) == 3

    def test_run_hard_ko_distance_score(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        assert isinstance(result.distance_score, float)
        assert result.distance_score > 0


class TestRunSoftPTM:
    def test_run_soft_ptm_returns_result(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="soft_ptm", magnitude=0.5)
        result = adapter.run(request)
        assert result.gene_symbol == "TP53"
        assert result.mode == "soft_ptm"
        assert result.distance_score > 0

    def test_run_soft_ptm_applies_profile(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="soft_ptm", magnitude=0.5)
        result = adapter.run(request)
        assert "TP53" not in result.ranked_genes
        assert len(result.ranked_genes) == 3

    def test_run_soft_ptm_magnitude_effect(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        req1 = adapter.build_request(gene_symbol="TP53", mode="soft_ptm", magnitude=0.1)
        req2 = adapter.build_request(gene_symbol="TP53", mode="soft_ptm", magnitude=0.9)
        r1 = adapter.run(req1)
        r2 = adapter.run(req2)
        assert r1.distance_score != r2.distance_score


class TestRunVirtualKO:
    def test_run_virtual_ko_calls_run(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        result = adapter.run_virtual_ko("TP53")
        assert result.gene_symbol == "TP53"
        assert result.mode == "hard_ko"

    def test_run_virtual_ko_top_k(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        result = adapter.run_virtual_ko("TP53", top_k=2)
        assert len(result.ranked_genes) <= 2
        assert result.metadata.get("top_k") == 2


class TestErrorHandling:
    def test_unknown_gene_raises_keyerror(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="UNKNOWN", mode="hard_ko", magnitude=1.0)
        with pytest.raises(KeyError):
            adapter.run(request)

    def test_missing_files_raises_filenotfound(self) -> None:
        adapter = GenKIAdapter(ref_root="/nonexistent/path/that/has/no/files")
        with pytest.raises(FileNotFoundError):
            adapter.load_reference_data()

    def test_invalid_mode_raises_valueerror(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="unknown_mode", magnitude=1.0)
        with pytest.raises(ValueError):
            adapter.run(request)


class TestSignificanceComputation:
    def test_compute_significance_returns_dicts(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        ref = adapter.load_reference_data()
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        gene_names = ref["gene_names"]
        gene_index = gene_names.index("TP53")
        baseline_counts = np.asarray(ref["counts"], dtype=float)
        baseline_network = np.asarray(ref["network"], dtype=float)
        perturbed_counts = baseline_counts.copy()
        perturbed_network = baseline_network.copy()
        perturbed_counts[:, gene_index] = 0.0
        perturbed_network[:, gene_index] = 0.0
        perturbed_network[gene_index, :] = 0.0
        combined_shift = adapter._score_from_dense_matrices(
            baseline_counts, baseline_network, perturbed_counts, perturbed_network
        )
        (
            gene_scores,
            gene_indices,
            empirical_pvalues,
            adjusted_pvalues,
            bagging_hits,
            bagging_frequencies,
            null_summary,
        ) = adapter._compute_significance(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
        )
        assert isinstance(gene_scores, dict)
        assert isinstance(gene_indices, dict)
        assert isinstance(empirical_pvalues, dict)
        assert isinstance(adjusted_pvalues, dict)
        assert isinstance(bagging_hits, dict)
        assert isinstance(bagging_frequencies, dict)
        assert isinstance(null_summary, dict)

    def test_empirical_pvalues_range(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=8,
            null_seed=0,
        )
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        for pv in result.metadata["empirical_pvalues"].values():
            assert 0.0 <= pv <= 1.0

    def test_adjusted_pvalues_range(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=8,
            null_seed=0,
        )
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        for pv in result.metadata["adjusted_pvalues"].values():
            assert 0.0 <= pv <= 1.0

    def test_benjamini_hochberg_correction(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        pvalues = np.array([0.05, 0.01, 0.5])
        adjusted = adapter._benjamini_hochberg(pvalues)
        assert adjusted.shape == pvalues.shape
        assert all(0.0 <= v <= 1.0 for v in adjusted)

    def test_gene_scores_computed(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        assert set(result.metadata["gene_scores"]) == set(result.ranked_genes)


class TestNullDistribution:
    def test_build_null_distribution_shape(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=4,
            null_seed=0,
        )
        ref = adapter.load_reference_data()
        gene_names = ref["gene_names"]
        gene_index = gene_names.index("TP53")
        baseline_counts = np.asarray(ref["counts"], dtype=float)
        baseline_network = np.asarray(ref["network"], dtype=float)
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        combined_shift = adapter._score_from_dense_matrices(
            baseline_counts, baseline_network, baseline_counts, baseline_network
        )
        # _build_null_distribution deletes the target gene index from pseudo_scores,
        # so observed_scores must be the same length as len(gene_names)-1.
        observed_scores = np.delete(combined_shift, gene_index)
        null_scores = adapter._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            observed_scores=observed_scores,
        )
        assert null_scores.shape[0] == 4

    def test_null_permutations_parameter(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=0,
        )
        ref = adapter.load_reference_data()
        gene_index = ref["gene_names"].index("TP53")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        combined_shift = adapter._score_from_dense_matrices(
            np.asarray(ref["counts"]), np.asarray(ref["network"]),
            np.asarray(ref["counts"]), np.asarray(ref["network"]),
        )
        null_scores = adapter._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=np.asarray(ref["counts"]),
            baseline_network=np.asarray(ref["network"]),
            observed_scores=combined_shift,
        )
        assert null_scores.size == 0

    def test_pseudo_target_resampling_method(self) -> None:
        from src.integration.genki_adapter import GenKIAdapter as GA
        adapter = GA(
            ref_root="tests/fixtures/genki",
            null_permutations=4,
            scoring_method="shift",
        )
        ref = adapter.load_reference_data()
        gene_index = ref["gene_names"].index("TP53")
        baseline_counts = np.asarray(ref["counts"], dtype=float)
        baseline_network = np.asarray(ref["network"], dtype=float)
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        combined_shift = np.zeros(len(ref["gene_names"]) - 1)
        null_scores = adapter._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            observed_scores=combined_shift,
        )
        assert null_scores.shape[0] == 4
        res = adapter.run(request)
        assert res.metadata["null_distribution_summary"]["method"] in ("pseudo_target_resampling", "score_permutation")

    def test_score_permutation_method(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=4,
            scoring_method="latent_vgae",
        )
        ref = adapter.load_reference_data()
        gene_index = ref["gene_names"].index("TP53")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        combined_shift = np.zeros(len(ref["gene_names"]) - 1)
        null_scores = adapter._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=np.asarray(ref["counts"]),
            baseline_network=np.asarray(ref["network"]),
            observed_scores=combined_shift,
        )
        assert null_scores.shape[0] == 4


class TestBaggingStatistics:
    def test_compute_bagging_statistics(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        null_scores = np.array([
            [0.1, 0.5, 0.9],
            [0.2, 0.4, 0.8],
            [0.3, 0.6, 0.7],
        ])
        hits, frequencies = adapter._compute_bagging_statistics(null_scores)
        assert hits.shape == (3,)
        assert frequencies.shape == (3,)

    def test_bagging_hits_non_negative(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        null_scores = np.random.rand(10, 5)
        hits, _ = adapter._compute_bagging_statistics(null_scores)
        assert all(h >= 0 for h in hits)

    def test_bagging_frequencies_range(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        null_scores = np.random.rand(10, 5)
        _, frequencies = adapter._compute_bagging_statistics(null_scores)
        assert all(0.0 <= f <= 1.0 for f in frequencies)


class TestScoreMetadata:
    def test_build_score_metadata_structure(self) -> None:
        adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        meta = result.metadata
        assert "gene_scores" in meta
        assert "empirical_pvalues" in meta
        assert "adjusted_pvalues" in meta
        assert "significant_genes" in meta
        assert "null_distribution_summary" in meta

    def test_fdr_significant_genes(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=8,
            null_seed=0,
            significance_alpha=0.2,
        )
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        fdr_genes = result.metadata.get("fdr_significant_genes", [])
        assert isinstance(fdr_genes, list)

    def test_stable_significant_genes(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=8,
            null_seed=0,
        )
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        stable_genes = result.metadata.get("stable_significant_genes", [])
        assert isinstance(stable_genes, list)

    def test_null_summary(self) -> None:
        adapter = GenKIAdapter(
            ref_root="tests/fixtures/genki",
            null_permutations=8,
            null_seed=0,
        )
        request = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = adapter.run(request)
        summary = result.metadata["null_distribution_summary"]
        assert "n_permutations" in summary
        assert "mean_score" in summary
        assert "std_score" in summary
