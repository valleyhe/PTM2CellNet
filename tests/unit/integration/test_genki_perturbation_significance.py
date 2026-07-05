import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.integration.contracts import PerturbationResult
from src.integration.genki.perturbation import PerturbationExecutor
from src.integration.genki.reference_data import ReferenceDataLoader
from src.integration.genki.significance import SignificanceAnalyzer


class TestPerturbationExecutorBranches:
    def test_run_requires_significance_analyzer(self) -> None:
        ref_loader = ReferenceDataLoader(
            ref_root="tests/fixtures/genki",
        )
        executor = PerturbationExecutor(
            ref_loader=ref_loader,
            significance_analyzer=None,
        )
        request = executor.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        with pytest.raises(RuntimeError) as exc_info:
            executor.run(request)
        assert "SignificanceAnalyzer is not configured" in str(exc_info.value)

    def test_run_unsupported_mode_raises(self) -> None:
        ref_loader = ReferenceDataLoader(
            ref_root="tests/fixtures/genki",
        )
        sig = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=MagicMock(),
        )
        executor = PerturbationExecutor(
            ref_loader=ref_loader,
            significance_analyzer=sig,
        )
        request = executor.build_request(gene_symbol="TP53", mode="unsupported", magnitude=1.0)
        with pytest.raises(ValueError) as exc_info:
            executor.run(request)
        assert "Unsupported perturbation mode" in str(exc_info.value)

    def test_run_virtual_ko_top_k(self) -> None:
        ref_loader = ReferenceDataLoader(
            ref_root="tests/fixtures/genki",
        )
        sig = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=MagicMock(),
        )
        executor = PerturbationExecutor(
            ref_loader=ref_loader,
            significance_analyzer=sig,
        )
        result = executor.run_virtual_ko("TP53", top_k=2)
        assert len(result.ranked_genes) <= 2
        assert result.metadata.get("top_k") == 2


class TestPerturbationGenKIBackendMocked:
    def test_run_with_genki_source_hard_ko_mocked(self, monkeypatch) -> None:
        ref_loader = ReferenceDataLoader(
            ref_root="tests/fixtures/genki",
            adata_file="fake.h5ad",
            grn_file_dir="fake_grn",
        )
        monkeypatch.setattr(ref_loader, "validate_runtime_ready", lambda: None)
        ref_loader._reference_cache = {
            "backend": "genki_source",
            "gene_names": ["TP53", "EGFR"],
        }

        executor = PerturbationExecutor(
            ref_loader=ref_loader,
            adata_file="fake.h5ad",
            grn_file_dir="fake_grn",
            scoring_method="shift",
        )
        sig = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=executor,
        )
        executor.set_significance_analyzer(sig)

        mock_wt = MagicMock()
        mock_wt.x = MagicMock()
        mock_wt.x.detach().cpu().numpy.return_value = np.ones((2, 2))
        mock_wt.y = ["TP53", "EGFR"]
        mock_wt.edge_index = MagicMock()
        mock_wt.edge_index.detach().cpu().numpy.return_value = np.array([[0, 1], [1, 0]])
        mock_wt.to.return_value = mock_wt
        mock_wt.num_features = 2

        mock_ko = MagicMock()
        mock_ko.x = MagicMock()
        mock_ko.x.detach().cpu().numpy.return_value = np.zeros((2, 2))
        mock_ko.edge_index = MagicMock()
        mock_ko.edge_index.detach().cpu().numpy.return_value = np.array([[0, 1], [1, 0]])

        mock_loader = MagicMock()
        mock_loader.load_data.return_value = mock_wt
        mock_loader.load_kodata.return_value = mock_ko
        mock_loader.counts = np.ones((2, 2))
        mock_loader.net = np.ones((2, 2))
        monkeypatch.setattr(executor, "_build_genki_loader", lambda gene_symbol: mock_loader)

        request = executor.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        result = executor.run(request)
        assert isinstance(result, PerturbationResult)
        assert result.distance_score >= 0
        assert "TP53" not in result.ranked_genes

    def test_run_with_genki_source_soft_ptm_mocked(self, monkeypatch) -> None:
        ref_loader = ReferenceDataLoader(
            ref_root="tests/fixtures/genki",
            adata_file="fake.h5ad",
            grn_file_dir="fake_grn",
        )
        monkeypatch.setattr(ref_loader, "validate_runtime_ready", lambda: None)
        ref_loader._reference_cache = {
            "backend": "genki_source",
            "gene_names": ["TP53", "EGFR"],
        }

        executor = PerturbationExecutor(
            ref_loader=ref_loader,
            adata_file="fake.h5ad",
            grn_file_dir="fake_grn",
            scoring_method="shift",
        )
        sig = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=executor,
        )
        executor.set_significance_analyzer(sig)

        mock_wt = MagicMock()
        mock_wt.x.detach().cpu().numpy.return_value = np.ones((2, 2))
        mock_wt.y = ["TP53", "EGFR"]
        mock_wt.edge_index.detach().cpu().numpy.return_value = np.array([[0, 1], [1, 0]])
        mock_wt.to.return_value = mock_wt
        mock_wt.num_features = 2

        mock_loader = MagicMock()
        mock_loader.load_data.return_value = mock_wt
        mock_loader.counts = np.ones((2, 2))
        mock_loader.net = np.ones((2, 2))
        mock_loader._build_edges.return_value = np.array([[0, 1], [1, 0]])
        monkeypatch.setattr(executor, "_build_genki_loader", lambda gene_symbol: mock_loader)

        request = executor.build_request(gene_symbol="TP53", mode="soft_ptm", magnitude=0.5)
        result = executor.run(request)
        assert result.mode == "soft_ptm"

    def test_score_with_latent_vgae_mocked(self, monkeypatch) -> None:
        import torch

        ref_loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        executor = PerturbationExecutor(ref_loader=ref_loader)

        fake_vgae_cls = MagicMock()
        fake_encoder_cls = MagicMock()
        fake_get_distance = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

        fake_train_mod = SimpleNamespace(VGAE=fake_vgae_cls, VariationalGCNEncoder=fake_encoder_cls)
        fake_utils_mod = SimpleNamespace(get_distance=fake_get_distance)

        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "GenKI.train":
                return fake_train_mod
            if name == "GenKI.utils":
                return fake_utils_mod
            if name == "torch_geometric.data":
                return SimpleNamespace(Data=MagicMock())
            return real_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        mock_model = MagicMock()
        mock_z = torch.ones((3, 2))
        mock_model.encode.return_value = mock_z
        mock_model.recon_loss.return_value = torch.tensor(0.5, requires_grad=True)
        mock_model.kl_loss.return_value = torch.tensor(0.1, requires_grad=True)
        mock_model.__mu__ = mock_z
        mock_model.__logstd__ = torch.zeros((3, 2))
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor(1.0))]
        fake_vgae_cls.return_value.to.return_value = mock_model

        mock_wt = MagicMock()
        mock_wt.to.return_value = mock_wt
        mock_wt.y = ["A", "B", "C"]

        perturbed_counts = np.zeros((3, 3), dtype=float)
        perturbed_network = np.zeros((3, 3), dtype=float)

        shift = executor._score_with_latent_vgae(
            wt_data=mock_wt,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )
        assert isinstance(shift, np.ndarray)
        assert shift.shape == (3,)

    def test_build_genki_loader_missing_paths_raises(self) -> None:
        ref_loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        executor = PerturbationExecutor(ref_loader=ref_loader)
        with pytest.raises(ValueError) as exc_info:
            executor._build_genki_loader("TP53")
        assert "requires adata_file and grn_file_dir" in str(exc_info.value)


class TestSignificanceAnalyzerBranches:
    def test_compute_significance_empty_observed(self) -> None:
        ref_loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        executor = PerturbationExecutor(ref_loader=ref_loader)
        analyzer = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=executor,
        )
        request = executor.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        gene_names = ["A"]
        gene_index = 0
        baseline_counts = np.zeros((1, 1), dtype=float)
        baseline_network = np.zeros((1, 1), dtype=float)
        combined_shift = np.array([0.0])

        (
            gene_scores,
            gene_indices,
            empirical_pvalues,
            adjusted_pvalues,
            bagging_hits,
            bagging_frequencies,
            null_summary,
        ) = analyzer._compute_significance(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
        )
        assert null_summary["method"] == "empty"

    def test_compute_significance_zero_null_permutations(self) -> None:
        ref_loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        executor = PerturbationExecutor(ref_loader=ref_loader)
        analyzer = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=executor,
            null_permutations=0,
        )
        request = executor.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
        gene_names = ["A", "B", "C"]
        gene_index = 1
        baseline_counts = np.zeros((3, 3), dtype=float)
        baseline_network = np.zeros((3, 3), dtype=float)
        combined_shift = np.array([0.1, 0.2, 0.3])

        (
            gene_scores,
            gene_indices,
            empirical_pvalues,
            adjusted_pvalues,
            bagging_hits,
            bagging_frequencies,
            null_summary,
        ) = analyzer._compute_significance(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
        )
        assert all(v == 1.0 for v in empirical_pvalues.values())
        assert all(v == 1.0 for v in adjusted_pvalues.values())
        assert all(v == 0 for v in bagging_hits.values())
        assert null_summary["method"] == "disabled"

    def test_benjamini_hochberg_empty(self) -> None:
        ref_loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        executor = PerturbationExecutor(ref_loader=ref_loader)
        analyzer = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=executor,
        )
        result = analyzer._benjamini_hochberg(np.array([]))
        assert result.size == 0

    def test_compute_bagging_statistics_empty_null(self) -> None:
        ref_loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        executor = PerturbationExecutor(ref_loader=ref_loader)
        analyzer = SignificanceAnalyzer(
            ref_loader=ref_loader,
            perturbation_executor=executor,
        )
        hits, frequencies = analyzer._compute_bagging_statistics(np.zeros((0, 0), dtype=float))
        assert hits.shape == (0,)
        assert frequencies.shape == (0,)
