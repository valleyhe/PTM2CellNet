import numpy as np
import pytest

from src.integration.genki.graph_utils import GraphUtilities


class TestEdgeIndexConversion:
    def test_edge_index_to_adjacency_basic(self) -> None:
        edge_index = np.array([[0, 1], [1, 0]])
        adjacency = GraphUtilities._edge_index_to_adjacency(edge_index, num_nodes=2)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert np.array_equal(adjacency, expected)

    def test_adjacency_to_edge_index_basic(self) -> None:
        adjacency = np.zeros((3, 3), dtype=float)
        adjacency[0, 1] = 1.0
        adjacency[1, 2] = 1.0
        edge_index = GraphUtilities._adjacency_to_edge_index(adjacency)
        assert edge_index.shape == (2, 2)
        assert np.array_equal(edge_index[:, 0], np.array([0, 1]))
        assert np.array_equal(edge_index[:, 1], np.array([1, 2]))


class TestScoreFromDenseMatrices:
    def test_score_from_dense_matrices_shape(self) -> None:
        baseline_counts = np.zeros((3, 4), dtype=float)
        perturbed_counts = np.ones((3, 4), dtype=float)
        baseline_network = np.zeros((4, 4), dtype=float)
        perturbed_network = np.zeros((4, 4), dtype=float)
        combined_shift = GraphUtilities._score_from_dense_matrices(
            baseline_counts, baseline_network, perturbed_counts, perturbed_network
        )
        assert combined_shift.shape == (4,)

    def test_score_positive_when_perturbed_differs(self) -> None:
        baseline_counts = np.zeros((3, 4), dtype=float)
        perturbed_counts = np.ones((3, 4), dtype=float)
        baseline_network = np.zeros((4, 4), dtype=float)
        perturbed_network = np.zeros((4, 4), dtype=float)
        combined_shift = GraphUtilities._score_from_dense_matrices(
            baseline_counts, baseline_network, perturbed_counts, perturbed_network
        )
        assert np.all(combined_shift >= 0)


class MockData:
    def __init__(self):
        self.x = None
        self.edge_index = None


class TestExtractLatentVars:
    def test_extract_latent_vars_flattens_singleton_dim(self) -> None:
        import torch

        class MockModel:
            def eval(self):
                pass

            def encode(self, x, edge_index):
                pass

            def __init__(self):
                self.__mu__ = torch.ones((5, 1))
                self.__logstd__ = torch.zeros((5, 1))

        model = MockModel()
        z_mu, z_std = GraphUtilities._extract_latent_vars(model, MockData())
        assert z_mu.shape == (5,)
        assert z_std.shape == (5,)

    def test_extract_latent_vars_keeps_2d(self) -> None:
        import torch

        class MockModel:
            def eval(self):
                pass

            def encode(self, x, edge_index):
                pass

            def __init__(self):
                self.__mu__ = torch.ones((5, 3))
                self.__logstd__ = torch.zeros((5, 3))

        model = MockModel()
        z_mu, z_std = GraphUtilities._extract_latent_vars(model, MockData())
        assert z_mu.shape == (5, 3)
        assert z_std.shape == (5, 3)
