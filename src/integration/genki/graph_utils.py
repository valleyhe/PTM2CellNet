"""Dense/sparse graph conversion and shared scoring utilities."""

from typing import Any

import numpy as np


class GraphUtilities:
    """Utility functions shared across GenKI adapter components."""

    @staticmethod
    def _edge_index_to_adjacency(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        adjacency = np.zeros((num_nodes, num_nodes), dtype=float)
        adjacency[edge_index[0], edge_index[1]] = 1.0
        return adjacency

    @staticmethod
    def _adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
        return np.asarray(np.where(np.asarray(adjacency, dtype=float) > 0), dtype=int)

    @staticmethod
    def _score_from_dense_matrices(
        baseline_counts: np.ndarray,
        baseline_network: np.ndarray,
        perturbed_counts: np.ndarray,
        perturbed_network: np.ndarray,
    ) -> np.ndarray:
        count_shift = np.abs(perturbed_counts.mean(axis=0) - baseline_counts.mean(axis=0))
        edge_shift = np.abs(perturbed_network - baseline_network).sum(axis=0) + np.abs(
            perturbed_network - baseline_network
        ).sum(axis=1)
        return count_shift + edge_shift

    @staticmethod
    def _extract_latent_vars(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        _ = model.encode(data.x, data.edge_index)
        z_mu = model.__mu__.detach().cpu().numpy()
        z_std = (model.__logstd__.exp() ** 2).detach().cpu().numpy()
        if z_mu.ndim == 2 and z_mu.shape[1] == 1:
            z_mu = z_mu.flatten()
            z_std = z_std.flatten()
        return z_mu, z_std
