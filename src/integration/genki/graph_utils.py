"""Dense/sparse graph conversion and shared scoring utilities."""

from typing import Any, Union

import numpy as np
import scipy.sparse as sp


class GraphUtilities:
    """Utility functions shared across GenKI adapter components."""

    # Dense adjacency matrices with this many nodes or more are built/returned
    # as sparse structures to avoid the O(N^2) memory blow-up on large
    # (>20k gene) networks. Below the threshold dense arrays are used because
    # they are faster and the memory cost is negligible.
    SPARSE_THRESHOLD = 5000

    @staticmethod
    def _edge_index_to_adjacency(
        edge_index: np.ndarray,
        num_nodes: int,
        sparse: bool = False,
    ) -> Union[np.ndarray, "sp.csr_matrix"]:
        """Convert an edge_index array to an adjacency matrix.

        Args:
            edge_index: (2, E) array of [source, target] node indices.
            num_nodes: Number of nodes in the graph.
            sparse: When True (or when ``num_nodes`` exceeds
                :data:`SPARSE_THRESHOLD`), return a ``scipy.sparse.csr_matrix``
                instead of a dense array to avoid O(N^2) memory on large
                graphs.
        """
        if sparse or num_nodes > GraphUtilities.SPARSE_THRESHOLD:
            data = np.ones(edge_index.shape[1], dtype=float)
            return sp.csr_matrix(
                (data, (edge_index[0], edge_index[1])),
                shape=(num_nodes, num_nodes),
            )
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
        count_shift = np.asarray(np.abs(perturbed_counts.mean(axis=0) - baseline_counts.mean(axis=0)))
        edge_shift = np.asarray(
            np.abs(perturbed_network - baseline_network).sum(axis=0) + np.abs(
                perturbed_network - baseline_network
            ).sum(axis=1)
        )
        return np.asarray(count_shift + edge_shift)

    @staticmethod
    def _score_from_sparse_matrices(
        baseline_counts: np.ndarray,
        baseline_network: "sp.spmatrix",
        perturbed_counts: np.ndarray,
        perturbed_network: "sp.spmatrix",
    ) -> np.ndarray:
        """Sparse-aware analogue of :meth:`_score_from_dense_matrices`.

        Accepts scipy sparse adjacency matrices for the network arguments and
        performs the absolute-difference and reduction using native sparse
        operations, avoiding the O(N^2) memory cost of densifying large
        (>20k gene) gene regulatory networks.

        Args:
            baseline_counts: Dense (cells, genes) count matrix.
            baseline_network: Sparse (genes, genes) baseline adjacency.
            perturbed_counts: Dense (cells, genes) perturbed count matrix.
            perturbed_network: Sparse (genes, genes) perturbed adjacency.

        Returns:
            Per-gene combined shift scores (genes,) as a dense array.
        """
        # Densify counts' means are already dense (1D over genes).
        baseline_counts_mean = np.asarray(baseline_counts, dtype=float).mean(axis=0)
        perturbed_counts_mean = np.asarray(perturbed_counts, dtype=float).mean(axis=0)
        count_shift = np.abs(perturbed_counts_mean - baseline_counts_mean)

        # Sparse absolute difference: |A - B| is computed via element-wise
        # subtraction and abs (scipy sparse supports both, returning sparse).
        diff = perturbed_network - baseline_network
        abs_diff = abs(diff)
        # sum(axis=0) on a csr/csc returns a 1 x N matrix; ravel to (N,).
        in_degree = np.asarray(abs_diff.sum(axis=0)).ravel()
        out_degree = np.asarray(abs_diff.sum(axis=1)).ravel()
        edge_shift = in_degree + out_degree
        return np.asarray(np.asarray(count_shift, dtype=float) + np.asarray(edge_shift, dtype=float))

    @staticmethod
    def _score_from_matrices(
        baseline_counts: np.ndarray,
        baseline_network: Any,
        perturbed_counts: np.ndarray,
        perturbed_network: Any,
    ) -> np.ndarray:
        """Dispatch to the dense or sparse scoring implementation.

        Selects :meth:`_score_from_sparse_matrices` when either network matrix
        is a scipy sparse matrix, otherwise :meth:`_score_from_dense_matrices`.
        """
        if sp.issparse(baseline_network) or sp.issparse(perturbed_network):
            # Coerce both networks to sparse so the sparse path can subtract.
            base_sp = baseline_network if sp.issparse(baseline_network) else sp.csr_matrix(
                np.asarray(baseline_network, dtype=float)
            )
            pert_sp = perturbed_network if sp.issparse(perturbed_network) else sp.csr_matrix(
                np.asarray(perturbed_network, dtype=float)
            )
            return GraphUtilities._score_from_sparse_matrices(
                baseline_counts=baseline_counts,
                baseline_network=base_sp,
                perturbed_counts=perturbed_counts,
                perturbed_network=pert_sp,
            )
        return GraphUtilities._score_from_dense_matrices(
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )

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
