"""Significance and stability calculations for perturbation outputs."""

from typing import Any, Dict, List, TYPE_CHECKING, cast

import numpy as np
import scipy.sparse as sp

from ..contracts import GenePerturbationRequest
from .graph_utils import GraphUtilities
from .reference_data import ReferenceDataLoader

if TYPE_CHECKING:
    from .perturbation import PerturbationExecutor


class SignificanceAnalyzer:
    """Compute p-values, FDR correction, and bagging stability statistics."""

    def __init__(
        self,
        ref_loader: ReferenceDataLoader,
        perturbation_executor: "PerturbationExecutor",
        gene_list_file: str | None = None,
        network_file: str | None = None,
        counts_file: str | None = None,
        adata_file: str | None = None,
        grn_file_dir: str | None = None,
        pcnet_name: str = "pcNet",
        cutoff: int = 85,
        target_cell: str | None = None,
        obs_label: str = "ident",
        scoring_method: str = "shift",
        trainer_epochs: int = 10,
        trainer_lr: float = 7e-4,
        trainer_beta: float = 1e-4,
        trainer_seed: int | None = None,
        trainer_out_channels: int = 2,
        null_permutations: int = 32,
        null_seed: int | None = None,
        significance_alpha: float = 0.05,
        bagging_threshold: float = 0.05,
        bagging_cutoff: float = 0.95,
        graph: GraphUtilities | None = None,
    ) -> None:
        self._ref_loader = ref_loader
        self._perturbation_executor = perturbation_executor
        self.gene_list_file = gene_list_file
        self.network_file = network_file
        self.counts_file = counts_file
        self.adata_file = adata_file
        self.grn_file_dir = grn_file_dir
        self.pcnet_name = pcnet_name
        self.cutoff = cutoff
        self.target_cell = target_cell
        self.obs_label = obs_label
        self.scoring_method = scoring_method
        self.trainer_epochs = trainer_epochs
        self.trainer_lr = trainer_lr
        self.trainer_beta = trainer_beta
        self.trainer_seed = trainer_seed
        self.trainer_out_channels = trainer_out_channels
        self.null_permutations = max(0, int(null_permutations))
        self.null_seed = null_seed
        self.significance_alpha = float(significance_alpha)
        self.bagging_threshold = float(bagging_threshold)
        self.bagging_cutoff = float(bagging_cutoff)
        self._graph = graph or GraphUtilities()

    def _build_score_metadata(
        self,
        gene_names: List[str],
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: np.ndarray,
        baseline_network: np.ndarray,
        combined_shift: np.ndarray,
        backend: str,
    ) -> Dict[str, Any]:
        (
            gene_scores,
            gene_indices,
            empirical_pvalues,
            adjusted_pvalues,
            bagging_hits,
            bagging_frequencies,
            null_summary,
        ) = self._compute_significance(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
        )
        fdr_significant_genes = [
            gene_name
            for gene_name, adjusted_pvalue in adjusted_pvalues.items()
            if adjusted_pvalue <= self.significance_alpha
        ]
        stable_significant_genes = [
            gene_name
            for gene_name, frequency in bagging_frequencies.items()
            if frequency >= self.bagging_cutoff and gene_scores.get(gene_name, 0.0) != 0.0
        ]
        significant_genes = sorted(
            set(fdr_significant_genes) | set(stable_significant_genes),
            key=lambda gene: -gene_scores[gene],
        )
        return {
            "target_gene_index": gene_index,
            "ref_root": self._ref_loader.ref_root,
            "magnitude": request.magnitude,
            "backend": backend,
            "scoring_method": self.scoring_method,
            "gene_scores": gene_scores,
            "gene_indices": gene_indices,
            "empirical_pvalues": empirical_pvalues,
            "adjusted_pvalues": adjusted_pvalues,
            "bagging_hits": bagging_hits,
            "bagging_frequencies": bagging_frequencies,
            "fdr_significant_genes": fdr_significant_genes,
            "stable_significant_genes": stable_significant_genes,
            "significant_genes": significant_genes,
            "null_distribution_summary": null_summary,
        }

    def _compute_significance(
        self,
        gene_names: List[str],
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: np.ndarray,
        baseline_network: np.ndarray,
        combined_shift: np.ndarray,
    ) -> tuple[
        Dict[str, float],
        Dict[str, int],
        Dict[str, float],
        Dict[str, float],
        Dict[str, int],
        Dict[str, float],
        Dict[str, Any],
    ]:
        observed_pairs = [
            (gene_name, float(combined_shift[idx]))
            for idx, gene_name in enumerate(gene_names)
            if idx != gene_index
        ]
        gene_scores = {gene_name: score for gene_name, score in observed_pairs}
        gene_indices = {gene_name: idx for idx, gene_name in enumerate(gene_names) if idx != gene_index}
        observed_scores = np.asarray([score for _, score in observed_pairs], dtype=float)
        if observed_scores.size == 0:
            empty_summary = {
                "n_permutations": self.null_permutations,
                "method": "empty",
                "mean_score": 0.0,
                "std_score": 0.0,
            }
            return gene_scores, gene_indices, {}, {}, {}, {}, empty_summary

        null_scores = self._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            observed_scores=observed_scores,
        )
        if null_scores.size == 0:
            empirical_pvalues = {gene_name: 1.0 for gene_name in gene_scores}
            adjusted_pvalues = empirical_pvalues.copy()
            bagging_hits = {gene_name: 0 for gene_name in gene_scores}
            bagging_frequencies = {gene_name: 0.0 for gene_name in gene_scores}
            summary = {
                "n_permutations": 0,
                "method": "disabled",
                "mean_score": float(observed_scores.mean()),
                "std_score": float(observed_scores.std()),
            }
            return (
                gene_scores,
                gene_indices,
                empirical_pvalues,
                adjusted_pvalues,
                bagging_hits,
                bagging_frequencies,
                summary,
            )

        empirical = (null_scores >= observed_scores[None, :]).mean(axis=0)
        empirical = np.maximum(empirical, 1.0 / float(null_scores.shape[0]))
        adjusted = self._benjamini_hochberg(empirical)
        bagging_hits_array, bagging_frequencies_array = self._compute_bagging_statistics(null_scores)
        empirical_pvalues = {
            gene_name: float(empirical[idx])
            for idx, (gene_name, _) in enumerate(observed_pairs)
        }
        adjusted_pvalues = {
            gene_name: float(adjusted[idx])
            for idx, (gene_name, _) in enumerate(observed_pairs)
        }
        bagging_hits = {
            gene_name: int(bagging_hits_array[idx])
            for idx, (gene_name, _) in enumerate(observed_pairs)
        }
        bagging_frequencies = {
            gene_name: float(bagging_frequencies_array[idx])
            for idx, (gene_name, _) in enumerate(observed_pairs)
        }
        summary = {
            "n_permutations": int(null_scores.shape[0]),
            "method": "pseudo_target_resampling" if self.scoring_method == "shift" else "score_permutation",
            "mean_score": float(null_scores.mean()),
            "std_score": float(null_scores.std()),
            "bagging_threshold": self.bagging_threshold,
            "bagging_cutoff": self.bagging_cutoff,
        }
        return (
            gene_scores,
            gene_indices,
            empirical_pvalues,
            adjusted_pvalues,
            bagging_hits,
            bagging_frequencies,
            summary,
        )

    def _build_null_distribution(
        self,
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: np.ndarray,
        baseline_network: np.ndarray,
        observed_scores: np.ndarray,
    ) -> np.ndarray:
        if self.null_permutations <= 0:
            return np.empty((0, observed_scores.shape[0]), dtype=float)

        rng = np.random.default_rng(self.null_seed)
        if self.scoring_method != "shift":
            null_scores = np.zeros((self.null_permutations, observed_scores.shape[0]), dtype=float)
            for perm_index in range(self.null_permutations):
                null_scores[perm_index] = rng.permutation(observed_scores)
            return null_scores

        candidate_indices = [idx for idx in range(baseline_counts.shape[1]) if idx != gene_index]
        if not candidate_indices:
            return np.empty((0, observed_scores.shape[0]), dtype=float)

        # Pre-sample all pseudo-target indices in one RNG call instead of
        # calling rng.choice per permutation (avoids per-iteration overhead).
        pseudo_targets = rng.choice(
            np.asarray(candidate_indices, dtype=int),
            size=self.null_permutations,
            replace=True,
        )

        use_sparse = sp.issparse(baseline_network)
        baseline_counts_dense = np.asarray(baseline_counts, dtype=float)
        baseline_network_dense = (
            cast(Any, baseline_network).toarray() if use_sparse else np.asarray(baseline_network, dtype=float)
        )
        n_genes = baseline_counts_dense.shape[1]

        null_scores = np.zeros((self.null_permutations, observed_scores.shape[0]), dtype=float)

        if request.mode == "hard_ko":
            # Vectorized hard-knockout: build a (P, N, N) stack of perturbed
            # networks and (P, cells, N) counts in one shot, then score the
            # whole batch with broadcasting. This replaces the per-permutation
            # Python loop + full-matrix copies and is dramatically faster on
            # large (>5k gene) networks.
            P = self.null_permutations
            # Counts: zero out the pseudo-target column for each permutation.
            perturbed_counts_batch = np.broadcast_to(
                baseline_counts_dense, (P, *baseline_counts_dense.shape)
            ).copy()
            row_idx = np.arange(P)
            perturbed_counts_batch[row_idx, :, pseudo_targets] = 0.0

            # Network: zero out the pseudo-target row and column per layer.
            perturbed_network_batch = np.broadcast_to(
                baseline_network_dense, (P, n_genes, n_genes)
            ).copy()
            perturbed_network_batch[row_idx, :, pseudo_targets] = 0.0
            perturbed_network_batch[row_idx, pseudo_targets, :] = 0.0

            # Broadcast scoring over the batch axis.
            count_shift = np.abs(
                perturbed_counts_batch.mean(axis=1) - baseline_counts_dense.mean(axis=0)[None, :]
            )  # (P, N)
            diff = np.abs(perturbed_network_batch - baseline_network_dense[None, :, :])
            edge_shift = diff.sum(axis=1) + diff.sum(axis=2)  # (P, N)
            batch_scores = count_shift + edge_shift  # (P, N)
            for perm_index in range(P):
                null_scores[perm_index] = np.delete(batch_scores[perm_index], gene_index)
            return null_scores

        # soft_ptm path: vectorized like the hard_ko path. apply_soft_perturbation
        # only scales the pseudo-target count column by node_decay and the
        # pseudo-target network row/column by edge_scale, so a (P, N, N) batch
        # of perturbed matrices can be built and scored with broadcasting in
        # one pass instead of a per-permutation Python loop + full-matrix
        # copies. node_decay == edge_scale for soft PTM.
        P = self.null_permutations
        decay = max(0.0, 1.0 - request.magnitude)

        # Counts: scale the pseudo-target column per permutation layer.
        perturbed_counts_batch = np.broadcast_to(
            baseline_counts_dense, (P, *baseline_counts_dense.shape)
        ).copy()
        row_idx = np.arange(P)
        perturbed_counts_batch[row_idx, :, pseudo_targets] *= decay

        # Network: scale the pseudo-target row and column per layer.
        perturbed_network_batch = np.broadcast_to(
            baseline_network_dense, (P, n_genes, n_genes)
        ).copy()
        perturbed_network_batch[row_idx, :, pseudo_targets] *= decay
        perturbed_network_batch[row_idx, pseudo_targets, :] *= decay

        # Broadcast scoring over the batch axis.
        count_shift = np.abs(
            perturbed_counts_batch.mean(axis=1) - baseline_counts_dense.mean(axis=0)[None, :]
        )  # (P, N)
        diff = np.abs(perturbed_network_batch - baseline_network_dense[None, :, :])
        edge_shift = diff.sum(axis=1) + diff.sum(axis=2)  # (P, N)
        batch_scores = count_shift + edge_shift  # (P, N)
        for perm_index in range(P):
            null_scores[perm_index] = np.delete(batch_scores[perm_index], gene_index)
        return null_scores

    def _benjamini_hochberg(self, pvalues: np.ndarray) -> np.ndarray:
        if pvalues.size == 0:
            return pvalues
        order = np.argsort(pvalues)
        ranked = pvalues[order]
        adjusted = np.empty_like(ranked)
        total = float(len(ranked))
        running = 1.0
        for idx in range(len(ranked) - 1, -1, -1):
            rank = float(idx + 1)
            running = min(running, ranked[idx] * total / rank)
            adjusted[idx] = running
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        return restored

    def _compute_bagging_statistics(self, null_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if null_scores.size == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
        threshold_index = int(null_scores.shape[1] * (1.0 - self.bagging_threshold))
        threshold_index = min(max(threshold_index, 0), null_scores.shape[1] - 1)
        # Only the top tail (indices >= threshold_index) is needed; use
        # argpartition to find the threshold position in O(n) per row instead
        # of fully sorting each permutation's scores.
        k = null_scores.shape[1] - threshold_index
        if k <= 0:
            top_indices = np.empty((null_scores.shape[0], 0), dtype=int)
        else:
            # argpartition gives the k largest elements unsorted at the tail;
            # that is sufficient for counting hits per gene.
            partitioned = np.argpartition(-null_scores, kth=k - 1, axis=1)
            top_indices = partitioned[:, threshold_index:]
        hits = np.zeros((null_scores.shape[1],), dtype=int)
        for row in top_indices:
            hits[row] += 1
        frequencies = hits.astype(float) / float(null_scores.shape[0])
        return hits, frequencies
