"""Thin adapter facade around specialized GenKI integration components."""

from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, cast, runtime_checkable

import numpy as np
import scipy.sparse as sp
import torch

from .contracts import GenePerturbationRequest, PerturbationResult
from .genki import GraphUtilities, PerturbationExecutor, ReferenceDataLoader, SignificanceAnalyzer


# ---------------------------------------------------------------------------
# Type alias for the shared kwargs dict passed to sub-components
# ---------------------------------------------------------------------------

# The shared_kwargs dict contains mixed types (str, int, float, None) that
# are forwarded to multiple constructors. Dict[str, Any] is the correct type
# for this genuinely heterogeneous dict; the alias makes intent explicit.
_SharedKwargs = Dict[str, Any]  # noqa: TY102


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dict returns / caches
# ---------------------------------------------------------------------------

class BackendInfo(TypedDict, total=False):
    """Shape returned by ``get_backend_info`` and consumed by ``validate_runtime_ready``."""

    backend: str
    runtime_ready: bool
    ref_root: str
    missing_dependencies: List[str]
    missing_files: List[str]
    uses_explicit_files: bool
    has_adata_file: bool
    has_grn_dir: bool
    scoring_method: str
    null_permutations: int
    bagging_threshold: float
    bagging_cutoff: float


class ReferenceData(TypedDict, total=False):
    """Shape returned by ``load_reference_data`` and stored in ``_reference_cache``."""

    gene_names: List[str]
    network: Union[np.ndarray, sp.spmatrix]
    counts: np.ndarray
    backend: str
    adata_file: str
    grn_file_dir: str
    loaded_at: float


class ScoreMetadata(TypedDict, total=False):
    """Shape returned by ``_build_score_metadata``."""

    target_gene_index: int
    ref_root: str
    magnitude: float
    backend: str
    scoring_method: str
    gene_scores: Dict[str, float]
    gene_indices: Dict[str, int]
    empirical_pvalues: Dict[str, float]
    adjusted_pvalues: Dict[str, float]
    bagging_hits: Dict[str, int]
    bagging_frequencies: Dict[str, float]
    fdr_significant_genes: List[str]
    stable_significant_genes: List[str]
    significant_genes: List[str]
    null_distribution_summary: Dict[str, Union[int, float]]


# ---------------------------------------------------------------------------
# Type alias for the 7-tuple returned by _compute_significance
# ---------------------------------------------------------------------------

ComputeSignificanceResult = tuple[
    Dict[str, float],
    Dict[str, int],
    Dict[str, float],
    Dict[str, float],
    Dict[str, int],
    Dict[str, float],
    Dict[str, Union[int, float]],
]


# ---------------------------------------------------------------------------
# Protocol definitions for dynamically-imported objects
# ---------------------------------------------------------------------------

@runtime_checkable
class _GenKIDataLoaderProtocol(Protocol):
    """Structural type for the GenKI DataLoader returned by ``_build_genki_loader``.

    The actual class is imported at runtime from the GenKI package; this
    protocol describes the attributes accessed by the adapter.
    """

    def load_data(self) -> "_PyGDataProtocol": ...
    def load_kodata(self) -> "_PyGDataProtocol": ...
    @property
    def counts(self) -> Union[np.ndarray, sp.spmatrix]: ...
    @property
    def net(self) -> Union[np.ndarray, sp.spmatrix]: ...


@runtime_checkable
class _VGAEModelProtocol(Protocol):
    """Structural type for the VGAE model used in latent scoring.

    Covers the encode / recon_loss / kl_loss API and the ``__mu__`` /
    ``__logstd__`` attributes read by ``_extract_latent_vars``.
    """

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor: ...
    def recon_loss(self, z: torch.Tensor, edge_index: torch.Tensor) -> float: ...
    def kl_loss(self) -> float: ...
    def eval(self) -> "_VGAEModelProtocol": ...
    def parameters(self) -> Any: ...  # torch.nn.Parameter iterator — opaque to numpy-only typing
    def to(self, device: Union[torch.device, str]) -> "_VGAEModelProtocol": ...
    @property
    def __mu__(self) -> torch.Tensor: ...
    @property
    def __logstd__(self) -> torch.Tensor: ...


@runtime_checkable
class _PyGDataProtocol(Protocol):
    """Structural type for ``torch_geometric.data.Data`` objects.

    Only the attributes accessed by the adapter are declared.
    """

    @property
    def x(self) -> torch.Tensor: ...
    @property
    def edge_index(self) -> torch.Tensor: ...
    @property
    def y(self) -> Union[List[str], torch.Tensor]: ...
    @property
    def num_features(self) -> int: ...
    def to(self, device: Union[torch.device, str]) -> "_PyGDataProtocol": ...


class GenKIAdapter:
    """Facade that preserves the original GenKIAdapter API.

    Private helper methods are forwarded explicitly to the underlying
    components (rather than via ``__getattr__``) so that static type
    checkers, IDE jump-to-definition, and refactor tools can resolve them.
    ``__getattr__`` is retained only as a cached fallback for any delegate
    not yet promoted to an explicit method, preserving backward
    compatibility for external callers that relied on dynamic delegation.
    """

    def __init__(
        self,
        ref_root: str,
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
    ) -> None:
        self.ref_root = ref_root
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

        shared_kwargs = {
            "gene_list_file": gene_list_file,
            "network_file": network_file,
            "counts_file": counts_file,
            "adata_file": adata_file,
            "grn_file_dir": grn_file_dir,
            "pcnet_name": pcnet_name,
            "cutoff": cutoff,
            "target_cell": target_cell,
            "obs_label": obs_label,
            "scoring_method": scoring_method,
            "trainer_epochs": trainer_epochs,
            "trainer_lr": trainer_lr,
            "trainer_beta": trainer_beta,
            "trainer_seed": trainer_seed,
            "trainer_out_channels": trainer_out_channels,
            "null_permutations": self.null_permutations,
            "null_seed": null_seed,
            "significance_alpha": self.significance_alpha,
            "bagging_threshold": self.bagging_threshold,
            "bagging_cutoff": self.bagging_cutoff,
        }
        self._graph = GraphUtilities()
        self._ref_loader = ReferenceDataLoader(ref_root=ref_root, **cast(_SharedKwargs, shared_kwargs))
        self._perturbation = PerturbationExecutor(ref_loader=self._ref_loader, graph=self._graph, **cast(_SharedKwargs, shared_kwargs))
        self._significance = SignificanceAnalyzer(
            ref_loader=self._ref_loader,
            perturbation_executor=self._perturbation,
            graph=self._graph,
            **cast(_SharedKwargs, shared_kwargs),
        )
        self._perturbation.set_significance_analyzer(self._significance)

    # ------------------------------------------------------------------
    # Explicit delegation methods. Each forwards to the matching
    # component method.
    # ------------------------------------------------------------------

    def _load_reference_data_from_genki_source(self) -> ReferenceData:
        """Forward to ReferenceDataLoader._load_reference_data_from_genki_source."""
        return cast(ReferenceData, self._ref_loader._load_reference_data_from_genki_source())

    def _probe_dependencies(self, module_names: List[str]) -> List[str]:
        """Forward to ReferenceDataLoader._probe_dependencies."""
        return self._ref_loader._probe_dependencies(module_names)

    def _run_with_genki_source(self, request: GenePerturbationRequest) -> PerturbationResult:
        """Forward to PerturbationExecutor._run_with_genki_source."""
        return self._perturbation._run_with_genki_source(request)

    def _build_genki_loader(self, gene_symbol: str) -> _GenKIDataLoaderProtocol:
        """Forward to PerturbationExecutor._build_genki_loader."""
        return cast(_GenKIDataLoaderProtocol, self._perturbation._build_genki_loader(gene_symbol))

    def _score_with_latent_vgae(
        self,
        wt_data: _PyGDataProtocol,
        perturbed_counts: np.ndarray,
        perturbed_network: np.ndarray,
    ) -> np.ndarray:
        """Forward to PerturbationExecutor._score_with_latent_vgae."""
        return self._perturbation._score_with_latent_vgae(
            wt_data=wt_data,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )

    def _edge_index_to_adjacency(self, edge_index, num_nodes) -> np.ndarray:
        """Forward to GraphUtilities._edge_index_to_adjacency."""
        return self._graph._edge_index_to_adjacency(edge_index, num_nodes)

    def _adjacency_to_edge_index(self, adjacency) -> torch.Tensor:
        """Forward to GraphUtilities._adjacency_to_edge_index."""
        return torch.from_numpy(self._graph._adjacency_to_edge_index(adjacency))

    def _score_from_dense_matrices(self, baseline_counts, baseline_network, perturbed_counts, perturbed_network) -> np.ndarray:
        """Forward to GraphUtilities._score_from_dense_matrices."""
        return self._graph._score_from_dense_matrices(
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )

    def _extract_latent_vars(self, model, data) -> Dict[str, torch.Tensor]:
        """Forward to GraphUtilities._extract_latent_vars."""
        z_mu, z_std = self._graph._extract_latent_vars(model, data)
        return {"mu": torch.from_numpy(z_mu), "std": torch.from_numpy(z_std)}

    def _build_score_metadata(
        self,
        gene_names: List[str],
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: np.ndarray,
        baseline_network: Union[np.ndarray, sp.spmatrix],
        combined_shift: np.ndarray,
        backend: str,
    ) -> ScoreMetadata:
        """Forward to SignificanceAnalyzer._build_score_metadata."""
        return cast(ScoreMetadata, self._significance._build_score_metadata(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
            backend=backend,
        ))

    def _compute_significance(
        self,
        gene_names: List[str],
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: np.ndarray,
        baseline_network: Union[np.ndarray, sp.spmatrix],
        combined_shift: np.ndarray,
    ) -> ComputeSignificanceResult:
        """Forward to SignificanceAnalyzer._compute_significance."""
        return self._significance._compute_significance(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
        )

    def _build_null_distribution(
        self,
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: np.ndarray,
        baseline_network: Union[np.ndarray, sp.spmatrix],
        observed_scores: np.ndarray,
    ) -> np.ndarray:
        """Forward to SignificanceAnalyzer._build_null_distribution."""
        return self._significance._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            observed_scores=observed_scores,
        )

    def _benjamini_hochberg(self, pvalues: np.ndarray) -> np.ndarray:
        """Forward to SignificanceAnalyzer._benjamini_hochberg."""
        return self._significance._benjamini_hochberg(pvalues)

    def _compute_bagging_statistics(self, null_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward to SignificanceAnalyzer._compute_bagging_statistics."""
        return self._significance._compute_bagging_statistics(null_scores)

    @property
    def _reference_cache(self) -> ReferenceData | None:
        return cast(Optional[ReferenceData], self._ref_loader._reference_cache)

    @_reference_cache.setter
    def _reference_cache(self, value: ReferenceData | None) -> None:
        self._ref_loader._reference_cache = value  # type: ignore[assignment]  # setting private attr on third-party ref_loader

    def get_backend_info(self) -> BackendInfo:
        return cast(BackendInfo, self._ref_loader.get_backend_info())

    def validate_runtime_ready(self) -> None:
        backend = self.get_backend_info()
        if backend["runtime_ready"]:
            return
        missing = ", ".join(backend["missing_dependencies"]) or "unknown"
        raise RuntimeError(
            f"Backend {backend['backend']} is not ready; missing or broken dependencies: {missing}"
        )

    def build_request(
        self,
        gene_symbol: str,
        mode: str,
        magnitude: float,
        source_protein_id: str = "",
        source_ptm_type: str = "",
        source_ptm_position: int = -1,
    ) -> GenePerturbationRequest:
        return self._perturbation.build_request(
            gene_symbol,
            mode,
            magnitude,
            source_protein_id=source_protein_id,
            source_ptm_type=source_ptm_type,
            source_ptm_position=source_ptm_position,
        )

    def load_reference_data(self) -> ReferenceData:
        return cast(ReferenceData, self._ref_loader.load_reference_data())

    def run(self, request: GenePerturbationRequest) -> PerturbationResult:
        return self._perturbation.run(request)

    def run_batch(self, requests: List[GenePerturbationRequest]) -> List[PerturbationResult]:
        return self._perturbation.run_batch(requests)

    def clear_cache(self) -> None:
        """Invalidate all cached reference data and trained VGAE models.

        Forces the next call to ``load_reference_data`` to reload from disk
        and the next latent-VGAE scoring pass to retrain the encoder. Use
        this when the underlying reference files have been replaced or when
        switching to a different dataset.
        """
        self._ref_loader._reference_cache = None
        self._ref_loader._cache_signature = None
        self._perturbation._vgae_cache.clear()

    def invalidate_entry(self, gene_symbol: str) -> None:
        """Invalidate cached results for a single gene.

        Clears the reference-data cache (so the next load is fresh) and
        removes any VGAE model entries that might have been trained on
        data associated with *gene_symbol*. Because the VGAE cache is
        keyed by a graph fingerprint rather than gene name, this method
        conservatively clears the entire VGAE cache to guarantee
        correctness.

        Args:
            gene_symbol: The gene whose cached results should be
                invalidated. The reference-data cache is always cleared
                because gene-level granularity is not tracked there.
        """
        self._ref_loader._reference_cache = None
        self._ref_loader._cache_signature = None
        self._perturbation._vgae_cache.clear()

    def run_virtual_ko(
        self,
        gene_symbol: str,
        top_k: int = 20,
        mode: str = "hard_ko",
        magnitude: float = 1.0,
    ) -> PerturbationResult:
        return self._perturbation.run_virtual_ko(
            gene_symbol,
            top_k,
            mode=mode,
            magnitude=magnitude,
        )
