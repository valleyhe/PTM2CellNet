"""Thin adapter facade around specialized GenKI integration components."""

from typing import Any, Dict, List

from .contracts import GenePerturbationRequest, PerturbationResult
from .genki import GraphUtilities, PerturbationExecutor, ReferenceDataLoader, SignificanceAnalyzer


class GenKIAdapter:
    """Facade that preserves the original GenKIAdapter API.

    Private helper methods are forwarded explicitly to the underlying
    components (rather than via ``__getattr__``) so that static type
    checkers, IDE jump-to-definition, and refactor tools can resolve them.
    ``__getattr__`` is retained only as a cached fallback for any delegate
    not yet promoted to an explicit method, preserving backward
    compatibility for external callers that relied on dynamic delegation.
    """

    _PRIVATE_DELEGATES = {
        "_load_reference_data_from_genki_source": ("_ref_loader", "_load_reference_data_from_genki_source"),
        "_probe_dependencies": ("_ref_loader", "_probe_dependencies"),
        "_run_with_genki_source": ("_perturbation", "_run_with_genki_source"),
        "_build_genki_loader": ("_perturbation", "_build_genki_loader"),
        "_score_with_latent_vgae": ("_perturbation", "_score_with_latent_vgae"),
        "_edge_index_to_adjacency": ("_graph", "_edge_index_to_adjacency"),
        "_adjacency_to_edge_index": ("_graph", "_adjacency_to_edge_index"),
        "_score_from_dense_matrices": ("_graph", "_score_from_dense_matrices"),
        "_extract_latent_vars": ("_graph", "_extract_latent_vars"),
        "_build_score_metadata": ("_significance", "_build_score_metadata"),
        "_compute_significance": ("_significance", "_compute_significance"),
        "_build_null_distribution": ("_significance", "_build_null_distribution"),
        "_benjamini_hochberg": ("_significance", "_benjamini_hochberg"),
        "_compute_bagging_statistics": ("_significance", "_compute_bagging_statistics"),
    }

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
        self._ref_loader = ReferenceDataLoader(ref_root=ref_root, **shared_kwargs)  # type: ignore[arg-type]
        self._perturbation = PerturbationExecutor(ref_loader=self._ref_loader, graph=self._graph, **shared_kwargs)  # type: ignore[arg-type]
        self._significance = SignificanceAnalyzer(
            ref_loader=self._ref_loader,
            perturbation_executor=self._perturbation,
            graph=self._graph,
            **shared_kwargs,  # type: ignore[arg-type]
        )
        self._perturbation.set_significance_analyzer(self._significance)

    def __getattr__(self, name: str) -> Any:
        """Cached fallback delegation for private helper methods.

        Promoted delegates are defined as explicit methods below and resolved
        by normal attribute lookup (this method is only consulted when the
        normal lookup fails). Results are memoized per (adapter, name) pair so
        repeated access does not re-resolve the bound method each call.
        """
        # Avoid recursing during __init__ before components are set.
        if name.startswith("_") and not name.startswith("__"):
            try:
                delegates = type(self)._PRIVATE_DELEGATES
            except AttributeError:
                delegates = {}
            delegate = delegates.get(name)
            if delegate is not None:
                component_name, target_name = delegate
                try:
                    component = getattr(self, component_name)
                except AttributeError as exc:
                    raise AttributeError(
                        f"{self.__class__.__name__!r} object has no attribute {name!r}"
                    ) from exc
                bound = getattr(component, target_name)
                return bound
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

    # ------------------------------------------------------------------
    # Explicit delegation methods (promoted from __getattr__ so static
    # type checkers / IDEs can resolve them). Each forwards to the matching
    # component method.
    # ------------------------------------------------------------------

    def _load_reference_data_from_genki_source(self) -> Dict[str, Any]:
        """Forward to ReferenceDataLoader._load_reference_data_from_genki_source."""
        return self._ref_loader._load_reference_data_from_genki_source()

    def _probe_dependencies(self, module_names: List[str]) -> List[str]:
        """Forward to ReferenceDataLoader._probe_dependencies."""
        return self._ref_loader._probe_dependencies(module_names)

    def _run_with_genki_source(self, request: GenePerturbationRequest) -> PerturbationResult:
        """Forward to PerturbationExecutor._run_with_genki_source."""
        return self._perturbation._run_with_genki_source(request)

    def _build_genki_loader(self, gene_symbol: str) -> Any:
        """Forward to PerturbationExecutor._build_genki_loader."""
        return self._perturbation._build_genki_loader(gene_symbol)

    def _score_with_latent_vgae(
        self,
        wt_data: Any,
        perturbed_counts: Any,
        perturbed_network: Any,
    ) -> Any:
        """Forward to PerturbationExecutor._score_with_latent_vgae."""
        return self._perturbation._score_with_latent_vgae(
            wt_data=wt_data,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )

    def _edge_index_to_adjacency(self, edge_index: Any, num_nodes: int) -> Any:
        """Forward to GraphUtilities._edge_index_to_adjacency."""
        return self._graph._edge_index_to_adjacency(edge_index, num_nodes)

    def _adjacency_to_edge_index(self, adjacency: Any) -> Any:
        """Forward to GraphUtilities._adjacency_to_edge_index."""
        return self._graph._adjacency_to_edge_index(adjacency)

    def _score_from_dense_matrices(
        self,
        baseline_counts: Any,
        baseline_network: Any,
        perturbed_counts: Any,
        perturbed_network: Any,
    ) -> Any:
        """Forward to GraphUtilities._score_from_dense_matrices."""
        return self._graph._score_from_dense_matrices(
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            perturbed_counts=perturbed_counts,
            perturbed_network=perturbed_network,
        )

    def _extract_latent_vars(self, model: Any, data: Any) -> Any:
        """Forward to GraphUtilities._extract_latent_vars."""
        return self._graph._extract_latent_vars(model, data)

    def _build_score_metadata(
        self,
        gene_names: List[str],
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: Any,
        baseline_network: Any,
        combined_shift: Any,
        backend: str,
    ) -> Dict[str, Any]:
        """Forward to SignificanceAnalyzer._build_score_metadata."""
        return self._significance._build_score_metadata(
            gene_names=gene_names,
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            combined_shift=combined_shift,
            backend=backend,
        )

    def _compute_significance(
        self,
        gene_names: List[str],
        gene_index: int,
        request: GenePerturbationRequest,
        baseline_counts: Any,
        baseline_network: Any,
        combined_shift: Any,
    ) -> Any:
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
        baseline_counts: Any,
        baseline_network: Any,
        observed_scores: Any,
    ) -> Any:
        """Forward to SignificanceAnalyzer._build_null_distribution."""
        return self._significance._build_null_distribution(
            gene_index=gene_index,
            request=request,
            baseline_counts=baseline_counts,
            baseline_network=baseline_network,
            observed_scores=observed_scores,
        )

    def _benjamini_hochberg(self, pvalues: Any) -> Any:
        """Forward to SignificanceAnalyzer._benjamini_hochberg."""
        return self._significance._benjamini_hochberg(pvalues)

    def _compute_bagging_statistics(self, null_scores: Any) -> Any:
        """Forward to SignificanceAnalyzer._compute_bagging_statistics."""
        return self._significance._compute_bagging_statistics(null_scores)

    @property
    def _reference_cache(self) -> Dict[str, Any] | None:
        return self._ref_loader._reference_cache

    @_reference_cache.setter
    def _reference_cache(self, value: Dict[str, Any] | None) -> None:
        self._ref_loader._reference_cache = value

    def get_backend_info(self) -> Dict[str, Any]:
        return self._ref_loader.get_backend_info()

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

    def load_reference_data(self) -> Dict[str, Any]:
        return self._ref_loader.load_reference_data()

    def run(self, request: GenePerturbationRequest) -> PerturbationResult:
        return self._perturbation.run(request)

    def run_batch(self, requests: List[GenePerturbationRequest]) -> List[PerturbationResult]:
        return self._perturbation.run_batch(requests)

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
