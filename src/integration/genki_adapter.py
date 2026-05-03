"""Thin adapter facade around specialized GenKI integration components."""

from typing import Any, Dict

from .contracts import GenePerturbationRequest, PerturbationResult
from .genki import GraphUtilities, PerturbationExecutor, ReferenceDataLoader, SignificanceAnalyzer


class GenKIAdapter:
    """Facade that preserves the original GenKIAdapter API."""

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
        delegate = self._PRIVATE_DELEGATES.get(name)
        if delegate is None:
            raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")
        component_name, target_name = delegate
        component = getattr(self, component_name)
        return getattr(component, target_name)

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

    def build_request(self, gene_symbol: str, mode: str, magnitude: float) -> GenePerturbationRequest:
        return self._perturbation.build_request(gene_symbol, mode, magnitude)

    def load_reference_data(self) -> Dict[str, Any]:
        return self._ref_loader.load_reference_data()

    def run(self, request: GenePerturbationRequest) -> PerturbationResult:
        return self._perturbation.run(request)

    def run_virtual_ko(self, gene_symbol: str, top_k: int = 20) -> PerturbationResult:
        return self._perturbation.run_virtual_ko(gene_symbol, top_k)
