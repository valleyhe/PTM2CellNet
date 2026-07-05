"""PTM-aware virtual perturbation pipeline.

Provides an end-to-end pipeline for PTM-driven virtual perturbation
experiments, integrating PTMPerturbationProfile with GenKI-based
perturbation execution and significance analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class PerturbationParams(TypedDict, total=False):
    """Shape of a single perturbation dict in ``run_batch_ptm_perturbations``."""

    protein_id: str
    ptm_type: str
    ptm_position: int
    gene_symbol: str
    mode: str
    magnitude: float
    node_decay: float
    edge_scale: float


class StrategyConfig(TypedDict, total=False):
    """Shape of a strategy config dict in ``compare_perturbation_strategies``."""

    name: str
    protein_id: str
    ptm_type: str
    ptm_position: int
    mode: str
    magnitude: float
    node_decay: float
    edge_scale: float


@dataclass(frozen=True)
class PTMPerturbationProfile:
    target_gene_index: int
    node_decay: float
    edge_scale: float


def apply_soft_perturbation(
    counts: np.ndarray,
    net: np.ndarray,
    profile: PTMPerturbationProfile,
    method: str = "multiply",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a soft PTM perturbation to counts and network matrices.

    Args:
        counts: Gene expression counts matrix [batch, n_genes].
        net: Gene-gene interaction network [n_genes, n_genes].
        profile: PTM perturbation profile specifying which gene to target
            and by how much to decay/scale.
        method: Perturbation method. ``"multiply"`` (default) scales the
            target gene's counts and edges by ``node_decay`` / ``edge_scale``.
            ``"add"`` adds ``node_decay`` to the target gene's counts column
            and ``edge_scale`` to the target gene's row and column in the
            network (additive perturbation, e.g. for over-expression).

    Returns:
        Tuple of (perturbed_counts, perturbed_network).

    Raises:
        ValueError: If *method* is not ``"multiply"`` or ``"add"``.
    """
    if method not in ("multiply", "add"):
        raise ValueError(f"method must be 'multiply' or 'add', got {method!r}")

    counts_new = np.array(counts, dtype=float, copy=True)
    net_new = np.array(net, dtype=float, copy=True)

    if method == "multiply":
        counts_new[:, profile.target_gene_index] *= profile.node_decay
        net_new[:, profile.target_gene_index] *= profile.edge_scale
        net_new[profile.target_gene_index, :] *= profile.edge_scale
    else:  # method == "add"
        counts_new[:, profile.target_gene_index] += profile.node_decay
        net_new[:, profile.target_gene_index] += profile.edge_scale
        net_new[profile.target_gene_index, :] += profile.edge_scale
    return counts_new, net_new


@dataclass
class PTMPerturbationPipelineConfig:
    """Configuration for the PTM virtual perturbation pipeline.

    Attributes:
        ref_root: Root directory for reference data files.
        gene_list_file: Optional path to gene list file.
        network_file: Optional path to network file.
        counts_file: Optional path to counts file.
        significance_alpha: Significance threshold for FDR correction.
        null_permutations: Number of null permutations for significance.
        device: Device string for computation ("cpu" or "cuda").
    """

    ref_root: str
    gene_list_file: Optional[str] = None
    network_file: Optional[str] = None
    counts_file: Optional[str] = None
    significance_alpha: float = 0.05
    null_permutations: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.significance_alpha <= 0 or self.significance_alpha >= 1:
            raise ValueError(
                f"significance_alpha must be in (0, 1), got {self.significance_alpha}"
            )
        if self.null_permutations < 1:
            raise ValueError(
                f"null_permutations must be >= 1, got {self.null_permutations}"
            )


class PTMVirtualPerturbationEngine:
    """Orchestration engine for PTM-aware virtual perturbation experiments.

    Connects PTM perturbation profiles (soft knock-out via node/edge decay)
    with the GenKI perturbation execution pipeline for scoring and
    significance analysis.
    """

    def __init__(
        self,
        adapter: Any,
        config: Optional[PTMPerturbationPipelineConfig] = None,
    ):
        """Initialize the perturbation engine.

        Args:
            adapter: GenKIAdapter instance for executing perturbations
                and loading reference data.
            config: Optional pipeline configuration. If not provided,
                defaults are used.
        """
        self._adapter = adapter
        self._config = config or PTMPerturbationPipelineConfig(ref_root="")

    def run_ptm_perturbation(
        self,
        protein_id: str,
        ptm_type: str,
        ptm_position: int,
        gene_symbol: str,
        mode: str = "soft_ko",
        magnitude: float = 0.5,
        node_decay: float = 0.3,
        edge_scale: float = 0.5,
    ) -> Any:
        """Run a single PTM-driven virtual perturbation.

        Creates a PTMPerturbationProfile, applies it to reference
        counts/network, and executes the GenKI perturbation pipeline.

        Args:
            protein_id: UniProt protein identifier.
            ptm_type: Type of PTM (e.g., "phosphorylation").
            ptm_position: Residue position of the PTM.
            gene_symbol: HGNC gene symbol of the target gene.
            mode: Perturbation mode ("soft_ko", "hard_ko", "overexpression").
            magnitude: Perturbation magnitude.
            node_decay: Decay factor for gene node expression.
            edge_scale: Scaling factor for gene-gene edges.

        Returns:
            PerturbationResult from the GenKI adapter.

        Raises:
            RuntimeError: If the perturbation pipeline fails.
        """
        ref_data = self._adapter.load_reference_data()
        gene_names: List[str] = list(ref_data.get("gene_names", []))
        try:
            gene_index = gene_names.index(gene_symbol)
        except ValueError:
            raise ValueError(
                f"Gene symbol '{gene_symbol}' not found in reference data. "
                f"Available genes ({len(gene_names)}): "
                f"{', '.join(gene_names[:10])}{'...' if len(gene_names) > 10 else ''}"
            ) from None

        profile = PTMPerturbationProfile(
            target_gene_index=gene_index,
            node_decay=node_decay,
            edge_scale=edge_scale,
        )

        baseline_counts = np.array(ref_data.get("counts", np.ones((1, len(gene_names)))), dtype=float)
        baseline_network = np.array(ref_data.get("network", np.eye(len(gene_names))), dtype=float)

        perturbed_counts, perturbed_network = apply_soft_perturbation(
            baseline_counts, baseline_network, profile,
        )

        request = self._adapter.build_request(
            gene_symbol=gene_symbol,
            mode=mode,
            magnitude=magnitude,
            source_protein_id=protein_id,
            source_ptm_type=ptm_type,
            source_ptm_position=ptm_position,
        )
        return self._adapter.run(request)

    def run_batch_ptm_perturbations(
        self,
        perturbations: List[PerturbationParams],
    ) -> List[Any]:
        """Run multiple PTM-driven virtual perturbations in batch.

        Args:
            perturbations: List of dicts, each containing keys:
                protein_id, ptm_type, ptm_position, gene_symbol, mode,
                magnitude, node_decay, edge_scale.

        Returns:
            List of PerturbationResult objects.
        """
        requests = []
        for p in perturbations:
            request = self._adapter.build_request(
                gene_symbol=p["gene_symbol"],
                mode=p.get("mode", "soft_ko"),
                magnitude=p.get("magnitude", 0.5),
                source_protein_id=p.get("protein_id", ""),
                source_ptm_type=p.get("ptm_type", ""),
                source_ptm_position=p.get("ptm_position", -1),
            )
            requests.append(request)
        return list(self._adapter.run_batch(requests))

    def compare_perturbation_strategies(
        self,
        gene_symbol: str,
        strategies: List[StrategyConfig],
    ) -> Dict[str, Any]:
        """Compare different perturbation strategies for a target gene.

        Args:
            gene_symbol: The target gene symbol.
            strategies: List of strategy config dicts, each containing:
                name (str), mode (str), magnitude (float), and optional
                node_decay (float), edge_scale (float).

        Returns:
            Dict mapping strategy name to PerturbationResult.
        """
        results: Dict[str, Union[Any, Dict[str, Any]]] = {}
        for strategy in strategies:
            name = strategy.get("name", strategy.get("mode", "unknown"))
            result = self.run_ptm_perturbation(
                protein_id=strategy.get("protein_id", ""),
                ptm_type=strategy.get("ptm_type", ""),
                ptm_position=strategy.get("ptm_position", -1),
                gene_symbol=gene_symbol,
                mode=strategy.get("mode", "soft_ko"),
                magnitude=strategy.get("magnitude", 0.5),
                node_decay=strategy.get("node_decay", 0.3),
                edge_scale=strategy.get("edge_scale", 0.5),
            )
            results[name] = result
        return results

    @property
    def adapter(self) -> Any:
        return self._adapter

    @property
    def config(self) -> PTMPerturbationPipelineConfig:
        return self._config


def create_perturbation_pipeline(
    config: PTMPerturbationPipelineConfig,
    **adapter_kwargs: Any,
) -> PTMVirtualPerturbationEngine:
    """Factory function that creates a fully configured perturbation pipeline.

    Args:
        config: Pipeline configuration.
        **adapter_kwargs: Additional keyword arguments forwarded to
            GenKIAdapter constructor.

    Returns:
        A configured PTMVirtualPerturbationEngine instance.

    Example:
        >>> config = PTMPerturbationPipelineConfig(ref_root="/data/genki")
        >>> engine = create_perturbation_pipeline(config)
        >>> result = engine.run_ptm_perturbation(
        ...     protein_id="P12345", ptm_type="phosphorylation",
        ...     ptm_position=42, gene_symbol="TP53",
        ... )
    """
    from .genki_adapter import GenKIAdapter

    adapter = GenKIAdapter(
        ref_root=config.ref_root,
        gene_list_file=config.gene_list_file,
        network_file=config.network_file,
        counts_file=config.counts_file,
        significance_alpha=config.significance_alpha,
        null_permutations=config.null_permutations,
        **adapter_kwargs,
    )
    return PTMVirtualPerturbationEngine(adapter=adapter, config=config)
