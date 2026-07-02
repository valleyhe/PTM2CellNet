"""Analysis modules for variant parsing, gene mapping, and pathway analysis."""
import logging
from typing import Any

_logger = logging.getLogger(__name__)


class _UnavailableProxy:
    """Lazy proxy that raises a clear error when an optional analysis symbol is used.

    Optional analysis submodules may fail to import when their external dependencies
    (e.g. ``sspa``, ``hgvs``) are not installed. Rather than leaving the public
    symbols as ``None`` and causing late ``AttributeError``/``TypeError``, this proxy
    gives callers an explicit message explaining that the feature is unavailable.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{self._name} is not available because an optional dependency is missing. "
            "Install the required optional dependencies for src.analysis and re-import."
        )

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"{self._name}.{name} is not available because an optional dependency is missing. "
            "Install the required optional dependencies for src.analysis and re-import."
        )

    def __repr__(self) -> str:
        return f"_UnavailableProxy({self._name!r})"


# pathway_integration — depends on sspa (optional) and networkx
_PathwayDatabaseIntegration: Any = _UnavailableProxy("PathwayDatabaseIntegration")
_load_kegg_pathways: Any = _UnavailableProxy("load_kegg_pathways")
_load_reactome_pathways: Any = _UnavailableProxy("load_reactome_pathways")

try:
    from .pathway_integration import (
        PathwayDatabaseIntegration as _PathwayDatabaseIntegration,
        load_kegg_pathways as _load_kegg_pathways,
        load_reactome_pathways as _load_reactome_pathways,
    )
except ImportError as exc:
    _logger.warning("pathway_integration module not available: %s", exc)

# gene_mapper — depends on UniProtMapper
_GeneMapper: Any = _UnavailableProxy("GeneMapper")
_map_gene_to_uniprot: Any = _UnavailableProxy("map_gene_to_uniprot")

try:
    from .gene_mapper import (
        GeneMapper as _GeneMapper,
        map_gene_to_uniprot as _map_gene_to_uniprot,
    )
except ImportError as exc:
    _logger.warning("gene_mapper module not available: %s", exc)

# variant_parser — depends on hgvs
_VariantComponents: Any = _UnavailableProxy("VariantComponents")
_HGVSVariantParser: Any = _UnavailableProxy("HGVSVariantParser")
_parse_variant: Any = _UnavailableProxy("parse_variant")

try:
    from .variant_parser import (
        VariantComponents as _VariantComponents,
        HGVSVariantParser as _HGVSVariantParser,
        parse_variant as _parse_variant,
    )
except ImportError as exc:
    _logger.warning("variant_parser module not available: %s", exc)

# variant_workflow — depends on models subpackage
_VariantEffectResult: Any = _UnavailableProxy("VariantEffectResult")
_VariantEffectWorkflow: Any = _UnavailableProxy("VariantEffectWorkflow")

try:
    from .variant_workflow import (
        VariantEffectResult as _VariantEffectResult,
        VariantEffectWorkflow as _VariantEffectWorkflow,
    )
except ImportError as exc:
    _logger.warning("variant_workflow module not available: %s", exc)

PathwayDatabaseIntegration: Any = _PathwayDatabaseIntegration
load_kegg_pathways: Any = _load_kegg_pathways
load_reactome_pathways: Any = _load_reactome_pathways
GeneMapper: Any = _GeneMapper
map_gene_to_uniprot: Any = _map_gene_to_uniprot
VariantComponents: Any = _VariantComponents
HGVSVariantParser: Any = _HGVSVariantParser
parse_variant: Any = _parse_variant
VariantEffectResult: Any = _VariantEffectResult
VariantEffectWorkflow: Any = _VariantEffectWorkflow

__all__ = [
    "PathwayDatabaseIntegration",
    "load_kegg_pathways",
    "load_reactome_pathways",
    "GeneMapper",
    "map_gene_to_uniprot",
    "VariantComponents",
    "HGVSVariantParser",
    "parse_variant",
    "VariantEffectResult",
    "VariantEffectWorkflow",
]
