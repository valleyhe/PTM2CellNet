"""Analysis modules for variant parsing, gene mapping, and pathway analysis."""
import logging

from src.utils.lazy_import import LazyImport

_logger = logging.getLogger(__name__)

# Optional analysis submodules are imported lazily so that a missing
# optional dependency (e.g. ``sspa``, ``hgvs``) does not break importing
# src.analysis. Accessing a symbol whose import failed re-raises the
# original ImportError instead of leaving a None/AttributeError landmine.
# Use ``<symbol>.is_available()`` to probe without raising.

# pathway_integration — depends on sspa (optional) and networkx
PathwayDatabaseIntegration: LazyImport = LazyImport(
    f"{__name__}.pathway_integration", "PathwayDatabaseIntegration"
)
load_kegg_pathways: LazyImport = LazyImport(f"{__name__}.pathway_integration", "load_kegg_pathways")
load_reactome_pathways: LazyImport = LazyImport(
    f"{__name__}.pathway_integration", "load_reactome_pathways"
)

# gene_mapper — depends on UniProtMapper
GeneMapper: LazyImport = LazyImport(f"{__name__}.gene_mapper", "GeneMapper")
map_gene_to_uniprot: LazyImport = LazyImport(f"{__name__}.gene_mapper", "map_gene_to_uniprot")

# variant_parser — depends on hgvs
VariantComponents: LazyImport = LazyImport(f"{__name__}.variant_parser", "VariantComponents")
HGVSVariantParser: LazyImport = LazyImport(f"{__name__}.variant_parser", "HGVSVariantParser")
parse_variant: LazyImport = LazyImport(f"{__name__}.variant_parser", "parse_variant")

# variant_workflow — depends on models subpackage
VariantEffectResult: LazyImport = LazyImport(f"{__name__}.variant_workflow", "VariantEffectResult")
VariantEffectWorkflow: LazyImport = LazyImport(f"{__name__}.variant_workflow", "VariantEffectWorkflow")

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
