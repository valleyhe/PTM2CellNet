"""Integration utilities for PTM2CellNet external explainability pipelines."""

from .contracts import CandidateRecord, GenePerturbationRequest, PerturbationResult
from .genki_reports import (
    build_comparison_summary_payload,
    build_generank_dataframe,
    build_gsea_ranked_dataframe,
    build_result_summary_payload,
    build_significant_gene_dataframe,
    build_two_stage_summary_payload,
    render_comparison_summary_markdown,
    render_two_stage_summary_markdown,
    save_gsea_ranked_tsv,
)
from .ptm_gene_mapper import ProteinGeneMapper
from .ptm_virtual_perturbation import PTMPerturbationProfile, apply_soft_perturbation

try:
    from .genki import GraphUtilities, PerturbationExecutor, ReferenceDataLoader, SignificanceAnalyzer
except ImportError as _genki_import_error:
    _GENKI_IMPORT_ERROR = _genki_import_error

    def _make_unavailable_stub(name: str):
        """Create a stub class that raises ImportError on any attribute access."""
        class _Unavailable:
            __qualname__ = name
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"{name} is unavailable because the genki optional dependency "
                    f"is not installed. Original error: {_genki_import_error}. "
                    f"Install with: pip install -r requirements-analysis.txt"
                )
            def __getattr__(self, attr):
                raise ImportError(
                    f"{name}.{attr} is unavailable because the genki optional dependency "
                    f"is not installed. Original error: {_genki_import_error}."
                )
        _Unavailable.__name__ = name
        return _Unavailable

    ReferenceDataLoader = _make_unavailable_stub("ReferenceDataLoader")
    PerturbationExecutor = _make_unavailable_stub("PerturbationExecutor")
    SignificanceAnalyzer = _make_unavailable_stub("SignificanceAnalyzer")
    GraphUtilities = _make_unavailable_stub("GraphUtilities")

try:
    from .genki_adapter import GenKIAdapter
except ImportError as _genki_adapter_import_error:
    _GENKI_ADAPTER_IMPORT_ERROR = _genki_adapter_import_error

    def _make_genki_adapter_stub():
        class _UnavailableGenKIAdapter:
            __qualname__ = "GenKIAdapter"
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"GenKIAdapter is unavailable because the genki optional dependency "
                    f"is not installed. Original error: {_genki_adapter_import_error}. "
                    f"Install with: pip install -r requirements-analysis.txt"
                )
            def __getattr__(self, attr):
                raise ImportError(
                    f"GenKIAdapter.{attr} is unavailable because the genki optional dependency "
                    f"is not installed. Original error: {_genki_adapter_import_error}."
                )
        _UnavailableGenKIAdapter.__name__ = "GenKIAdapter"
        return _UnavailableGenKIAdapter

    GenKIAdapter = _make_genki_adapter_stub()

__all__ = [
    "CandidateRecord",
    "GenePerturbationRequest",
    "PerturbationResult",
    "GenKIAdapter",
    "ReferenceDataLoader",
    "PerturbationExecutor",
    "SignificanceAnalyzer",
    "GraphUtilities",
    "build_comparison_summary_payload",
    "build_generank_dataframe",
    "build_gsea_ranked_dataframe",
    "build_result_summary_payload",
    "build_significant_gene_dataframe",
    "build_two_stage_summary_payload",
    "render_comparison_summary_markdown",
    "render_two_stage_summary_markdown",
    "save_gsea_ranked_tsv",
    "ProteinGeneMapper",
    "PTMPerturbationProfile",
    "apply_soft_perturbation",
]
