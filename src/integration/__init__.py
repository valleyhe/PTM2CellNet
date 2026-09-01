"""Integration adapters for external tools (GenKI, PTM virtual perturbation).

Lazy import pattern: submodules are imported only when their names are
first accessed, so optional dependencies (anndata, torch_geometric, etc.)
do not raise ImportError at package import time.
"""

from typing import Any

# Public API names
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
    "PerturbGenCandidateEvidence",
    "PerturbGenRunner",
    "evaluate_dual_path_candidate",
    "MainlineDecision",
    "evaluate_davf_perturbgen_candidate",
]

# Module-level import error tracking for diagnostic purposes
_IMPORT_ERRORS: dict[str, Exception] = {}


def __getattr__(name: str) -> Any:
    """Lazy import: only load submodules when their names are first accessed.

    This prevents ImportError from bubbling up at package import time when
    optional dependencies (anndata, torch_geometric) are not installed.
    """
    _LAZY_IMPORTS: dict[str, tuple[str, str]] = {
        # contracts
        "CandidateRecord": (".contracts", "CandidateRecord"),
        "GenePerturbationRequest": (".contracts", "GenePerturbationRequest"),
        "PerturbationResult": (".contracts", "PerturbationResult"),
        # genki
        "GraphUtilities": (".genki", "GraphUtilities"),
        "PerturbationExecutor": (".genki", "PerturbationExecutor"),
        "ReferenceDataLoader": (".genki", "ReferenceDataLoader"),
        "SignificanceAnalyzer": (".genki", "SignificanceAnalyzer"),
        # genki_adapter
        "GenKIAdapter": (".genki_adapter", "GenKIAdapter"),
        # genki_reports
        "build_comparison_summary_payload": (".genki_reports", "build_comparison_summary_payload"),
        "build_generank_dataframe": (".genki_reports", "build_generank_dataframe"),
        "build_gsea_ranked_dataframe": (".genki_reports", "build_gsea_ranked_dataframe"),
        "build_result_summary_payload": (".genki_reports", "build_result_summary_payload"),
        "build_significant_gene_dataframe": (".genki_reports", "build_significant_gene_dataframe"),
        "build_two_stage_summary_payload": (".genki_reports", "build_two_stage_summary_payload"),
        "render_comparison_summary_markdown": (".genki_reports", "render_comparison_summary_markdown"),
        "render_two_stage_summary_markdown": (".genki_reports", "render_two_stage_summary_markdown"),
        "save_gsea_ranked_tsv": (".genki_reports", "save_gsea_ranked_tsv"),
        # ptm_gene_mapper
        "ProteinGeneMapper": (".ptm_gene_mapper", "ProteinGeneMapper"),
        # ptm_virtual_perturbation
        "PTMPerturbationProfile": (".ptm_virtual_perturbation", "PTMPerturbationProfile"),
        "apply_soft_perturbation": (".ptm_virtual_perturbation", "apply_soft_perturbation"),
        # perturbgen (main-process bridge only; never imports external package)
        "PerturbGenCandidateEvidence": (".perturbgen.contracts", "CandidateEvidence"),
        "PerturbGenRunner": (".perturbgen.runner", "PerturbGenRunner"),
        "evaluate_dual_path_candidate": (
            ".perturbgen.dual_path",
            "evaluate_dual_path_candidate",
        ),
        "MainlineDecision": (".perturbgen.mainline", "MainlineDecision"),
        "evaluate_davf_perturbgen_candidate": (
            ".perturbgen.mainline",
            "evaluate_davf_perturbgen_candidate",
        ),
    }

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        try:
            module = importlib.import_module(module_path, __package__)
            return getattr(module, attr_name)
        except ImportError as exc:
            _IMPORT_ERRORS[name] = exc
            raise ImportError(
                f"Cannot import {name!r} from src.integration: "
                f"{exc}. Install the required dependencies "
                f"(e.g., pip install -r requirements-analysis.txt)."
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_import_errors() -> dict[str, Exception]:
    """Return any import errors encountered during lazy loading."""
    return dict(_IMPORT_ERRORS)
