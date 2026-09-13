"""Secondary pathway / GSEA evidence for PerturbGen utility.

L-2026-0901-01 mentions pathway utility; scheme §4.7 hard PASS does **not**
list GSEA.  This module records optional secondary evidence and is forbidden
from writing a dual-path verdict.  A missing GSEA backend is inconclusive,
not a silent pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PATHWAY_EVIDENCE_SCHEMA_VERSION = "perturbgen_pathway_evidence/v1"


class PathwayEvidenceError(ValueError):
    """Raised when pathway evidence is claimed as a dual-path hard gate."""


def evaluate_pathway_evidence(
    ranked_genes: Sequence[str],
    *,
    gene_effects: Mapping[str, float] | None = None,
    gmt_path: str | Path | None = None,
    backend: str = "none",
) -> dict[str, Any]:
    """Return secondary pathway evidence that never affects dual-path PASS.

    ``backend='none'`` is the default production behaviour until a signed-off
    GSEA contract exists.  ``backend='gseapy'`` is attempted only when the
    optional extra is installed; absence is inconclusive.
    """

    if isinstance(ranked_genes, (str, bytes)) or not isinstance(ranked_genes, Sequence):
        raise PathwayEvidenceError("ranked_genes must be a sequence of gene identifiers")
    genes = [str(gene).strip() for gene in ranked_genes]
    if any(not gene for gene in genes):
        raise PathwayEvidenceError("ranked_genes must not contain empty identifiers")
    if len(genes) != len(set(genes)):
        raise PathwayEvidenceError("ranked_genes must be unique")

    chosen = str(backend).strip().lower()
    if chosen not in {"none", "gseapy"}:
        raise PathwayEvidenceError("backend must be 'none' or 'gseapy'")

    payload: dict[str, Any] = {
        "schema_version": PATHWAY_EVIDENCE_SCHEMA_VERSION,
        "evidence_class": "secondary",
        "affects_dual_path_verdict": False,
        "backend": chosen,
        "ranked_gene_count": len(genes),
        "gmt_path": None if gmt_path is None else str(Path(gmt_path)),
        "terms": [],
        "status": "inconclusive",
        "reasons": (),
    }
    if chosen == "none":
        payload["reasons"] = ("pathway_backend_not_configured",)
        return payload

    try:
        import gseapy  # noqa: F401
    except ImportError:
        payload["reasons"] = ("gseapy_not_installed",)
        return payload
    if gmt_path is None:
        payload["reasons"] = ("gmt_path_required_for_gseapy",)
        return payload
    resolved = Path(gmt_path).expanduser()
    if not resolved.is_file():
        payload["reasons"] = ("gmt_path_missing",)
        payload["gmt_path"] = str(resolved)
        return payload
    payload["gmt_path"] = str(resolved.resolve())
    payload["status"] = "inconclusive"
    payload["reasons"] = ("gseapy_execution_not_wired_to_dual_path",)
    payload["note"] = (
        "gseapy is present but dual-path hard PASS must not consume pathway terms; "
        "record this payload as secondary evidence only"
    )
    if gene_effects is not None:
        payload["gene_effect_count"] = len(dict(gene_effects))
    return payload


def assert_not_used_for_dual_path(payload: Mapping[str, Any]) -> None:
    """Hard-fail if a caller tries to treat pathway evidence as a hard gate."""

    if payload.get("affects_dual_path_verdict") is True:
        raise PathwayEvidenceError("pathway evidence must not affect dual-path verdict")
    if payload.get("evidence_class") != "secondary":
        raise PathwayEvidenceError("pathway evidence_class must remain 'secondary'")
