"""Shared report formatting helpers for GenKI-style perturbation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from scipy import stats
from typing_extensions import TypedDict

from .contracts import PerturbationResult


class _GeneRankRow(TypedDict):
    affected_gene: str
    dis: float | None
    rank: int
    gene_index: int | None
    hit: int
    frequency: float
    empirical_pvalue: float | None
    adjusted_pvalue: float | None
    is_significant: bool


class _ComparisonMetrics(TypedDict, total=False):
    top_gene_overlap_at_3: int
    top_gene_overlap_at_10: int


class _ResultSummaryPayload(TypedDict):
    gene_symbol: str
    mode: str
    distance_score: float
    scoring_method: str | None
    significant_gene_count: int
    top_ranked_genes: List[str]
    top_significant_genes: List[str]
    top_gsea_genes: List[str]
    null_distribution_summary: Dict[str, Any]


class _ComparisonSummaryPayload(TypedDict):
    gene_symbol: str
    backend: Dict[str, Any]
    hard_ko: _ResultSummaryPayload
    soft_ptm: _ResultSummaryPayload
    comparison: _ComparisonMetrics


class _TwoStageSummaryPayload(TypedDict):
    total_results: int
    mode_counts: Dict[str, int]
    unique_genes: List[str]
    top_results: List[_ResultSummaryPayload]
    results: List[_ResultSummaryPayload]


def build_generank_dataframe(result: PerturbationResult) -> pd.DataFrame:
    """Format ranked genes into a stable GenKI-like tabular report."""
    metadata = result.metadata
    gene_scores = metadata.get("gene_scores", {})
    gene_indices = metadata.get("gene_indices", {})
    empirical = metadata.get("empirical_pvalues", {})
    adjusted = metadata.get("adjusted_pvalues", {})
    hits = metadata.get("bagging_hits", {})
    frequencies = metadata.get("bagging_frequencies", {})
    significant = set(metadata.get("significant_genes", []))

    rows: List[_GeneRankRow] = []
    for rank, gene_name in enumerate(result.ranked_genes, start=1):
        rows.append(
            {
                "affected_gene": gene_name,
                "dis": gene_scores.get(gene_name),
                "rank": rank,
                "gene_index": gene_indices.get(gene_name),
                "hit": hits.get(gene_name, 0),
                "frequency": frequencies.get(gene_name, 0.0),
                "empirical_pvalue": empirical.get(gene_name),
                "adjusted_pvalue": adjusted.get(gene_name),
                "is_significant": gene_name in significant,
            }
        )
    return pd.DataFrame(rows)


def build_significant_gene_dataframe(result: PerturbationResult) -> pd.DataFrame:
    """Return the filtered significant-gene subset from a generank report."""
    generank = build_generank_dataframe(result)
    if generank.empty:
        return generank
    return cast(pd.DataFrame, generank[generank["is_significant"]].reset_index(drop=True))


def build_gsea_ranked_dataframe(result: PerturbationResult) -> pd.DataFrame:
    """Format a GSEA-style ranked list from positive perturbation scores."""
    generank = build_generank_dataframe(result)
    if generank.empty:
        return pd.DataFrame(columns=["affected_gene", "dis", "dis_norm", "gsea_rank"])

    gsea = generank[["affected_gene", "dis"]].copy()
    gsea["dis"] = pd.to_numeric(gsea["dis"], errors="coerce")
    gsea = gsea[gsea["dis"].fillna(0.0) > 0].reset_index(drop=True)
    if gsea.empty:
        gsea["dis_norm"] = pd.Series(dtype=float)
        gsea["gsea_rank"] = pd.Series(dtype=int)
        return cast(pd.DataFrame, gsea[["affected_gene", "dis", "dis_norm", "gsea_rank"]])

    gsea["dis_norm"] = _normalize_gsea_scores(gsea["dis"].to_numpy(dtype=float))
    gsea = gsea.sort_values(by="dis_norm", ascending=False).reset_index(drop=True)
    gsea["gsea_rank"] = np.arange(len(gsea)) + 1
    return cast(pd.DataFrame, gsea[["affected_gene", "dis", "dis_norm", "gsea_rank"]])


def save_gsea_ranked_tsv(result: PerturbationResult, file_path: str) -> None:
    """Persist a GSEA-style ranked list as a TSV artifact."""
    gsea = build_gsea_ranked_dataframe(result)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gsea.to_csv(path, sep="\t", index=False, encoding="utf-8")


def build_result_summary_payload(result: PerturbationResult, top_k: int = 5) -> _ResultSummaryPayload:
    """Build a compact JSON-serializable summary for one perturbation result."""
    gsea = build_gsea_ranked_dataframe(result)
    significant_genes = list(result.metadata.get("significant_genes", []))
    return {
        "gene_symbol": result.gene_symbol,
        "mode": result.mode,
        "distance_score": result.distance_score,
        "scoring_method": result.metadata.get("scoring_method"),
        "significant_gene_count": len(significant_genes),
        "top_ranked_genes": list(result.ranked_genes[:top_k]),
        "top_significant_genes": significant_genes[:top_k],
        "top_gsea_genes": gsea["affected_gene"].head(top_k).tolist(),
        "null_distribution_summary": result.metadata.get("null_distribution_summary", {}),
    }


def build_comparison_summary_payload(
    hard: PerturbationResult,
    soft: PerturbationResult,
    comparison: _ComparisonMetrics,
    backend_info: Dict[str, Any],
    top_k: int = 5,
) -> _ComparisonSummaryPayload:
    """Build a unified summary payload for hard-vs-soft perturbation comparison."""
    return {
        "gene_symbol": hard.gene_symbol,
        "backend": backend_info,
        "hard_ko": build_result_summary_payload(hard, top_k=top_k),
        "soft_ptm": build_result_summary_payload(soft, top_k=top_k),
        "comparison": comparison,
    }


def render_comparison_summary_markdown(payload: _ComparisonSummaryPayload) -> str:
    """Render a human-readable markdown summary for perturbation comparison."""
    hard = payload["hard_ko"]
    soft = payload["soft_ptm"]
    comparison = payload["comparison"]
    lines = [
        "# PTM-Aware Virtual Perturbation Summary",
        "",
        f"- Gene: {payload['gene_symbol']}",
        f"- Backend: {payload['backend'].get('backend', 'unknown')}",
        f"- Hard KO distance: {hard['distance_score']:.4f}",
        f"- Soft PTM distance: {soft['distance_score']:.4f}",
        f"- Hard significant genes: {hard['significant_gene_count']}",
        f"- Soft significant genes: {soft['significant_gene_count']}",
        f"- Top-3 overlap: {comparison.get('top_gene_overlap_at_3', 0)}",
        f"- Top-10 overlap: {comparison.get('top_gene_overlap_at_10', 0)}",
        "",
        "## Hard KO",
        f"- Top ranked genes: {', '.join(hard['top_ranked_genes'])}",
        f"- Top significant genes: {', '.join(hard['top_significant_genes'])}",
        f"- Top GSEA genes: {', '.join(hard['top_gsea_genes'])}",
        "",
        "## Soft PTM",
        f"- Top ranked genes: {', '.join(soft['top_ranked_genes'])}",
        f"- Top significant genes: {', '.join(soft['top_significant_genes'])}",
        f"- Top GSEA genes: {', '.join(soft['top_gsea_genes'])}",
    ]
    return "\n".join(lines) + "\n"


def build_two_stage_summary_payload(results: List[PerturbationResult], top_k: int = 5) -> _TwoStageSummaryPayload:
    """Build an aggregate JSON summary for two-stage explanation results."""
    per_result = [build_result_summary_payload(result, top_k=top_k) for result in results]
    mode_counts: Dict[str, int] = {}
    for result in results:
        mode_counts[result.mode] = mode_counts.get(result.mode, 0) + 1
    sorted_by_distance = sorted(results, key=lambda item: item.distance_score, reverse=True)
    return {
        "total_results": len(results),
        "mode_counts": mode_counts,
        "unique_genes": sorted({result.gene_symbol for result in results}),
        "top_results": [build_result_summary_payload(result, top_k=top_k) for result in sorted_by_distance[:top_k]],
        "results": per_result,
    }


def render_two_stage_summary_markdown(results: List[PerturbationResult], top_k: int = 5) -> str:
    """Render a concise human-readable markdown summary for two-stage outputs."""
    payload = build_two_stage_summary_payload(results, top_k=top_k)
    lines = [
        "# Two-Stage Explanation Summary",
        "",
        f"- Total results: {payload['total_results']}",
        f"- Unique genes: {', '.join(payload['unique_genes'])}",
    ]
    for mode, count in payload["mode_counts"].items():
        lines.append(f"- {mode}: {count}")
    for item in payload["top_results"]:
        lines.extend(
            [
                "",
                f"## {item['gene_symbol']}",
                f"- Mode: {item['mode']}",
                f"- Distance: {item['distance_score']:.4f}",
                f"- Significant genes: {item['significant_gene_count']}",
                f"- Top ranked genes: {', '.join(item['top_ranked_genes'])}",
                f"- Top GSEA genes: {', '.join(item['top_gsea_genes'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def _normalize_gsea_scores(scores: np.ndarray) -> np.ndarray:
    positive_scores = np.asarray(scores, dtype=float)
    if positive_scores.size == 0:
        return positive_scores
    if positive_scores.size == 1:
        return np.asarray([0.0], dtype=float)
    if np.allclose(positive_scores, positive_scores[0]):
        return np.zeros_like(positive_scores, dtype=float)

    transformed, _ = stats.boxcox(positive_scores)
    std = float(transformed.std())
    if std == 0.0:
        return np.zeros_like(transformed, dtype=float)
    return np.asarray((transformed - transformed.mean()) / std)
