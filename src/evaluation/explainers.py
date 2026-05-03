"""Explainability helpers for PTM candidate ranking and two-stage interpretation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch

from ..integration.contracts import CandidateRecord, PerturbationResult
from ..integration.genki_reports import (
    build_generank_dataframe,
    build_gsea_ranked_dataframe,
    build_two_stage_summary_payload,
    render_two_stage_summary_markdown,
)
from ..utils.io import save_dataframe, save_json


def _parse_ptm_sites(ptm_sites: Any) -> List[Dict[str, Any]]:
    if isinstance(ptm_sites, list):
        return [site for site in ptm_sites if isinstance(site, dict)]
    if isinstance(ptm_sites, str) and ptm_sites.strip():
        parsed = json.loads(ptm_sites)
        if isinstance(parsed, list):
            return [site for site in parsed if isinstance(site, dict)]
    return []


def _encode_ptm_mask(
    ptm_sites: Iterable[Dict[str, Any]],
    sequence_length: int,
    drop_site: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    mask = torch.zeros((1, sequence_length), dtype=torch.float32)
    for site in ptm_sites:
        if drop_site is not None and site == drop_site:
            continue
        position = int(site.get("position", 0)) - 1
        if 0 <= position < sequence_length:
            mask[0, position] = 1.0
    return mask


class LeaveOnePTMOutScorer:
    """Ranks PTM sites by the probability drop observed after masking each site."""

    def __init__(
        self,
        target_class_index: int,
        label_names: Optional[List[str]] = None,
        amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
        ptm_type_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        self.target_class_index = target_class_index
        self.label_names = label_names
        self.aa_to_index = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
        self.ptm_type_to_index = ptm_type_to_index or {}

    def rank_row(self, model: Any, row: pd.Series) -> List[CandidateRecord]:
        sequence = str(row.get("sequence", ""))
        if not sequence:
            return []

        ptm_sites = _parse_ptm_sites(row.get("ptm_sites", "[]"))
        if not ptm_sites:
            return []

        baseline = self._predict(model, sequence, ptm_sites)
        candidates: List[CandidateRecord] = []
        for site in ptm_sites:
            perturbed = self._predict(model, sequence, ptm_sites, drop_site=site)
            candidates.append(
                CandidateRecord(
                    sample_id=str(row.get("sample_id", "")),
                    protein_id=str(row.get("protein_id", "")),
                    ptm_type=str(site.get("type", "")),
                    ptm_position=int(site.get("position", -1)),
                    baseline_label=baseline["label"],
                    baseline_probability=baseline["probability"],
                    perturbed_probability=perturbed["probability"],
                    delta_probability=baseline["probability"] - perturbed["probability"],
                )
            )
        return sorted(candidates, key=lambda item: item.delta_probability, reverse=True)

    def _predict(
        self,
        model: Any,
        sequence: str,
        ptm_sites: List[Dict[str, Any]],
        drop_site: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        batch = self._build_batch(sequence, ptm_sites, drop_site=drop_site)
        outputs = model(batch)
        probabilities = outputs["probabilities"][0]
        prediction_index = int(outputs["predictions"][0].item())
        if self.label_names and 0 <= prediction_index < len(self.label_names):
            label = self.label_names[prediction_index]
        else:
            label = str(prediction_index)
        return {
            "label": label,
            "probability": float(probabilities[self.target_class_index].item()),
        }

    def _build_batch(
        self,
        sequence: str,
        ptm_sites: List[Dict[str, Any]],
        drop_site: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        sequence_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        for index, amino_acid in enumerate(sequence):
            sequence_tensor[0, index] = self.aa_to_index.get(amino_acid, 0)

        ptm_mask = _encode_ptm_mask(ptm_sites, len(sequence), drop_site=drop_site)
        ptm_types = torch.zeros((1, len(sequence)), dtype=torch.long)
        for site in ptm_sites:
            if drop_site is not None and site == drop_site:
                continue
            position = int(site.get("position", 0)) - 1
            if 0 <= position < len(sequence):
                ptm_types[0, position] = self.ptm_type_to_index.get(str(site.get("type", "")), 0)

        return {
            "sequence": sequence_tensor,
            "ptm_mask": ptm_mask,
            "ptm_types": ptm_types,
            "ptm_positions": torch.arange(len(sequence), dtype=torch.long).unsqueeze(0),
        }


def aggregate_by_protein(candidates: List[CandidateRecord]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for candidate in candidates:
        scores[candidate.protein_id] = max(scores.get(candidate.protein_id, 0.0), candidate.delta_probability)
    return scores


def _build_gene_ranking_rows(
    candidate: CandidateRecord,
    result: PerturbationResult,
) -> List[Dict[str, Any]]:
    generank = build_generank_dataframe(result)
    rows: List[Dict[str, Any]] = []
    for _, ranked_row in generank.iterrows():
        rows.append(
            {
                "sample_id": candidate.sample_id,
                "protein_id": candidate.protein_id,
                "ptm_type": candidate.ptm_type,
                "ptm_position": candidate.ptm_position,
                "perturbation_gene": result.gene_symbol,
                "mode": result.mode,
                "rank": int(ranked_row["rank"]),
                "affected_gene": ranked_row["affected_gene"],
                "dis": ranked_row["dis"],
                "gene_index": ranked_row["gene_index"],
                "hit": ranked_row["hit"],
                "frequency": ranked_row["frequency"],
                "empirical_pvalue": ranked_row["empirical_pvalue"],
                "adjusted_pvalue": ranked_row["adjusted_pvalue"],
                "is_significant": ranked_row["is_significant"],
            }
        )
    return rows


class TwoStageExplanationPipeline:
    """Runs candidate ranking followed by GenKI-style perturbation explanation."""

    def __init__(
        self,
        scorer: Any,
        mapper: Any,
        genki_adapter: Any,
        top_k_candidates: int = 1,
        perturbation_mode: str = "hard_ko",
    ) -> None:
        self.scorer = scorer
        self.mapper = mapper
        self.genki_adapter = genki_adapter
        self.top_k_candidates = top_k_candidates
        self.perturbation_mode = perturbation_mode

    def run(self, model: Any, df: pd.DataFrame, output_dir: Any) -> List[PerturbationResult]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        candidate_rows: List[Dict[str, Any]] = []
        explanation_rows: List[Dict[str, Any]] = []
        gene_ranking_rows: List[Dict[str, Any]] = []
        significant_gene_rows: List[Dict[str, Any]] = []
        gsea_rows: List[Dict[str, Any]] = []
        unmapped_rows: List[Dict[str, Any]] = []
        results: List[PerturbationResult] = []

        for _, row in df.iterrows():
            ranked = self.scorer.rank_row(model, row)
            if not ranked:
                continue
            candidate_rows.extend(asdict(candidate) for candidate in ranked)
            for candidate in ranked[: max(1, self.top_k_candidates)]:
                try:
                    gene_symbol = self.mapper.map_candidate(candidate)
                except KeyError:
                    unmapped_rows.append(asdict(candidate))
                    continue

                request = self.genki_adapter.build_request(
                    gene_symbol=gene_symbol,
                    mode=self.perturbation_mode,
                    magnitude=candidate.delta_probability,
                )
                result = self.genki_adapter.run(request)
                results.append(result)
                ranked_rows = _build_gene_ranking_rows(candidate, result)
                gene_ranking_rows.extend(ranked_rows)
                significant_gene_rows.extend([row for row in ranked_rows if row["is_significant"]])
                gsea = build_gsea_ranked_dataframe(result)
                if not gsea.empty:
                    for _, gsea_row in gsea.iterrows():
                        gsea_rows.append(
                            {
                                "sample_id": candidate.sample_id,
                                "protein_id": candidate.protein_id,
                                "ptm_type": candidate.ptm_type,
                                "ptm_position": candidate.ptm_position,
                                "perturbation_gene": result.gene_symbol,
                                "mode": result.mode,
                                "affected_gene": gsea_row["affected_gene"],
                                "dis": gsea_row["dis"],
                                "dis_norm": gsea_row["dis_norm"],
                                "gsea_rank": int(gsea_row["gsea_rank"]),
                            }
                        )
                explanation_rows.append(
                    {
                        "sample_id": candidate.sample_id,
                        "protein_id": candidate.protein_id,
                        "ptm_type": candidate.ptm_type,
                        "ptm_position": candidate.ptm_position,
                        "gene_symbol": result.gene_symbol,
                        "mode": result.mode,
                        "distance_score": result.distance_score,
                        "ranked_genes": result.ranked_genes,
                        "metadata": result.metadata,
                    }
                )

        save_dataframe(pd.DataFrame(candidate_rows), str(output_path / "candidate_scores.csv"))
        save_dataframe(pd.DataFrame(gene_ranking_rows), str(output_path / "gene_rankings.csv"))
        save_dataframe(pd.DataFrame(significant_gene_rows), str(output_path / "significant_gene_rankings.csv"))
        pd.DataFrame(gsea_rows).to_csv(output_path / "gsea_rankings.tsv", sep="\t", index=False, encoding="utf-8")
        save_dataframe(pd.DataFrame(unmapped_rows), str(output_path / "unmapped_candidates.csv"))
        save_json(explanation_rows, str(output_path / "genki_explanations.json"))
        save_json(build_two_stage_summary_payload(results), str(output_path / "two_stage_summary.json"))
        self._write_markdown_summary(results, output_path)
        return results

    def _write_markdown_summary(self, results: List[PerturbationResult], output_dir: Path) -> None:
        (output_dir / "two_stage_summary.md").write_text(
            render_two_stage_summary_markdown(results),
            encoding="utf-8",
        )
