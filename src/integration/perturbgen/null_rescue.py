"""Bind runner perturb stage results to matched-null ``NullStageRecord`` rows.

``run_matched_null_stages`` requires a ``rescue_extractor(request, stage_result)``
callable for GPU execution.  This module provides that binding: it reads the
runner's stage manifest for the registered ``result_h5ad`` output and its
sha256, then computes ``rescue_excl_target`` with the exact same donor
aggregation (baseline ``pred_counts``, perturbed ``X``) and held-out DEG
signature scoring used for candidate path extraction, excluding the perturbed
null gene itself.  DEG column names, the h5ad donor/var columns, and the gene
namespace shared by both must be supplied explicitly by the caller; nothing is
guessed or silently substituted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .results import (  # noqa: SLF001
    _aggregate_donor_expression_from_h5ad,
    _coerce_dataframe,
    _require_columns,
    summarize_rescue_by_donor,
    validate_perturbgen_h5ad_schema,
)
from .null_generation import NullGenerationError, NullStageRecord, _stage_output_record

RescueExtractor = Callable[[Mapping[str, Any], Any], NullStageRecord]


def build_stage_rescue_extractor(
    deg_table: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    donor_obs_column: str,
    var_gene_column: str,
    donor_column: str = "donor",
    gene_column: str = "gene",
    effect_column: str = "log2fc",
    fdr_column: str = "fdr",
    fdr_threshold: float = 0.05,
    min_training_donors: int = 2,
    min_evaluable_donors: int = 3,
    top_k: int = 50,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int | None = 0,
) -> RescueExtractor:
    """Return a ``rescue_extractor`` bound to one donor-level DEG table.

    The DEG table must already use the same gene namespace as
    ``var[var_gene_column]`` in the perturb result h5ad; a namespace mismatch
    fails loudly as ``insufficient evaluable donors`` because per-donor
    signature scoring cannot resolve the DEG genes.
    """

    if min_evaluable_donors < 3:
        raise NullGenerationError("min_evaluable_donors must be >= 3")
    frame = _coerce_dataframe(deg_table)
    _require_columns(frame, donor_column, gene_column, effect_column, fdr_column)

    def extract(request: Mapping[str, Any], stage_result: Any) -> NullStageRecord:
        status = getattr(stage_result, "status", None)
        manifest_path = getattr(stage_result, "manifest_path", None)
        artifacts = getattr(stage_result, "artifacts", None)
        if status != "success" or manifest_path is None or not isinstance(artifacts, Mapping):
            raise NullGenerationError(
                "rescue extractor requires a successful StageExecutionResult with manifest_path and artifacts"
            )
        manifest_file = Path(str(manifest_path)).expanduser().resolve(strict=True)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise NullGenerationError(f"stage manifest must contain a JSON object: {manifest_file}")

        raw_result = artifacts.get("result_h5ad")
        if not isinstance(raw_result, str) or not raw_result.strip():
            raise NullGenerationError("stage result artifacts must contain the 'result_h5ad' output")
        result_h5ad = Path(raw_result).expanduser().resolve(strict=True)
        output_record = _stage_output_record(manifest, result_h5ad=result_h5ad)

        null_symbol = str(request["null_gene_symbol"]).strip()
        if not null_symbol:
            raise NullGenerationError("null request gene_symbol must be a non-empty string")

        validate_perturbgen_h5ad_schema(
            result_h5ad,
            required_layers=("true_counts", "pred_counts"),
            required_obs=(donor_obs_column,),
            required_var=() if var_gene_column == "__index__" else (var_gene_column,),
        )
        baseline_by_donor, perturbed_by_donor = _aggregate_donor_expression_from_h5ad(
            output_h5ad=result_h5ad,
            donor_obs_column=donor_obs_column,
            var_gene_column=var_gene_column,
        )
        rescue = summarize_rescue_by_donor(
            baseline_by_donor,
            perturbed_by_donor,
            frame,
            target_gene=null_symbol,
            donor_column=donor_column,
            gene_column=gene_column,
            effect_column=effect_column,
            fdr_column=fdr_column,
            fdr_threshold=fdr_threshold,
            min_training_donors=min_training_donors,
            top_k=top_k,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        if rescue.evaluable_donors < min_evaluable_donors:
            reasons = [score.reason_code or "unevaluable_donor" for score in rescue.donor_scores if not score.evaluable]
            raise NullGenerationError(
                f"null {request['null_ensembl_id']} rescue has only {rescue.evaluable_donors} "
                f"evaluable donors (<{min_evaluable_donors}); reasons: {sorted(set(reasons))}"
            )
        median_rescue = rescue.median_rescue
        if median_rescue is None or not math.isfinite(median_rescue):
            raise NullGenerationError(f"null {request['null_ensembl_id']} median rescue is not finite")

        return NullStageRecord(
            null_ensembl_id=str(request["null_ensembl_id"]),
            null_gene_symbol=null_symbol,
            candidate_ensembl_id=str(request["candidate_ensembl_id"]),
            path=str(request["path"]),
            mode=str(request["mode"]),
            seed=int(request["seed"]),
            rescue_excl_target=float(median_rescue),
            stage_manifest=str(manifest_file),
            result_h5ad=str(result_h5ad),
            result_h5ad_sha256=str(output_record["sha256"]),
        )

    return extract
