"""Candidate-level empirical p-value aggregation.

Per-run calibration already uses ``(k+1)/(n+1)`` in
:func:`evaluate_null_calibration`.  Scheme §4.7 condition 6 needs a *candidate*
p that can enter BH-FDR.  A multi-run estimand (path × mode × seed) was not
signed off by a statistics owner, so this module implements one conservative,
fully declared rule and refuses to invent Fisher/Tippett combinations.

Estimand
--------
``conservative_max_required_runs``: among the required *primary-mode* runs
(both paths × each required seed), take the maximum finite empirical p-value.

This is conservative relative to an AND of path-level tests: the candidate is
only as significant as its weakest required primary run.  Sensitivity modes
(pad/delete) are excluded from the p-value; they remain the KO 2-of-3 gate in
``dual_path``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from src.models.gene_vocabulary import normalize_ensembl_id

CANDIDATE_PVALUE_SCHEMA_VERSION = "perturbgen_candidate_empirical_pvalue/v1"
CONSERVATIVE_MAX_ESTIMAND = "conservative_max_required_runs"
_VALID_DIRECTIONS = {"up", "down"}
_VALID_ROUTES = {"KO", "KD"}
_VALID_PATHS = {"source_intervention", "within_state"}


class EmpiricalPvalueError(ValueError):
    """Raised when required empirical p-values are missing or malformed."""


def primary_mode_for_direction(observed_direction: str) -> str:
    direction = str(observed_direction).strip().lower()
    if direction not in _VALID_DIRECTIONS:
        raise EmpiricalPvalueError("observed_direction must be 'up' or 'down'")
    return "mask" if direction == "up" else "overexpress"


def _normalize_run(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise EmpiricalPvalueError("each run result must be a mapping")
    path = str(result.get("path", "")).strip()
    if path not in _VALID_PATHS:
        raise EmpiricalPvalueError(f"run path must be one of {sorted(_VALID_PATHS)}")
    mode = str(result.get("mode", "")).strip()
    seed_raw = result.get("seed")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int) or seed_raw < 0:
        raise EmpiricalPvalueError("run seed must be a non-negative integer")
    pvalue = result.get("empirical_pvalue")
    if pvalue is None and isinstance(result.get("path_result"), Mapping):
        pvalue = result["path_result"].get("empirical_pvalue")
    if pvalue is None and isinstance(result.get("metadata"), Mapping):
        pvalue = result["metadata"].get("empirical_pvalue")
    converted: float | None
    if pvalue is None:
        converted = None
    else:
        try:
            converted = float(pvalue)
        except (TypeError, ValueError) as exc:
            raise EmpiricalPvalueError("empirical_pvalue must be numeric") from exc
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise EmpiricalPvalueError("empirical_pvalue must be finite and within [0, 1]")
    return {
        "path": path,
        "mode": mode,
        "seed": int(seed_raw),
        "empirical_pvalue": converted,
    }


def aggregate_candidate_empirical_pvalues(
    run_results: Sequence[Mapping[str, Any]],
    intervention_type: str,
    observed_direction: str,
    *,
    ensembl_id: str,
    expected_seed_count: int = 3,
) -> dict[str, Any]:
    """Aggregate per-run empirical p-values into one candidate p with provenance.

    Formal callers must not substitute a uniform or hand-filled p when this
    function raises: missing coverage is ``INCONCLUSIVE``, not a fabricated p.
    """

    route = str(intervention_type).strip().upper()
    if route not in _VALID_ROUTES:
        raise EmpiricalPvalueError("intervention_type must be explicitly 'KO' or 'KD'")
    if isinstance(expected_seed_count, bool) or expected_seed_count < 1:
        raise EmpiricalPvalueError("expected_seed_count must be a positive integer")
    if isinstance(run_results, (str, bytes)) or not isinstance(run_results, Sequence) or not run_results:
        raise EmpiricalPvalueError("run_results must be a non-empty sequence")

    candidate_id = normalize_ensembl_id(ensembl_id)
    primary_mode = primary_mode_for_direction(observed_direction)
    normalized = [_normalize_run(item) for item in run_results]
    indexed: dict[tuple[str, str, int], float] = {}
    for item in normalized:
        key = (item["path"], item["mode"], item["seed"])
        if key in indexed:
            raise EmpiricalPvalueError(f"duplicate run identity {key}")
        if item["empirical_pvalue"] is not None:
            indexed[key] = item["empirical_pvalue"]

    primary_seeds = sorted(
        {seed for path, mode, seed in indexed if path == "source_intervention" and mode == primary_mode}
    )
    if len(primary_seeds) < expected_seed_count:
        raise EmpiricalPvalueError(
            f"required at least {expected_seed_count} primary-mode seeds on source_intervention; "
            f"got {len(primary_seeds)}"
        )
    seeds = tuple(primary_seeds[:expected_seed_count])
    required_keys = tuple(
        (path, primary_mode, seed) for path in ("source_intervention", "within_state") for seed in seeds
    )
    missing = [key for key in required_keys if key not in indexed]
    if missing:
        pretty = ", ".join(f"{path}/{mode}/seed={seed}" for path, mode, seed in missing)
        raise EmpiricalPvalueError(f"missing finite empirical_pvalue for required runs: {pretty}")

    values = [indexed[key] for key in required_keys]
    pvalue = max(values)
    provenance_runs = [
        {
            "path": path,
            "mode": mode,
            "seed": seed,
            "empirical_pvalue": indexed[(path, mode, seed)],
        }
        for path, mode, seed in required_keys
    ]
    return {
        "schema_version": CANDIDATE_PVALUE_SCHEMA_VERSION,
        "ensembl_id": candidate_id,
        "pvalue": pvalue,
        "estimand": CONSERVATIVE_MAX_ESTIMAND,
        "intervention_type": route,
        "observed_direction": str(observed_direction).strip().lower(),
        "primary_mode": primary_mode,
        "required_run_count": len(required_keys),
        "provenance": {
            "runs": provenance_runs,
            "aggregation": "max",
            "excluded_from_estimand": "sensitivity_modes_pad_delete",
        },
    }
