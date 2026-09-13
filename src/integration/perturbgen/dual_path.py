"""双路径判定：严格 AND、PASS/FAIL/INCONCLUSIVE。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Protocol, Sequence


_VALID_INTERVENTION_TYPES = {"KO", "KD"}


class SupportsPathResult(Protocol):
    status: str
    path: str
    mode: str
    rescue_excl_target: float | None
    evaluable_donors: int
    donor_consistency: float | None
    seed: int
    reason_code: str | None


@dataclass(frozen=True)
class SeedModeSummary:
    seed: int
    mode: str
    status: str
    rescue_excl_target: float | None
    evaluable_donors: int
    donor_consistency: float | None
    matched_null_count: int | None
    empirical_pvalue: float | None
    reason_code: str | None


@dataclass(frozen=True)
class PathDecision:
    path: str
    verdict: str
    primary_mode: str
    median_rescue: float | None
    worst_seed_rescue: float | None
    seed_count: int
    evaluable_donors_min: int | None
    donor_consistency_min: float | None
    reasons: tuple[str, ...]
    seed_summaries: tuple[SeedModeSummary, ...]
    mode_direction_support: Mapping[str, bool]


@dataclass(frozen=True)
class DualPathDecision:
    verdict: str
    q_value: float | None
    intervention_type: str
    reasons: tuple[str, ...]
    path_decisions: tuple[PathDecision, ...]
    candidate_gene: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_path_results(
    path_results: Sequence[SupportsPathResult | Mapping[str, Any]],
    *,
    observed_direction: str,
    intervention_type: str,
    formal_null_min: int = 99,
    smoke_null_count: int = 20,
    expected_seed_count: int = 3,
) -> PathDecision:
    """单路径门控。数据不齐直接 inconclusive，数据齐但未达门才 fail。"""

    route = _normalize_intervention_type(intervention_type)
    if not path_results:
        raise ValueError("path_results must not be empty")
    normalized = [_normalize_result(item) for item in path_results]
    path_names = {item["path"] for item in normalized}
    if len(path_names) != 1:
        raise ValueError("all path_results must belong to the same path")
    identities = [(item["mode"], item["seed"]) for item in normalized]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (mode, seed) result within one path")

    if observed_direction not in {"up", "down"}:
        raise ValueError("observed_direction must be 'up' or 'down'")

    primary_mode = "mask" if observed_direction == "up" else "overexpress"
    required_modes = (
        ("mask", "pad", "delete")
        if route == "KO" and observed_direction == "up"
        else ("mask",)
        if observed_direction == "up"
        else ("overexpress",)
    )
    reasons: list[str] = []
    has_inconclusive = False
    has_fail = False

    mode_groups: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        mode_groups.setdefault(item["mode"], []).append(item)

    if primary_mode not in mode_groups:
        return PathDecision(
            path=next(iter(path_names)),
            verdict="inconclusive",
            primary_mode=primary_mode,
            median_rescue=None,
            worst_seed_rescue=None,
            seed_count=0,
            evaluable_donors_min=None,
            donor_consistency_min=None,
            reasons=("missing_primary_mode",),
            seed_summaries=tuple(),
            mode_direction_support={},
        )

    primary_rows = sorted(mode_groups[primary_mode], key=lambda item: item["seed"])
    seed_summaries = tuple(_to_seed_summary(item) for item in primary_rows)
    seed_count = len({item["seed"] for item in primary_rows})
    if seed_count < expected_seed_count:
        has_inconclusive = True
        reasons.append("insufficient_seeds")

    primary_rescue_values: list[float] = []
    donor_counts: list[int] = []
    donor_consistencies: list[float] = []
    for item in primary_rows:
        status = item["status"]
        if status != "evaluable":
            has_inconclusive = True
            reasons.append(f"seed_{item['seed']}_status_{status}")
            continue

        matched_null_count = item["matched_null_count"]
        if matched_null_count is None:
            has_inconclusive = True
            reasons.append(f"seed_{item['seed']}_missing_null_count")
        elif matched_null_count < formal_null_min:
            has_inconclusive = True
            if matched_null_count == smoke_null_count:
                reasons.append(f"seed_{item['seed']}_smoke_only_null")
            else:
                reasons.append(f"seed_{item['seed']}_insufficient_null")

        evaluable_donors = item["evaluable_donors"]
        donor_consistency = item["donor_consistency"]
        rescue_value = item["rescue_excl_target"]
        if evaluable_donors is None or evaluable_donors < 3:
            has_inconclusive = True
            reasons.append(f"seed_{item['seed']}_insufficient_donors")
            continue
        if donor_consistency is None or rescue_value is None:
            has_inconclusive = True
            reasons.append(f"seed_{item['seed']}_missing_statistics")
            continue

        donor_counts.append(evaluable_donors)
        donor_consistencies.append(donor_consistency)
        primary_rescue_values.append(rescue_value)

        if donor_consistency <= 0.5:
            has_fail = True
            reasons.append(f"seed_{item['seed']}_donor_consistency_not_strict_majority")
        if rescue_value <= 0:
            has_fail = True
            reasons.append(f"seed_{item['seed']}_non_positive_rescue")

    mode_direction_support: dict[str, bool] = {}
    for mode_name in required_modes:
        rows = mode_groups.get(mode_name)
        if not rows:
            has_inconclusive = True
            reasons.append(f"missing_mode_{mode_name}")
            continue

        evaluable_rescue = [
            row["rescue_excl_target"]
            for row in rows
            if row["status"] == "evaluable" and row["rescue_excl_target"] is not None
        ]
        if len(evaluable_rescue) < expected_seed_count:
            has_inconclusive = True
            reasons.append(f"mode_{mode_name}_insufficient_evaluable_seeds")
            continue

        median_rescue = _median(evaluable_rescue)
        mode_direction_support[mode_name] = median_rescue is not None and median_rescue > 0

    if route == "KO" and observed_direction == "up":
        positive_modes = [mode_name for mode_name, supported in mode_direction_support.items() if supported]
        if "mask" in mode_direction_support and not mode_direction_support["mask"]:
            has_fail = True
            reasons.append("mask_not_primary_positive_mode")
        if len(positive_modes) < 2:
            has_fail = True
            reasons.append("ko_modes_not_two_of_three_positive")

    if primary_rescue_values:
        positive_seed_count = sum(value > 0 for value in primary_rescue_values)
        if positive_seed_count < 2:
            has_fail = True
            reasons.append("seed_direction_support_below_two_of_three")
        if min(primary_rescue_values) < 0:
            has_fail = True
            reasons.append("worst_seed_strong_reverse_effect")

    verdict = "pass"
    if has_inconclusive:
        verdict = "inconclusive"
    elif has_fail:
        verdict = "fail"

    return PathDecision(
        path=next(iter(path_names)),
        verdict=verdict,
        primary_mode=primary_mode,
        median_rescue=_median(primary_rescue_values),
        worst_seed_rescue=min(primary_rescue_values) if primary_rescue_values else None,
        seed_count=seed_count,
        evaluable_donors_min=min(donor_counts) if donor_counts else None,
        donor_consistency_min=min(donor_consistencies) if donor_consistencies else None,
        reasons=tuple(dict.fromkeys(reasons)),
        seed_summaries=seed_summaries,
        mode_direction_support=mode_direction_support,
    )


def evaluate_dual_path_candidate(
    path_results: Sequence[SupportsPathResult | Mapping[str, Any]],
    *,
    observed_direction: str,
    intervention_type: str,
    q_value: float | None,
    candidate_gene: str | None = None,
    formal_null_min: int = 99,
    smoke_null_count: int = 20,
    expected_seed_count: int = 3,
    unperturbed_quality_status: str = "inconclusive",
) -> DualPathDecision:
    """双路径严格 AND。"""

    route = _normalize_intervention_type(intervention_type)
    if not path_results:
        raise ValueError("path_results must not be empty")
    if unperturbed_quality_status not in {"pass", "fail", "inconclusive"}:
        raise ValueError("unperturbed_quality_status must be pass/fail/inconclusive")
    if q_value is not None and (not math.isfinite(q_value) or not 0.0 <= q_value <= 1.0):
        raise ValueError("q_value must be finite and within [0, 1]")

    normalized = [_normalize_result(item) for item in path_results]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        grouped.setdefault(item["path"], []).append(item)

    required_paths = ("source_intervention", "within_state")
    path_decisions: list[PathDecision] = []
    reasons: list[str] = []
    has_inconclusive = False
    has_fail = False

    for path_name in required_paths:
        if path_name not in grouped:
            path_decisions.append(
                PathDecision(
                    path=path_name,
                    verdict="inconclusive",
                    primary_mode="mask" if observed_direction == "up" else "overexpress",
                    median_rescue=None,
                    worst_seed_rescue=None,
                    seed_count=0,
                    evaluable_donors_min=None,
                    donor_consistency_min=None,
                    reasons=("missing_path",),
                    seed_summaries=tuple(),
                    mode_direction_support={},
                )
            )
            has_inconclusive = True
            reasons.append(f"{path_name}_missing")
            continue

        decision = evaluate_path_results(
            grouped[path_name],
            observed_direction=observed_direction,
            intervention_type=route,
            formal_null_min=formal_null_min,
            smoke_null_count=smoke_null_count,
            expected_seed_count=expected_seed_count,
        )
        path_decisions.append(decision)
        if decision.verdict == "inconclusive":
            has_inconclusive = True
            reasons.append(f"{path_name}_inconclusive")
        elif decision.verdict == "fail":
            has_fail = True
            reasons.append(f"{path_name}_fail")

    if q_value is None:
        has_inconclusive = True
        reasons.append("missing_q_value")
    elif q_value >= 0.05:
        has_fail = True
        reasons.append("candidate_q_value_not_significant")

    if unperturbed_quality_status == "inconclusive":
        has_inconclusive = True
        reasons.append("unperturbed_quality_inconclusive")
    elif unperturbed_quality_status == "fail":
        has_fail = True
        reasons.append("unperturbed_quality_fail")

    verdict = "pass"
    if has_inconclusive:
        verdict = "inconclusive"
    elif has_fail:
        verdict = "fail"

    return DualPathDecision(
        verdict=verdict,
        q_value=q_value,
        intervention_type=route,
        reasons=tuple(dict.fromkeys(reasons)),
        path_decisions=tuple(path_decisions),
        candidate_gene=candidate_gene,
    )


def _normalize_intervention_type(value: Any) -> str:
    route = str(value).strip().upper()
    if route not in _VALID_INTERVENTION_TYPES:
        raise ValueError("intervention_type must be explicitly 'KO' or 'KD'; no route default is allowed")
    return route


def _normalize_result(result: SupportsPathResult | Mapping[str, Any]) -> dict[str, Any]:
    status = _read_field(result, "status")
    path = _read_field(result, "path")
    mode = _read_field(result, "mode")
    rescue_excl_target = _read_optional_float(result, "rescue_excl_target")
    evaluable_donors = _read_optional_int(result, "evaluable_donors")
    donor_consistency = _read_optional_float(result, "donor_consistency")
    seed = _read_field(result, "seed")
    reason_code = _read_optional_str(result, "reason_code")

    metadata = _read_metadata(result)
    matched_null_count = _read_optional_int(result, "matched_null_count")
    if matched_null_count is None:
        matched_null_count = _coerce_optional_int(metadata.get("matched_null_count"))
    empirical_pvalue = _read_optional_float(result, "empirical_pvalue")
    if empirical_pvalue is None:
        empirical_pvalue = _coerce_optional_float(metadata.get("empirical_pvalue"))

    return {
        "status": str(status),
        "path": str(path),
        "mode": str(mode),
        "rescue_excl_target": rescue_excl_target,
        "evaluable_donors": evaluable_donors,
        "donor_consistency": donor_consistency,
        "seed": int(seed),
        "reason_code": reason_code,
        "matched_null_count": matched_null_count,
        "empirical_pvalue": empirical_pvalue,
    }


def _to_seed_summary(item: Mapping[str, Any]) -> SeedModeSummary:
    return SeedModeSummary(
        seed=int(item["seed"]),
        mode=str(item["mode"]),
        status=str(item["status"]),
        rescue_excl_target=_coerce_optional_float(item.get("rescue_excl_target")),
        evaluable_donors=_coerce_optional_int(item.get("evaluable_donors")) or 0,
        donor_consistency=_coerce_optional_float(item.get("donor_consistency")),
        matched_null_count=_coerce_optional_int(item.get("matched_null_count")),
        empirical_pvalue=_coerce_optional_float(item.get("empirical_pvalue")),
        reason_code=_coerce_optional_str(item.get("reason_code")),
    )


def _read_metadata(result: SupportsPathResult | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        metadata = result.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}
    metadata = getattr(result, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _read_field(result: SupportsPathResult | Mapping[str, Any], name: str) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise KeyError(f"missing field '{name}'")
        return result[name]
    return getattr(result, name)


def _read_optional_float(result: SupportsPathResult | Mapping[str, Any], name: str) -> float | None:
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
    return _coerce_optional_float(value)


def _read_optional_int(result: SupportsPathResult | Mapping[str, Any], name: str) -> int | None:
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
    return _coerce_optional_int(value)


def _read_optional_str(result: SupportsPathResult | Mapping[str, Any], name: str) -> str | None:
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
    return _coerce_optional_str(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    size = len(ordered)
    midpoint = size // 2
    if size % 2 == 1:
        return ordered[midpoint]
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)
