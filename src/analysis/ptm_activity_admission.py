"""Frozen activity-admission gate executed before signed propagation (方案 §4.4/§5.2).

方案 §5.2 要求 activity 在进入 signed 传播前执行*预先冻结*的
q-value / substrate / coverage 准入（阈值登记在 ``ptm_research_config.yaml``
的 ``activity_admission`` 段，由
:class:`~src.analysis.ptm_research_config.ActivityAdmissionPolicy` 承载），
且 benchmark 未通过时只能保留 exploratory 输出。

本模块是传播的唯一选择入口：``select_activities_for_propagation`` 返回
admitted regulators、逐行拒绝原因与 policy hash；``activities_for_propagation``
时代的“method/contrast 之外全选”行为自此删除，不再存在绕过 admission 的
传播路径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.analysis.ptm_activity import PTMActivityContractError
from src.analysis.ptm_activity_benchmark import ActivityBenchmarkReport
from src.analysis.ptm_research_config import ActivityAdmissionPolicy

_ADMISSION_COLUMNS = ("activity_qvalue", "n_substrates", "network_coverage")


@dataclass(frozen=True)
class ActivitySelectionResult:
    """Admitted regulators plus per-row rejection reasons for one method/contrast."""

    admitted: Mapping[str, float]
    rejected: pd.DataFrame
    policy: ActivityAdmissionPolicy
    method: str
    condition_or_contrast: str | None
    n_selected_rows: int
    #: Optional verdict from ``evaluate_activity_benchmark``; when present the
    #: manifest's ``benchmark_gate`` reflects it, otherwise the gate stays
    #: ``available=false`` and admitted activities remain exploratory-only.
    benchmark_report: ActivityBenchmarkReport | None = None

    def as_dict(self) -> dict[str, Any]:
        rejected = self.rejected
        if self.benchmark_report is None:
            benchmark_gate: dict[str, Any] = {
                # 方案 §5.2：无独立 kinase-perturbation benchmark 校准时，
                # admitted 结果仍只有 exploratory 效力；这里显式登记该边界，
                # 不伪造 benchmark PASS。
                "available": False,
                "requirement": "independent kinase perturbation benchmark (方案 §5.2)",
                "effect": "admitted activities are exploratory only; may_enter_lineage stays false",
            }
        else:
            report = self.benchmark_report
            benchmark_gate = {
                "available": True,
                "passed": report.passed,
                "verdict": report.verdict,
                "effect": (
                    "benchmark PASS: admitted activities may enter formal lineage"
                    if report.passed
                    else "benchmark FAIL: admitted activities are exploratory only; may_enter_lineage stays false"
                ),
                "report": report.as_dict(),
            }
        return {
            "policy": self.policy.as_dict(),
            "policy_hash": self.policy.policy_hash,
            "method": self.method,
            "condition_or_contrast": self.condition_or_contrast,
            "n_selected_rows": self.n_selected_rows,
            "n_admitted": len(self.admitted),
            "n_rejected": int(len(rejected)),
            "rejected": rejected.to_dict("records") if not rejected.empty else [],
            "benchmark_gate": benchmark_gate,
        }


def _rejection_reason(record: Mapping[str, Any], policy: ActivityAdmissionPolicy) -> str | None:
    reasons: list[str] = []
    if float(record["activity_qvalue"]) > policy.max_activity_qvalue:
        reasons.append("activity_qvalue_above_max")
    if int(record["n_substrates"]) < policy.min_substrates:
        reasons.append("n_substrates_below_min")
    if float(record["network_coverage"]) < policy.min_network_coverage:
        reasons.append("network_coverage_below_min")
    return ";".join(reasons) if reasons else None


def select_activities_for_propagation(
    activity_frame: pd.DataFrame,
    *,
    method: str,
    condition_or_contrast: str | None = None,
    policy: ActivityAdmissionPolicy | None = None,
    benchmark_report: ActivityBenchmarkReport | None = None,
) -> ActivitySelectionResult:
    """Select signed regulator activities for one method/contrast under a frozen policy.

    The propagation input is the *primary* method's scores; a sensitivity
    method (方案 §4.2) must be propagated separately and compared, never
    averaged into one truth.  Every selected row is checked against the
    frozen ``ActivityAdmissionPolicy``; rejected rows keep their reason and
    never reach the signed network.  ``benchmark_report`` (from
    ``evaluate_activity_benchmark``) only annotates the manifest gate — a
    FAIL keeps outputs exploratory, it never silently drops regulators.
    """

    frozen_policy = policy if policy is not None else ActivityAdmissionPolicy()
    selected = activity_frame[activity_frame["method"].astype(str).str.strip() == method]
    if selected.empty:
        raise PTMActivityContractError(
            f"activity table has no rows for primary method {method!r}; available: "
            f"{sorted(set(activity_frame['method'].astype(str)))}"
        )
    if condition_or_contrast is not None:
        selected = selected[selected["condition_or_contrast"].astype(str).str.strip() == condition_or_contrast]
        if selected.empty:
            raise PTMActivityContractError(
                f"activity table has no rows for contrast {condition_or_contrast!r} under method {method!r}"
            )
    missing = [column for column in _ADMISSION_COLUMNS if column not in selected.columns]
    if missing:
        raise PTMActivityContractError(
            "activity admission requires columns " + ", ".join(missing) + "; load via load_ptm_activity_table"
        )

    admitted: dict[str, float] = {}
    rejected_rows: list[dict[str, Any]] = []
    for record in selected.to_dict("records"):
        regulator_id = str(record["regulator_id"]).strip()
        if not regulator_id:
            raise PTMActivityContractError("regulator_id must not be empty")
        score = float(record["activity_score"])
        if not math.isfinite(score):
            raise PTMActivityContractError(f"regulator {regulator_id} has a non-finite activity score")
        if regulator_id in admitted and admitted[regulator_id] != score:
            raise PTMActivityContractError(
                f"regulator {regulator_id} has conflicting activity scores for method {method!r}"
            )
        reason = _rejection_reason(record, frozen_policy)
        if reason is None:
            admitted[regulator_id] = score
        else:
            rejected_rows.append({"regulator_id": regulator_id, "activity_score": score, "reason": reason})
    if not admitted:
        raise PTMActivityContractError(
            "activity admission rejected every regulator under the frozen policy "
            f"(max_activity_qvalue={frozen_policy.max_activity_qvalue}, "
            f"min_substrates={frozen_policy.min_substrates}, "
            f"min_network_coverage={frozen_policy.min_network_coverage}); "
            f"rejected reasons: {rejected_rows}"
        )
    rejected = pd.DataFrame(rejected_rows, columns=["regulator_id", "activity_score", "reason"])
    return ActivitySelectionResult(
        admitted=admitted,
        rejected=rejected,
        policy=frozen_policy,
        method=method,
        condition_or_contrast=condition_or_contrast,
        n_selected_rows=len(selected),
        benchmark_report=benchmark_report,
    )


__all__ = [
    "ActivitySelectionResult",
    "select_activities_for_propagation",
]
