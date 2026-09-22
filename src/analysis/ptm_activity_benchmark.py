"""Independent kinase-perturbation benchmark evaluation for activity ranking (方案 §5.2).

方案 §5.2 第 4–5 条要求用*独立*的已知 kinase perturbation phosphoproteomics
benchmark 验证 activity 排名，且 benchmark 未通过时 activity 结果只能保留
exploratory 效力。本模块把该要求落成可执行合同：

* ``load_activity_benchmark_table`` 校验 benchmark 表契约（每行一个被扰动
  kinase 及其 signed perturbation effect，regulator 唯一）；
* ``evaluate_activity_benchmark`` 计算 paired regulator 集上的方向
  concordance、Spearman 秩相关与 bootstrap CI，按预注册的
  ``ActivityBenchmarkCriteria`` 给出 PASS/FAIL，并记录输入内容 hash。

阈值没有"宽松默认"：criteria 必须由研究负责人在 frozen config 的
``activity_benchmark`` 段显式预注册（方案 §8.7），未注册时传播 CLI 拒绝
benchmark 评估而不是替用户挑阈值。PASS/FAIL 判据使用点估计，CI 仅供复核。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.ptm_activity import PTMActivityContractError
from src.analysis.ptm_research_config import ActivityBenchmarkCriteria

BENCHMARK_REQUIRED_COLUMNS: tuple[str, ...] = (
    "regulator_id",
    "perturbation_effect",
)

BENCHMARK_SCHEMA_VERSION = "ptm2cellnet.activity-benchmark/v1"


@dataclass(frozen=True)
class ActivityBenchmarkReport:
    """Immutable benchmark verdict consumed by the admission gate."""

    passed: bool
    n_paired: int
    direction_concordance: float
    spearman_rho: float
    paired_regulators: tuple[str, ...]
    criteria: ActivityBenchmarkCriteria
    activity_input_hash: str
    benchmark_input_hash: str
    n_activity_rows: int
    n_benchmark_rows: int
    method: str | None
    condition_or_contrast: str | None
    bootstrap: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def criteria_hash(self) -> str:
        payload = self.criteria.as_dict()
        digest = hashlib.sha256()
        digest.update(BENCHMARK_SCHEMA_VERSION.encode("utf-8"))
        digest.update(repr(sorted(payload.items())).encode("utf-8"))
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "verdict": self.verdict,
            "passed": self.passed,
            "n_paired": self.n_paired,
            "direction_concordance": self.direction_concordance,
            "spearman_rho": self.spearman_rho,
            "paired_regulators": list(self.paired_regulators),
            "criteria": self.criteria.as_dict(),
            "criteria_hash": self.criteria_hash,
            "activity_input_hash": self.activity_input_hash,
            "benchmark_input_hash": self.benchmark_input_hash,
            "n_activity_rows": self.n_activity_rows,
            "n_benchmark_rows": self.n_benchmark_rows,
            "method": self.method,
            "condition_or_contrast": self.condition_or_contrast,
            "bootstrap": dict(self.bootstrap),
        }


def load_activity_benchmark_table(path: str | Path) -> pd.DataFrame:
    """Load and validate the kinase-perturbation benchmark table (方案 §5.2)."""

    frame = pd.read_csv(path, sep="\t")
    missing = [column for column in BENCHMARK_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PTMActivityContractError(
            "activity benchmark table requires columns " + ", ".join(missing) + "; one row per perturbed kinase"
        )
    if frame.empty:
        raise PTMActivityContractError("activity benchmark table must not be empty")
    seen: set[str] = set()
    for record in frame.to_dict("records"):
        regulator_id = str(record["regulator_id"]).strip()
        if not regulator_id:
            raise PTMActivityContractError("activity benchmark regulator_id must not be empty")
        if regulator_id in seen:
            raise PTMActivityContractError(f"activity benchmark has duplicate rows for regulator {regulator_id!r}")
        seen.add(regulator_id)
        effect = float(record["perturbation_effect"])
        if not math.isfinite(effect):
            raise PTMActivityContractError(
                f"activity benchmark perturbation_effect for {regulator_id!r} must be finite"
            )
    return frame


def _frame_content_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    hashed = pd.util.hash_pandas_object(frame.astype(str), index=False)
    for value in sorted(int(item) for item in hashed.tolist()):
        digest.update(value.to_bytes(8, "big"))
    return digest.hexdigest()


def _bootstrap_ci(
    scores: np.ndarray,
    effects: np.ndarray,
    criteria: ActivityBenchmarkCriteria,
) -> dict[str, Any]:
    if criteria.bootstrap_iterations == 0:
        return {"iterations": 0, "seed": criteria.seed, "note": "bootstrap disabled by pre-registered criteria"}
    rng = np.random.default_rng(criteria.seed)
    n = int(scores.shape[0])
    alpha = (1.0 - criteria.ci_level) / 2.0
    concordance_samples: list[float] = []
    spearman_samples: list[float] = []
    for _ in range(criteria.bootstrap_iterations):
        indices = rng.integers(0, n, n)
        sample_scores = scores[indices]
        sample_effects = effects[indices]
        concordance_samples.append(float(np.mean(np.sign(sample_scores) == np.sign(sample_effects))))
        rho = float(spearmanr(sample_scores, sample_effects).statistic)
        if math.isnan(rho):
            continue
        spearman_samples.append(rho)
    concordance_ci = (
        [float(np.quantile(concordance_samples, alpha)), float(np.quantile(concordance_samples, 1.0 - alpha))]
        if concordance_samples
        else None
    )
    spearman_ci = (
        [float(np.quantile(spearman_samples, alpha)), float(np.quantile(spearman_samples, 1.0 - alpha))]
        if spearman_samples
        else None
    )
    return {
        "iterations": criteria.bootstrap_iterations,
        "seed": criteria.seed,
        "ci_level": criteria.ci_level,
        "direction_concordance_ci": concordance_ci,
        "spearman_rho_ci": spearman_ci,
        "n_spearman_valid": len(spearman_samples),
    }


def evaluate_activity_benchmark(
    activity_table: pd.DataFrame,
    benchmark_table: pd.DataFrame,
    *,
    criteria: ActivityBenchmarkCriteria,
    method: str | None = None,
    condition_or_contrast: str | None = None,
) -> ActivityBenchmarkReport:
    """Evaluate activity ranking against an independent perturbation benchmark.

    The evaluated scope is the activity table filtered to ``method`` /
    ``condition_or_contrast`` (both optional, ``None`` keeps all rows); the
    benchmark is evaluated on the full filtered scope, not on the admitted
    subset, so the verdict calibrates the *method* and stays independent of the
    admission policy.  PASS requires the paired-regulator count, direction
    concordance and Spearman rho to all meet the pre-registered criteria;
    a FAIL keeps downstream outputs exploratory (方案 §5.2 第 5 条).
    """

    selected = activity_table
    if method is not None:
        selected = selected[selected["method"].astype(str).str.strip() == method]
        if selected.empty:
            raise PTMActivityContractError(
                f"activity table has no rows for benchmark method {method!r}; available: "
                f"{sorted(set(activity_table['method'].astype(str)))}"
            )
    if condition_or_contrast is not None:
        selected = selected[selected["condition_or_contrast"].astype(str).str.strip() == condition_or_contrast]
        if selected.empty:
            raise PTMActivityContractError(
                f"activity table has no rows for contrast {condition_or_contrast!r} under method {method!r}"
            )

    activity_scores: dict[str, float] = {}
    for record in selected.to_dict("records"):
        regulator_id = str(record["regulator_id"]).strip()
        score = float(record["activity_score"])
        if not math.isfinite(score):
            raise PTMActivityContractError(f"regulator {regulator_id} has a non-finite activity score")
        if regulator_id in activity_scores and activity_scores[regulator_id] != score:
            raise PTMActivityContractError(
                f"regulator {regulator_id} has conflicting activity scores within the benchmark scope"
            )
        activity_scores[regulator_id] = score

    benchmark_scores = {
        str(record["regulator_id"]).strip(): float(record["perturbation_effect"])
        for record in benchmark_table.to_dict("records")
    }
    paired_ids = sorted(set(activity_scores) & set(benchmark_scores))
    n_paired = len(paired_ids)
    if n_paired == 0:
        raise PTMActivityContractError(
            "activity and benchmark tables share no regulator_id; benchmark evaluation is undefined "
            "(check that both use the same kinase identifier space)"
        )
    scores = np.array([activity_scores[regulator] for regulator in paired_ids], dtype=float)
    effects = np.array([benchmark_scores[regulator] for regulator in paired_ids], dtype=float)
    direction_concordance = float(np.mean(np.sign(scores) == np.sign(effects)))
    spearman_rho = float(spearmanr(scores, effects).statistic)

    passed = (
        n_paired >= criteria.min_paired_regulators
        and direction_concordance >= criteria.min_direction_concordance
        and not math.isnan(spearman_rho)
        and spearman_rho >= criteria.min_abs_spearman
    )
    bootstrap = _bootstrap_ci(scores, effects, criteria)
    return ActivityBenchmarkReport(
        passed=passed,
        n_paired=n_paired,
        direction_concordance=direction_concordance,
        spearman_rho=spearman_rho,
        paired_regulators=tuple(paired_ids),
        criteria=criteria,
        activity_input_hash=_frame_content_hash(selected.reset_index(drop=True)),
        benchmark_input_hash=_frame_content_hash(benchmark_table.reset_index(drop=True)),
        n_activity_rows=int(len(selected)),
        n_benchmark_rows=int(len(benchmark_table)),
        method=method,
        condition_or_contrast=condition_or_contrast,
        bootstrap=bootstrap,
    )


__all__ = [
    "BENCHMARK_REQUIRED_COLUMNS",
    "BENCHMARK_SCHEMA_VERSION",
    "ActivityBenchmarkReport",
    "evaluate_activity_benchmark",
    "load_activity_benchmark_table",
]
