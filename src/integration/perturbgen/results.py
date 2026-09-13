"""统计核心：held-out signature、rescue、null 与 FDR。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence, cast

from .contracts import PathKind, PerturbationMode

import numpy as np
import pandas as pd

from src.integration.perturbgen.contracts import PathResult
from src.integration.perturbgen.env_guard import sha256_file


@dataclass(frozen=True)
class HeldOutSignature:
    """基于训练 donor 构建、对 held-out donor 评分的疾病 signature。"""

    held_out_donor: str
    training_donors: tuple[str, ...]
    up_genes: tuple[str, ...]
    down_genes: tuple[str, ...]
    excluded_target_gene: str | None = None


@dataclass(frozen=True)
class DonorRescueScore:
    donor: str
    baseline_score: float | None
    perturbed_score: float | None
    rescue_score: float | None
    evaluable: bool
    reason_code: str | None = None


@dataclass(frozen=True)
class RescueSummary:
    evaluable_donors: int
    donor_consistency: float | None
    median_rescue: float | None
    bootstrap_ci: tuple[float, float] | None
    donor_scores: tuple[DonorRescueScore, ...]


@dataclass(frozen=True)
class NullCalibrationResult:
    observed_effect: float
    empirical_pvalue: float
    matched_null_count: int
    smoke_only: bool
    formal_test: bool


@dataclass(frozen=True)
class PerturbGenH5ADSummary:
    """Schema evidence for one PerturbGen perturbation output."""

    n_obs: int
    n_vars: int
    layers: tuple[str, ...]
    obs_columns: tuple[str, ...]
    var_columns: tuple[str, ...]


@dataclass(frozen=True)
class UnperturbedQualityResult:
    """Release gate for the unperturbed model on held-out data."""

    status: str
    median_deg_direction_recovery: float | None
    worst_deg_direction_recovery: float | None
    minimum_signature_correlation: float | None
    seeds: int
    reasons: tuple[str, ...]
    source: str = "evaluate_unperturbed_quality"
    h5ad_paths: tuple[str, ...] = ()
    seed_metrics: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ExtractedPathResult:
    """从真实 PerturbGen h5ad、DEG 和 null 明确提取的路径结果。"""

    path_result: PathResult
    h5ad_summary: PerturbGenH5ADSummary
    h5ad_provenance: Mapping[str, Any]
    baseline_matrix: str
    perturbed_matrix: str
    bootstrap_ci: tuple[float, float] | None
    donor_scores: tuple[DonorRescueScore, ...]


def evaluate_unperturbed_quality(
    deg_direction_recovery: Sequence[float],
    signature_correlation: Sequence[float],
    *,
    required_seeds: int = 3,
    median_recovery_threshold: float = 0.60,
    worst_recovery_threshold: float = 0.50,
) -> UnperturbedQualityResult:
    """Evaluate the §5.4 unperturbed prediction quality gate."""

    recovery = np.asarray(list(deg_direction_recovery), dtype=float)
    correlations = np.asarray(list(signature_correlation), dtype=float)
    if recovery.size != correlations.size:
        raise ValueError("quality metric arrays must have the same number of seeds")
    if recovery.size < required_seeds:
        return UnperturbedQualityResult(
            status="inconclusive",
            median_deg_direction_recovery=None,
            worst_deg_direction_recovery=None,
            minimum_signature_correlation=None,
            seeds=int(recovery.size),
            reasons=("insufficient_quality_seeds",),
        )
    if not np.isfinite(recovery).all() or not np.isfinite(correlations).all():
        raise ValueError("quality metrics must be finite")
    median_recovery = float(np.median(recovery))
    worst_recovery = float(np.min(recovery))
    minimum_correlation = float(np.min(correlations))
    reasons: list[str] = []
    if median_recovery < median_recovery_threshold:
        reasons.append("median_deg_direction_recovery_below_threshold")
    if worst_recovery < worst_recovery_threshold:
        reasons.append("worst_deg_direction_recovery_below_threshold")
    if minimum_correlation <= 0:
        reasons.append("signature_correlation_not_positive")
    return UnperturbedQualityResult(
        status="fail" if reasons else "pass",
        median_deg_direction_recovery=median_recovery,
        worst_deg_direction_recovery=worst_recovery,
        minimum_signature_correlation=minimum_correlation,
        seeds=int(recovery.size),
        reasons=tuple(reasons),
        source="evaluate_unperturbed_quality",
    )


def extract_unperturbed_quality_from_h5ad(
    h5ad_by_seed: Mapping[int, str | Path],
    deg_table: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    donor_obs_column: str,
    var_gene_column: str,
    target_gene: str | None = None,
    donor_column: str = "donor",
    gene_column: str = "gene",
    effect_column: str = "log2fc",
    fdr_column: str = "fdr",
    fdr_threshold: float = 0.05,
    min_training_donors: int = 2,
    top_k: int = 50,
    required_seeds: int = 3,
    median_recovery_threshold: float = 0.60,
    worst_recovery_threshold: float = 0.50,
) -> UnperturbedQualityResult:
    """Compute §5.4 unperturbed quality from pred_counts vs true_counts.

    DEG direction recovery (per seed)
        Among significant DEGs excluding the target, the fraction whose
        predicted mean (pred_counts) sits on the same side of the transcriptome
        median as the observed median log2fc sign.

    Signature correlation (per seed)
        Pearson correlation of held-out signature scores S(pred) vs S(true)
        across donors.  The signature itself is built from training donors
        exactly as in :func:`build_held_out_signature`.
    """

    if isinstance(h5ad_by_seed, (str, bytes)) or not isinstance(h5ad_by_seed, Mapping) or not h5ad_by_seed:
        raise ValueError("h5ad_by_seed must map seed integers to h5ad paths")
    seeds = sorted(h5ad_by_seed)
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("h5ad_by_seed keys must be non-negative integers")

    frame = _coerce_dataframe(deg_table)
    _require_columns(frame, donor_column, gene_column, effect_column, fdr_column)
    observed_signs = _significant_gene_signs(
        frame,
        gene_column=gene_column,
        effect_column=effect_column,
        fdr_column=fdr_column,
        fdr_threshold=fdr_threshold,
        target_gene=target_gene,
    )
    recoveries: list[float] = []
    correlations: list[float] = []
    seed_metrics: list[dict[str, Any]] = []
    resolved_paths: list[str] = []
    for seed in seeds:
        output_path = Path(h5ad_by_seed[seed]).expanduser().resolve(strict=True)
        resolved_paths.append(str(output_path))
        validate_perturbgen_h5ad_schema(
            output_path,
            required_layers=("true_counts", "pred_counts"),
            required_obs=(donor_obs_column,),
            required_var=() if var_gene_column == "__index__" else (var_gene_column,),
        )
        pred_by_donor, true_by_donor = _aggregate_unperturbed_expression_from_h5ad(
            output_h5ad=output_path,
            donor_obs_column=donor_obs_column,
            var_gene_column=var_gene_column,
        )
        recovery = _deg_direction_recovery(pred_by_donor, observed_signs)
        correlation = _signature_score_correlation(
            pred_by_donor,
            true_by_donor,
            frame,
            target_gene=target_gene,
            donor_column=donor_column,
            gene_column=gene_column,
            effect_column=effect_column,
            fdr_column=fdr_column,
            fdr_threshold=fdr_threshold,
            min_training_donors=min_training_donors,
            top_k=top_k,
        )
        recoveries.append(recovery)
        correlations.append(correlation)
        seed_metrics.append(
            {
                "seed": seed,
                "deg_direction_recovery": recovery,
                "signature_correlation": correlation,
                "h5ad": str(output_path),
            }
        )

    result = evaluate_unperturbed_quality(
        recoveries,
        correlations,
        required_seeds=required_seeds,
        median_recovery_threshold=median_recovery_threshold,
        worst_recovery_threshold=worst_recovery_threshold,
    )
    return UnperturbedQualityResult(
        status=result.status,
        median_deg_direction_recovery=result.median_deg_direction_recovery,
        worst_deg_direction_recovery=result.worst_deg_direction_recovery,
        minimum_signature_correlation=result.minimum_signature_correlation,
        seeds=result.seeds,
        reasons=result.reasons,
        source="extract_unperturbed_quality_from_h5ad",
        h5ad_paths=tuple(resolved_paths),
        seed_metrics=tuple(seed_metrics),
    )


def validate_perturbgen_h5ad_schema(
    path: str | Path,
    *,
    required_layers: Sequence[str] = ("true_counts", "pred_counts"),
    required_obs: Sequence[str] = (),
    required_var: Sequence[str] = (),
    required_obsm: Sequence[str] = (
        "true_cls",
        "perturbed_cls",
        "mean_cos_similarity",
    ),
    required_varm: Sequence[str] = ("gene_cos_similarity",),
) -> PerturbGenH5ADSummary:
    """Validate the documented PerturbGen result contract without loading X.

    PerturbGen output uses ``X`` for perturbed counts and keeps control/model
    counts in layers.  Missing fields or shape drift are hard errors; callers
    must not infer a replacement from another matrix.
    """

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to validate PerturbGen h5ad output") from exc

    output_path = Path(path).expanduser().resolve(strict=True)
    adata = ad.read_h5ad(output_path, backed="r")
    try:
        if adata.n_obs <= 0 or adata.n_vars <= 0:
            raise ValueError("h5ad output must contain at least one cell and one gene")
        if adata.X is None or adata.X.shape != adata.shape:
            raise ValueError("h5ad X must contain perturbed counts with the AnnData shape")
        missing_layers = [name for name in required_layers if name not in adata.layers]
        if missing_layers:
            raise KeyError(f"missing required layers: {missing_layers}")
        for name in required_layers:
            if adata.layers[name].shape != adata.shape:
                raise ValueError(f"layer {name!r} shape does not match AnnData")
        missing_obs = [name for name in required_obs if name not in adata.obs.columns]
        if missing_obs:
            raise KeyError(f"missing required obs columns: {missing_obs}")
        missing_var = [name for name in required_var if name not in adata.var.columns]
        if missing_var:
            raise KeyError(f"missing required var columns: {missing_var}")
        missing_obsm = [name for name in required_obsm if name not in adata.obsm]
        if missing_obsm:
            raise KeyError(f"missing required obsm fields: {missing_obsm}")
        for name in required_obsm:
            if adata.obsm[name].shape[0] != adata.n_obs:
                raise ValueError(f"obsm {name!r} first dimension must equal n_obs")
        missing_varm = [name for name in required_varm if name not in adata.varm]
        if missing_varm:
            raise KeyError(f"missing required varm fields: {missing_varm}")
        for name in required_varm:
            if adata.varm[name].shape[0] != adata.n_vars:
                raise ValueError(f"varm {name!r} first dimension must equal n_vars")
        return PerturbGenH5ADSummary(
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars),
            layers=tuple(sorted(str(name) for name in adata.layers.keys())),
            obs_columns=tuple(str(name) for name in adata.obs.columns),
            var_columns=tuple(str(name) for name in adata.var.columns),
        )
    finally:
        if adata.file is not None:
            adata.file.close()


def build_held_out_signature(
    deg_table: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    held_out_donor: str,
    target_gene: str | None = None,
    donor_column: str = "donor",
    gene_column: str = "gene",
    effect_column: str = "log2fc",
    fdr_column: str = "fdr",
    fdr_threshold: float = 0.05,
    min_training_donors: int = 2,
    top_k: int = 50,
) -> HeldOutSignature:
    """对每个 held-out donor 用剩余 donor 构建稳定方向 signature。"""

    frame = _coerce_dataframe(deg_table)
    _require_columns(frame, donor_column, gene_column, effect_column, fdr_column)

    train = frame.loc[frame[donor_column] != held_out_donor].copy()
    if target_gene is not None:
        train = train.loc[train[gene_column] != target_gene].copy()

    if train.empty:
        raise ValueError("no training donors remain after held-out exclusion")

    significant = train.loc[train[fdr_column] <= fdr_threshold].copy()
    if significant.empty:
        raise ValueError("no significant genes available for held-out signature")

    grouped_rows: list[tuple[str, float]] = []
    grouped_rows_down: list[tuple[str, float]] = []
    for gene_name, gene_frame in significant.groupby(gene_column, sort=False):
        donor_count = int(gene_frame[donor_column].nunique())
        if donor_count < min_training_donors:
            continue

        effects = gene_frame[effect_column].astype(float).to_numpy()
        if np.all(effects > 0):
            grouped_rows.append((str(gene_name), float(np.median(effects))))
        elif np.all(effects < 0):
            grouped_rows_down.append((str(gene_name), float(np.median(effects))))

    grouped_rows.sort(key=lambda item: item[1], reverse=True)
    grouped_rows_down.sort(key=lambda item: item[1])

    up_genes = tuple(gene for gene, _ in grouped_rows[:top_k])
    down_genes = tuple(gene for gene, _ in grouped_rows_down[:top_k])
    if not up_genes or not down_genes:
        raise ValueError("held-out signature requires both up and down genes")

    return HeldOutSignature(
        held_out_donor=str(held_out_donor),
        training_donors=tuple(sorted(str(value) for value in train[donor_column].unique())),
        up_genes=up_genes,
        down_genes=down_genes,
        excluded_target_gene=target_gene,
    )


def compute_signature_score(
    expression: Mapping[str, float] | pd.Series,
    signature: HeldOutSignature,
) -> float:
    """S(X)=mean(D_up)-mean(D_down)。"""

    values: Mapping[Any, Any] = dict(expression.items()) if isinstance(expression, pd.Series) else dict(expression)
    up_values = [_coerce_float(values, gene) for gene in signature.up_genes]
    down_values = [_coerce_float(values, gene) for gene in signature.down_genes]
    return float(np.mean(up_values) - np.mean(down_values))


def compute_rescue_score(
    baseline_expression: Mapping[str, float] | pd.Series,
    perturbed_expression: Mapping[str, float] | pd.Series,
    signature: HeldOutSignature,
) -> float:
    """R = S(unperturbed) - S(perturbed)。"""

    baseline_score = compute_signature_score(baseline_expression, signature)
    perturbed_score = compute_signature_score(perturbed_expression, signature)
    return float(baseline_score - perturbed_score)


def summarize_rescue_by_donor(
    baseline_by_donor: Mapping[str, Mapping[str, float] | pd.Series],
    perturbed_by_donor: Mapping[str, Mapping[str, float] | pd.Series],
    deg_table: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    target_gene: str | None = None,
    donor_column: str = "donor",
    gene_column: str = "gene",
    effect_column: str = "log2fc",
    fdr_column: str = "fdr",
    fdr_threshold: float = 0.05,
    min_training_donors: int = 2,
    top_k: int = 50,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int | None = 0,
) -> RescueSummary:
    """对每个 donor 独立 held-out 打分，再汇总 donor 一致性。"""

    donor_scores: list[DonorRescueScore] = []
    common_donors = sorted(set(baseline_by_donor) & set(perturbed_by_donor))
    for donor in common_donors:
        try:
            signature = build_held_out_signature(
                deg_table,
                held_out_donor=donor,
                target_gene=target_gene,
                donor_column=donor_column,
                gene_column=gene_column,
                effect_column=effect_column,
                fdr_column=fdr_column,
                fdr_threshold=fdr_threshold,
                min_training_donors=min_training_donors,
                top_k=top_k,
            )
            baseline_score = compute_signature_score(baseline_by_donor[donor], signature)
            perturbed_score = compute_signature_score(perturbed_by_donor[donor], signature)
            rescue = float(baseline_score - perturbed_score)
            donor_scores.append(
                DonorRescueScore(
                    donor=str(donor),
                    baseline_score=baseline_score,
                    perturbed_score=perturbed_score,
                    rescue_score=rescue,
                    evaluable=True,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            donor_scores.append(
                DonorRescueScore(
                    donor=str(donor),
                    baseline_score=None,
                    perturbed_score=None,
                    rescue_score=None,
                    evaluable=False,
                    reason_code=str(exc),
                )
            )

    rescue_values = [score.rescue_score for score in donor_scores if score.evaluable and score.rescue_score is not None]
    evaluable_donors = len(rescue_values)
    if evaluable_donors == 0:
        return RescueSummary(
            evaluable_donors=0,
            donor_consistency=None,
            median_rescue=None,
            bootstrap_ci=None,
            donor_scores=tuple(donor_scores),
        )

    donor_consistency = float(sum(value > 0 for value in rescue_values) / evaluable_donors)
    return RescueSummary(
        evaluable_donors=evaluable_donors,
        donor_consistency=donor_consistency,
        median_rescue=float(median(rescue_values)),
        bootstrap_ci=bootstrap_confidence_interval(
            rescue_values,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        donor_scores=tuple(donor_scores),
    )


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int | None = 0,
) -> tuple[float, float] | None:
    """bootstrap 95% CI，统计量固定用 median。"""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    if array.size == 1:
        value = float(array[0])
        return (value, value)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    rng = np.random.default_rng(seed)
    resampled = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sample = rng.choice(array, size=array.size, replace=True)
        resampled[idx] = float(np.median(sample))
    lower, upper = np.percentile(resampled, [2.5, 97.5])
    return (float(lower), float(upper))


def evaluate_null_calibration(
    observed_effect: float,
    null_distribution: Sequence[float],
    *,
    formal_min_count: int = 99,
    smoke_count: int = 20,
    alternative: str = "greater",
) -> NullCalibrationResult:
    """加一经验 p 值；20 个 null 只算 smoke，不算正式门。"""

    if not np.isfinite(float(observed_effect)):
        raise ValueError("observed_effect must be finite")
    values = np.asarray(list(null_distribution), dtype=float)
    if values.size == 0:
        raise ValueError("null_distribution must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("null_distribution must contain only finite values")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")

    if alternative == "greater":
        extreme = int(np.sum(values >= observed_effect))
    else:
        extreme = int(np.sum(values <= observed_effect))

    empirical = float((extreme + 1) / (values.size + 1))
    return NullCalibrationResult(
        observed_effect=float(observed_effect),
        empirical_pvalue=empirical,
        matched_null_count=int(values.size),
        smoke_only=int(values.size) == smoke_count and int(values.size) < formal_min_count,
        formal_test=int(values.size) >= formal_min_count,
    )


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """标准 BH-FDR，保持原顺序返回 q 值。"""

    array = np.asarray(list(pvalues), dtype=float)
    if array.size == 0:
        return []
    if not np.isfinite(array).all():
        raise ValueError("pvalues must be finite")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("pvalues must be within [0, 1]")

    order = np.argsort(array)
    ranked = array[order]
    n = ranked.size
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        raw = ranked[idx] * n / (idx + 1)
        running = min(running, raw)
        adjusted[idx] = running
    restored = np.empty(n, dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(value) for value in restored.tolist()]


def extract_path_result_from_perturbgen_h5ad(
    *,
    output_h5ad: str | Path,
    h5ad_provenance: Mapping[str, Any],
    deg_table: pd.DataFrame | Sequence[Mapping[str, Any]],
    null_distribution: Sequence[float],
    path: PathKind,
    mode: PerturbationMode,
    seed: int,
    donor_obs_column: str,
    var_gene_column: str,
    target_gene: str | None = None,
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
) -> ExtractedPathResult:
    """严格按 ``pred_counts`` 基线和 ``X`` 扰动矩阵提取 ``PathResult``。

    - baseline 固定使用 ``adata.layers["pred_counts"]``
    - perturbed 固定使用 ``adata.X``
    - donor 聚合固定使用显式 ``obs[donor_obs_column]``
    - gene 解析固定使用显式 ``var[var_gene_column]``
    - 不允许猜测列名、矩阵或 provenance
    """

    if min_evaluable_donors < 3:
        raise ValueError("min_evaluable_donors must be >= 3")

    output_path = Path(output_h5ad).expanduser().resolve(strict=True)
    provenance = _validate_h5ad_provenance(
        h5ad_provenance,
        output_h5ad=output_path,
        expected_stage=path,
    )
    summary = validate_perturbgen_h5ad_schema(
        output_path,
        required_layers=("true_counts", "pred_counts"),
        required_obs=(donor_obs_column,),
        required_var=() if var_gene_column == "__index__" else (var_gene_column,),
    )
    baseline_by_donor, perturbed_by_donor = _aggregate_donor_expression_from_h5ad(
        output_h5ad=output_path,
        donor_obs_column=donor_obs_column,
        var_gene_column=var_gene_column,
    )
    rescue = summarize_rescue_by_donor(
        baseline_by_donor,
        perturbed_by_donor,
        deg_table,
        target_gene=target_gene,
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
        result = PathResult(
            status="inconclusive",
            path=path,
            mode=mode,
            rescue_excl_target=None,
            evaluable_donors=rescue.evaluable_donors,
            donor_consistency=None,
            seed=int(seed),
            output_h5ad=str(output_path),
            reason_code="insufficient_evaluable_donors",
        )
        return ExtractedPathResult(
            path_result=result,
            h5ad_summary=summary,
            h5ad_provenance=provenance,
            baseline_matrix="pred_counts",
            perturbed_matrix="X",
            bootstrap_ci=rescue.bootstrap_ci,
            donor_scores=rescue.donor_scores,
        )

    if rescue.median_rescue is None or rescue.donor_consistency is None:
        raise ValueError("rescue summary is missing median_rescue or donor_consistency")

    calibration = evaluate_null_calibration(
        rescue.median_rescue,
        null_distribution,
        alternative="greater",
    )
    result = PathResult(
        status="evaluable",
        path=path,
        mode=mode,
        rescue_excl_target=rescue.median_rescue,
        evaluable_donors=rescue.evaluable_donors,
        donor_consistency=rescue.donor_consistency,
        seed=int(seed),
        output_h5ad=str(output_path),
        matched_null_count=calibration.matched_null_count,
        empirical_pvalue=calibration.empirical_pvalue,
    )
    return ExtractedPathResult(
        path_result=result,
        h5ad_summary=summary,
        h5ad_provenance=provenance,
        baseline_matrix="pred_counts",
        perturbed_matrix="X",
        bootstrap_ci=rescue.bootstrap_ci,
        donor_scores=rescue.donor_scores,
    )


def _coerce_dataframe(deg_table: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(deg_table, pd.DataFrame):
        return cast(pd.DataFrame, deg_table.copy())
    return pd.DataFrame(list(deg_table))


def _require_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")


def _coerce_float(values: Mapping[str, Any], gene: str) -> float:
    if gene not in values:
        raise KeyError(f"missing gene '{gene}' in expression values")
    return float(values[gene])


def _validate_h5ad_provenance(
    h5ad_provenance: Mapping[str, Any],
    *,
    output_h5ad: Path,
    expected_stage: str,
) -> dict[str, Any]:
    if not isinstance(h5ad_provenance, Mapping) or not h5ad_provenance:
        raise ValueError("h5ad_provenance must be a non-empty mapping")
    required_keys = ("stage_manifest", "sha256")
    missing = [key for key in required_keys if key not in h5ad_provenance]
    if missing:
        raise ValueError(f"h5ad_provenance missing required keys: {missing}")
    normalized: dict[str, Any] = {}
    for key, value in h5ad_provenance.items():
        text_key = str(key).strip()
        if not text_key:
            raise ValueError("h5ad_provenance keys must not be empty")
        if value is None:
            raise ValueError(f"h5ad_provenance[{text_key!r}] must not be None")
        if isinstance(value, str):
            text_value = value.strip()
            if not text_value:
                raise ValueError(f"h5ad_provenance[{text_key!r}] must not be empty")
            normalized[text_key] = text_value
        else:
            normalized[text_key] = value
    stage_manifest_path = Path(str(normalized["stage_manifest"])).expanduser().resolve(strict=True)
    manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise ValueError("stage_manifest status must be 'success'")
    if str(manifest.get("stage", "")).strip() != expected_stage:
        raise ValueError(f"stage_manifest stage {manifest.get('stage')!r} does not match run path {expected_stage!r}")
    artifacts = manifest.get("artifacts")
    outputs = manifest.get("outputs")
    if not isinstance(artifacts, Mapping):
        raise ValueError("stage_manifest artifacts must be a mapping")
    if not isinstance(outputs, Mapping):
        raise ValueError("stage_manifest outputs must be a mapping")
    if "result_h5ad" not in artifacts:
        raise ValueError("stage_manifest artifacts must contain result_h5ad")

    artifact_h5ad = _resolve_manifest_path(
        _require_non_empty_manifest_string(artifacts["result_h5ad"], "artifacts.result_h5ad"),
        stage_manifest_path.parent,
    )
    if artifact_h5ad != output_h5ad:
        raise ValueError("stage_manifest artifacts.result_h5ad does not match output_h5ad")

    matched_records = [
        record
        for key, record in outputs.items()
        if _resolve_manifest_path(str(key), stage_manifest_path.parent) == output_h5ad
    ]
    if len(matched_records) != 1:
        raise ValueError("stage_manifest outputs must contain exactly one record for output_h5ad")
    output_record = matched_records[0]
    if not isinstance(output_record, Mapping):
        raise ValueError("stage_manifest output record must be a mapping")
    manifest_sha256 = _require_non_empty_manifest_string(output_record.get("sha256"), "outputs.sha256")
    actual_sha256 = sha256_file(output_h5ad)
    if normalized["sha256"] != manifest_sha256 or normalized["sha256"] != actual_sha256:
        raise ValueError("h5ad_provenance sha256 does not match stage manifest or actual h5ad")
    normalized["stage_manifest"] = str(stage_manifest_path)
    normalized["sha256"] = manifest_sha256
    return normalized


def _aggregate_donor_expression_from_h5ad(
    *,
    output_h5ad: str | Path,
    donor_obs_column: str,
    var_gene_column: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to extract PerturbGen path results") from exc

    output_path = Path(output_h5ad).expanduser().resolve(strict=True)
    adata = ad.read_h5ad(output_path, backed="r")
    try:
        if donor_obs_column not in adata.obs.columns:
            raise KeyError(f"missing required obs column: {donor_obs_column}")
        if var_gene_column == "__index__":
            gene_index = [str(item).strip() for item in adata.var_names.tolist()]
        else:
            if var_gene_column not in adata.var.columns:
                raise KeyError(f"missing required var column: {var_gene_column}")
            gene_index = [str(item).strip() for item in adata.var[var_gene_column].tolist()]
        if any(not item for item in gene_index):
            raise ValueError(f"gene identifiers from {var_gene_column!r} must not contain empty values")
        duplicates = pd.Index(gene_index).duplicated(keep=False)
        if bool(np.any(duplicates)):
            repeated = sorted({gene_index[idx] for idx, flag in enumerate(duplicates) if flag})
            raise ValueError(f"gene identifiers from {var_gene_column!r} must be unique, got duplicates: {repeated}")

        raw_donor_series = adata.obs[donor_obs_column]
        if raw_donor_series.isna().any():
            raise ValueError(f"obs column {donor_obs_column!r} must not contain missing donors")
        donor_series = raw_donor_series.astype(str).str.strip()
        if donor_series.eq("").any():
            raise ValueError(f"obs column {donor_obs_column!r} must not contain empty donors")

        if adata.layers["pred_counts"].shape != adata.X.shape:
            raise ValueError("pred_counts and X must have the same shape")

        baseline_by_donor: dict[str, dict[str, float]] = {}
        perturbed_by_donor: dict[str, dict[str, float]] = {}
        for donor in sorted(donor_series.unique()):
            mask = donor_series.to_numpy() == donor
            donor_view = adata[mask]
            baseline_vector = _mean_vector(donor_view.layers["pred_counts"], matrix_name="pred_counts")
            perturbed_vector = _mean_vector(donor_view.X, matrix_name="X")
            baseline_by_donor[str(donor)] = {
                gene: float(value) for gene, value in zip(gene_index, baseline_vector, strict=True)
            }
            perturbed_by_donor[str(donor)] = {
                gene: float(value) for gene, value in zip(gene_index, perturbed_vector, strict=True)
            }
        return baseline_by_donor, perturbed_by_donor
    finally:
        if adata.file is not None:
            adata.file.close()


def _resolve_manifest_path(path_text: str, manifest_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"stage_manifest path does not exist: {path}") from exc


def _require_non_empty_manifest_string(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _mean_vector(matrix: Any, *, matrix_name: str) -> np.ndarray:
    mean_vector = matrix.mean(axis=0)
    if hasattr(mean_vector, "A1"):
        array = np.asarray(mean_vector.A1, dtype=float)
    elif hasattr(mean_vector, "toarray"):
        array = np.asarray(mean_vector.toarray(), dtype=float).reshape(-1)
    else:
        array = np.asarray(mean_vector, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{matrix_name} mean vector must be finite")
    return array


def _significant_gene_signs(
    frame: pd.DataFrame,
    *,
    gene_column: str,
    effect_column: str,
    fdr_column: str,
    fdr_threshold: float,
    target_gene: str | None,
) -> dict[str, float]:
    significant = frame.loc[frame[fdr_column] <= fdr_threshold].copy()
    if target_gene is not None:
        significant = significant.loc[significant[gene_column] != target_gene]
    if significant.empty:
        raise ValueError("no significant DEG genes remain after target exclusion")
    signs: dict[str, float] = {}
    for gene_name, gene_frame in significant.groupby(gene_column, sort=False):
        median_effect = float(np.median(gene_frame[effect_column].astype(float).to_numpy()))
        if median_effect == 0 or not np.isfinite(median_effect):
            continue
        signs[str(gene_name)] = float(np.sign(median_effect))
    if not signs:
        raise ValueError("significant DEG genes have no non-zero median log2fc")
    return signs


def _donor_gene_means(by_donor: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    genes: set[str] = set()
    for values in by_donor.values():
        genes.update(values)
    means: dict[str, float] = {}
    for gene in genes:
        collected = [float(values[gene]) for values in by_donor.values() if gene in values]
        if collected:
            means[gene] = float(np.mean(collected))
    return means


def _deg_direction_recovery(
    pred_by_donor: Mapping[str, Mapping[str, float]],
    observed_signs: Mapping[str, float],
) -> float:
    pred_means = _donor_gene_means(pred_by_donor)
    if not pred_means:
        raise ValueError("pred_counts donor means are empty")
    transcriptome_median = float(np.median(list(pred_means.values())))
    recovered = 0
    total = 0
    for gene, observed_sign in observed_signs.items():
        if gene not in pred_means:
            raise KeyError(f"DEG gene {gene!r} is missing from pred_counts")
        predicted_sign = float(np.sign(pred_means[gene] - transcriptome_median))
        if predicted_sign == 0:
            continue
        total += 1
        if predicted_sign == observed_sign:
            recovered += 1
    if total == 0:
        raise ValueError("no DEG genes had a non-zero predicted offset from the transcriptome median")
    return float(recovered / total)


def _signature_score_correlation(
    pred_by_donor: Mapping[str, Mapping[str, float]],
    true_by_donor: Mapping[str, Mapping[str, float]],
    deg_table: pd.DataFrame,
    *,
    target_gene: str | None,
    donor_column: str,
    gene_column: str,
    effect_column: str,
    fdr_column: str,
    fdr_threshold: float,
    min_training_donors: int,
    top_k: int,
) -> float:
    common_donors = sorted(set(pred_by_donor) & set(true_by_donor))
    pred_scores: list[float] = []
    true_scores: list[float] = []
    for donor in common_donors:
        try:
            signature = build_held_out_signature(
                deg_table,
                held_out_donor=donor,
                target_gene=target_gene,
                donor_column=donor_column,
                gene_column=gene_column,
                effect_column=effect_column,
                fdr_column=fdr_column,
                fdr_threshold=fdr_threshold,
                min_training_donors=min_training_donors,
                top_k=top_k,
            )
        except (KeyError, ValueError, TypeError):
            continue
        pred_scores.append(compute_signature_score(pred_by_donor[donor], signature))
        true_scores.append(compute_signature_score(true_by_donor[donor], signature))
    if len(pred_scores) < 2:
        raise ValueError("signature correlation requires at least two held-out donors")
    pred_array = np.asarray(pred_scores, dtype=float)
    true_array = np.asarray(true_scores, dtype=float)
    if np.allclose(pred_array, pred_array[0]) or np.allclose(true_array, true_array[0]):
        return 0.0
    correlation = float(np.corrcoef(pred_array, true_array)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("signature correlation must be finite")
    return correlation


def _aggregate_unperturbed_expression_from_h5ad(
    *,
    output_h5ad: str | Path,
    donor_obs_column: str,
    var_gene_column: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to extract unperturbed quality") from exc

    output_path = Path(output_h5ad).expanduser().resolve(strict=True)
    adata = ad.read_h5ad(output_path, backed="r")
    try:
        if donor_obs_column not in adata.obs.columns:
            raise KeyError(f"missing required obs column: {donor_obs_column}")
        if var_gene_column == "__index__":
            gene_index = [str(item).strip() for item in adata.var_names.tolist()]
        else:
            if var_gene_column not in adata.var.columns:
                raise KeyError(f"missing required var column: {var_gene_column}")
            gene_index = [str(item).strip() for item in adata.var[var_gene_column].tolist()]
        raw_donor_series = adata.obs[donor_obs_column]
        if raw_donor_series.isna().any():
            raise ValueError(f"obs column {donor_obs_column!r} must not contain missing donors")
        donor_series = raw_donor_series.astype(str).str.strip()
        pred_by_donor: dict[str, dict[str, float]] = {}
        true_by_donor: dict[str, dict[str, float]] = {}
        for donor in sorted(donor_series.unique()):
            mask = donor_series.to_numpy() == donor
            donor_view = adata[mask]
            pred_vector = _mean_vector(donor_view.layers["pred_counts"], matrix_name="pred_counts")
            true_vector = _mean_vector(donor_view.layers["true_counts"], matrix_name="true_counts")
            pred_by_donor[str(donor)] = {
                gene: float(value) for gene, value in zip(gene_index, pred_vector, strict=True)
            }
            true_by_donor[str(donor)] = {
                gene: float(value) for gene, value in zip(gene_index, true_vector, strict=True)
            }
        return pred_by_donor, true_by_donor
    finally:
        if adata.file is not None:
            adata.file.close()
