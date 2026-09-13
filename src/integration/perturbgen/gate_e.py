"""Gate-E: gene-vocabulary migration and DAVF non-inferiority evaluation.

Implements the scientific quality gate of the integration proposal §5.4
(``docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md``) that had
no code before (analysis ``project_analysis_20260910.md`` §4.1 N-3):

* benchmark — at least 200 traceable PTM→gene samples on a frozen split;
* vocabulary migration — benchmark gene coverage >= 99%, token collision
  count 0, and PTM action codes unchanged for 100% of the benchmark;
* DAVF non-inferiority — the primary downstream metric (direction
  accuracy) may drop by at most 1 percentage point versus the frozen old
  baseline, and the bootstrap 95% CI must not show clear deterioration.

Gate-E compares semantic resolution coverage and downstream behaviour.  It
never compares whether old and new numeric token IDs are equal, and it does
not treat mapper rule agreement as embedding quality.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.models.gene_vocabulary import normalize_ensembl_id

GATE_E_BENCHMARK_SCHEMA = "ptm2cellnet.gate-e-benchmark/v1"
GATE_E_REPORT_SCHEMA = "ptm2cellnet.gate-e-report/v1"

MIN_BENCHMARK_SAMPLES = 200
MIN_GENE_COVERAGE = 0.99
MAX_TOKEN_COLLISIONS = 0
MIN_ACTION_CODE_AGREEMENT = 1.0
DEFAULT_NONINFERIORITY_MARGIN = 0.01

_BENCHMARK_REQUIRED_COLUMNS = ("gene_symbol", "ensembl_id", "ptm_type")
_PAIRED_REQUIRED_COLUMNS = (
    "ensembl_id",
    "expected_direction",
    "old_direction",
    "new_direction",
)
_VALID_DIRECTIONS = {"up", "down", "none"}
_VALID_INTERVENTIONS = {"KO", "KD", "OE"}


class GateEError(ValueError):
    """Raised when Gate-E inputs violate the evaluation contract."""


@dataclass(frozen=True)
class GateEBenchmark:
    """Frozen PTM→gene benchmark set with a verifiable file digest."""

    rows: tuple[dict[str, str], ...]
    source_path: str
    source_sha256: str

    def validate(self, *, min_samples: int = MIN_BENCHMARK_SAMPLES) -> None:
        if len(self.rows) < min_samples:
            raise GateEError(f"Gate-E benchmark requires at least {min_samples} samples, got {len(self.rows)}")
        for index, row in enumerate(self.rows):
            for column in _BENCHMARK_REQUIRED_COLUMNS:
                value = row.get(column, "").strip()
                if not value:
                    raise GateEError(f"benchmark row {index} has an empty {column}")
            try:
                normalize_ensembl_id(row["ensembl_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GateEError(
                    f"benchmark row {index} has an invalid ensembl_id: {row.get('ensembl_id')!r}"
                ) from exc


def load_benchmark(path: str | Path) -> GateEBenchmark:
    """Load the frozen benchmark CSV (gene_symbol, ensembl_id, ptm_type[, position])."""

    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in _BENCHMARK_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise GateEError(f"benchmark is missing required columns: {', '.join(missing)}")
        rows = [{key: str(value or "").strip() for key, value in row.items() if key} for row in reader]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return GateEBenchmark(
        rows=tuple(rows),
        source_path=str(resolved),
        source_sha256=digest.hexdigest(),
    )


def load_vocabulary(path: str | Path) -> dict[str, int]:
    """Load a JSON gene→token vocabulary (Geneformer vocab.json or asset token dict)."""

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateEError(f"vocabulary root must be a JSON object: {resolved}")
    vocabulary: dict[str, int] = {}
    for key, value in payload.items():
        token = int(value)
        if token < 0:
            raise GateEError(f"vocabulary token for {key!r} must be non-negative")
        vocabulary[str(key)] = token
    if not vocabulary:
        raise GateEError(f"vocabulary is empty: {resolved}")
    return vocabulary


@dataclass(frozen=True)
class VocabularyMigrationMetrics:
    """Word-list migration metrics defined by proposal §5.4."""

    benchmark_genes: int
    covered_genes: int
    gene_coverage: float
    token_collision_groups: int
    action_code_mismatches: int
    action_code_agreement: float

    @property
    def passed(self) -> bool:
        return (
            self.gene_coverage >= MIN_GENE_COVERAGE
            and self.token_collision_groups <= MAX_TOKEN_COLLISIONS
            and self.action_code_agreement >= MIN_ACTION_CODE_AGREEMENT
        )


def evaluate_vocabulary_migration(
    benchmark: GateEBenchmark,
    old_vocabulary: Mapping[str, int],
    new_vocabulary: Mapping[str, int],
    *,
    intervention_type: str = "KO",
) -> VocabularyMigrationMetrics:
    """Evaluate coverage, collisions and action-code invariance.

    Coverage counts a benchmark gene as covered when its canonical Ensembl ID
    resolves in the new vocabulary.  Action codes are
    compared end-to-end through :class:`PTMDirectionMapper` so that any
    future coupling between the vocabulary and the PTM action rules would
    be caught here.
    """

    if intervention_type not in _VALID_INTERVENTIONS:
        raise GateEError(f"intervention_type must be one of {sorted(_VALID_INTERVENTIONS)}")

    covered = 0
    for index, row in enumerate(benchmark.rows):
        try:
            canonical_ensembl_id = normalize_ensembl_id(row.get("ensembl_id", ""))
        except (TypeError, ValueError) as exc:
            raise GateEError(
                f"benchmark row {index} has an invalid ensembl_id: {row.get('ensembl_id')!r}"
            ) from exc
        if canonical_ensembl_id in new_vocabulary:
            covered += 1
    coverage = covered / len(benchmark.rows) if benchmark.rows else 0.0

    token_groups: dict[int, list[str]] = {}
    for gene, token in new_vocabulary.items():
        token_groups.setdefault(token, []).append(gene)
    collisions = sum(1 for genes in token_groups.values() if len(genes) > 1)

    from src.data.schemas import PTMSite
    from src.models.ptm_direction_mapper import PTMDirectionMapper

    old_mapper = PTMDirectionMapper(gene_to_idx=dict(old_vocabulary))
    new_mapper = PTMDirectionMapper(gene_to_idx=dict(new_vocabulary))
    mismatches = 0
    for row in benchmark.rows:
        site = [PTMSite(position=int(row.get("position", "1") or 1), type=row["ptm_type"])]
        old_output = old_mapper.map_ptms(site, [row["gene_symbol"]])
        new_output = new_mapper.map_ptms(site, [row["gene_symbol"]])
        if not torch_tensor_equal(old_output.directions, new_output.directions):
            mismatches += 1
    agreement = 1.0 - mismatches / len(benchmark.rows) if benchmark.rows else 0.0

    return VocabularyMigrationMetrics(
        benchmark_genes=len(benchmark.rows),
        covered_genes=covered,
        gene_coverage=coverage,
        token_collision_groups=collisions,
        action_code_mismatches=mismatches,
        action_code_agreement=agreement,
    )


def torch_tensor_equal(left: Any, right: Any) -> bool:
    """Compare two direction tensors without importing torch at module scope."""

    return bool(list(left.tolist()) == list(right.tolist()))


@dataclass(frozen=True)
class DavfNonInferiorityResult:
    """Paired bootstrap non-inferiority verdict for the primary metric."""

    old_accuracy: float
    new_accuracy: float
    delta: float
    ci_lower: float
    ci_upper: float
    margin: float
    noninferior: bool
    ci_clear: bool

    @property
    def passed(self) -> bool:
        return self.noninferior and self.ci_clear


def evaluate_davf_noninferiority(
    paired_results: Sequence[Mapping[str, str]],
    *,
    margin: float = DEFAULT_NONINFERIORITY_MARGIN,
    bootstrap_iterations: int = 10000,
    seed: int = 0,
) -> DavfNonInferiorityResult:
    """Bootstrap the paired direction-accuracy difference (new − old).

    ``paired_results`` rows carry ``ensembl_id``, ``expected_direction``,
    ``old_direction`` and ``new_direction``; a direction of ``none`` (or a
    mismatched call) simply scores 0 for that side.
    """

    if not 0.0 < margin < 1.0:
        raise GateEError("non-inferiority margin must be within (0, 1)")
    if bootstrap_iterations < 1:
        raise GateEError("bootstrap_iterations must be >= 1")
    if not paired_results:
        raise GateEError("paired DAVF results must not be empty")

    old_scores = np.zeros(len(paired_results), dtype=float)
    new_scores = np.zeros(len(paired_results), dtype=float)
    for index, row in enumerate(paired_results):
        for column in _PAIRED_REQUIRED_COLUMNS:
            if column not in row:
                raise GateEError(f"paired result row {index} is missing {column}")
        expected = str(row["expected_direction"]).strip().lower()
        old_call = str(row["old_direction"]).strip().lower()
        new_call = str(row["new_direction"]).strip().lower()
        for label, value in (
            ("expected_direction", expected),
            ("old_direction", old_call),
            ("new_direction", new_call),
        ):
            if value not in _VALID_DIRECTIONS:
                raise GateEError(
                    f"paired result row {index} {label} must be one of {sorted(_VALID_DIRECTIONS)}, got {value!r}"
                )
        old_scores[index] = 1.0 if old_call == expected else 0.0
        new_scores[index] = 1.0 if new_call == expected else 0.0

    old_accuracy = float(old_scores.mean())
    new_accuracy = float(new_scores.mean())
    delta = new_accuracy - old_accuracy

    rng = np.random.default_rng(seed)
    n = len(paired_results)
    boot = np.empty(bootstrap_iterations, dtype=float)
    for iteration in range(bootstrap_iterations):
        sample = rng.integers(0, n, size=n)
        boot[iteration] = float(new_scores[sample].mean() - old_scores[sample].mean())
    ci_lower, ci_upper = (float(value) for value in np.percentile(boot, [2.5, 97.5]))
    if not all(math.isfinite(value) for value in (ci_lower, ci_upper)):
        raise GateEError("bootstrap produced a non-finite confidence interval")

    return DavfNonInferiorityResult(
        old_accuracy=old_accuracy,
        new_accuracy=new_accuracy,
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        margin=margin,
        noninferior=delta >= -margin - 1e-9,
        ci_clear=ci_lower >= -margin - 1e-9,
    )


def load_paired_results(path: str | Path) -> list[dict[str, str]]:
    """Load the paired old/new DAVF prediction table."""

    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in _PAIRED_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise GateEError(f"paired results are missing columns: {', '.join(missing)}")
        return [{key: str(value or "").strip() for key, value in row.items() if key} for row in reader]


def build_gate_e_report(
    *,
    benchmark: GateEBenchmark,
    vocabulary_metrics: VocabularyMigrationMetrics | None,
    davf_metrics: DavfNonInferiorityResult | None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the Gate-E verdict with explicit thresholds and evidence."""

    limits = {
        "min_benchmark_samples": MIN_BENCHMARK_SAMPLES,
        "min_gene_coverage": MIN_GENE_COVERAGE,
        "max_token_collisions": MAX_TOKEN_COLLISIONS,
        "min_action_code_agreement": MIN_ACTION_CODE_AGREEMENT,
        "noninferiority_margin": DEFAULT_NONINFERIORITY_MARGIN,
    }
    limits.update({key: float(value) for key, value in (thresholds or {}).items()})

    sections: dict[str, Any] = {
        "schema_version": GATE_E_REPORT_SCHEMA,
        "benchmark": {
            "n_samples": len(benchmark.rows),
            "source": benchmark.source_path,
            "source_sha256": benchmark.source_sha256,
            "passed": len(benchmark.rows) >= int(limits["min_benchmark_samples"]),
        },
    }
    if vocabulary_metrics is not None:
        sections["vocabulary_migration"] = {
            "benchmark_genes": vocabulary_metrics.benchmark_genes,
            "covered_genes": vocabulary_metrics.covered_genes,
            "gene_coverage": vocabulary_metrics.gene_coverage,
            "token_collision_groups": vocabulary_metrics.token_collision_groups,
            "action_code_mismatches": vocabulary_metrics.action_code_mismatches,
            "action_code_agreement": vocabulary_metrics.action_code_agreement,
            "passed": vocabulary_metrics.passed,
        }
    if davf_metrics is not None:
        sections["davf_noninferiority"] = {
            "old_accuracy": davf_metrics.old_accuracy,
            "new_accuracy": davf_metrics.new_accuracy,
            "delta": davf_metrics.delta,
            "ci95": [davf_metrics.ci_lower, davf_metrics.ci_upper],
            "margin": davf_metrics.margin,
            "noninferior": davf_metrics.noninferior,
            "ci_clear": davf_metrics.ci_clear,
            "passed": davf_metrics.passed,
        }
    sections["thresholds"] = limits
    missing_sections = [
        name
        for name, present in (
            ("benchmark", True),
            ("vocabulary_migration", vocabulary_metrics is not None),
            ("davf_noninferiority", davf_metrics is not None),
        )
        if not present
    ]
    sections["missing_sections"] = missing_sections
    required = [sections["benchmark"]["passed"], vocabulary_metrics is not None, davf_metrics is not None]
    if vocabulary_metrics is not None:
        required.append(vocabulary_metrics.passed)
    if davf_metrics is not None:
        required.append(davf_metrics.passed)
    sections["gate_e_passed"] = all(required)
    return sections
