"""Gate-E evaluation tests (vocabulary migration + DAVF non-inferiority)."""

from __future__ import annotations

import json

import pytest

from src.integration.perturbgen.gate_e import (
    GateEError,
    build_gate_e_report,
    evaluate_davf_noninferiority,
    evaluate_vocabulary_migration,
    load_benchmark,
    load_paired_results,
    load_vocabulary,
)


def _benchmark_csv(tmp_path, rows):
    path = tmp_path / "benchmark.csv"
    header = "gene_symbol,ensembl_id,ptm_type,position\n"
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def _three_row_benchmark(tmp_path):
    return _benchmark_csv(
        tmp_path,
        [
            "STAT3,ENSG00000168610,phosphorylation,12\n",
            "BRAF,ENSG00000157757,phosphorylation,601\n",
            "TP53,ENSG00000141510,acetylation,382\n",
        ],
    )


class TestBenchmark:
    def test_load_records_digest_and_rows(self, tmp_path):
        benchmark = load_benchmark(_three_row_benchmark(tmp_path))
        assert len(benchmark.rows) == 3
        assert len(benchmark.source_sha256) == 64
        assert benchmark.rows[0]["gene_symbol"] == "STAT3"

    def test_min_samples_enforced(self, tmp_path):
        benchmark = load_benchmark(_three_row_benchmark(tmp_path))
        with pytest.raises(GateEError, match="at least 200"):
            benchmark.validate()
        benchmark.validate(min_samples=3)

    def test_missing_columns_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("gene_symbol,ptm_type\nX,phosphorylation\n", encoding="utf-8")
        with pytest.raises(GateEError, match="missing required columns"):
            load_benchmark(path)


class TestVocabularyMigration:
    def _run(self, tmp_path, old_vocab, new_vocab):
        benchmark = load_benchmark(_three_row_benchmark(tmp_path))
        return evaluate_vocabulary_migration(benchmark, old_vocab, new_vocab)

    def test_full_coverage_no_collision_passes(self, tmp_path):
        old_vocab = {"STAT3": 1, "BRAF": 2, "TP53": 3}
        new_vocab = {"STAT3": 10, "BRAF": 11, "TP53": 12}
        metrics = self._run(tmp_path, old_vocab, new_vocab)
        assert metrics.gene_coverage == 1.0
        assert metrics.token_collision_groups == 0
        assert metrics.action_code_agreement == 1.0
        assert metrics.passed

    def test_ensembl_keyed_vocabulary_counts_as_coverage(self, tmp_path):
        new_vocab = {"ENSG00000168610": 0, "ENSG00000157757": 1, "ENSG00000141510": 2}
        metrics = self._run(tmp_path, {"STAT3": 1}, new_vocab)
        assert metrics.gene_coverage == 1.0

    def test_partial_coverage_fails_threshold(self, tmp_path):
        new_vocab = {"STAT3": 0, "BRAF": 1}  # TP53 missing → 2/3 coverage
        metrics = self._run(tmp_path, {"STAT3": 1, "BRAF": 2, "TP53": 3}, new_vocab)
        assert metrics.covered_genes == 2
        assert metrics.gene_coverage == pytest.approx(2 / 3)
        assert not metrics.passed

    def test_token_collision_detected(self, tmp_path):
        new_vocab = {"STAT3": 7, "BRAF": 7, "TP53": 8}  # STAT3/BRAF share token 7
        metrics = self._run(tmp_path, {"STAT3": 1}, new_vocab)
        assert metrics.token_collision_groups == 1
        assert not metrics.passed

    def test_load_vocabulary_rejects_bad_payload(self, tmp_path):
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps({"G1": -3}), encoding="utf-8")
        with pytest.raises(GateEError, match="non-negative"):
            load_vocabulary(path)


class TestDavfNonInferiority:
    def _paired(self, old_calls, new_calls, expected=None):
        n = len(old_calls)
        expected = expected or ["up"] * n
        return [
            {
                "ensembl_id": f"ENSG{i:011d}",
                "expected_direction": expected[i],
                "old_direction": old_calls[i],
                "new_direction": new_calls[i],
            }
            for i in range(n)
        ]

    def test_identical_results_pass(self):
        rows = self._paired(["up"] * 50, ["up"] * 50)
        result = evaluate_davf_noninferiority(rows, bootstrap_iterations=200, seed=1)
        assert result.delta == 0.0
        assert result.noninferior
        assert result.ci_clear
        assert result.passed

    def test_small_drop_within_margin_passes(self):
        rows = self._paired(["up"] * 100, ["down"] * 1 + ["up"] * 99)
        result = evaluate_davf_noninferiority(rows, bootstrap_iterations=200, seed=1)
        assert result.delta == pytest.approx(-0.01)
        assert result.noninferior  # delta >= -0.01

    def test_large_drop_fails(self):
        rows = self._paired(["up"] * 100, ["down"] * 30 + ["up"] * 70)
        result = evaluate_davf_noninferiority(rows, bootstrap_iterations=200, seed=1)
        assert result.delta == pytest.approx(-0.30)
        assert not result.noninferior
        assert not result.passed

    def test_none_predictions_score_zero(self):
        rows = self._paired(["up", "up"], ["none", "up"])
        result = evaluate_davf_noninferiority(rows, bootstrap_iterations=50, seed=1)
        assert result.old_accuracy == 1.0
        assert result.new_accuracy == 0.5

    def test_invalid_direction_rejected(self):
        rows = self._paired(["sideways"], ["up"])
        with pytest.raises(GateEError, match="must be one of"):
            evaluate_davf_noninferiority(rows, bootstrap_iterations=10)

    def test_load_paired_results_validates_columns(self, tmp_path):
        path = tmp_path / "paired.csv"
        path.write_text("ensembl_id,expected_direction\nX,up\n", encoding="utf-8")
        with pytest.raises(GateEError, match="missing columns"):
            load_paired_results(path)


class TestReport:
    def test_report_aggregates_sections_and_thresholds(self, tmp_path):
        benchmark = load_benchmark(_three_row_benchmark(tmp_path))
        vocab = evaluate_vocabulary_migration(benchmark, {"STAT3": 1}, {"STAT3": 0, "BRAF": 1, "TP53": 2})
        davf = evaluate_davf_noninferiority(self_test_paired(), bootstrap_iterations=100, seed=2)
        report = build_gate_e_report(
            benchmark=benchmark,
            vocabulary_metrics=vocab,
            davf_metrics=davf,
        )
        assert report["schema_version"] == "ptm2cellnet.gate-e-report/v1"
        assert report["benchmark"]["n_samples"] == 3
        assert report["benchmark"]["passed"] is False  # 3 < default 200
        assert report["vocabulary_migration"]["passed"] is True
        assert report["davf_noninferiority"]["passed"] is True
        assert report["gate_e_passed"] is False
        assert report["thresholds"]["noninferiority_margin"] == 0.01

    def test_cli_writes_report(self, tmp_path):
        from scripts.evaluate_gate_e import main

        benchmark = _three_row_benchmark(tmp_path)
        old_vocab = tmp_path / "old.json"
        old_vocab.write_text(json.dumps({"STAT3": 1, "BRAF": 2, "TP53": 3}), encoding="utf-8")
        new_vocab = tmp_path / "new.json"
        new_vocab.write_text(json.dumps({"STAT3": 9, "BRAF": 8, "TP53": 7}), encoding="utf-8")
        output = tmp_path / "report.json"
        code = main(
            [
                "--benchmark",
                str(benchmark),
                "--old-vocabulary",
                str(old_vocab),
                "--new-vocabulary",
                str(new_vocab),
                "--min-samples",
                "3",
                "--bootstrap-iterations",
                "100",
                "--output",
                str(output),
            ]
        )
        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["gate_e_passed"] is True


def self_test_paired():
    return [
        {
            "ensembl_id": f"ENSG{i:011d}",
            "expected_direction": "down",
            "old_direction": "down",
            "new_direction": "down",
        }
        for i in range(40)
    ]
