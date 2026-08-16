"""Unit tests: TD-H02 pLM embedding precompute CLI (2026-08-16).

Covers: TSV parsing, deterministic mock backbone encoding, length
truncation, manifest provenance, fail-soft ledger, resume semantics and
CLI exit codes.  All tests run offline (mock backbone).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_cross_scale_embeddings import (
    BackboneSpec,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    EmbeddingPrecomputeError,
    _make_mock_backbone,
    build_parser,
    main,
    precompute,
    read_fasta_sequences,
    read_gene_request_records,
    read_sequence_records,
)

MOCK_BACKBONES = [_make_mock_backbone("mock"), _make_mock_backbone("mock2", dim=16)]


def _write_records(path: Path, records) -> None:
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

class TestReadSequenceRecords:
    def test_parses_tab_separated(self, tmp_path) -> None:
        f = tmp_path / "in.tsv"
        _write_records(f, ["s1\tACDEFG", "s2\tACDE", "# comment", "", "s3\tG"])
        assert read_sequence_records(f) == [("s1", "ACDEFG"), ("s2", "ACDE"), ("s3", "G")]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(EmbeddingPrecomputeError):
            read_sequence_records(tmp_path / "nope.tsv")

    def test_malformed_line_raises_with_lineno(self, tmp_path) -> None:
        f = tmp_path / "in.tsv"
        _write_records(f, ["s1\tACDE", "bad-line-without-tab"])
        with pytest.raises(EmbeddingPrecomputeError, match=":2"):
            read_sequence_records(f)

    def test_empty_sample_id_rejected(self, tmp_path) -> None:
        f = tmp_path / "in.tsv"
        _write_records(f, ["\tACDE"])
        with pytest.raises(EmbeddingPrecomputeError, match="sample_id"):
            read_sequence_records(f)

    def test_empty_file_rejected(self, tmp_path) -> None:
        f = tmp_path / "in.tsv"
        f.write_text("# only a comment\n\n", encoding="utf-8")
        with pytest.raises(EmbeddingPrecomputeError, match="无有效记录"):
            read_sequence_records(f)

    def test_duplicate_sample_id_rejected(self, tmp_path) -> None:
        f = tmp_path / "in.tsv"
        _write_records(f, ["s1\tACDE", "s1\tFGHI"])
        with pytest.raises(EmbeddingPrecomputeError, match="重复"):
            read_sequence_records(f)


class TestGeneRequests:
    def test_resolves_sequence_requests_against_fasta_aliases(self, tmp_path) -> None:
        request = tmp_path / "sequence_requests.json"
        request.write_text(json.dumps({"dataset": "demo", "genes": ["AHR", "MISSING"]}), encoding="utf-8")
        fasta = tmp_path / "sequences.fasta"
        fasta.write_text(">sp|P11111|AHR_HUMAN GN=AHR OS=Homo sapiens\nACDE\n", encoding="utf-8")
        assert read_gene_request_records(request, sequence_fasta=fasta) == [
            ("AHR", "ACDE"),
            ("MISSING", ""),
        ]
        assert read_fasta_sequences(fasta)["P11111"] == "ACDE"

    def test_direct_sequence_records_do_not_need_fasta(self, tmp_path) -> None:
        request = tmp_path / "requests.json"
        request.write_text(json.dumps({"records": [{"id": "s1", "sequence": "ACDE"}]}), encoding="utf-8")
        assert read_gene_request_records(request) == [("s1", "ACDE")]


# ---------------------------------------------------------------------------
# Encoding + persistence
# ---------------------------------------------------------------------------

class TestPrecompute:
    def test_batch_size_is_used_by_backbone(self, tmp_path) -> None:
        calls = []

        def encode(sequences):
            calls.append(len(sequences))
            max_len = max(len(sequence) for sequence in sequences)
            embeddings = np.zeros((len(sequences), max_len, 4), dtype=np.float32)
            masks = np.zeros((len(sequences), max_len), dtype=np.float32)
            for index, sequence in enumerate(sequences):
                masks[index, : len(sequence)] = 1.0
            return embeddings, masks

        spec = BackboneSpec("tracked", None, 4, "test", encode)
        result = precompute(
            [(f"s{index}", "ACDE") for index in range(5)],
            [spec],
            tmp_path,
            batch_size=2,
        )
        assert result.n_processed == 5
        assert calls == [2, 2, 1]

    def test_mock_encoding_shapes_and_finite(self, tmp_path) -> None:
        result = precompute(
            [("s1", "ACDEFG"), ("s2", "ACDE")],
            MOCK_BACKBONES,
            tmp_path,
        )
        assert result.n_processed == 2
        assert result.n_failed == 0
        assert result.sequence_length == 6
        assert result.backbone_dims == {"mock": 8, "mock2": 16}

        archive = np.load(tmp_path / "embeddings.npz", allow_pickle=False)
        assert archive["schema_version"].item() == EMBEDDING_CACHE_SCHEMA_VERSION
        assert list(archive["sample_id"]) == ["s1", "s2"]
        for key, dim in (("mock_embeddings", 8), ("mock2_embeddings", 16)):
            emb = archive[key]
            assert emb.shape == (2, 6, dim)
            assert np.isfinite(emb).all()
            mask = archive[f"{key.split('_')[0]}_attention_mask"]
            assert mask.shape == (2, 6)
            assert mask[0, :6].tolist() == [1.0] * 6
            assert mask[1, 4:].tolist() == [0.0, 0.0]

    def test_deterministic_encoding(self, tmp_path) -> None:
        records = [("s1", "ACDEFG")]
        result_a = precompute(records, MOCK_BACKBONES[:1], tmp_path / "a")
        result_b = precompute(records, MOCK_BACKBONES[:1], tmp_path / "b")
        a = np.load(tmp_path / "a" / "embeddings.npz", allow_pickle=False)["mock_embeddings"]
        b = np.load(tmp_path / "b" / "embeddings.npz", allow_pickle=False)["mock_embeddings"]
        assert np.array_equal(a, b)

    def test_max_length_truncation(self, tmp_path) -> None:
        result = precompute(
            [("s1", "ACDEFGHIK"), ("s2", "ACDE")],
            MOCK_BACKBONES[:1],
            tmp_path,
            max_length=4,
        )
        assert result.sequence_length == 4
        archive = np.load(tmp_path / "embeddings.npz", allow_pickle=False)
        assert archive["mock_embeddings"].shape == (2, 4, 8)

    def test_empty_sequence_goes_to_failure_ledger(self, tmp_path) -> None:
        result = precompute(
            [("s1", "ACDE"), ("s2", ""), ("s3", "G")],
            MOCK_BACKBONES[:1],
            tmp_path,
        )
        assert result.n_processed == 2
        assert result.n_failed == 1
        assert result.failures[0]["sample_id"] == "s2"
        failures = [
            json.loads(line)
            for line in (tmp_path / "failed_samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert failures[0]["error"] == "empty sequence"
        archive = np.load(tmp_path / "embeddings.npz", allow_pickle=False)
        assert list(archive["sample_id"]) == ["s1", "s3"]

    def test_manifest_provenance(self, tmp_path) -> None:
        records = [("s1", "ACDE")]
        precompute(records, MOCK_BACKBONES[:1], tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == EMBEDDING_CACHE_SCHEMA_VERSION
        assert manifest["input"]["n_samples"] == 1
        assert manifest["input"]["sha256"]
        assert manifest["backbones"]["mock"]["embedding_dim"] == 8
        assert manifest["backbones"]["mock"]["provenance"] == "mock"
        assert manifest["n_processed"] == 1
        assert manifest["n_failed"] == 0
        assert manifest["processed_sample_ids"] == ["s1"]
        assert manifest["outputs"]["embeddings_npz"].endswith("embeddings.npz")
        assert len(manifest["outputs"]["sha256"]) == 64

    def test_resume_skips_processed_samples(self, tmp_path) -> None:
        records = [("s1", "ACDE"), ("s2", "ACDEFG")]
        first = precompute(records, MOCK_BACKBONES[:1], tmp_path)
        assert first.n_processed == 2
        # 第二次运行只含 s2（输入 sha 变化 => 全量重算的语义先验证）
        second = precompute([("s1", "ACDE"), ("s2", "ACDEFG")], MOCK_BACKBONES[:1], tmp_path, resume=True)
        assert second.n_processed == 2
        # 输入 sha 相同 + resume => 全部跳过
        third = precompute([("s1", "ACDE"), ("s2", "ACDEFG")], MOCK_BACKBONES[:1], tmp_path, resume=True)
        assert third.n_processed == 2
        assert third.sample_ids == ["s1", "s2"]

    def test_resume_rebuilds_complete_cache_from_prior_samples(self, tmp_path) -> None:
        records = [("s1", "ACDE"), ("s2", "ACDEFG")]
        precompute(records, MOCK_BACKBONES[:1], tmp_path)
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Simulate an interrupted manifest that recorded only the first sample
        # while the NPZ already contains both rows.
        manifest["processed_sample_ids"] = ["s1"]
        manifest["n_processed"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = precompute(records, MOCK_BACKBONES[:1], tmp_path, resume=True)
        assert result.sample_ids == ["s1", "s2"]
        archive = np.load(tmp_path / "embeddings.npz", allow_pickle=False)
        assert list(archive["sample_id"]) == ["s1", "s2"]

    def test_input_change_invalidates_resume(self, tmp_path) -> None:
        precompute([("s1", "ACDE"), ("s2", "ACDEFG")], MOCK_BACKBONES[:1], tmp_path)
        # 输入内容变化：即使 resume，也必须全部重算（s2 新序列）
        second = precompute([("s1", "ACDE"), ("s2", "XXXXXX")], MOCK_BACKBONES[:1], tmp_path, resume=True)
        assert second.n_processed == 2
        archive = np.load(tmp_path / "embeddings.npz", allow_pickle=False)
        assert list(archive["sample_id"]) == ["s1", "s2"]

    def test_all_failed_still_writes_manifest(self, tmp_path) -> None:
        result = precompute([("s1", "")], MOCK_BACKBONES[:1], tmp_path)
        assert result.n_processed == 0
        assert result.n_failed == 1
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_processed"] == 0
        assert manifest["n_failed"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_end_to_end(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        _write_records(input_path, ["s1\tACDEFG", "s2\tACDE"])
        out = tmp_path / "out"
        exit_code = main(["--input", str(input_path), "--output", str(out), "--backbones", "mock"])
        assert exit_code == 0
        assert (out / "embeddings.npz").is_file()
        assert (out / "manifest.json").is_file()

    def test_cli_exit_1_with_failures(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        _write_records(input_path, ["s1\tACDE", "s2\t"])
        out = tmp_path / "out"
        assert main(["--input", str(input_path), "--output", str(out), "--backbones", "mock"]) == 1

    def test_cli_exit_2_on_invalid_input(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        _write_records(input_path, ["not-a-tsv-line"])
        out = tmp_path / "out"
        assert main(["--input", str(input_path), "--output", str(out), "--backbones", "mock"]) == 2

    def test_cli_unsupported_backbone(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        _write_records(input_path, ["s1\tACDE"])
        out = tmp_path / "out"
        assert main(["--input", str(input_path), "--output", str(out), "--backbones", "bogus"]) == 2

    def test_cli_missing_model_dir_fails_loudly(self, tmp_path) -> None:
        input_path = tmp_path / "in.tsv"
        _write_records(input_path, ["s1\tACDE"])
        out = tmp_path / "out"
        code = main(
            [
                "--input", str(input_path),
                "--output", str(out),
                "--backbones", "esm2",
                "--model-dir", str(tmp_path / "no-models"),
            ]
        )
        assert code == 2

    def test_build_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--input", "x", "--output", "y"])
        assert args.backbones == "esm2,prott5"
        assert args.device == "cpu"
        assert args.max_length == 1024
        assert args.batch_size == 8

    def test_build_parser_accepts_genes_json_source(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--genes-json", "requests.json", "--sequence-fasta", "seq.fasta", "--output", "y"])
        assert args.genes_json == "requests.json"
        assert args.sequence_fasta == "seq.fasta"
