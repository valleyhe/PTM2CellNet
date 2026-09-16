"""Unit tests for Gate-E vocabulary-migration benchmark assembly."""

import pickle
import tempfile
from pathlib import Path

import pytest

from src.analysis.gate_e_benchmark import (
    BENCHMARK_COLUMNS,
    GateEBenchmarkBuildError,
    build_gate_e_benchmark,
    write_gate_e_benchmark,
)
from src.integration.perturbgen.gate_e import GateEBenchmark, load_benchmark


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mapping(path, mapping):
    with path.open("wb") as handle:
        pickle.dump(mapping, handle)


def _cplm_lines():
    return [
        # symbol gene GENE1 (acc P00001): ubiquitination at 10, acetylation at 20
        "CPLM000001\tP00001\t10\tUbiquitination\tGENE1\tHomo sapiens\tPEPTIDE",
        "CPLM000001\tP00001\t20\tAcetylation\tGENE1\tHomo sapiens\tPEPTIDE",
        # GENE2 (acc P00002): sumoylation twice (same gene+type deduplicates)
        "CPLM000002\tP00002\t5\tSumoylation\tGENE2\tHomo sapiens\tPEPTIDE",
        "CPLM000002\tP00002\t7\tSumoylation\tGENE2\tHomo sapiens\tPEPTIDE",
        # non-human rows are excluded
        "CPLM000003\tP00003\t1\tUbiquitination\tGENE3\tMus musculus\tPEPTIDE",
    ]


def _dbptm_lines():
    return [
        # phosphorylation rows anchored by CPLM accessions
        "GENE1_HUMAN\tP00001\t99\tPhosphorylation\t12345\tPEPTIDE",
        "GENE2_HUMAN\tP00002\t77\tPhosphorylation\t12345\tPEPTIDE",
        # unknown accession is dropped and counted
        "GENE9_HUMAN\tP00009\t12\tPhosphorylation\t12345\tPEPTIDE",
    ]


def _build(_unused=None, cplm_lines=None, dbptm_lines=None, mapping=None, min_rows=3):
    tmp_path = Path(tempfile.mkdtemp())
    cplm = tmp_path / "cplm.txt"
    dbptm = tmp_path / "phosph.txt"
    mapping_path = tmp_path / "mapping.pkl"
    _write_lines(cplm, cplm_lines if cplm_lines is not None else _cplm_lines())
    _write_lines(dbptm, dbptm_lines if dbptm_lines is not None else _dbptm_lines())
    _write_mapping(
        mapping_path, mapping if mapping is not None else {"GENE1": "ENSG00000000001", "GENE2": "ENSG00000000002"}
    )
    return build_gate_e_benchmark(
        cplm_path=cplm,
        dbptm_phosphorylation_path=dbptm,
        ensembl_mapping_path=mapping_path,
        min_rows=min_rows,
    )


def test_builds_deduplicated_rows_with_cplm_and_dbptm_sources():
    frame, manifest = _build()

    assert list(frame.columns) == list(BENCHMARK_COLUMNS)
    keys = set(zip(frame["gene_symbol"], frame["ptm_type"], strict=True))
    assert keys == {
        ("GENE1", "ubiquitination"),
        ("GENE1", "acetylation"),
        ("GENE1", "phosphorylation"),
        ("GENE2", "sumoylation"),
        ("GENE2", "phosphorylation"),
    }
    # deduplicated GENE2 sumoylation keeps the first annotated position
    sumo = frame[(frame["gene_symbol"] == "GENE2") & (frame["ptm_type"] == "sumoylation")]
    assert len(sumo) == 1 and int(sumo.iloc[0]["position"]) == 5
    assert manifest["benchmark_rows"] == len(frame)
    assert manifest["dbptm_phosphorylation_unmapped_accessions"] >= 1
    assert manifest["ptm_type_counts"]["phosphorylation"] == 2


def test_output_satisfies_gate_e_loader_contract(tmp_path):
    frame, manifest = _build()
    output = tmp_path / "benchmark.csv"
    manifest_output = tmp_path / "manifest.json"
    write_gate_e_benchmark(frame, manifest, output, manifest_output)

    loaded: GateEBenchmark = load_benchmark(output)
    loaded.validate(min_samples=3)
    assert len(loaded.rows) == len(frame)


def test_unmapped_symbols_are_counted_not_invented():
    frame, manifest = _build(None, mapping={"GENE1": "ENSG00000000001"})
    assert set(frame["gene_symbol"]) == {"GENE1"}
    assert manifest["unmapped_symbols"] >= 1


def test_rejects_below_min_rows():
    with pytest.raises(GateEBenchmarkBuildError, match="at least 200"):
        _build(None, min_rows=200)


def test_rejects_conflicting_cplm_accession_symbols():
    lines = _cplm_lines() + ["CPLM000004\tP00001\t30\tUbiquitination\tGENE_OTHER\tHomo sapiens\tPEPTIDE"]
    with pytest.raises(GateEBenchmarkBuildError, match="conflicting UniProt"):
        _build(None, cplm_lines=lines)
