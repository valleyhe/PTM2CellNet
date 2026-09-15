"""Tests for the strict public GSE normal/disease preparation path."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from src.data.gse_normal_disease import (
    GSE10xSample,
    GSENormalDiseaseError,
    build_gse_sample_specs,
    prepare_gse_normal_disease,
    summarize_normal_disease_directions,
)


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def _matrix_text(values: list[list[int]]) -> str:
    rows = len(values)
    columns = len(values[0])
    entries = [
        (row_index + 1, column_index + 1, value)
        for row_index, row in enumerate(values)
        for column_index, value in enumerate(row)
        if value
    ]
    return (
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                "%",
                f"{rows} {columns} {len(entries)}",
                *(f"{row} {column} {value}" for row, column, value in entries),
            ]
        )
        + "\n"
    )


def _make_sample(root: Path, *, index: int, state: str) -> GSE10xSample:
    label = f"HC-{index}" if state == "normal" else f"CD-{index}"
    sample = f"HC{index}" if state == "normal" else f"CD{index}"
    accession = f"GSM{214695000 + index + (0 if state == 'normal' else 20)}"
    matrix_path = root / f"{accession}_{label}_matrix.mtx.gz"
    features_path = root / f"{accession}_{label}_features.tsv.gz"
    barcodes_path = root / f"{accession}_{label}_barcodes.tsv.gz"
    gene_rows = [f"ENSG0000000000{gene_index}\tG{gene_index}\tGene Expression" for gene_index in range(1, 6)]
    _write_gz(features_path, "\n".join(gene_rows) + "\n")
    _write_gz(barcodes_path, "b0\nb1\nunannotated\n")
    first_gene = 30 if state == "disease" else 10
    _write_gz(
        matrix_path,
        _matrix_text(
            [
                [first_gene, first_gene + 2, 0],
                [2, 2, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ]
        ),
    )
    return GSE10xSample(
        sample=sample,
        donor=sample,
        state=state,
        accession=accession,
        matrix_path=matrix_path,
        features_path=features_path,
        barcodes_path=barcodes_path,
        tissue="colon",
    )


def _make_cohort(tmp_path: Path) -> tuple[tuple[GSE10xSample, ...], Path]:
    samples = tuple(
        [_make_sample(tmp_path, index=index, state="normal") for index in range(1, 4)]
        + [_make_sample(tmp_path, index=index, state="disease") for index in range(1, 4)]
    )
    rows = [
        {"sample": sample.sample, "cell_id": barcode, "annotation": "T-cell"}
        for sample in samples
        for barcode in ("b0", "b1")
    ]
    annotation_path = tmp_path / "annotation.csv.gz"
    pd.DataFrame(rows).to_csv(annotation_path, index=False, compression="gzip")
    return samples, annotation_path


def test_prepare_excludes_unannotated_raw_barcodes_and_preserves_gene_contract(tmp_path):
    samples, annotation_path = _make_cohort(tmp_path)
    output_path = tmp_path / "prepared.h5ad"

    result = prepare_gse_normal_disease(
        samples,
        annotation_path=annotation_path,
        candidate_gene_ids={f"ENSG0000000000{index}" for index in range(1, 6)},
        n_genes=4,
        output_path=output_path,
        min_donors=3,
    )

    assert result.adata.shape == (12, 4)
    assert result.adata.layers["counts"].shape == (12, 4)
    assert set(result.adata.obs["state"]) == {"normal", "disease"}
    assert result.adata.obs["donor"].nunique() == 6
    assert result.adata.uns["gse_normal_disease"]["observational_only"] is True
    assert result.adata.uns["gse_normal_disease"]["formal_perturbgen_ready"] is False
    assert all(item["raw_n_cells"] == 3 for item in result.report["samples"])
    assert all(item["excluded_unannotated_cells"] == 1 for item in result.report["samples"])
    assert output_path.is_file()
    assert output_path.with_suffix(".manifest.json").is_file()

    evidence = summarize_normal_disease_directions(result.adata, cell_types=["T-cell"])
    assert len(evidence) == 4
    assert set(evidence["observed_direction"]) == {"up", "down"}
    assert set(evidence["n_normal_donors"]) == {3}
    assert set(evidence["n_disease_donors"]) == {3}


def test_prepare_fails_when_annotation_references_missing_barcode(tmp_path):
    samples, annotation_path = _make_cohort(tmp_path)
    annotation = pd.read_csv(annotation_path)
    annotation.loc[0, "cell_id"] = "missing-barcode"
    annotation.to_csv(annotation_path, index=False, compression="gzip")

    with pytest.raises(GSENormalDiseaseError, match="missing barcodes"):
        prepare_gse_normal_disease(
            samples,
            annotation_path=annotation_path,
            candidate_gene_ids={f"ENSG0000000000{index}" for index in range(1, 6)},
            n_genes=4,
        )


def test_prepare_excludes_conflicting_annotation_rows_without_guessing_label(tmp_path):
    samples, annotation_path = _make_cohort(tmp_path)
    annotation = pd.read_csv(annotation_path)
    annotation = pd.concat(
        [annotation, pd.DataFrame([{"sample": "CD1", "cell_id": "b0", "annotation": "B-cell"}])],
        ignore_index=True,
    )
    annotation.to_csv(annotation_path, index=False, compression="gzip")

    result = prepare_gse_normal_disease(
        samples,
        annotation_path=annotation_path,
        candidate_gene_ids={f"ENSG0000000000{index}" for index in range(1, 6)},
        n_genes=4,
    )

    assert result.adata.n_obs == 11
    assert result.report["ambiguous_annotation_cells"] == {"CD1": ["b0"]}
    assert "B-cell" not in set(result.adata.obs["cell_type"])


def test_discovery_requires_complete_per_sample_file_triplets(tmp_path):
    _make_sample(tmp_path, index=1, state="normal")
    _make_sample(tmp_path, index=1, state="disease")
    specs = build_gse_sample_specs(tmp_path, normal_samples=("HC1",), disease_samples=("CD1",))
    assert [(spec.sample, spec.state) for spec in specs] == [("HC1", "normal"), ("CD1", "disease")]


def test_prepare_rejects_non_integer_counts(tmp_path):
    samples, annotation_path = _make_cohort(tmp_path)
    sample = samples[0]
    _write_gz(
        sample.matrix_path,
        _matrix_text([[1, 1.5, 0], [2, 2, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]).replace(
            "matrix coordinate integer", "matrix coordinate real", 1
        ),
    )

    with pytest.raises(GSENormalDiseaseError, match="integer-like"):
        prepare_gse_normal_disease(
            samples,
            annotation_path=annotation_path,
            candidate_gene_ids={f"ENSG0000000000{index}" for index in range(1, 6)},
            n_genes=4,
        )
