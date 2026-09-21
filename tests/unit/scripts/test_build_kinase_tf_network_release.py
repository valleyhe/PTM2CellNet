"""Contract tests for the kinase+TF signed-network release assembler."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_kinase_tf_network_release import (
    NetworkReleaseAssemblyError,
    assemble_network_release,
    main,
)
from src.analysis.ptm_research_config import PropagationConfig
from src.analysis.signed_network import load_signed_network, propagate_signed_scores


COLUMNS = [
    "source_id",
    "target_id",
    "edge_type",
    "effect_sign",
    "site",
    "species",
    "evidence",
    "confidence",
    "release",
]


def _write_table(path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None) -> Path:
    pd.DataFrame(rows, columns=columns or COLUMNS).to_csv(path, sep="\t", index=False)
    return path


def _kinase_rows(release: str = "kinase-2026-09") -> list[dict[str, object]]:
    return [
        {
            "source_id": "ENSG00000000001",
            "target_id": "ENSG00000000002",
            "edge_type": "kinase_substrate:phosphorylation",
            "effect_sign": "+1",
            "site": "S9",
            "species": "9606",
            "evidence": "synthetic-kinase",
            "confidence": "0.9",
            "release": release,
        }
    ]


def _tf_rows(release: str = "omnipath-2026-09-16") -> list[dict[str, object]]:
    return [
        {
            "source_id": "ENSG00000000002",
            "target_id": "ENSG00000000003",
            "edge_type": "tf_regulation",
            "effect_sign": "-1",
            "site": "",
            "species": "9606",
            "evidence": "synthetic-tf",
            "confidence": "0.8",
            "release": release,
        }
    ]


def _paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    kinase = _write_table(tmp_path / "kinase.tsv", _kinase_rows())
    tf = _write_table(tmp_path / "tf.tsv", _tf_rows())
    output = tmp_path / "combined.tsv"
    manifest = tmp_path / "combined.manifest.json"
    return kinase, tf, output, manifest


def test_assembles_new_release_deduplicates_and_records_inputs(tmp_path: Path):
    kinase, tf, output, manifest_path = _paths(tmp_path)
    duplicate = pd.read_csv(kinase, sep="\t", dtype=str)
    duplicate.to_csv(kinase, sep="\t", index=False, mode="a", header=False)

    assert (
        main(
            [
                "--kinase-network-tsv",
                str(kinase),
                "--tf-network-tsv",
                str(tf),
                "--release",
                "omnipath-enzsub+tf-20260920",
                "--output-tsv",
                str(output),
                "--manifest-output",
                str(manifest_path),
            ]
        )
        == 0
    )

    result = pd.read_csv(output, sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == 2
    assert set(result["release"]) == {"omnipath-enzsub+tf-20260920"}
    assert set(result["evidence"]) == {"synthetic-kinase", "synthetic-tf"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "ptm2cellnet.signed-network-release/v1"
    assert manifest["biology_pass"] is False
    assert manifest["inputs"]["tf_network"]["release"] == "omnipath-2026-09-16"
    assert manifest["counts"]["duplicate_rows_removed"] == 1
    assert manifest["counts"]["output_rows"] == 2


def test_rejects_tf_edge_in_kinase_input(tmp_path: Path):
    _, tf, output, manifest = _paths(tmp_path)
    bad_kinase = _write_table(tmp_path / "bad-kinase.tsv", _tf_rows("kinase-2026-09"))
    with pytest.raises(NetworkReleaseAssemblyError, match="tf_regulation"):
        assemble_network_release(
            kinase_network_tsv=bad_kinase,
            tf_network_tsv=tf,
            release="new-release",
            output_tsv=output,
            manifest_output=manifest,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("species", "species"),
        ("columns", "nine"),
        ("release", "release"),
    ],
)
def test_rejects_species_schema_and_mixed_release(tmp_path: Path, mutation: str, match: str):
    kinase, tf, output, manifest = _paths(tmp_path)
    if mutation == "species":
        frame = pd.read_csv(kinase, sep="\t", dtype=str)
        frame.loc[0, "species"] = "10090"
        frame.to_csv(kinase, sep="\t", index=False)
    elif mutation == "columns":
        frame = pd.read_csv(kinase, sep="\t")
        frame = frame.drop(columns=["evidence"])
        frame.to_csv(kinase, sep="\t", index=False)
    else:
        frame = pd.DataFrame(_kinase_rows() + [{**_kinase_rows()[0], "release": "other-release"}])
        frame.to_csv(kinase, sep="\t", index=False)
    with pytest.raises(NetworkReleaseAssemblyError, match=match):
        assemble_network_release(
            kinase_network_tsv=kinase,
            tf_network_tsv=tf,
            release="new-release",
            output_tsv=output,
            manifest_output=manifest,
        )


def test_rejects_empty_or_reused_release_and_in_place_output(tmp_path: Path):
    kinase, tf, output, manifest = _paths(tmp_path)
    with pytest.raises(NetworkReleaseAssemblyError, match="must not be empty"):
        assemble_network_release(
            kinase_network_tsv=kinase,
            tf_network_tsv=tf,
            release=" ",
            output_tsv=output,
            manifest_output=manifest,
        )
    with pytest.raises(NetworkReleaseAssemblyError, match="new release"):
        assemble_network_release(
            kinase_network_tsv=kinase,
            tf_network_tsv=tf,
            release="omnipath-2026-09-16",
            output_tsv=output,
            manifest_output=manifest,
        )
    with pytest.raises(NetworkReleaseAssemblyError, match="overwrite"):
        assemble_network_release(
            kinase_network_tsv=kinase,
            tf_network_tsv=tf,
            release="new-release",
            output_tsv=tf,
            manifest_output=manifest,
        )


def test_accepts_tf_freeze_sign_notation_without_rewriting_input(tmp_path: Path):
    kinase, _, output, manifest = _paths(tmp_path)
    tf = _write_table(tmp_path / "tf-plain-sign.tsv", [{**_tf_rows()[0], "effect_sign": "1"}])
    original = tf.read_bytes()
    assemble_network_release(
        kinase_network_tsv=kinase,
        tf_network_tsv=tf,
        release="omnipath-kinase+tf-2026-09-21",
        output_tsv=output,
        manifest_output=manifest,
    )
    assert tf.read_bytes() == original
    result = pd.read_csv(output, sep="\t", dtype=str, keep_default_na=False)
    assert set(result["effect_sign"]) == {"+1"}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["inputs"]["tf_network"]["sha256"]
    assert payload["biology_pass"] is False


def test_output_loads_and_intermediate_edge_reaches_tf_gene(tmp_path: Path):
    kinase, tf, output, manifest = _paths(tmp_path)
    assemble_network_release(
        kinase_network_tsv=kinase,
        tf_network_tsv=tf,
        release="omnipath-enzsub+tf-20260920",
        output_tsv=output,
        manifest_output=manifest,
    )
    network = load_signed_network(output, expected_species="9606", expected_release="omnipath-enzsub+tf-20260920")
    result = propagate_signed_scores(
        network,
        {"ENSG00000000001": 1.0},
        config=PropagationConfig(max_depth=2, decay=1.0, gene_edge_types=("tf_regulation",)),
    )
    assert [(score.source_id, score.target_id, score.n_paths) for score in result.scores] == [
        ("ENSG00000000001", "ENSG00000000003", 1)
    ]
    assert result.scores[0].gene_score == pytest.approx(-0.72)
