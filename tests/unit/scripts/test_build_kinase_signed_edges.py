"""Contract tests for kinase signed-edge construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_kinase_signed_edges import KinaseSignedEdgeError, assemble_kinase_signed_edges, main
from scripts.build_kinase_tf_network_release import assemble_network_release
from src.analysis.ptm_research_config import PropagationConfig
from src.analysis.signed_network import load_signed_network, propagate_signed_scores


def _mapping(tmp_path: Path) -> Path:
    path = tmp_path / "map.tsv"
    path.write_text(
        "ensembl_id\tgene_symbol\nENSG00000000001\tGSK3B\nENSG00000000002\tCTNNB1\nENSG00000000003\tAPP\n",
        encoding="utf-8",
    )
    return path


def _omnipath(tmp_path: Path) -> Path:
    path = tmp_path / "interactions.tsv"
    pd.DataFrame(
        [
            {
                "source": "P49841",
                "target": "P35222",
                "source_genesymbol": "GSK3B",
                "target_genesymbol": "CTNNB1",
                "is_directed": "true",
                "is_stimulation": "false",
                "is_inhibition": "true",
                "consensus_stimulation": "false",
                "consensus_inhibition": "true",
                "sources": "SIGNOR;PhosphoSite",
            },
            {
                "source": "P49841",
                "target": "P05067",
                "source_genesymbol": "GSK3B",
                "target_genesymbol": "APP",
                "is_directed": "true",
                "is_stimulation": "true",
                "is_inhibition": "true",
                "consensus_stimulation": "false",
                "consensus_inhibition": "false",
                "sources": "SIGNOR",
            },
        ]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_builds_signed_kinase_edges_and_rejects_tf_type(tmp_path: Path):
    enzymes = tmp_path / "enzymes.tsv"
    enzymes.write_text("source_gene\nGSK3B\n", encoding="utf-8")
    output = tmp_path / "kinase.tsv"
    manifest = tmp_path / "kinase.manifest.json"
    assert (
        main(
            [
                "--omnipath-tsv",
                str(_omnipath(tmp_path)),
                "--ensembl-mapping",
                str(_mapping(tmp_path)),
                "--release",
                "omnipath-kinase-2026-09-21",
                "--output-tsv",
                str(output),
                "--manifest-output",
                str(manifest),
                "--enzyme-source-tsv",
                str(enzymes),
            ]
        )
        == 0
    )
    frame = pd.read_csv(output, sep="\t", dtype=str, keep_default_na=False)
    assert list(frame["edge_type"].unique()) == ["kinase_substrate:signaling"]
    assert set(frame["effect_sign"]) == {"-1"}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["biology_pass"] is False
    assert payload["counts"]["unsigned_or_conflicting_dropped"] == 1
    with pytest.raises(KinaseSignedEdgeError, match="tf_regulation"):
        assemble_kinase_signed_edges(
            omnipath_tsv=_omnipath(tmp_path),
            ensembl_mapping=_mapping(tmp_path),
            release="new-release",
            output_tsv=tmp_path / "bad.tsv",
            manifest_output=tmp_path / "bad.json",
            edge_type="tf_regulation",
        )


def test_assembled_kinase_plus_tf_reaches_gene_and_does_not_rewrite_tf(tmp_path: Path):
    kinase_output = tmp_path / "kinase.tsv"
    assemble_kinase_signed_edges(
        omnipath_tsv=_omnipath(tmp_path),
        ensembl_mapping=_mapping(tmp_path),
        release="omnipath-kinase-2026-09-21",
        output_tsv=kinase_output,
        manifest_output=tmp_path / "kinase.json",
    )
    tf = tmp_path / "tf.tsv"
    tf.write_text(
        "source_id\ttarget_id\tedge_type\teffect_sign\tsite\tspecies\tevidence\tconfidence\trelease\n"
        "ENSG00000000002\tENSG00000000003\ttf_regulation\t1\t\t9606\tsynthetic-tf\t0.800\tomnipath-2026-09-16\n",
        encoding="utf-8",
    )
    original = tf.read_bytes()
    combined = tmp_path / "combined.tsv"
    assemble_network_release(
        kinase_network_tsv=kinase_output,
        tf_network_tsv=tf,
        release="omnipath-kinase+tf-2026-09-21",
        output_tsv=combined,
        manifest_output=tmp_path / "combined.json",
    )
    assert tf.read_bytes() == original
    network = load_signed_network(combined, expected_species="9606", expected_release="omnipath-kinase+tf-2026-09-21")
    result = propagate_signed_scores(
        network,
        {"ENSG00000000001": 1.0},
        config=PropagationConfig(max_depth=2, decay=1.0, gene_edge_types=("tf_regulation",)),
    )
    assert result.seeds_without_node == ()
    assert [(score.source_id, score.target_id) for score in result.scores] == [("ENSG00000000001", "ENSG00000000003")]
    assert result.scores[0].gene_score < 0
