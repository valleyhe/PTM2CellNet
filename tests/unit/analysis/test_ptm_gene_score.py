"""Tests for gene-score table construction and PTM–AD intersections (方案 §4.4/§4.6/§5.4)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.ptm_gene_score import (
    PTMGeneScoreError,
    build_gene_score_table,
    intersect_gene_scores_with_deg,
    load_deg_table,
    load_gene_score_table,
    load_network_id_map,
    write_gene_score_table,
    write_intersection_outputs,
)
from src.analysis.ptm_research_config import parse_ptm_research_config
from src.analysis.signed_network import PropagationConfig, propagate_signed_scores
from src.analysis.signed_network import load_signed_network

EDGE_COLUMNS = (
    "source_id",
    "target_id",
    "edge_type",
    "effect_sign",
    "site",
    "species",
    "evidence",
    "confidence",
    "release",
)


def _config(**overrides):
    payload = {
        "schema_version": "ptm2cellnet.ptm-research-config/v1",
        "research_objective": "association",
        "reference_axis": "disease_minus_normal",
        "contrast": "disease-minus-normal",
        "primary_activity_method": "KSTAR",
        "network_release": "2026-08",
        "cell_types": ["EX", "IN"],
        "cohort_h5ad": "data/AD/standardized/GSE174367_ad_cohort.h5ad",
        "cohort_pairing": "between_donor",
        "species": "9606",
        "ptm_cohort": "CPTAC_TEST",
        "deg_max_fdr": 0.05,
        "min_donors_per_state": 3,
        "replicate_policy": "mean",
        "propagation": {"max_depth": 3, "decay": 0.5, "gene_edge_types": ["tf_regulation"]},
    }
    payload.update(overrides)
    return parse_ptm_research_config(payload)


def _propagation_result(tmp_path, decay: float = 0.5):
    edges = pd.DataFrame(
        {
            "source_id": ["KIN", "TF"],
            "target_id": ["TF", "GENE_A"],
            "edge_type": ["protein_protein", "tf_regulation"],
            "effect_sign": ["+1", "-1"],
            "site": ["", "s1"],
            "species": ["9606", "9606"],
            "evidence": ["test", "test"],
            "confidence": [1.0, 1.0],
            "release": ["2026-08", "2026-08"],
        }
    )
    network_path = tmp_path / "network.tsv"
    edges.to_csv(network_path, sep="\t", index=False)
    network = load_signed_network(network_path)
    return propagate_signed_scores(
        network, {"KIN": 2.0}, config=PropagationConfig(max_depth=3, decay=decay, gene_edge_types=("tf_regulation",))
    )


def _id_map_file(tmp_path):
    path = tmp_path / "network_id_map.tsv"
    path.write_text(
        "network_id\tgene_symbol\temsembl_dummy\n".replace("emsembl_dummy", "ensembl_id")
        + "GENE_A\tGENEA\tENSG00000000001\n"
        + "TF\tTFA\tENSG00000000002\n",
        encoding="utf-8",
    )
    return path


class TestNetworkIdMap:
    def test_conflict_fails(self, tmp_path):
        path = tmp_path / "network_id_map.tsv"
        path.write_text(
            "network_id\tgene_symbol\temsembl_dummy\n".replace("emsembl_dummy", "ensembl_id")
            + "GENE_A\tGENEA\tENSG00000000001\n"
            + "GENE_A\tGENEB\tENSG00000000009\n",
            encoding="utf-8",
        )
        with pytest.raises(PTMGeneScoreError, match="conflicting"):
            load_network_id_map(path)


class TestBuildGeneScoreTable:
    def test_rows_follow_propagation_with_direction(self, tmp_path):
        frame, audit = build_gene_score_table(
            _propagation_result(tmp_path),
            network_id_map=load_network_id_map(_id_map_file(tmp_path)),
            config=_config(),
            provenance="activity=stub;network=stub",
        )
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["source_activity_id"] == "KIN"
        assert row["target_ensembl_id"] == "ENSG00000000001"
        assert row["gene_score"] == pytest.approx(2.0 * 1.0 * -1.0 * 0.25)
        assert row["predicted_gene_direction"] == "down"
        assert row["prediction_status"] == "direction_only"
        assert audit.n_targets_unmapped == 0

    def test_unmapped_targets_excluded_and_counted(self, tmp_path):
        frame, audit = build_gene_score_table(
            _propagation_result(tmp_path),
            network_id_map={"TF": ("TFA", "ENSG00000000002")},
            config=_config(),
            provenance="stub",
        )
        assert frame.empty
        assert audit.unmapped_targets == ("GENE_A",)

    def test_empty_provenance_fails(self, tmp_path):
        with pytest.raises(PTMGeneScoreError, match="provenance"):
            build_gene_score_table(
                _propagation_result(tmp_path),
                network_id_map=load_network_id_map(_id_map_file(tmp_path)),
                config=_config(),
                provenance="  ",
            )

    def test_round_trip_through_tsv(self, tmp_path):
        frame, _ = build_gene_score_table(
            _propagation_result(tmp_path),
            network_id_map=load_network_id_map(_id_map_file(tmp_path)),
            config=_config(),
            provenance="stub",
        )
        path = write_gene_score_table(frame, tmp_path / "scores.tsv")
        loaded = load_gene_score_table(path)
        assert loaded.iloc[0]["target_ensembl_id"] == "ENSG00000000001"

    def test_direction_score_sign_disagreement_fails_on_load(self, tmp_path):
        frame, _ = build_gene_score_table(
            _propagation_result(tmp_path),
            network_id_map=load_network_id_map(_id_map_file(tmp_path)),
            config=_config(),
            provenance="stub",
        )
        frame.loc[0, "gene_score"] = 0.5  # sign no longer matches 'down'
        path = tmp_path / "scores.tsv"
        frame.to_csv(path, sep="\t", index=False)
        with pytest.raises(PTMGeneScoreError, match="disagree"):
            load_gene_score_table(path)

    def test_duplicate_pair_fails_on_load(self, tmp_path):
        frame, _ = build_gene_score_table(
            _propagation_result(tmp_path),
            network_id_map=load_network_id_map(_id_map_file(tmp_path)),
            config=_config(),
            provenance="stub",
        )
        doubled = pd.concat([frame, frame], ignore_index=True)
        path = tmp_path / "scores.tsv"
        doubled.to_csv(path, sep="\t", index=False)
        with pytest.raises(PTMGeneScoreError, match="duplicates"):
            load_gene_score_table(path)


DEG_COLUMNS = (
    "cell_type",
    "ensembl_id",
    "gene_symbol",
    "log2fc",
    "fdr",
    "observed_direction",
    "n_normal_donors",
    "n_disease_donors",
)


def _deg_frame(rows=None) -> pd.DataFrame:
    rows = rows or [
        # EX: concordant-formal (down/down, fdr 0.01, donors 4/8)
        ["EX", "ENSG00000000001", "GENEA", -0.7, 0.01, "down", 4, 8],
        # EX: AD-only gene
        ["EX", "ENSG00000000009", "GENE9", 2.0, 0.01, "up", 4, 8],
        # IN: same gene observed up (discordant) with weak FDR (exploratory)
        ["IN", "ENSG00000000001", "GENEA", 0.5, 0.2, "up", 3, 3],
    ]
    return pd.DataFrame(rows, columns=DEG_COLUMNS)


def _score_frame(tmp_path) -> pd.DataFrame:
    frame, _ = build_gene_score_table(
        _propagation_result(tmp_path),
        network_id_map=load_network_id_map(_id_map_file(tmp_path)),
        config=_config(),
        provenance="stub",
    )
    return frame


class TestLoadDegTable:
    def test_missing_donor_counts_fail(self, tmp_path):
        frame = _deg_frame()
        frame.loc[0, "n_normal_donors"] = None
        path = tmp_path / "deg.tsv"
        frame.to_csv(path, sep="\t", index=False)
        with pytest.raises(PTMGeneScoreError, match="n_normal_donors"):
            load_deg_table(path)

    def test_direction_sign_disagreement_fails(self, tmp_path):
        frame = _deg_frame([["EX", "ENSG00000000001", "GENEA", -0.7, 0.01, "up", 4, 8]])
        path = tmp_path / "deg.tsv"
        frame.to_csv(path, sep="\t", index=False)
        with pytest.raises(PTMGeneScoreError, match="disagree"):
            load_deg_table(path)

    def test_duplicate_cell_type_gene_fails(self, tmp_path):
        frame = pd.concat([_deg_frame().head(1), _deg_frame().head(1)], ignore_index=True)
        path = tmp_path / "deg.tsv"
        frame.to_csv(path, sep="\t", index=False)
        with pytest.raises(PTMGeneScoreError, match="duplicates"):
            load_deg_table(path)


class TestIntersections:
    def test_membership_and_tiers(self, tmp_path):
        score_frame = _score_frame(tmp_path)
        intersections = intersect_gene_scores_with_deg(score_frame, _deg_frame(), config=_config())
        ex_frame, ex_summary = intersections["EX"]
        concordant = ex_frame[ex_frame["membership"] == "concordant"]
        assert len(concordant) == 1
        assert concordant.iloc[0]["target_ensembl_id"] == "ENSG00000000001"
        assert concordant.iloc[0]["evidence_tier"] == "formal"
        assert concordant.iloc[0]["observed_significant"]
        assert ex_summary.n_formal_concordant == 1
        ad_only = ex_frame[ex_frame["membership"] == "AD_only"]
        assert len(ad_only) == 1
        assert (ad_only["evidence_tier"] == "ad_only").all()
        # score row not in EX DEG? GENE_A is in EX DEG so no PTM_only there.
        assert ex_summary.n_ptm_only == 0
        in_frame, in_summary = intersections["IN"]
        discordant = in_frame[in_frame["membership"] == "discordant"]
        assert len(discordant) == 1
        assert discordant.iloc[0]["evidence_tier"] == "exploratory"  # fdr 0.2 > 0.05
        assert not bool(discordant.iloc[0]["observed_significant"])

    def test_signed_admission_keeps_concordant_nonsignificant_as_formal(self, tmp_path):
        score_frame = _score_frame(tmp_path)
        deg = _deg_frame(
            [
                ["EX", "ENSG00000000001", "GENEA", -0.7, 0.52, "down", 4, 8],
                ["IN", "ENSG00000000001", "GENEA", -0.7, 0.52, "down", 3, 3],
            ]
        )
        config = _config(observed_admission_rule="signed_direction_without_fdr_cutoff")
        intersections = intersect_gene_scores_with_deg(score_frame, deg, config=config)
        ex_frame, ex_summary = intersections["EX"]
        concordant = ex_frame[ex_frame["membership"] == "concordant"]
        assert len(concordant) == 1
        assert concordant.iloc[0]["evidence_tier"] == "formal"
        assert not bool(concordant.iloc[0]["observed_significant"])
        assert ex_summary.n_formal_concordant == 1

    def test_ptm_only_rows_are_kept(self, tmp_path):
        score_frame = _score_frame(tmp_path)
        deg = _deg_frame(
            [
                ["EX", "ENSG00000000009", "GENE9", 2.0, 0.01, "up", 4, 8],
                ["IN", "ENSG00000000009", "GENE9", 2.0, 0.01, "up", 4, 8],
            ]
        )
        intersections = intersect_gene_scores_with_deg(score_frame, deg, config=_config())
        ex_frame, ex_summary = intersections["EX"]
        ptm_only = ex_frame[ex_frame["membership"] == "PTM_only"]
        assert len(ptm_only) == 1
        assert (ptm_only["evidence_tier"] == "ptm_only").all()
        assert ex_summary.n_ptm_only == 1

    def test_low_donor_support_marks_exploratory(self, tmp_path):
        score_frame = _score_frame(tmp_path)
        deg = _deg_frame(
            [
                ["EX", "ENSG00000000001", "GENEA", -0.7, 0.01, "down", 2, 8],
                ["IN", "ENSG00000000001", "GENEA", -0.7, 0.01, "down", 3, 3],
            ]
        )
        intersections = intersect_gene_scores_with_deg(score_frame, deg, config=_config())
        ex_frame, _ = intersections["EX"]
        concordant = ex_frame[ex_frame["membership"] == "concordant"]
        assert concordant.iloc[0]["evidence_tier"] == "exploratory"

    def test_frozen_cell_type_without_deg_fails(self, tmp_path):
        score_frame = _score_frame(tmp_path)
        deg = _deg_frame().head(1)  # only EX rows
        with pytest.raises(PTMGeneScoreError, match="IN"):
            intersect_gene_scores_with_deg(score_frame, deg, config=_config())

    def test_outputs_written_with_target_sets(self, tmp_path):
        score_frame = _score_frame(tmp_path)
        intersections = intersect_gene_scores_with_deg(score_frame, _deg_frame(), config=_config())
        output_dir = tmp_path / "intersections"
        outputs = write_intersection_outputs(
            intersections,
            output_dir=output_dir,
            config=_config(),
            score_table_path=tmp_path / "scores.tsv",
            deg_table_path=tmp_path / "deg.tsv",
        )
        assert (output_dir / "ptm_ad_intersection_EX.tsv").is_file()
        assert (output_dir / "ptm_ad_intersection_IN.tsv").is_file()
        summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
        assert summary["cell_types"]["EX"]["n_formal_concordant"] == 1
        assert summary["thresholds"]["observed_admission_rule"] == "fdr_cutoff"
        assert summary["n_cell_types_supported"]["ENSG00000000001|GENEA"]["n_cell_types"] == 1
        manifest = json.loads(outputs["target_set_manifest"].read_text(encoding="utf-8"))
        ex_sources = manifest["cell_types"]["EX"]["sources"]
        assert list(ex_sources) == ["KIN"]
        assert ex_sources["KIN"]["targets"][0]["target_ensembl_id"] == "ENSG00000000001"
