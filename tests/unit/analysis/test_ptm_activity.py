"""Tests for the PTM input/activity contracts (方案 §4.1/§4.2/§5.1)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.ptm_activity import (
    PTMActivityContractError,
    activities_for_propagation,
    load_gene_map,
    load_ptm_activity_table,
    load_ptm_site_quantification,
    standardize_ptm_input,
    write_ptm_input_manifest,
)
from src.analysis.ptm_research_config import parse_ptm_research_config

INPUT_COLUMNS = (
    "sample_id",
    "donor_id",
    "condition",
    "protein_id",
    "gene_symbol",
    "residue",
    "ptm_type",
    "ptm_value",
    "value_scale",
    "ptm_qvalue",
    "total_protein_value",
    "species",
    "source_dataset",
)


def _input_frame(**overrides) -> pd.DataFrame:
    rows = [
        {
            "sample_id": "S1",
            "donor_id": "D1",
            "condition": "disease",
            "protein_id": "P12345",
            "gene_symbol": "GSK3B",
            "residue": "S9",
            "ptm_type": "phosphorylation",
            "ptm_value": 2.0,
            "value_scale": "linear",
            "ptm_qvalue": 0.01,
            "total_protein_value": 4.0,
            "species": "9606",
            "source_dataset": "CPTAC_TEST",
        },
        {
            "sample_id": "S2",
            "donor_id": "",
            "condition": "normal",
            "protein_id": "P67890",
            "gene_symbol": "MAPT",
            "residue": "T205",
            "ptm_type": "phosphorylation",
            "ptm_value": 3.0,
            "value_scale": "linear",
            "ptm_qvalue": 0.02,
            "total_protein_value": "",
            "species": "9606",
            "source_dataset": "CPTAC_TEST",
        },
    ]
    rows[0].update(overrides)
    frame = pd.DataFrame(rows)
    return frame


def _config():
    return parse_ptm_research_config(
        {
            "schema_version": "ptm2cellnet.ptm-research-config/v1",
            "research_objective": "association",
            "reference_axis": "disease_minus_normal",
            "contrast": "disease-minus-normal",
            "primary_activity_method": "KSTAR",
            "network_release": "2026-08",
            "cell_types": ["EX"],
            "cohort_h5ad": "data/AD/standardized/GSE174367_ad_cohort.h5ad",
            "cohort_pairing": "between_donor",
            "species": "9606",
            "ptm_cohort": "CPTAC_TEST",
            "deg_max_fdr": 0.05,
            "min_donors_per_state": 3,
            "replicate_policy": "mean",
            "propagation": {"max_depth": 3, "decay": 0.5, "gene_edge_types": ["tf_regulation"]},
        }
    )


def _gene_map_file(tmp_path):
    path = tmp_path / "gene_map.tsv"
    path.write_text(
        "protein_id\tgene_symbol\temsembl_dummy\n".replace("emsembl_dummy", "ensembl_id")
        + "P12345\tGSK3B\tENSG00000082701\n"
        + "P67890\tMAPT\tENSG00000186868\n",
        encoding="utf-8",
    )
    return path


def _write_tsv(tmp_path, frame, name="ptm.tsv"):
    path = tmp_path / name
    frame.to_csv(path, sep="\t", index=False)
    return path


class TestLoadPtmSiteQuantification:
    def _write(self, tmp_path, frame):
        return _write_tsv(tmp_path, frame)

    def test_valid_input_passes(self, tmp_path):
        frame = load_ptm_site_quantification(self._write(tmp_path, _input_frame()))
        assert len(frame) == 2
        assert not frame.loc[0, "has_donor"] if "has_donor" in frame else True
        # has_donor is added by standardize; here donor presence is preserved verbatim
        assert frame.loc[1, "donor_id"] == ""

    def test_missing_column_fails(self, tmp_path):
        frame = _input_frame().drop(columns=["residue"])
        with pytest.raises(PTMActivityContractError, match="residue"):
            load_ptm_site_quantification(self._write(tmp_path, frame))

    def test_qvalue_out_of_range_fails(self, tmp_path):
        frame = _input_frame(ptm_qvalue=1.5)
        with pytest.raises(PTMActivityContractError, match="ptm_qvalue"):
            load_ptm_site_quantification(self._write(tmp_path, frame))

    def test_mixed_value_scale_fails(self, tmp_path):
        frame = _input_frame()
        frame.loc[1, "value_scale"] = "log2"
        with pytest.raises(PTMActivityContractError, match="value scales"):
            load_ptm_site_quantification(self._write(tmp_path, frame))

    def test_unknown_value_scale_fails(self, tmp_path):
        frame = _input_frame(value_scale="ln")
        with pytest.raises(PTMActivityContractError, match="value_scale"):
            load_ptm_site_quantification(self._write(tmp_path, frame))

    def test_nonpositive_total_protein_fails(self, tmp_path):
        frame = _input_frame(total_protein_value=0.0)
        with pytest.raises(PTMActivityContractError, match="total_protein_value"):
            load_ptm_site_quantification(self._write(tmp_path, frame))


class TestStandardize:
    def _write(self, tmp_path, frame):
        return _write_tsv(tmp_path, frame)

    def test_linear_normalization_and_donor_flag(self, tmp_path):
        frame = load_ptm_site_quantification(self._write(tmp_path, _input_frame()))
        gene_map = load_gene_map(_gene_map_file(tmp_path))
        standardized, audit = standardize_ptm_input(frame, config=_config(), gene_map=gene_map)
        assert standardized.loc[0, "ptm_value_normalized"] == pytest.approx(0.5)
        assert pd.isna(standardized.loc[1, "ptm_value_normalized"])
        assert bool(standardized.loc[0, "has_donor"]) is True
        assert bool(standardized.loc[1, "has_donor"]) is False
        assert audit.n_rows_without_donor == 1
        assert audit.n_unmapped_proteins == 0
        assert audit.n_rows_with_total_protein == 1

    def test_log2_normalization_uses_difference(self, tmp_path):
        frame = _input_frame()
        frame["value_scale"] = "log2"
        frame["ptm_value"] = [4.0, 5.0]
        frame["total_protein_value"] = [2.0, ""]
        frame = load_ptm_site_quantification(self._write(tmp_path, frame))
        gene_map = load_gene_map(_gene_map_file(tmp_path))
        standardized, _ = standardize_ptm_input(frame, config=_config(), gene_map=gene_map)
        assert standardized.loc[0, "ptm_value_normalized"] == pytest.approx(2.0)

    def test_replicate_policy_fail_is_default_hard_failure(self, tmp_path):
        frame = pd.concat([_input_frame(), _input_frame().head(1)], ignore_index=True)
        frame = load_ptm_site_quantification(self._write(tmp_path, frame))
        config = _config()
        object.__setattr__(config, "replicate_policy", "fail")
        gene_map = load_gene_map(_gene_map_file(tmp_path))
        with pytest.raises(PTMActivityContractError, match="replicate_policy=fail"):
            standardize_ptm_input(frame, config=config, gene_map=gene_map)

    def test_replicate_policy_mean_collapses(self, tmp_path):
        frame = pd.concat([_input_frame(), _input_frame().head(1)], ignore_index=True)
        frame = load_ptm_site_quantification(self._write(tmp_path, frame))
        gene_map = load_gene_map(_gene_map_file(tmp_path))
        standardized, audit = standardize_ptm_input(frame, config=_config(), gene_map=gene_map)
        assert len(standardized) == 2
        assert audit.n_replicate_groups == 1
        assert audit.n_rows_input == 3

    def test_unmapped_protein_is_kept_and_counted(self, tmp_path):
        frame = load_ptm_site_quantification(self._write(tmp_path, _input_frame()))
        gene_map = load_gene_map(_gene_map_file(tmp_path))
        gene_map.pop("P67890")
        standardized, audit = standardize_ptm_input(frame, config=_config(), gene_map=gene_map)
        assert pd.isna(standardized.loc[1, "ensembl_id"])
        assert audit.unmapped_proteins == ("P67890",)

    def test_manifest_written_with_provenance(self, tmp_path):
        frame = load_ptm_site_quantification(self._write(tmp_path, _input_frame()))
        gene_map_path = _gene_map_file(tmp_path)
        gene_map = load_gene_map(gene_map_path)
        standardized, audit = standardize_ptm_input(frame, config=_config(), gene_map=gene_map)
        output_path = tmp_path / "standardized.tsv"
        standardized.to_csv(output_path, sep="\t", index=False)
        manifest_path = write_ptm_input_manifest(
            tmp_path / "manifest.json",
            input_path=self._write(tmp_path, _input_frame()),
            audit=audit,
            config=_config(),
            gene_map_path=gene_map_path,
            output_path=output_path,
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "ptm2cellnet.ptm-input-manifest/v1"
        assert len(payload["source"]["sha256"]) == 64
        assert payload["standardization"]["n_rows_without_donor"] == 1


ACTIVITY_COLUMNS = (
    "activity_unit",
    "regulator_id",
    "regulator_type",
    "condition_or_contrast",
    "activity_score",
    "activity_direction",
    "activity_pvalue",
    "activity_qvalue",
    "n_substrates",
    "network_coverage",
    "method",
    "method_version",
    "input_manifest",
)


def _activity_frame(**overrides) -> pd.DataFrame:
    row = {
        "activity_unit": "zscore",
        "regulator_id": "GSK3B",
        "regulator_type": "kinase",
        "condition_or_contrast": "disease-minus-normal",
        "activity_score": 1.5,
        "activity_direction": "up",
        "activity_pvalue": 0.01,
        "activity_qvalue": 0.05,
        "n_substrates": 12,
        "network_coverage": 0.8,
        "method": "KSTAR",
        "method_version": "1.0",
        "input_manifest": "ptm_input_manifest.json",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestLoadPtmActivityTable:
    def _write(self, tmp_path, frame):
        return _write_tsv(tmp_path, frame, name="activity.tsv")

    def _write(self, tmp_path, frame):
        path = tmp_path / "activity.tsv"
        frame.to_csv(path, sep="\t", index=False)
        return path

    def test_valid_table_passes(self, tmp_path):
        frame = load_ptm_activity_table(self._write(tmp_path, _activity_frame()))
        assert frame.loc[0, "regulator_id"] == "GSK3B"

    def test_missing_column_fails(self, tmp_path):
        frame = _activity_frame().drop(columns=["n_substrates"])
        with pytest.raises(PTMActivityContractError, match="n_substrates"):
            load_ptm_activity_table(self._write(tmp_path, frame))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("activity_score", 0.0, "non-zero"),
            ("activity_direction", "flat", "activity_direction"),
            ("activity_qvalue", 1.2, "activity_qvalue"),
            ("network_coverage", 2.0, "network_coverage"),
        ],
    )
    def test_invalid_values_fail(self, tmp_path, field, value, message):
        frame = _activity_frame(**{field: value})
        with pytest.raises(PTMActivityContractError, match=message):
            load_ptm_activity_table(self._write(tmp_path, frame))

    def test_direction_score_sign_disagreement_fails(self, tmp_path):
        frame = _activity_frame(activity_score=-1.5, activity_direction="up")
        with pytest.raises(PTMActivityContractError, match="disagree"):
            load_ptm_activity_table(self._write(tmp_path, frame))

    def test_duplicate_regulator_contrast_method_fails(self, tmp_path):
        frame = pd.concat([_activity_frame(), _activity_frame()], ignore_index=True)
        with pytest.raises(PTMActivityContractError, match="duplicates"):
            load_ptm_activity_table(self._write(tmp_path, frame))


class TestActivitiesForPropagation:
    def test_selects_primary_method_only(self):
        frame = pd.concat(
            [
                _activity_frame(),
                _activity_frame(regulator_id="CDK5", method="PhosR"),
                _activity_frame(regulator_id="MAPK", condition_or_contrast="other-contrast"),
            ],
            ignore_index=True,
        )
        activities = activities_for_propagation(frame, method="KSTAR", condition_or_contrast="disease-minus-normal")
        assert set(activities) == {"GSK3B"}

    def test_unknown_method_fails(self):
        frame = _activity_frame()
        with pytest.raises(PTMActivityContractError, match="no rows for primary method"):
            activities_for_propagation(frame, method="PhosR")

    def test_unknown_contrast_fails(self):
        frame = _activity_frame()
        with pytest.raises(PTMActivityContractError, match="no rows for contrast"):
            activities_for_propagation(frame, method="KSTAR", condition_or_contrast="missing")


class TestLoadGeneMap:
    def test_conflicting_entries_fail(self, tmp_path):
        path = tmp_path / "gene_map.tsv"
        path.write_text(
            "protein_id\tgene_symbol\temsembl_dummy\n".replace("emsembl_dummy", "ensembl_id")
            + "P12345\tGSK3B\tENSG00000082701\n"
            + "P12345\tGSK3B\tENSG00000186868\n",
            encoding="utf-8",
        )
        with pytest.raises(PTMActivityContractError, match="conflicting"):
            load_gene_map(path)

    def test_versioned_ensembl_is_canonicalized(self, tmp_path):
        path = tmp_path / "gene_map.tsv"
        path.write_text(
            "protein_id\tgene_symbol\temsembl_dummy\n".replace("emsembl_dummy", "ensembl_id")
            + "P12345\tGSK3B\tENSG00000082701.5\n",
            encoding="utf-8",
        )
        mapping = load_gene_map(path)
        assert mapping["P12345"] == ("GSK3B", "ENSG00000082701")
