"""Unit tests for the IBD metadata contract."""

from pathlib import Path

import pandas as pd

from src.analysis.ibd_dataset import (
    REQUIRED_METADATA_COLUMNS,
    build_metadata_master,
    load_metadata,
    parse_geo_family_soft,
    write_metadata_outputs,
)


def test_parse_geo_family_soft_extracts_accession_and_characteristics(tmp_path: Path):
    path = tmp_path / "sample.family.soft"
    path.write_text(
        "!Sample_title = S1\n"
        "!Sample_geo_accession = GSM1\n"
        "!Sample_characteristics_ch1 = disease: Healthy\n"
        "!Sample_characteristics_ch1 = site: Ascending_Colon\n"
        "!Sample_title = S2\n"
        "!Sample_geo_accession = GSM2\n"
        "!Sample_characteristics_ch1 = disease: UC\n",
        encoding="utf-8",
    )

    frame = parse_geo_family_soft(path)

    assert frame.loc[0, ["GSM", "disease", "site"]].to_dict() == {
        "GSM": "GSM1",
        "disease": "Healthy",
        "site": "Ascending_Colon",
    }
    assert frame.loc[1, ["GSM", "disease"]].to_dict() == {
        "GSM": "GSM2",
        "disease": "UC",
    }
    assert pd.isna(frame.loc[1, "site"])


def test_real_ibd_metadata_has_documented_inclusion_counts():
    frame = build_metadata_master("data/raw")

    assert len(frame) == 331
    assert set(REQUIRED_METADATA_COLUMNS).issubset(frame.columns)
    assert frame.groupby("dataset").size().to_dict() == {
        "GSE214695": 18,
        "GSE231993": 12,
        "GSE266616": 85,
        "GSE282122": 216,
    }
    assert int(frame["include_atlas"].sum()) == 126
    assert int(frame["include_primary_DE"].sum()) == 91
    assert int(
        ((frame["dataset"] == "GSE282122") & frame["include_atlas"] & (frame["intestinal_region"] == "colon")).sum()
    ) == 96
    assert not frame["GSM"].duplicated().any()


def test_metadata_outputs_round_trip(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                column: (True if column.startswith("include_") else "x")
                for column in REQUIRED_METADATA_COLUMNS
            }
        ]
    )
    paths = write_metadata_outputs(frame, tmp_path)
    loaded = load_metadata(paths[0])
    assert bool(loaded.loc[0, "include_atlas"])
    assert bool(loaded.loc[0, "include_primary_DE"])
