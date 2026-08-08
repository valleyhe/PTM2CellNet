"""Branch-focused tests for local CSV/FASTA/JSON loading contracts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.loaders import DataLoader


@pytest.mark.parametrize("method", ["load_from_csv", "load_from_fasta", "load_from_json"])
def test_local_loaders_reject_missing_files(tmp_path, method):
    loader = DataLoader()

    with pytest.raises(FileNotFoundError):
        getattr(loader, method)(str(tmp_path / "missing.data"))


def test_csv_and_hdf_suffix_loading(tmp_path, monkeypatch):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("id,value\na,1\n", encoding="utf-8")
    loader = DataLoader()

    assert loader.load_from_csv(str(csv_path)).to_dict("records") == [{"id": "a", "value": 1}]

    hdf_path = tmp_path / "records.h5"
    hdf_path.touch()
    expected = pd.DataFrame({"id": ["hdf"]})
    monkeypatch.setattr(pd, "read_hdf", lambda path: expected)
    assert loader.load_from_csv(str(hdf_path)) is expected


@pytest.mark.parametrize("fallback", [pd.DataFrame({"id": ["frame"]}), {"only": pd.DataFrame({"id": ["dict"]})}])
def test_hdf_loading_falls_back_to_shared_safe_loader(tmp_path, monkeypatch, fallback):
    path = tmp_path / "records.h5"
    path.touch()
    monkeypatch.setattr(pd, "read_hdf", lambda path: (_ for _ in ()).throw(ValueError("no key")))
    monkeypatch.setattr("src.utils.io.load_hdf5", lambda path: fallback)

    loaded = DataLoader().load_from_csv(str(path))

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.iloc[0, 0] in {"frame", "dict"}


def test_hdf_fallback_reraises_original_error_for_ambiguous_payload(tmp_path, monkeypatch):
    path = tmp_path / "records.h5"
    path.touch()
    monkeypatch.setattr(pd, "read_hdf", lambda path: (_ for _ in ()).throw(ValueError("no key")))
    monkeypatch.setattr("src.utils.io.load_hdf5", lambda path: {"one": [1], "two": [2]})

    with pytest.raises(ValueError, match="no key"):
        DataLoader().load_from_csv(str(path))


def test_hdf_fallback_reraises_original_error_for_non_dataframe_value(tmp_path, monkeypatch):
    path = tmp_path / "records.h5"
    path.touch()
    monkeypatch.setattr(pd, "read_hdf", lambda path: (_ for _ in ()).throw(ValueError("no key")))
    monkeypatch.setattr("src.utils.io.load_hdf5", lambda path: {"only": [1, 2]})

    with pytest.raises(ValueError, match="no key"):
        DataLoader().load_from_csv(str(path))


def test_hdf_table_rejects_series_payload(tmp_path, monkeypatch):
    path = tmp_path / "records.h5"
    path.touch()
    monkeypatch.setattr(pd, "read_hdf", lambda path: pd.Series([1, 2]))

    with pytest.raises(TypeError, match="must contain a pandas DataFrame"):
        DataLoader().load_from_csv(str(path))


def test_fasta_loader_cleans_valid_sequences_and_skips_invalid(tmp_path):
    path = tmp_path / "proteins.fasta"
    path.write_text(">valid\nacd efg\n>invalid\nACDZ\n", encoding="utf-8")

    sequences = DataLoader().load_from_fasta(str(path))

    assert sequences == {"valid": "ACDEFG"}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({"id": "one"}, [{"id": "one"}]), ([{"id": "one"}, {"id": "two"}], [{"id": "one"}, {"id": "two"}])],
)
def test_json_loader_normalizes_object_and_object_array(tmp_path, payload, expected):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert DataLoader().load_from_json(str(path)) == expected


@pytest.mark.parametrize("payload", [[{"id": 1}, "bad"], 3, "bad"])
def test_json_loader_rejects_non_object_records(tmp_path, payload):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError):
        DataLoader().load_from_json(str(path))


def test_sample_data_respects_configured_bounds_and_empty_request():
    loader = DataLoader(
        {
            "data": {
                "min_sequence_length": 4,
                "max_sequence_length": 4,
                "ptm_types": ["phosphorylation"],
                "cell_states": ["quiescence"],
            }
        }
    )

    generated = loader.load_sample_data(5)

    assert generated["sequence"].str.len().eq(4).all()
    assert generated["cell_state"].eq("quiescence").all()
    for sequence, encoded_sites in zip(generated["sequence"], generated["ptm_sites"], strict=True):
        for site in json.loads(encoded_sites):
            assert 1 <= site["position"] <= len(sequence)
            assert site["type"] == "phosphorylation"
            assert site["amino_acid"] == sequence[site["position"] - 1]
    assert DataLoader().load_sample_data(0).empty


def test_combined_loader_handles_empty_single_merge_and_concat(tmp_path):
    loader = DataLoader()
    assert loader.load_combined_data().empty

    sequence_path = tmp_path / "sequence.csv"
    sequence_path.write_text("id,sequence\np1,ACDE\n", encoding="utf-8")
    single = loader.load_combined_data(sequence_file=str(sequence_path))
    assert single.to_dict("records") == [{"id": "p1", "sequence": "ACDE"}]

    ptm_path = tmp_path / "ptm.csv"
    ptm_path.write_text("id,position\np1,2\n", encoding="utf-8")
    merged = loader.load_combined_data(sequence_file=str(sequence_path), ptm_file=str(ptm_path))
    assert merged.to_dict("records") == [{"id": "p1", "sequence": "ACDE", "position": 2}]

    label_path = tmp_path / "labels.csv"
    label_path.write_text("state\nactive\n", encoding="utf-8")
    concatenated = loader.load_combined_data(sequence_file=str(sequence_path), label_file=str(label_path))
    assert concatenated.loc[0].to_dict() == {"id": "p1", "sequence": "ACDE", "state": "active"}
