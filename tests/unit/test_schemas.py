import pytest

from src.data.schemas import PTMSite, PTMRecord, validate_protein_data


def test_ptm_site_to_dict_round_trip():
    site = PTMSite(position=10, type="phosphorylation", amino_acid="S")
    data = site.to_dict()
    assert data == {"position": 10, "type": "phosphorylation", "amino_acid": "S"}

    restored = PTMSite.from_dict(data)
    assert restored == site


def test_ptm_site_from_dict_without_amino_acid():
    site = PTMSite.from_dict({"position": 5, "type": "acetylation"})
    assert site.position == 5
    assert site.type == "acetylation"
    assert site.amino_acid is None


def test_ptm_record_to_dict_round_trip():
    record = PTMRecord(
        protein_accession="P12345",
        position=42,
        ptm_type="ubiquitination",
        amino_acid="K",
        confidence=0.95,
        source="test",
    )
    data = record.to_dict()
    restored = PTMRecord.from_dict(data)
    assert restored == record


def test_ptm_record_from_dict_with_none_confidence():
    record = PTMRecord.from_dict(
        {"protein_accession": "P12345", "position": 7, "ptm_type": "methylation"}
    )
    assert record.confidence is None
    assert record.amino_acid is None
    assert record.source is None


def test_validate_protein_data_accepts_valid_input():
    assert validate_protein_data(
        {
            "accession": "P12345",
            "sequence": "ACDEFG",
            "ptm_sites": [{"position": 2, "type": "phosphorylation"}],
        }
    )


def test_validate_protein_data_accepts_no_ptm_sites():
    assert validate_protein_data({"accession": "P12345", "sequence": "ACDEFG"})


def test_validate_protein_data_rejects_missing_accession():
    assert not validate_protein_data({"sequence": "ACDEFG"})


def test_validate_protein_data_rejects_missing_sequence():
    assert not validate_protein_data({"accession": "P12345"})


def test_validate_protein_data_rejects_non_string_accession():
    assert not validate_protein_data({"accession": 123, "sequence": "ACDEFG"})


def test_validate_protein_data_rejects_invalid_ptm_position():
    assert not validate_protein_data(
        {
            "accession": "P12345",
            "sequence": "ACDEFG",
            "ptm_sites": [{"position": 0, "type": "phosphorylation"}],
        }
    )


def test_validate_protein_data_rejects_missing_ptm_type():
    assert not validate_protein_data(
        {
            "accession": "P12345",
            "sequence": "ACDEFG",
            "ptm_sites": [{"position": 2, "type": ""}],
        }
    )


def test_validate_protein_data_rejects_malformed_ptm_sites():
    assert not validate_protein_data(
        {
            "accession": "P12345",
            "sequence": "ACDEFG",
            "ptm_sites": {"position": 2, "type": "phosphorylation"},
        }
    )
