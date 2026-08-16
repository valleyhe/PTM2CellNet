"""Unit tests for public PTM database loaders."""

from __future__ import annotations

import gzip
from io import BytesIO, StringIO

import pandas as pd
import requests

from src.data.loaders import DataLoader


PHOSPHOSITEPLUS_COLUMNS = [
    "protein_accession",
    "gene",
    "position",
    "ptm_type",
    "amino_acid",
    "confidence",
    "source",
]

DBPTM_COLUMNS = [
    "protein_accession",
    "position",
    "ptm_type",
    "amino_acid",
    "source",
]

UNIPROT_COLUMNS = ["sequence", "gene_symbol", "accession"]


class _FakeResponse:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        for index in range(0, len(self.payload), chunk_size):
            yield self.payload[index:index + chunk_size]


def test_load_from_phosphositeplus_local_gzip(tmp_path) -> None:
    loader = DataLoader()
    content = StringIO(
        "PROTEIN\tACC_ID\tGENE\tMOD_RSD\tSITE_GRP_ID\tORGANISM\n"
        "AKT1\tP31749\tAKT1\tS473-p\t1001\thuman\n"
        "AKT1\tP31749\tAKT1\tT308-p\t1002\tmouse\n"
        "STAT3\tP40763\tSTAT3\tY705-p\t1003\tHuman\n"
    )
    file_path = tmp_path / "Phosphorylation_site_dataset.gz"
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        handle.write(content.getvalue().encode("utf-8"))
    file_path.write_bytes(buffer.getvalue())

    df = loader.load_from_phosphositeplus(str(file_path), organism="human")

    assert list(df.columns) == PHOSPHOSITEPLUS_COLUMNS
    assert df.to_dict("records") == [
        {
            "protein_accession": "P31749",
            "gene": "AKT1",
            "position": 473,
            "ptm_type": "phosphorylation",
            "amino_acid": "S",
            # 测试数据仅含 SITE_GRP_ID（site group id），并非真实置信度。
            # 修复后 loaders 不再把 SITE_GRP_ID 误作 confidence，故 confidence 为缺失值。
            "confidence": None,
            "source": "PhosphoSitePlus",
        },
        {
            "protein_accession": "P40763",
            "gene": "STAT3",
            "position": 705,
            "ptm_type": "phosphorylation",
            "amino_acid": "Y",
            "confidence": None,
            "source": "PhosphoSitePlus",
        },
    ]


def test_load_from_phosphositeplus_download_url(tmp_path, monkeypatch) -> None:
    loader = DataLoader({"paths": {"data_raw": str(tmp_path)}})
    content = StringIO(
        "PROTEIN\tACC_ID\tGENE\tMOD_RSD\tSITE_GRP_ID\tORGANISM\n"
        "MAPK1\tP28482\tMAPK1\tT185-p\t2001\thuman\n"
    )
    payload = gzip.compress(content.getvalue().encode("utf-8"))

    def fake_get(url: str, stream: bool = False, timeout: int = 0):
        assert url == "https://www.phosphosite.org/downloads/Phosphorylation_site_dataset.gz"
        assert stream is True
        assert timeout == 30
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data.loaders.base.requests.get", fake_get)

    df = loader.load_from_phosphositeplus(
        "https://www.phosphosite.org/downloads/Phosphorylation_site_dataset.gz"
    )

    assert list(df.columns) == PHOSPHOSITEPLUS_COLUMNS
    assert df.loc[0, "protein_accession"] == "P28482"
    assert df.loc[0, "position"] == 185
    assert df.loc[0, "amino_acid"] == "T"
    # 测试数据仅含 SITE_GRP_ID（非真实置信度），修复后 confidence 为缺失值。
    assert pd.isna(df.loc[0, "confidence"])


def test_load_from_phosphositeplus_network_failure_returns_empty(monkeypatch) -> None:
    loader = DataLoader()

    def fake_get(url: str, stream: bool = False, timeout: int = 0):
        raise requests.RequestException("network down")

    monkeypatch.setattr("src.data.loaders.base.requests.get", fake_get)

    df = loader.load_from_phosphositeplus(
        "https://www.phosphosite.org/downloads/Phosphorylation_site_dataset.gz"
    )

    assert list(df.columns) == PHOSPHOSITEPLUS_COLUMNS
    assert df.empty


def test_load_from_dbptm_parses_delimited_file(tmp_path) -> None:
    loader = DataLoader()
    content = StringIO(
        "UniProtKB Accession\tPosition\tResidue\n"
        "P31749\t473\tS\n"
        "Q9Y243\t321\tT\n"
    )
    file_path = tmp_path / "dbptm.tsv"
    file_path.write_text(content.getvalue(), encoding="utf-8")

    df = loader.load_from_dbptm(str(file_path))

    assert list(df.columns) == DBPTM_COLUMNS
    assert df.to_dict("records") == [
        {
            "protein_accession": "P31749",
            "position": 473,
            "ptm_type": "phosphorylation",
            "amino_acid": "S",
            "source": "dbPTM",
        },
        {
            "protein_accession": "Q9Y243",
            "position": 321,
            "ptm_type": "phosphorylation",
            "amino_acid": "T",
            "source": "dbPTM",
        },
    ]


def test_load_from_cplm_parses_tab_delimited_download(tmp_path) -> None:
    loader = DataLoader()
    content = StringIO(
        "UniProt Accession\tPosition\tAmino Acid\n"
        "P11142\t138\tC\n"
        "Q96HC4\t42\tC\n"
    )
    file_path = tmp_path / "cplm.tsv"
    file_path.write_text(content.getvalue(), encoding="utf-8")

    df = loader.load_from_cplm(str(file_path))

    assert list(df.columns) == DBPTM_COLUMNS
    assert df.to_dict("records") == [
        {
            "protein_accession": "P11142",
            "position": 138,
            "ptm_type": "cysteine",
            "amino_acid": "C",
            "source": "CPLM",
        },
        {
            "protein_accession": "Q96HC4",
            "position": 42,
            "ptm_type": "cysteine",
            "amino_acid": "C",
            "source": "CPLM",
        },
    ]


def test_load_from_epsd_parses_tab_delimited_download(tmp_path) -> None:
    loader = DataLoader()
    content = StringIO(
        "EPSD ID\tUniProt ID\tAA\tPosition\tSource\tReference\n"
        "EP0000005\tO00110\tY\t29\tExp\t15174125\n"
        "EP0000006\tP31749\tS\t473\tExp\t28150246\n"
    )
    file_path = tmp_path / "epsd.tsv"
    file_path.write_text(content.getvalue(), encoding="utf-8")

    df = loader.load_from_epsd(str(file_path))

    assert list(df.columns) == DBPTM_COLUMNS
    assert df.to_dict("records") == [
        {
            "protein_accession": "O00110",
            "position": 29,
            "ptm_type": "phosphorylation",
            "amino_acid": "Y",
            "source": "EPSD",
        },
        {
            "protein_accession": "P31749",
            "position": 473,
            "ptm_type": "phosphorylation",
            "amino_acid": "S",
            "source": "EPSD",
        },
    ]


def test_load_from_uniprot_uses_api_and_cache(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    loader = DataLoader({"paths": {"data_raw": str(raw_dir)}})
    calls: list[str] = []

    class _UniProtResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "primaryAccession": "P04637",
                "sequence": {"value": "MEEPQSDPSV"},
                "genes": [{"geneName": {"value": "TP53"}}],
            }

    def fake_get(url: str, timeout: int = 0):
        calls.append(url)
        assert url == "https://rest.uniprot.org/uniprotkb/P04637.json"
        assert timeout == 15
        return _UniProtResponse()

    monkeypatch.setattr("src.data.loaders.uniprot_loader.requests.get", fake_get)

    first = loader.load_from_uniprot(["P04637"])
    second = loader.load_from_uniprot(["P04637"])

    assert list(first.columns) == UNIPROT_COLUMNS
    assert first.to_dict("records") == [
        {
            "sequence": "MEEPQSDPSV",
            "gene_symbol": "TP53",
            "accession": "P04637",
        }
    ]
    assert second.to_dict("records") == first.to_dict("records")
    assert calls == ["https://rest.uniprot.org/uniprotkb/P04637.json"]
    assert (tmp_path / "data" / "uniprot_cache" / "P04637.json").exists()


def test_load_from_uniprot_network_failure_returns_empty(monkeypatch) -> None:
    loader = DataLoader()

    def fake_get(url: str, timeout: int = 0):
        raise requests.RequestException("network down")

    monkeypatch.setattr("src.data.loaders.uniprot_loader.requests.get", fake_get)

    df = loader.load_from_uniprot(["P04637"])

    assert list(df.columns) == UNIPROT_COLUMNS
    assert df.empty
