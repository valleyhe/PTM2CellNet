"""Unit tests for the PDC API client (F-03 / TD-H3).

These tests never hit the network — they inject a fake ``requests.Session``
that records the GraphQL payload and returns canned responses. The
real-network acceptance test lives in ``tests/real_assets/``.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.analysis.pdc_client import PDCClient, PDCAPIError


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    """Records the last POST and returns a queued response."""

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_url: str | None = None
        self.last_json: dict[str, Any] | None = None
        self.last_headers: dict[str, str] | None = None

    def post(self, url: str, json: dict[str, Any], timeout: float, headers: dict[str, str]):
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        return self._response


def _make_study_payload(study_id: str = "abc-123") -> dict[str, Any]:
    return {
        "data": {
            "study": {
                "study_id": study_id,
                "study_submitter_id": "CPTAC-BRCA",
                "study_name": "CPTAC Breast Cancer",
                "disease_type": "Breast Invasive Carcinoma",
                "primary_site": "Breast",
                "files": [
                    {
                        "file_id": "file-1",
                        "file_name": "phospho.tsv",
                        "file_type": "TSV",
                        "data_category": "Phosphoproteomics",
                        "download_url": "https://example.org/phospho.tsv",
                    }
                ],
            }
        }
    }


def test_get_study_returns_normalized_pdcstudy():
    fake = _FakeSession(_FakeResponse(200, _make_study_payload("s1")))
    client = PDCClient(session=fake)
    study = client.get_study("s1")
    assert study.study_id == "s1"
    assert study.study_submitter_id == "CPTAC-BRCA"
    assert study.disease_type == "Breast Invasive Carcinoma"
    assert len(study.files) == 1
    assert study.files[0]["file_name"] == "phospho.tsv"


def test_get_study_sends_graphql_query_with_variables():
    fake = _FakeSession(_FakeResponse(200, _make_study_payload()))
    client = PDCClient(session=fake)
    client.get_study("uuid-1")
    assert fake.last_url == "https://pdc.cancer.gov/graphql"
    assert fake.last_json is not None
    assert "query" in fake.last_json
    assert "variables" in fake.last_json
    assert fake.last_json["variables"] == {"study_id": "uuid-1"}
    # The query must ask for the study(...) field we parse.
    assert "study(study_id:" in fake.last_json["query"]
    assert "files" in fake.last_json["query"]


def test_get_study_raises_when_study_missing():
    payload = {"data": {"study": None}}
    fake = _FakeSession(_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="no study"):
        client.get_study("missing")


def test_get_study_raises_on_http_error():
    fake = _FakeSession(_FakeResponse(503, {}))
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="HTTP 503"):
        client.get_study("s1")


def test_get_study_raises_on_graphql_errors():
    payload = {"errors": [{"message": "Unauthorized"}], "data": None}
    fake = _FakeSession(_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="Unauthorized"):
        client.get_study("s1")


def test_get_study_raises_when_data_key_missing():
    payload = {"unexpected": 1}
    fake = _FakeSession(_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="missing 'data'"):
        client.get_study("s1")


def test_get_study_files_filters_by_file_type():
    fake = _FakeSession(_FakeResponse(200, _make_study_payload()))
    client = PDCClient(session=fake)
    files = client.get_study_files("s1", file_type="tsv")
    assert len(files) == 1
    assert files[0]["file_type"] == "TSV"
    # Filter that excludes everything returns empty list, not error.
    none_files = client.get_study_files("s1", file_type="bogus")
    assert none_files == []


def test_list_studies_truncates_to_limit():
    studies_raw = [
        {"study_id": f"s{i}", "study_submitter_id": f"sub-{i}",
         "study_name": f"name-{i}", "disease_type": "X", "primary_site": "Y"}
        for i in range(10)
    ]
    payload = {"data": {"allStudies": studies_raw}}
    fake = _FakeSession(_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    out = client.list_studies(limit=3)
    assert len(out) == 3
    assert [s.study_id for s in out] == ["s0", "s1", "s2"]


def test_session_post_raises_pdcapierror_on_transport_error(monkeypatch):
    import requests

    class _BoomSession:
        def post(self, *a, **kw):
            raise requests.ConnectionError("boom")

    client = PDCClient(session=_BoomSession())  # type: ignore[arg-type]
    with pytest.raises(PDCAPIError, match="transport failed"):
        client.get_study("s1")
