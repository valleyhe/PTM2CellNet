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
    def __init__(self, status_code: int, payload: dict[str, Any], text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeDownloadResponse:
    """Streaming GET response fake for ``download_file`` tests."""

    def __init__(self, status_code: int, body: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = body.decode("utf-8", errors="replace")[:200]

    def iter_content(self, chunk_size: int = 1):
        # Split body into chunk_size pieces so the cap path is exercised.
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class _FakeSession:
    """Records the last POST/GET and returns queued responses.

    ``response`` is sticky — every POST returns it. For more control over
    multiple successive POSTs (e.g. distinct responses), pass
    ``post_responses`` (consumed in order; falls back to the sticky
    ``response`` once exhausted). ``get_response`` is returned for GET
    calls used by ``download_file``.
    """

    def __init__(
        self,
        response: _FakeResponse | None = None,
        *,
        post_responses: list[_FakeResponse] | None = None,
        get_response: Any = None,
    ):
        self._sticky = response
        self._post_queue = list(post_responses) if post_responses else []
        self._get_response = get_response
        self.last_url: str | None = None
        self.last_json: dict[str, Any] | None = None
        self.last_headers: dict[str, str] | None = None
        # Expose for assertions.
        self.post_calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.get_calls: list[tuple[str, dict[str, str]]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float, headers: dict[str, str]):
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        self.post_calls.append((url, json, headers))
        if self._post_queue:
            return self._post_queue.pop(0)
        if self._sticky is not None:
            return self._sticky
        raise AssertionError("FakeSession.post called but no responses queued")

    def get(self, url: str, stream: bool = False, timeout: float = 30.0, headers: dict[str, str] | None = None):
        self.last_url = url
        self.last_headers = headers or {}
        self.get_calls.append((url, headers or {}))
        if self._get_response is None:
            raise AssertionError("FakeSession.get called but no get_response set")
        return self._get_response


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
        {
            "study_id": f"s{i}",
            "study_submitter_id": f"sub-{i}",
            "study_name": f"name-{i}",
            "disease_type": "X",
            "primary_site": "Y",
        }
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


# ---------------------------------------------------------------------------
# F-03 v17: get_file_url + download_file
# ---------------------------------------------------------------------------


def _make_file_payload(download_url: str = "https://pdc.cancer.gov/file.something.tsv") -> dict[str, Any]:
    return {
        "data": {
            "file": {
                "file_id": "file-1",
                "file_name": "phospho.tsv",
                "file_type": "TSV",
                "file_location": "/data/pdc/phospho.tsv",
                "downloadUrl": download_url,
            }
        }
    }


def test_get_file_url_returns_download_url():
    fake = _FakeSession(response=_FakeResponse(200, _make_file_payload()))
    client = PDCClient(session=fake)
    url = client.get_file_url("file-1")
    assert url == "https://pdc.cancer.gov/file.something.tsv"
    # The GraphQL query must reference file(file_id: ...) and downloadUrl.
    assert fake.last_json is not None
    assert fake.last_json["variables"] == {"file_id": "file-1"}
    assert "file(file_id:" in fake.last_json["query"]
    assert "downloadUrl" in fake.last_json["query"]


def test_get_file_url_falls_back_to_file_location_when_no_download_url():
    payload = {
        "data": {
            "file": {
                "file_id": "file-2",
                "file_name": "x.tsv",
                "file_type": "TSV",
                "file_location": "https://pdc.cancer.gov/loc/x.tsv",
                "downloadUrl": None,
            }
        }
    }
    fake = _FakeSession(response=_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    url = client.get_file_url("file-2")
    assert url == "https://pdc.cancer.gov/loc/x.tsv"


def test_get_file_url_raises_when_file_missing():
    payload = {"data": {"file": None}}
    fake = _FakeSession(response=_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="no file"):
        client.get_file_url("missing")


def test_get_file_url_raises_when_no_url_field():
    payload = {"data": {"file": {"file_id": "x", "file_name": "y.tsv"}}}
    fake = _FakeSession(response=_FakeResponse(200, payload))
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="no downloadUrl"):
        client.get_file_url("x")


def test_download_file_streams_to_target_dir(tmp_path):
    """Happy path: GET streamed, file written, atomic rename to final name."""
    body = b"gene\tpos\tscore\nP53\t100S\t0.9\n"
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload()),
        get_response=_FakeDownloadResponse(200, body, headers={"content-length": str(len(body))}),
    )
    client = PDCClient(session=fake)
    out = client.download_file("file-1", tmp_path, filename="phospho.tsv")
    assert out == tmp_path / "phospho.tsv"
    assert out.read_bytes() == body
    # GET hit the resolved URL.
    assert len(fake.get_calls) == 1
    assert fake.get_calls[0][0] == "https://pdc.cancer.gov/file.something.tsv"


def test_download_file_rejects_non_pdc_host(monkeypatch, tmp_path):
    """Default allowlist is pdc.cancer.gov; foreign hosts must be refused."""
    monkeypatch.delenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", raising=False)
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload("https://evil.example.com/x.tsv")),
        get_response=_FakeDownloadResponse(200, b"x"),
    )
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match=r"is not 'pdc.cancer.gov'"):
        client.download_file("file-1", tmp_path)


def test_download_file_respects_operator_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "evil.example.com")
    body = b"x\n"
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload("https://evil.example.com/x.tsv")),
        get_response=_FakeDownloadResponse(200, body, headers={"content-length": "2"}),
    )
    client = PDCClient(session=fake)
    out = client.download_file("file-1", tmp_path)
    assert out.read_bytes() == body


def test_download_file_enforces_declared_size_cap(tmp_path):
    """Pre-check: declared Content-Length over the cap aborts before streaming."""
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload()),
        # Declare 1 TiB even though the body is tiny.
        get_response=_FakeDownloadResponse(200, b"x", headers={"content-length": str(1024**4)}),
    )
    client = PDCClient(session=fake, max_download_bytes=1024)
    with pytest.raises(PDCAPIError, match="declared size"):
        client.download_file("file-1", tmp_path)


def test_download_file_enforces_streamed_size_cap(tmp_path):
    """Runtime cap: server omits Content-Length, mid-stream bytes exceed cap."""
    body = b"a" * 50
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload()),
        # No content-length header.
        get_response=_FakeDownloadResponse(200, body, headers={}),
    )
    client = PDCClient(session=fake, max_download_bytes=10)
    with pytest.raises(PDCAPIError, match="exceeded 10 bytes"):
        client.download_file("file-1", tmp_path)
    # No partial file leaked.
    assert list(tmp_path.iterdir()) == []


def test_download_file_raises_on_http_error(tmp_path):
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload()),
        get_response=_FakeDownloadResponse(403, b"forbidden"),
    )
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="HTTP 403"):
        client.download_file("file-1", tmp_path)


def test_download_file_rejects_non_http_scheme(tmp_path):
    fake = _FakeSession(
        response=_FakeResponse(200, _make_file_payload("file:///etc/passwd")),
        get_response=_FakeDownloadResponse(200, b"x"),
    )
    client = PDCClient(session=fake)
    with pytest.raises(PDCAPIError, match="scheme"):
        client.download_file("file-1", tmp_path)
