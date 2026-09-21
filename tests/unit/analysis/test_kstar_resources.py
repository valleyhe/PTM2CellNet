"""Tests for isolated KSTAR resource hash pinning."""

from __future__ import annotations

import json
from email.message import EmailMessage
from pathlib import Path
from urllib.error import HTTPError

import pytest

from src.analysis.kstar_resources import (
    KSTAR_PINNED_VERSION,
    KSTAR_RESOURCE_HASH_SCHEMA,
    KSTAR_ST_UNIQUE_NETWORK_ID,
    KSTAR_UNIQUE_REFERENCE_ID,
    KSTAR_Y_UNIQUE_NETWORK_ID,
    KSTARResourceError,
    default_resource_hash_manifest,
    download_url_to_file,
    install_kstar_networks,
    load_resource_hash_manifest,
    sha256_file,
    verify_kstar_network_dir,
    verify_kstar_resource_files,
    verify_pinned_python_packages,
)


def _write_network_dir(root: Path, *, n_files: int = 50, file_bytes: int = 16) -> Path:
    """Write a compliance-shaped ST/Y Default network directory."""

    for kind, network_id in (("ST", KSTAR_ST_UNIQUE_NETWORK_ID), ("Y", KSTAR_Y_UNIQUE_NETWORK_ID)):
        individual = root / kind / "Default" / "INDIVIDUAL_NETWORKS"
        individual.mkdir(parents=True)
        for index in range(n_files):
            (individual / f"NetworKIN_{index}_{kind}_compendia_2500_limit_20.tsv").write_bytes(b"x" * file_bytes)
        (root / kind / "Default" / "RUN_INFORMATION.txt").write_text(
            f"Unique Network ID: {network_id}\nUnique Reference ID: {KSTAR_UNIQUE_REFERENCE_ID}\n",
            encoding="utf-8",
        )
    return root


def test_resource_hash_manifest_is_frozen():
    payload = load_resource_hash_manifest()
    assert payload["schema_version"] == KSTAR_RESOURCE_HASH_SCHEMA
    assert payload["kstar_version"] == KSTAR_PINNED_VERSION
    assert payload["unique_reference_id"] == KSTAR_UNIQUE_REFERENCE_ID
    files = payload["files"]
    assert set(files) >= {
        "HumanPhosphoProteome.csv",
        "humanProteome.fasta",
        "reference_info.json",
        "citations_compendia.txt",
        "readme.txt",
    }
    assert Path(default_resource_hash_manifest()).is_file()


def test_verify_kstar_resource_files_matches_and_detects_drift(tmp_path):
    source = default_resource_hash_manifest()
    payload = json.loads(source.read_text(encoding="utf-8"))
    resource_dir = tmp_path / "RESOURCE_FILES"
    resource_dir.mkdir()
    for name, spec in payload["files"].items():
        content = f"fixture-{name}\n".encode("utf-8")
        path = resource_dir / name
        path.write_bytes(content)
        spec["sha256"] = sha256_file(path)
        spec["bytes"] = path.stat().st_size
    (resource_dir / "reference_info.json").write_text(
        json.dumps({"unique_reference_id": KSTAR_UNIQUE_REFERENCE_ID}) + "\n",
        encoding="utf-8",
    )
    payload["files"]["reference_info.json"]["sha256"] = sha256_file(resource_dir / "reference_info.json")
    payload["files"]["reference_info.json"]["bytes"] = (resource_dir / "reference_info.json").stat().st_size
    manifest = tmp_path / "resource_hashes.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    verified = verify_kstar_resource_files(resource_dir, manifest_path=manifest)
    assert "HumanPhosphoProteome.csv" in verified

    (resource_dir / "HumanPhosphoProteome.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(KSTARResourceError, match="mismatch"):
        verify_kstar_resource_files(resource_dir, manifest_path=manifest)


def test_verify_pinned_packages_and_network_reference(tmp_path):
    verify_pinned_python_packages(
        {
            "kstar": "1.2.0",
            "pandas": "3.0.6",
            "numpy": "2.5.3",
            "scipy": "1.18.1",
            "requests": "2.34.2",
            "tqdm": "4.70.1",
            "matplotlib": "3.11.2",
            "seaborn": "0.13.2",
            "fpdf2": "2.8.8",
        }
    )
    with pytest.raises(KSTARResourceError, match="pandas"):
        verify_pinned_python_packages({"kstar": "1.2.0", "pandas": "2.0.0"})

    network = _write_network_dir(tmp_path / "NETWORKS")
    audit = verify_kstar_network_dir(network)
    assert audit.reference_ids["ST"] == KSTAR_UNIQUE_REFERENCE_ID
    assert audit.network_ids["Y"] == KSTAR_Y_UNIQUE_NETWORK_ID
    assert audit.n_files == {"ST": 50, "Y": 50}
    assert audit.total_bytes["ST"] == 50 * 16
    (network / "ST" / "Default" / "RUN_INFORMATION.txt").write_text(
        "Unique Network ID: deadbeef\nUnique Reference ID: deadbeef\n", encoding="utf-8"
    )
    with pytest.raises(KSTARResourceError, match="unique_reference_id"):
        verify_kstar_network_dir(network)
    _write_network_dir(tmp_path / "NETWORKS2")
    drift = tmp_path / "NETWORKS2"
    (drift / "ST" / "Default" / "RUN_INFORMATION.txt").write_text(
        f"Unique Network ID: {'0' * 64}\nUnique Reference ID: {KSTAR_UNIQUE_REFERENCE_ID}\n", encoding="utf-8"
    )
    with pytest.raises(KSTARResourceError, match="Unique Network ID"):
        verify_kstar_network_dir(drift)


def test_network_verifier_rejects_empty_partial_and_truncated_dirs(tmp_path):
    # Empty INDIVIDUAL_NETWORKS directories (the old false-pass) must fail.
    empty = tmp_path / "empty"
    for kind in ("ST", "Y"):
        default = empty / kind / "Default"
        (default / "INDIVIDUAL_NETWORKS").mkdir(parents=True)
        (default / "RUN_INFORMATION.txt").write_text(
            f"Unique Network ID: {KSTAR_ST_UNIQUE_NETWORK_ID if kind == 'ST' else KSTAR_Y_UNIQUE_NETWORK_ID}\n"
            f"Unique Reference ID: {KSTAR_UNIQUE_REFERENCE_ID}\n",
            encoding="utf-8",
        )
    with pytest.raises(KSTARResourceError, match="exactly 50"):
        verify_kstar_network_dir(empty)

    partial = _write_network_dir(tmp_path / "partial")
    next(iter((partial / "ST" / "Default" / "INDIVIDUAL_NETWORKS").iterdir())).unlink()
    with pytest.raises(KSTARResourceError, match="exactly 50"):
        verify_kstar_network_dir(partial)

    truncated = _write_network_dir(tmp_path / "truncated")
    zero = truncated / "Y" / "Default" / "INDIVIDUAL_NETWORKS" / "NetworKIN_0_Y_compendia_2500_limit_20.tsv"
    zero.write_bytes(b"")
    with pytest.raises(KSTARResourceError, match="empty files"):
        verify_kstar_network_dir(truncated)


def test_install_reuses_package_adjacent_network_without_download(tmp_path, monkeypatch):
    package_dir = tmp_path / "site-packages" / "kstar"
    network = _write_network_dir(package_dir / "NETWORKS")

    def fail_download(*args, **kwargs):
        raise AssertionError("an existing package-adjacent NETWORKS/ must be reused")

    monkeypatch.setattr("src.analysis.kstar_resources.download_url_to_file", fail_download)
    assert install_kstar_networks(package_dir) == network.resolve()


def test_download_url_rejects_http_202_and_nongzip(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, status: int, body: bytes):
            self.status = status
            self._body = body

        def read(self, size: int = -1) -> bytes:
            if size == -1:
                chunk, self._body = self._body, b""
                return chunk
            chunk, self._body = self._body[:size], self._body[size:]
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    dest = tmp_path / "NETWORKS.tar.gz"

    def fake_202(request, timeout=45):
        return FakeResponse(202, b"")

    monkeypatch.setattr("src.analysis.kstar_resources.urlopen", fake_202)
    with pytest.raises(KSTARResourceError, match="HTTP 202"):
        download_url_to_file("https://example.invalid/networks", dest)

    def fake_html(request, timeout=45):
        return FakeResponse(200, b"<!doctype html>not gzip")

    monkeypatch.setattr("src.analysis.kstar_resources.urlopen", fake_html)
    with pytest.raises(KSTARResourceError, match="gzip"):
        download_url_to_file("https://example.invalid/networks", dest)

    def fake_http_error(request, timeout=45):
        raise HTTPError("https://example.invalid/networks", 403, "Forbidden", EmailMessage(), None)

    monkeypatch.setattr("src.analysis.kstar_resources.urlopen", fake_http_error)
    with pytest.raises(KSTARResourceError, match="HTTP 403"):
        download_url_to_file("https://example.invalid/networks", dest)
