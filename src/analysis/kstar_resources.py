"""Pin verification for the isolated KSTAR environment.

This module does not import KSTAR.  It hashes the PhosphoSitePlus-derived
companion files shipped with ``kstar==1.2.0`` and optionally downloads the
official ST/Y network archive.  Empty HTTP 202 bodies are not a network asset.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

KSTAR_RESOURCE_HASH_SCHEMA = "ptm2cellnet.kstar-resource-hashes/v1"
KSTAR_PINNED_VERSION = "1.2.0"
KSTAR_PINNED_PYTHON = "3.12.14"
KSTAR_UNIQUE_REFERENCE_ID = "a7dfa119afa4f833375dfaf0c503ee1bca28c9fe7d868e4249907442d7821c64"
#: Frozen Unique Network IDs of the official KSTAR 1.2.0 ST/Y networks.  A
#: network directory whose IDs drift (wrong archive, partial re-extract,
#: regenerated networks) fails verification instead of silently passing.
KSTAR_ST_UNIQUE_NETWORK_ID = "0c85777e396f8931cc6662138fef6e6273acf706ec3503c178cd31e40ec04810"
KSTAR_Y_UNIQUE_NETWORK_ID = "23ce4b6c19d912d314e6893a993c48ad18129e9d88766b994ca5fc6f8fd2e8b5"
KSTAR_EXPECTED_NETWORK_FILES = 50
KSTAR_NETWORK_URLS = (
    "https://figshare.com/ndownloader/files/60883384",
    "https://ndownloader.figshare.com/files/60883384",
)
KSTAR_DOWNLOAD_USER_AGENT = "PTM2CellNet-kstar-setup/2026-09-21"
_GZIP_MAGIC = b"\x1f\x8b"
_MIN_NETWORK_ARCHIVE_BYTES = 1_000_000
_REFERENCE_ID_PATTERN = re.compile(r"Unique Reference ID:\s+([a-fA-F0-9]+)")
_NETWORK_ID_PATTERN = re.compile(r"Unique Network ID:\s+([a-fA-F0-9]+)")
_PINNED_PACKAGES = {
    "kstar": KSTAR_PINNED_VERSION,
    "pandas": "3.0.6",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "requests": "2.34.2",
    "tqdm": "4.70.1",
    "matplotlib": "3.11.2",
    "seaborn": "0.13.2",
    "fpdf2": "2.8.8",
}


class KSTARResourceError(ValueError):
    """Raised when a KSTAR resource pin or network asset is invalid."""


@dataclass(frozen=True)
class KSTARNetworkAudit:
    """Inventory of a verified KSTAR ST/Y Default network directory."""

    reference_ids: Mapping[str, str]
    network_ids: Mapping[str, str]
    n_files: Mapping[str, int]
    total_bytes: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def default_resource_hash_manifest() -> Path:
    """Return the in-repo hash manifest path."""

    return Path(__file__).resolve().parents[2] / "environments" / "kstar" / "resource_hashes.json"


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest of a file."""

    resolved = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resource_hash_manifest(path: str | Path | None = None) -> dict[str, object]:
    """Load and validate the frozen KSTAR resource hash manifest."""

    resolved = Path(path).expanduser().resolve(strict=True) if path else default_resource_hash_manifest()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KSTARResourceError(f"KSTAR resource hash manifest is not valid JSON: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise KSTARResourceError("KSTAR resource hash manifest must contain a JSON object")
    schema = str(payload.get("schema_version", "")).strip()
    if schema != KSTAR_RESOURCE_HASH_SCHEMA:
        raise KSTARResourceError(f"unexpected KSTAR resource hash schema: {schema!r}")
    if str(payload.get("kstar_version", "")).strip() != KSTAR_PINNED_VERSION:
        raise KSTARResourceError("resource hash manifest kstar_version must be 1.2.0")
    unique_id = str(payload.get("unique_reference_id", "")).strip()
    if unique_id != KSTAR_UNIQUE_REFERENCE_ID:
        raise KSTARResourceError("resource hash manifest unique_reference_id does not match KSTAR 1.2.0")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise KSTARResourceError("resource hash manifest is missing files")
    required = (
        "HumanPhosphoProteome.csv",
        "humanProteome.fasta",
        "reference_info.json",
        "citations_compendia.txt",
        "readme.txt",
    )
    missing = [name for name in required if name not in files]
    if missing:
        raise KSTARResourceError(f"resource hash manifest is missing files: {missing}")
    return dict(payload)


def verify_kstar_resource_files(
    resource_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, str]:
    """Verify PhosphoSitePlus-derived KSTAR companion files against frozen hashes."""

    manifest = load_resource_hash_manifest(manifest_path)
    root = Path(resource_dir).expanduser().resolve(strict=True)
    files = manifest["files"]
    assert isinstance(files, Mapping)
    verified: dict[str, str] = {}
    for name, spec in files.items():
        if not isinstance(spec, Mapping):
            raise KSTARResourceError(f"resource hash entry {name!r} must be an object")
        expected = str(spec.get("sha256", "")).strip().lower()
        expected_bytes = spec.get("bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise KSTARResourceError(f"resource hash for {name!r} is not a SHA256 hex digest")
        path = root / str(name)
        if not path.is_file():
            raise KSTARResourceError(f"KSTAR resource file is missing: {path}")
        actual_bytes = path.stat().st_size
        if expected_bytes is not None and int(expected_bytes) != actual_bytes:
            raise KSTARResourceError(f"{name} size mismatch: expected {expected_bytes}, got {actual_bytes}")
        actual = sha256_file(path)
        if actual != expected:
            raise KSTARResourceError(f"{name} sha256 mismatch: expected {expected}, got {actual}")
        verified[name] = actual
    reference_info = json.loads((root / "reference_info.json").read_text(encoding="utf-8"))
    unique_id = str(reference_info.get("unique_reference_id", "")).strip()
    if unique_id != KSTAR_UNIQUE_REFERENCE_ID:
        raise KSTARResourceError(
            f"reference_info.json unique_reference_id {unique_id!r} does not match the frozen KSTAR 1.2.0 pin"
        )
    return verified


def verify_pinned_python_packages(versions: Mapping[str, str]) -> None:
    """Verify the isolated environment has the frozen KSTAR package versions."""

    for name, expected in _PINNED_PACKAGES.items():
        actual = str(versions.get(name, "")).strip()
        if actual != expected:
            raise KSTARResourceError(f"expected {name}=={expected}, found {actual or 'MISSING'}")


def download_url_to_file(url: str, destination: str | Path, *, timeout_s: int = 45) -> int:
    """Download one URL to a file.  HTTP 202 / HTML / tiny bodies hard-fail."""

    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": KSTAR_DOWNLOAD_USER_AGENT, "Accept": "*/*"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            status = int(getattr(response, "status", 200) or 200)
            if status == 202:
                raise KSTARResourceError(f"KSTAR network URL returned HTTP 202 without a usable body: {url}")
            if status != 200:
                raise KSTARResourceError(f"KSTAR network URL failed: HTTP {status} {url}")
            first = response.read(2)
            if first != _GZIP_MAGIC:
                preview = first + response.read(510)
                raise KSTARResourceError(
                    f"KSTAR network URL did not return a gzip archive: {url}; prefix={preview[:80]!r}"
                )
            written = len(first)
            with dest.open("wb") as handle:
                handle.write(first)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
    except HTTPError as exc:
        raise KSTARResourceError(f"KSTAR network URL failed: HTTP {exc.code} {url}") from exc
    except URLError as exc:
        raise KSTARResourceError(f"KSTAR network URL failed: {url}: {exc}") from exc
    if written < _MIN_NETWORK_ARCHIVE_BYTES:
        dest.unlink(missing_ok=True)
        raise KSTARResourceError(f"KSTAR network archive is too small to be valid ({written} bytes): {url}")
    return written


def _network_run_information(network_root: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for phospho_type in ("ST", "Y"):
        info = network_root / phospho_type / "Default" / "RUN_INFORMATION.txt"
        if not info.is_file():
            raise KSTARResourceError(f"KSTAR network is missing {info}")
        text = info.read_text(encoding="utf-8")
        reference_match = _REFERENCE_ID_PATTERN.search(text)
        if reference_match is None:
            raise KSTARResourceError(f"KSTAR RUN_INFORMATION.txt has no Unique Reference ID: {info}")
        network_match = _NETWORK_ID_PATTERN.search(text)
        if network_match is None:
            raise KSTARResourceError(f"KSTAR RUN_INFORMATION.txt has no Unique Network ID: {info}")
        found[phospho_type] = {
            "reference_id": reference_match.group(1),
            "network_id": network_match.group(1),
        }
    return found


def verify_kstar_network_dir(network_dir: str | Path) -> KSTARNetworkAudit:
    """Verify an extracted KSTAR ST/Y Default network matches the frozen 1.2.0 pins.

    An empty or partial ``INDIVIDUAL_NETWORKS`` directory no longer passes:
    each type must carry exactly the pinned file count of non-empty network
    files, and both the Unique Reference ID and the per-type Unique Network ID
    must match the frozen pins.
    """

    root = Path(network_dir).expanduser().resolve(strict=True)
    run_information = _network_run_information(root)
    mismatched = {
        kind: ids["reference_id"]
        for kind, ids in run_information.items()
        if ids["reference_id"] != KSTAR_UNIQUE_REFERENCE_ID
    }
    if mismatched:
        raise KSTARResourceError(
            "KSTAR network unique_reference_id does not match the frozen 1.2.0 proteome "
            f"{KSTAR_UNIQUE_REFERENCE_ID}: {mismatched}"
        )
    pinned_network_ids = {"ST": KSTAR_ST_UNIQUE_NETWORK_ID, "Y": KSTAR_Y_UNIQUE_NETWORK_ID}
    network_id_mismatch = {
        kind: ids["network_id"]
        for kind, ids in run_information.items()
        if ids["network_id"] != pinned_network_ids[kind]
    }
    if network_id_mismatch:
        raise KSTARResourceError(
            "KSTAR network Unique Network ID does not match the frozen 1.2.0 networks "
            f"{pinned_network_ids}: {network_id_mismatch}"
        )
    n_files: dict[str, int] = {}
    total_bytes: dict[str, int] = {}
    for phospho_type in ("ST", "Y"):
        individual = root / phospho_type / "Default" / "INDIVIDUAL_NETWORKS"
        if not individual.is_dir():
            raise KSTARResourceError(f"KSTAR network is missing {individual}")
        files = sorted(path for path in individual.iterdir() if path.is_file())
        if len(files) != KSTAR_EXPECTED_NETWORK_FILES:
            raise KSTARResourceError(
                f"KSTAR {phospho_type} network must contain exactly {KSTAR_EXPECTED_NETWORK_FILES} files; "
                f"found {len(files)} under {individual}"
            )
        empty = [path.name for path in files if path.stat().st_size == 0]
        if empty:
            raise KSTARResourceError(f"KSTAR {phospho_type} network contains empty files: {empty[:5]}")
        n_files[phospho_type] = len(files)
        total_bytes[phospho_type] = int(sum(path.stat().st_size for path in files))
    return KSTARNetworkAudit(
        reference_ids={kind: ids["reference_id"] for kind, ids in run_information.items()},
        network_ids={kind: ids["network_id"] for kind, ids in run_information.items()},
        n_files=n_files,
        total_bytes=total_bytes,
    )


def extract_kstar_network_archive(archive: str | Path, install_dir: str | Path) -> Path:
    """Extract NETWORKS.tar.gz and return the network root containing ST/Y."""

    archive_path = Path(archive).expanduser().resolve(strict=True)
    target = Path(install_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tarball:
        tarball.extractall(target, filter="data")
    candidates = [target / "NETWORKS", target]
    for candidate in candidates:
        if (candidate / "ST" / "Default" / "RUN_INFORMATION.txt").is_file():
            verify_kstar_network_dir(candidate)
            return candidate
    raise KSTARResourceError(f"extracted KSTAR archive has no ST/Y Default network under {target}")


def install_kstar_networks(install_dir: str | Path, *, archive_path: str | Path | None = None) -> Path:
    """Download the official KSTAR 1.2.0 networks or reuse a local archive."""

    target = Path(install_dir).expanduser().resolve()
    existing = target / "NETWORKS" if (target / "NETWORKS" / "ST").is_dir() else target
    if (existing / "ST" / "Default" / "RUN_INFORMATION.txt").is_file():
        verify_kstar_network_dir(existing)
        return existing

    archive = Path(archive_path).expanduser().resolve() if archive_path else target / "NETWORKS.tar.gz"
    if archive_path is None or not archive.is_file():
        last_error: Exception | None = None
        for url in KSTAR_NETWORK_URLS:
            try:
                download_url_to_file(url, archive)
                last_error = None
                break
            except KSTARResourceError as exc:
                last_error = exc
        if last_error is not None:
            raise KSTARResourceError(
                "official KSTAR 1.2.0 network download failed; not forging ST/Y assets. " + str(last_error)
            )
    return extract_kstar_network_archive(archive, target)


__all__ = [
    "KSTAR_DOWNLOAD_USER_AGENT",
    "KSTAR_EXPECTED_NETWORK_FILES",
    "KSTAR_NETWORK_URLS",
    "KSTAR_PINNED_PYTHON",
    "KSTAR_PINNED_VERSION",
    "KSTAR_RESOURCE_HASH_SCHEMA",
    "KSTAR_ST_UNIQUE_NETWORK_ID",
    "KSTAR_UNIQUE_REFERENCE_ID",
    "KSTAR_Y_UNIQUE_NETWORK_ID",
    "KSTARNetworkAudit",
    "KSTARResourceError",
    "default_resource_hash_manifest",
    "download_url_to_file",
    "extract_kstar_network_archive",
    "install_kstar_networks",
    "load_resource_hash_manifest",
    "sha256_file",
    "verify_kstar_network_dir",
    "verify_kstar_resource_files",
    "verify_pinned_python_packages",
]
