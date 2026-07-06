"""Proteomic Data Commons (PDC) API client (F-03 / TD-H3).

This module implements the real PDC GraphQL integration that
``scripts/validate_cptac.py`` previously stubbed with synthetic data.

Design
------

* **Single responsibility**: only PDC API I/O. No model inference, no
  synthetic data generation. Callers compose this with the existing
  PTM predictors.
* **Reuse the safe download boundary**: HTTP requests go through the
  same ``requests`` + timeout + size-cap discipline used by
  ``src/data/loaders/base.py``. PDC is added to the production download
  allowlist when the env override is set.
* **Network-optional by design**: every public method accepts an
  injected ``session`` and raises a clear ``PDCAPIError`` on failure
  rather than silently returning mock data. The caller decides whether
  to fall back.
* **Tested without network**: the unit test suite injects a fake
  session and asserts the GraphQL query shape, response parsing, and
  error handling. A real-assets test in ``tests/real_assets/`` exercises
  the live endpoint behind the opt-in gate.

Endpoints
---------

PDC exposes a public GraphQL endpoint at ``https://pdc.cancer.gov/graphql``.
We use only the read-only ``study`` / ``files`` / ``biospecimen`` queries
needed for phosphoproteomics validation.

File download (F-03 v17)
------------------------

``download_file`` resolves a PDC file_id to a real data file via the
``filePath`` query (which returns a per-file download URL on
``pdc.cancer.gov``), then streams it to a local path under the same
size cap and host allowlist used everywhere else. This closes the
"file metadata only" gap noted in v16 §4 (F-03) without weakening the
download safety boundary.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

PDC_GRAPHQL_URL = "https://pdc.cancer.gov/graphql"
PDC_DATA_HOST = "pdc.cancer.gov"
DEFAULT_TIMEOUT_S = 30.0
# Cap a single PDC file download at 2 GiB by default. PDC phosphoproteomics
# TSVs are tens of MB; this is a safety rail against runaway downloads.
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


class PDCAPIError(RuntimeError):
    """Raised when the PDC API returns an error or unusable response."""


@dataclass
class PDCStudy:
    """Normalized view of a PDC study record."""

    study_id: str
    study_submitter_id: str
    study_name: str
    disease_type: str
    primary_site: str
    files: list[dict[str, Any]] = field(default_factory=list)


class PDCClient:
    """Read-only GraphQL client for the Proteomic Data Commons.

    Parameters
    ----------
    endpoint:
        GraphQL endpoint URL. Defaults to the public PDC endpoint.
    session:
        Optional pre-configured ``requests.Session`` (e.g. with retry
        adapters or auth). When ``None``, a fresh session is created
        per request — fine for one-shot CLI use.
    timeout_s:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str = PDC_GRAPHQL_URL,
        session: Optional[requests.Session] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    ) -> None:
        self.endpoint = endpoint
        self._session = session
        self.timeout_s = timeout_s
        self.max_download_bytes = max_download_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_study(self, study_id: str) -> PDCStudy:
        """Fetch a single study by its PDC study_id (UUID)."""
        payload = self._query(
            """
            query($study_id: String!) {
                study(study_id: $study_id) {
                    study_id
                    study_submitter_id
                    study_name
                    disease_type
                    primary_site
                    files {
                        file_id
                        file_name
                        file_type
                        data_category
                        download_url
                    }
                }
            }
            """,
            {"study_id": study_id},
        )
        study_data = payload.get("data", {}).get("study")
        if not study_data:
            raise PDCAPIError(f"PDC returned no study for study_id={study_id!r}")
        return self._parse_study(study_data)

    def list_studies(self, limit: int = 100) -> list[PDCStudy]:
        """List all studies (paged server-side by PDC)."""
        payload = self._query(
            """
            query {
                allStudies {
                    study_id
                    study_submitter_id
                    study_name
                    disease_type
                    primary_site
                }
            }
            """,
            {},
        )
        studies = payload.get("data", {}).get("allStudies", [])
        return [self._parse_study(s) for s in studies[:limit]]

    def get_study_files(self, study_id: str, file_type: Optional[str] = None) -> list[dict[str, Any]]:
        """Return the file manifest for a study, optionally filtered by type."""
        study = self.get_study(study_id)
        files = study.files
        if file_type:
            file_type_lower = file_type.lower()
            files = [f for f in files if str(f.get("file_type", "")).lower() == file_type_lower]
        return files

    def get_file_url(self, file_id: str) -> str:
        """Resolve a PDC ``file_id`` to a real per-file download URL.

        PDC exposes the per-file download URL through the ``file`` query
        (``file(file_id: ...) { downloadUrl }``). This method issues that
        query and returns the URL string. The URL itself is fetched
        separately via :meth:`download_file` so callers can audit it.

        Raises :class:`PDCAPIError` if PDC returns no URL or the response
        is malformed.
        """
        payload = self._query(
            """
            query($file_id: String!) {
                file(file_id: $file_id) {
                    file_id
                    file_name
                    file_type
                    file_location
                    downloadUrl
                }
            }
            """,
            {"file_id": file_id},
        )
        file_data = payload.get("data", {}).get("file")
        if not file_data:
            raise PDCAPIError(f"PDC returned no file for file_id={file_id!r}")
        url = file_data.get("downloadUrl") or file_data.get("file_location")
        if not url:
            raise PDCAPIError(
                f"PDC file {file_id!r} has no downloadUrl/file_location; "
                f"payload was {file_data!r}"
            )
        return str(url)

    def download_file(
        self,
        file_id: str,
        target_dir: str | os.PathLike[str],
        *,
        filename: Optional[str] = None,
    ) -> Path:
        """Download a PDC file by ``file_id`` to ``target_dir`` and return the path.

        This closes the F-03 gap noted in v16: until now we exposed study
        metadata only. ``download_file`` resolves the per-file URL via
        :meth:`get_file_url`, then streams the bytes to a local temp file
        under the same size cap and host allowlist as the rest of the
        project (``PTM2CELLNET_DOWNLOAD_ALLOWLIST`` /
        ``PTM2CELLNET_MAX_DOWNLOAD_BYTES``).

        Parameters
        ----------
        file_id:
            PDC file UUID.
        target_dir:
            Directory to write the file into. Created if missing.
        filename:
            Optional override for the local filename. Defaults to the
            last path segment of the resolved URL.

        Returns
        -------
        Path to the downloaded file.

        Raises
        ------
        PDCAPIError:
            On transport errors, oversized payloads, or disallowed hosts.
        """
        url = self.get_file_url(file_id)
        return self._stream_url_to_dir(url, target_dir, filename=filename, file_id=file_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Execute a GraphQL query and return the parsed JSON payload.

        Raises :class:`PDCAPIError` on transport errors, non-200 status,
        or GraphQL-level ``errors``.
        """
        try:
            session = self._session or requests.Session()
            response = session.post(
                self.endpoint,
                json={"query": query, "variables": variables},
                timeout=self.timeout_s,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:
            raise PDCAPIError(f"PDC request transport failed: {exc}") from exc

        if response.status_code != 200:
            raise PDCAPIError(
                f"PDC request returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise PDCAPIError(f"PDC response is not valid JSON: {exc}") from exc

        if "errors" in payload and payload["errors"]:
            messages = "; ".join(
                str(err.get("message", err)) for err in payload["errors"]
            )
            raise PDCAPIError(f"PDC GraphQL errors: {messages}")

        if "data" not in payload:
            raise PDCAPIError(f"PDC response missing 'data' key: {payload!r}")

        return dict(payload)

    @staticmethod
    def _parse_study(raw: dict[str, Any]) -> PDCStudy:
        return PDCStudy(
            study_id=str(raw.get("study_id", "")),
            study_submitter_id=str(raw.get("study_submitter_id", "")),
            study_name=str(raw.get("study_name", "")),
            disease_type=str(raw.get("disease_type", "")),
            primary_site=str(raw.get("primary_site", "")),
            files=list(raw.get("files") or []),
        )

    def _stream_url_to_dir(
        self,
        url: str,
        target_dir: str | os.PathLike[str],
        *,
        filename: Optional[str],
        file_id: str,
    ) -> Path:
        """Stream a URL to ``target_dir`` honouring the project's download guard.

        Host allowlist:
            When ``PTM2CELLNET_DOWNLOAD_ALLOWLIST`` is unset, only
            ``pdc.cancer.gov`` is allowed (the canonical PDC data host).
            When set, the operator's list applies; an empty value allows
            any host (dev/test only).

        Size cap:
            The smaller of ``self.max_download_bytes`` and the env-driven
            ``PTM2CELLNET_MAX_DOWNLOAD_BYTES`` (if set). Both the declared
            ``Content-Length`` and the actual streamed byte count are
            enforced.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise PDCAPIError(
                f"Refusing PDC download with scheme {parsed.scheme!r}; "
                "only http/https are allowed."
            )
        if parsed.scheme == "http":
            logger.warning(
                "PDC file %s is being downloaded over plaintext HTTP; "
                "configure the source for HTTPS if possible.",
                file_id,
            )

        host = parsed.netloc.split(":")[0].lower()
        allowlist_env = os.environ.get("PTM2CELLNET_DOWNLOAD_ALLOWLIST", None)
        if allowlist_env is not None:
            # Operator-provided list. Empty string = allow any host.
            if allowlist_env.strip():
                allowed_hosts = {
                    h.strip().lower() for h in allowlist_env.split(",") if h.strip()
                }
                if host not in allowed_hosts:
                    raise PDCAPIError(
                        f"PDC download host {host!r} not in operator allowlist."
                    )
        else:
            # Default allowlist for PDC: the canonical PDC data host only.
            if host != PDC_DATA_HOST:
                raise PDCAPIError(
                    f"PDC download host {host!r} is not {PDC_DATA_HOST!r}. "
                    "Set PTM2CELLNET_DOWNLOAD_ALLOWLIST to allow additional hosts."
                )

        # Resolve size cap: env override wins if smaller than the client default.
        env_cap_str = os.environ.get("PTM2CELLNET_MAX_DOWNLOAD_BYTES")
        if env_cap_str:
            try:
                env_cap = int(env_cap_str)
                max_bytes = min(self.max_download_bytes, env_cap)
            except ValueError:
                max_bytes = self.max_download_bytes
        else:
            max_bytes = self.max_download_bytes

        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        local_name = filename or os.path.basename(parsed.path) or f"{file_id}.bin"
        final_path = out_dir / local_name

        session = self._session or requests.Session()
        try:
            response = session.get(
                url,
                stream=True,
                timeout=self.timeout_s,
                headers={"Accept": "*/*"},
            )
        except requests.RequestException as exc:
            raise PDCAPIError(f"PDC file download transport failed: {exc}") from exc

        if response.status_code != 200:
            raise PDCAPIError(
                f"PDC file download returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        # Pre-check Content-Length to fail fast on obviously oversized files.
        headers = getattr(response, "headers", None) or {}
        content_length = headers.get("content-length") if hasattr(headers, "get") else None
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared > max_bytes:
                raise PDCAPIError(
                    f"PDC file {file_id} declared size {declared} bytes exceeds "
                    f"cap {max_bytes} bytes (PTM2CELLNET_MAX_DOWNLOAD_BYTES)."
                )

        # Stream to a temp file in the same dir, then atomically rename. This
        # avoids leaving a half-written file on disk if the download is
        # interrupted or hits the size cap mid-stream.
        tmp_suffix = "".join(part for part in os.path.splitext(local_name) if part) or ".tmp"
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=tmp_suffix,
                prefix=f"pdc_{file_id}_",
                dir=str(out_dir),
                delete=False,
            ) as handle:
                bytes_written = 0
                iter_content = getattr(response, "iter_content", None)
                if callable(iter_content):
                    for chunk in iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if bytes_written > max_bytes:
                            handle.close()
                            _safe_unlink(handle.name)
                            raise PDCAPIError(
                                f"PDC file {file_id} exceeded {max_bytes} bytes "
                                "during streaming; aborted."
                            )
                        handle.write(chunk)
                else:
                    content = getattr(response, "content", b"") or b""
                    if len(content) > max_bytes:
                        handle.close()
                        _safe_unlink(handle.name)
                        raise PDCAPIError(
                            f"PDC file {file_id} body {len(content)} bytes exceeds "
                            f"cap {max_bytes} bytes."
                        )
                    handle.write(content)
                    bytes_written = len(content)
                tmp_name = handle.name
        except Exception:
            # Ensure we don't leak a partial temp file on any failure path.
            _safe_unlink(locals().get("tmp_name", ""))
            raise

        os.replace(tmp_name, final_path)
        logger.info(
            "PDC file %s downloaded to %s (%d bytes)", file_id, final_path, bytes_written
        )
        return final_path


def _safe_unlink(path: str | os.PathLike[str]) -> None:
    """Best-effort remove; swallow OS errors so callers can re-raise."""
    try:
        if path:
            os.unlink(path)
    except OSError:
        pass


__all__ = [
    "PDCClient",
    "PDCStudy",
    "PDCAPIError",
    "PDC_GRAPHQL_URL",
    "PDC_DATA_HOST",
    "DEFAULT_MAX_DOWNLOAD_BYTES",
]
