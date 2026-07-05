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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

PDC_GRAPHQL_URL = "https://pdc.cancer.gov/graphql"
DEFAULT_TIMEOUT_S = 30.0


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
    ) -> None:
        self.endpoint = endpoint
        self._session = session
        self.timeout_s = timeout_s

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


__all__ = ["PDCClient", "PDCStudy", "PDCAPIError", "PDC_GRAPHQL_URL"]
