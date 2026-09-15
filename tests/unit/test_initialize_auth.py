"""Tests for API key authentication in the /initialize endpoint.

Verifies that _require_api_key uses hmac.compare_digest (constant-time)
and that the endpoint responds correctly with missing, wrong, or correct keys.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A path that is within the default allowed root (project root) but does not
# exist, so authentication passes the path-escape check and proceeds to
# the "file not found" (404) stage.
_NONEXISTENT_PATH = "nonexistent_checkpoint_for_testing.pt"


def _valid_body() -> dict:
    """Return a body that passes Pydantic validation (required checkpoint_path)
    but will not reach actual model loading (path doesn't exist → 404)."""
    return {"checkpoint_path": _NONEXISTENT_PATH}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a minimal app instance for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """TestClient bound to the app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# _require_api_key behaviour (via /initialize endpoint)
# ---------------------------------------------------------------------------


class TestInitializeAuth:
    """Functional tests for /initialize API key enforcement."""

    def test_missing_header_returns_401(self, client, monkeypatch):
        """X-API-Key header not sent, key is set in env → 401."""
        monkeypatch.setenv("PTM2CELLNET_API_KEY", "secret-123")
        monkeypatch.setenv("PTM2CELLNET_ENV", "production")
        resp = client.post("/api/v1/initialize", json=_valid_body())
        assert resp.status_code == 401
        assert "X-API-Key" in resp.text

    def test_wrong_key_returns_403(self, client, monkeypatch):
        """Wrong key sent → 403."""
        monkeypatch.setenv("PTM2CELLNET_API_KEY", "secret-123")
        monkeypatch.setenv("PTM2CELLNET_ENV", "production")
        resp = client.post(
            "/api/v1/initialize",
            json=_valid_body(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403
        assert "Invalid API key" in resp.text

    def test_correct_key_allows_request(self, client, monkeypatch):
        """Correct key → the request reaches the route handler (expect 404
        because the checkpoint path does not exist, not 401/403)."""
        monkeypatch.setenv("PTM2CELLNET_API_KEY", "secret-123")
        monkeypatch.setenv("PTM2CELLNET_ENV", "production")
        resp = client.post(
            "/api/v1/initialize",
            json=_valid_body(),
            headers={"X-API-Key": "secret-123"},
        )
        # The key is correct, so we should NOT get 401 or 403.
        # With a nonexistent checkpoint path we expect 404, not an auth error.
        assert resp.status_code not in (401, 403), f"Expected non-auth status code, got {resp.status_code}: {resp.text}"
        assert resp.status_code == 404

    def test_uses_hmac_compare_digest(self, client, monkeypatch):
        """Verify that hmac.compare_digest is actually called (not plain !=)."""
        monkeypatch.setenv("PTM2CELLNET_API_KEY", "secret-123")
        monkeypatch.setenv("PTM2CELLNET_ENV", "production")

        import hmac

        original = hmac.compare_digest
        call_count = 0

        def tracking_digest(a, b):
            nonlocal call_count
            call_count += 1
            return original(a, b)

        with patch("hmac.compare_digest", side_effect=tracking_digest):
            client.post(
                "/api/v1/initialize",
                json=_valid_body(),
                headers={"X-API-Key": "secret-123"},
            )

        assert call_count >= 1, (
            "hmac.compare_digest was never called. The _require_api_key function may still be using plain '!='."
        )

    def test_no_key_in_dev_mode_skips_auth(self, client, monkeypatch):
        """No API key set + PTM2CELLNET_ENV=development → request allowed."""
        monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
        monkeypatch.setenv("PTM2CELLNET_ENV", "development")
        resp = client.post("/api/v1/initialize", json=_valid_body())
        # Not 503 (unavailable) — dev mode lets it through
        assert resp.status_code != 503

    def test_no_key_in_production_returns_503(self, client, monkeypatch):
        """No API key set + PTM2CELLNET_ENV=production → 503."""
        monkeypatch.delenv("PTM2CELLNET_API_KEY", raising=False)
        monkeypatch.setenv("PTM2CELLNET_ENV", "production")
        resp = client.post("/api/v1/initialize", json=_valid_body())
        assert resp.status_code == 503
