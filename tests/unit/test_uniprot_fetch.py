"""Tests for UniProt sequence fetching and caching functionality.

Converted from scripts/tools/test_uniprot_fetch.py to proper pytest format.
"""

from unittest.mock import MagicMock, patch

import pytest


# Skip the entire module if the scripts.process_pmads_data module is not importable
pytest.importorskip(
    "scripts.process_pmads_data",
    reason="scripts.process_pmads_data not available",
)


@pytest.fixture
def mock_uniprot_response():
    """Mock UniProt API response for testing without network access.

    The real fetch_uniprot_sequence calls response.raise_for_status() and
    response.json() -> {'results': [{'sequence': {'value': ...}}]}.
    """
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.text = (
        ">sp|P04217|A1BG_HUMAN Alpha-1B-glycoprotein OS=Homo sapiens OX=9606 GN=A1BG PE=1 SV=2\n"
        "MLRVLSLVLALLALLLWGRQPSAQEAAEAGAEAGAEVRVVNDEGNYTVRVTGEVREQLFHVDSYQ\n"
    )
    # The real function parses the JSON response body, not raw text
    mock_resp.json.return_value = {
        "results": [{"sequence": {"value": "MLRVLSLVLALLALLLWGRQPSAQEAAEAGAEAGAEVRVVNDEGNYTVRVTGEVREQLFHVDSYQ"}}]
    }
    return mock_resp


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory for UniProt sequences."""
    cache = tmp_path / "uniprot_cache"
    cache.mkdir()
    return cache


class TestUniProtFetch:
    """Tests for UniProt API fetching functionality."""

    @patch("scripts.process_pmads_data.requests.get")
    def test_fetch_uniprot_sequence_success(self, mock_get, mock_uniprot_response):
        """Test fetching a protein sequence from UniProt API."""
        from scripts.process_pmads_data import fetch_uniprot_sequence

        mock_get.return_value = mock_uniprot_response
        seq = fetch_uniprot_sequence("A1BG")
        assert seq is not None
        assert len(seq) > 0
        assert seq.isalpha()

    @patch("scripts.process_pmads_data.requests.get")
    def test_fetch_uniprot_sequence_network_failure(self, mock_get):
        """Test that network failure returns None gracefully."""
        from scripts.process_pmads_data import fetch_uniprot_sequence
        import requests

        mock_get.side_effect = requests.ConnectionError("Network error")
        seq = fetch_uniprot_sequence("A1BG")
        assert seq is None

    @patch("scripts.process_pmads_data.requests.get")
    def test_fetch_uniprot_sequence_not_found(self, mock_get):
        """Test that a 404 response returns None."""
        from scripts.process_pmads_data import fetch_uniprot_sequence
        import requests

        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Client Error: Not Found")
        mock_get.return_value = mock_resp
        seq = fetch_uniprot_sequence("NONEXISTENT")
        assert seq is None


class TestUniProtCache:
    """Tests for UniProt sequence caching functionality."""

    def test_save_and_read_cache(self, tmp_path):
        """Test saving and reading a cached sequence."""
        from scripts.process_pmads_data import save_cached_sequence, get_cached_sequence

        test_protein = "TEST_PROTEIN_CACHE"
        test_seq = "MKTLLILTGLAVLLGLLAHSAQLTPTGTF"

        # Patch the cache directory to use tmp_path
        with patch("scripts.process_pmads_data.UNIPROT_CACHE_DIR", tmp_path):
            save_cached_sequence(test_protein, test_seq)
            cached = get_cached_sequence(test_protein)
            assert cached == test_seq

    def test_read_missing_cache_returns_none(self, tmp_path):
        """Test that reading a non-existent cache entry returns None."""
        from scripts.process_pmads_data import get_cached_sequence

        with patch("scripts.process_pmads_data.UNIPROT_CACHE_DIR", tmp_path):
            cached = get_cached_sequence("NON_EXISTENT_PROTEIN_12345")
            assert cached is None


class TestSequenceGeneration:
    """Tests for protein sequence generation."""

    @patch("scripts.process_pmads_data.fetch_uniprot_sequence")
    def test_generate_with_real_sequence(self, mock_fetch):
        """Test sequence generation with a real UniProt sequence."""
        from scripts.process_pmads_data import generate_protein_sequence

        mock_fetch.return_value = "M" * 500
        seq = generate_protein_sequence(
            protein_name="A1BG",
            ptm_position=237,
            ptm_aa="S",
            ptm_positions=[237, 229, 231],
            seq_length=500,
        )
        assert len(seq) == 500

    @patch("scripts.process_pmads_data.fetch_uniprot_sequence")
    def test_generate_fallback_random_sequence(self, mock_fetch):
        """Test that fallback generates a random sequence when UniProt fails."""
        from scripts.process_pmads_data import generate_protein_sequence

        mock_fetch.return_value = None
        seq = generate_protein_sequence(
            protein_name="NON_EXISTENT_XYZ123",
            ptm_position=100,
            ptm_aa="K",
            ptm_positions=[100],
            seq_length=500,
        )
        assert len(seq) == 500
