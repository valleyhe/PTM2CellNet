"""Tests for src/models/external_tools.py — External bioinformatics tool integration.

Tests AlphaFoldClient, BLASTClient, ClustalWClient, and ChouFasmanClient
with mocked requests/subprocess and fallback behavior verification.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.models.external_tools import (
    AlphaFoldClient,
    BLASTClient,
    ClustalWClient,
    ChouFasmanClient,
    _needleman_wunsch,
    _nw_score,
)
from src.models.external_tools import alphafold as _af
from src.models.external_tools import blast as _blast
from src.models.external_tools import clustalw as _clustalw


# ---------------------------------------------------------------------------
# AlphaFoldClient tests
# ---------------------------------------------------------------------------


class TestAlphaFoldClient:
    """Tests for AlphaFoldClient with mocked requests."""

    def test_instantiation(self):
        """Client creates with default config."""
        client = AlphaFoldClient()
        assert client.config == {}

    def test_instantiation_with_config(self):
        """Client stores custom config."""
        client = AlphaFoldClient(config={"timeout": 10})
        assert client.config == {"timeout": 10}

    def test_check_available_requests_unavailable(self):
        """check_available returns False when requests module is missing."""
        with patch.object(_af, "REQUESTS_AVAILABLE", False):
            client = AlphaFoldClient()
            assert client.check_available() is False

    @patch.object(_af, "requests")
    def test_check_available_api_reachable(self, mock_requests):
        """check_available returns True when API responds OK."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_requests.get.return_value = mock_resp
        with patch.object(_af, "REQUESTS_AVAILABLE", True):
            client = AlphaFoldClient()
            assert client.check_available() is True

    @patch.object(_af, "requests")
    def test_check_available_api_unreachable(self, mock_requests):
        """check_available returns False when API raises exception."""
        mock_requests.get.side_effect = ConnectionError("timeout")
        with patch.object(_af, "REQUESTS_AVAILABLE", True):
            client = AlphaFoldClient()
            assert client.check_available() is False

    def test_predict_structure_fallback_when_no_requests(self):
        """predict_structure falls back to PDB when requests is unavailable."""
        with patch.object(_af, "REQUESTS_AVAILABLE", False):
            client = AlphaFoldClient()
            result = client.predict_structure("ACDEFGHIK")
            assert "pdb_string" in result
            assert "confidence" in result
            assert "predicted_aligned_error" in result
            assert len(result["pdb_string"]) > 0

    @patch.object(_af, "requests")
    def test_predict_structure_with_uniprot_id(self, mock_requests):
        """predict_structure fetches from EBI API with uniprot_id."""
        # Mock the prediction API response
        pred_resp = MagicMock()
        pred_resp.ok = True
        pred_resp.json.return_value = [
            {
                "pdbUrl": "https://example.com/pdb",
                "confidence": 90.0,
                "predictedAlignedError": [],
            }
        ]
        # Mock the PDB download response
        pdb_resp = MagicMock()
        pdb_resp.ok = True
        pdb_resp.text = "ATOM  ...fake PDB..."

        mock_requests.get.side_effect = [pred_resp, pdb_resp]

        with patch.object(_af, "REQUESTS_AVAILABLE", True):
            client = AlphaFoldClient()
            result = client.predict_structure("ACDEFGHIK", uniprot_id="P15056")

        assert result["pdb_string"] == "ATOM  ...fake PDB..."
        assert result["confidence"] == pytest.approx(0.9, abs=0.01)

    @patch.object(_af, "requests")
    def test_predict_structure_api_error_fallback(self, mock_requests):
        """predict_structure falls back when EBI API returns error."""
        pred_resp = MagicMock()
        pred_resp.ok = False
        mock_requests.get.return_value = pred_resp

        with patch.object(_af, "REQUESTS_AVAILABLE", True):
            client = AlphaFoldClient()
            result = client.predict_structure("ACDEFGHIK", uniprot_id="P15056")

        # Should fall back to built-in PDB
        assert "pdb_string" in result
        assert len(result["pdb_string"]) > 0

    def test_fallback_pdb_output_structure(self):
        """Fallback PDB contains ATOM records for each residue."""
        client = AlphaFoldClient()
        result = client._fallback_pdb("ACD")
        lines = result["pdb_string"].split("\n")
        assert len(lines) == 3  # one ATOM line per residue
        assert all(line.startswith("ATOM") for line in lines)

    def test_fallback_pdb_unknown_amino_acid(self):
        """Fallback PDB handles unknown amino acids as UNK."""
        client = AlphaFoldClient()
        result = client._fallback_pdb("X")
        assert "UNK" in result["pdb_string"]


# ---------------------------------------------------------------------------
# BLASTClient tests
# ---------------------------------------------------------------------------


class TestBLASTClient:
    """Tests for BLASTClient with mocked BioPython."""

    def test_instantiation(self):
        """Client creates with default config."""
        client = BLASTClient()
        assert client.config == {}

    def test_check_available_biopython_unavailable(self):
        """check_available returns False when BioPython BLAST is missing."""
        with patch.object(_blast, "BIO_BLAST_AVAILABLE", False):
            client = BLASTClient()
            assert client.check_available() is False

    def test_search_returns_empty_when_unavailable(self):
        """search returns empty list when BioPython BLAST is unavailable."""
        with patch.object(_blast, "BIO_BLAST_AVAILABLE", False):
            client = BLASTClient()
            result = client.search("ACDEFGHIK")
            assert result == []

    @patch.object(_blast, "NCBIXML")
    @patch.object(_blast, "NCBIWWW")
    def test_search_returns_hits(self, mock_www, mock_xml):
        """search returns parsed hits from NCBI BLAST."""
        # Build mock BLAST record
        mock_hsp = MagicMock()
        mock_hsp.expect = 1e-10
        mock_hsp.score = 200.0
        mock_hsp.identities = 50
        mock_hsp.align_length = 100

        mock_alignment = MagicMock()
        mock_alignment.accession = "NP_12345"
        mock_alignment.title = "Test protein"
        mock_alignment.hsps = [mock_hsp]

        mock_record = MagicMock()
        mock_record.alignments = [mock_alignment]

        mock_www.qblast.return_value = MagicMock()
        mock_xml.parse.return_value = [mock_record]

        with patch.object(_blast, "BIO_BLAST_AVAILABLE", True):
            client = BLASTClient()
            hits = client.search("ACDEFGHIK")

        assert len(hits) == 1
        assert hits[0]["accession"] == "NP_12345"
        assert hits[0]["e_value"] == 1e-10

    @patch.object(_blast, "NCBIWWW")
    def test_search_error_returns_empty(self, mock_www):
        """search returns empty list on NCBI error."""
        mock_www.qblast.side_effect = OSError("Network error")

        with patch.object(_blast, "BIO_BLAST_AVAILABLE", True):
            client = BLASTClient()
            hits = client.search("ACDEFGHIK")
            assert hits == []


# ---------------------------------------------------------------------------
# ClustalWClient tests
# ---------------------------------------------------------------------------


class TestClustalWClient:
    """Tests for ClustalWClient with built-in fallback alignment."""

    def test_instantiation(self):
        """Client creates with default config."""
        client = ClustalWClient()
        assert client.config == {}

    def test_align_empty_sequences(self):
        """align returns empty result for empty input."""
        client = ClustalWClient()
        result = client.align({})
        assert result["alignment"] == {}
        assert result["consensus"] == ""

    def test_align_single_sequence(self):
        """align returns the sequence itself for single input."""
        client = ClustalWClient()
        result = client.align({"seq1": "ACDE"})
        assert result["alignment"]["seq1"] == "ACDE"
        assert result["consensus"] == "ACDE"
        assert result["phylogenetic_tree"] == "(seq1);"

    def test_align_two_sequences_fallback(self):
        """align uses built-in progressive alignment when ClustalW unavailable."""
        client = ClustalWClient()
        # Force fallback by making external clustal return None
        with patch.object(client, "_run_external_clustal", return_value=None):
            result = client.align({"seq1": "ACD", "seq2": "ACD"})
        assert "alignment" in result
        assert "consensus" in result
        assert "phylogenetic_tree" in result
        assert len(result["alignment"]) == 2

    def test_compute_consensus(self):
        """_compute_consensus returns majority-rule consensus."""
        alignment = {
            "seq1": "ACD",
            "seq2": "ACD",
        }
        consensus = ClustalWClient._compute_consensus(alignment)
        assert consensus == "ACD"

    def test_compute_consensus_with_gaps(self):
        """_compute_consensus handles gaps correctly."""
        alignment = {
            "seq1": "A-D",
            "seq2": "ACD",
        }
        consensus = ClustalWClient._compute_consensus(alignment)
        assert len(consensus) == 3

    def test_check_available_with_biopython(self):
        """check_available returns True when BioPython Align is available."""
        with patch.object(_clustalw, "BIO_ALIGN_AVAILABLE", True), patch.object(_clustalw, "shutil") as mock_shutil:
            mock_shutil.which.return_value = None  # no external clustalw
            client = ClustalWClient()
            assert client.check_available() is True


# ---------------------------------------------------------------------------
# ChouFasmanClient tests
# ---------------------------------------------------------------------------


class TestChouFasmanClient:
    """Tests for ChouFasmanClient (Chou-Fasman built-in predictor)."""

    def test_instantiation(self):
        """Client creates with default config."""
        client = ChouFasmanClient()
        assert client.config == {}

    def test_check_available_always_true(self):
        """ChouFasmanClient is always available (built-in predictor)."""
        client = ChouFasmanClient()
        assert client.check_available() is True

    def test_predict_secondary_structure_output_keys(self):
        """Prediction returns ss_prediction and confidence_scores."""
        client = ChouFasmanClient()
        result = client.predict_secondary_structure("ACDEFGHIKL")
        assert "ss_prediction" in result
        assert "confidence_scores" in result

    def test_prediction_length_matches_sequence(self):
        """Prediction length equals input sequence length."""
        client = ChouFasmanClient()
        seq = "ACDEFGHIKL"
        result = client.predict_secondary_structure(seq)
        assert len(result["ss_prediction"]) == len(seq)
        assert len(result["confidence_scores"]) == len(seq)

    def test_ss_prediction_valid_chars(self):
        """Secondary structure prediction contains only H, E, C."""
        client = ChouFasmanClient()
        result = client.predict_secondary_structure("ACDEFGHIKLMNPQRSTVWY")
        valid_chars = {"H", "E", "C"}
        assert set(result["ss_prediction"]).issubset(valid_chars)

    def test_confidence_scores_in_range(self):
        """Confidence scores are in [0, 1]."""
        client = ChouFasmanClient()
        result = client.predict_secondary_structure("ACDEFGHIKL")
        for score in result["confidence_scores"]:
            assert 0.0 <= score <= 1.0

    def test_empty_sequence(self):
        """Empty sequence returns empty prediction."""
        client = ChouFasmanClient()
        result = client.predict_secondary_structure("")
        assert result["ss_prediction"] == ""
        assert result["confidence_scores"] == []

    def test_single_residue(self):
        """Single residue produces one-character prediction."""
        client = ChouFasmanClient()
        result = client.predict_secondary_structure("A")
        assert len(result["ss_prediction"]) == 1
        assert len(result["confidence_scores"]) == 1


# ---------------------------------------------------------------------------
# Needleman-Wunsch helper tests
# ---------------------------------------------------------------------------


class TestNeedlemanWunsch:
    """Tests for the built-in Needleman-Wunsch alignment."""

    def test_identical_sequences(self):
        """Identical sequences align without gaps."""
        a1, a2 = _needleman_wunsch("ACD", "ACD")
        assert a1 == "ACD"
        assert a2 == "ACD"

    def test_completely_different(self):
        """Completely different sequences produce gapped alignment."""
        a1, a2 = _needleman_wunsch("AAA", "CCC")
        # Should produce some alignment (exact result depends on scoring)
        assert len(a1) == len(a2)

    def test_nw_score_identical(self):
        """NW score for identical sequences is positive."""
        score = _nw_score("ACD", "ACD")
        assert score > 0

    def test_nw_score_different(self):
        """NW score for different sequences is lower than identical."""
        score_same = _nw_score("ACD", "ACD")
        score_diff = _nw_score("ACD", "EEE")
        assert score_same > score_diff


# ---------------------------------------------------------------------------
# PSIPREDClient alias tests (src/models/external_tools/psipred.py, N06)
# ---------------------------------------------------------------------------


class TestPSIPREDClientAlias:
    """PSIPREDClient is a deprecated alias for ChouFasmanClient.

    ``src/models/external_tools/psipred.py`` implements the built-in
    Chou-Fasman predictor under a PSIPRED-flavoured name; the alias keeps
    older imports working and must stay behaviour-identical.
    """

    def test_alias_identity(self):
        from src.models.external_tools.psipred import PSIPREDClient

        assert PSIPREDClient is ChouFasmanClient

    def test_alias_prediction_matches_choufasman(self):
        from src.models.external_tools.psipred import PSIPREDClient

        sequence = "AAAEEEKKK"
        via_alias = PSIPREDClient().predict_secondary_structure(sequence)
        via_canonical = ChouFasmanClient().predict_secondary_structure(sequence)
        assert via_alias.ss_prediction == via_canonical.ss_prediction
        assert via_alias.confidence_scores == via_canonical.confidence_scores

    def test_module_reports_builtin_availability(self):
        from src.models.external_tools.psipred import PSIPREDClient

        assert PSIPREDClient().check_available() is True
