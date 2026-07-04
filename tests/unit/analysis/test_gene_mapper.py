"""Tests for gene to UniProt mapping."""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from src.analysis.gene_mapper import (
    GeneMapper,
    _RequestsUniProtMapper,
    _build_mapper,
    map_gene_to_uniprot,
)


class TestGeneMapper:
    """Test gene symbol to UniProt ID mapping functionality."""

    @pytest.fixture
    def mock_protmapper(self):
        """Create mock ProtMapper."""
        with patch('src.analysis.gene_mapper.ProtMapper') as mock:
            yield mock

    def test_map_gene_to_uniprot_success(self, mock_protmapper):
        """Test successful gene to UniProt mapping."""
        # Setup mock
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Create mock result DataFrame
        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, [])

        # Create mapper after mock is set up
        mapper = GeneMapper()

        # Test
        result = mapper.map_gene_to_uniprot('BRAF')

        assert result == 'P15056'
        mock_instance.get.assert_called_once_with(
            ids=['BRAF'],
            from_db='Gene_Name',
            to_db='UniProtKB',
        )

    def test_map_gene_to_uniprot_not_found(self, mock_protmapper):
        """Test mapping returns None for unknown gene."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Empty result for unknown gene
        mock_result = pd.DataFrame(columns=['From', 'To'])
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('UNKNOWN_GENE')

        assert result is None

    def test_map_gene_to_uniprot_failed_mapping(self, mock_protmapper):
        """Test mapping returns None when API reports failure."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Failed mapping
        mock_result = pd.DataFrame(columns=['From', 'To'])
        mock_instance.get.return_value = (mock_result, ['UNKNOWN_GENE'])

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('UNKNOWN_GENE')

        assert result is None

    def test_map_gene_to_uniprot_caching(self, mock_protmapper):
        """Test that gene mappings are cached."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()

        # First call
        result1 = mapper.map_gene_to_uniprot('BRAF')
        assert result1 == 'P15056'

        # Second call should use cache
        result2 = mapper.map_gene_to_uniprot('BRAF')
        assert result2 == 'P15056'

        # ProtMapper.get should only be called once
        mock_instance.get.assert_called_once()

    def test_map_genes_batch_success(self, mock_protmapper):
        """Test batch gene mapping."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF', 'TP53'],
            'To': ['P15056', 'P04637'],
        })
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()
        results = mapper.map_genes_batch(['BRAF', 'TP53'])

        assert results['BRAF'] == 'P15056'
        assert results['TP53'] == 'P04637'

    def test_map_genes_batch_partial_failure(self, mock_protmapper):
        """Test batch mapping with some failures."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, ['UNKNOWN_GENE'])

        mapper = GeneMapper()
        results = mapper.map_genes_batch(['BRAF', 'UNKNOWN_GENE'])

        assert results['BRAF'] == 'P15056'
        assert results['UNKNOWN_GENE'] is None

    def test_map_genes_batch_empty_list(self, mock_protmapper):
        """Test batch mapping with empty list returns empty dict."""
        mapper = GeneMapper()
        results = mapper.map_genes_batch([])
        assert results == {}

    def test_map_genes_batch_uses_cache(self, mock_protmapper):
        """Test batch mapping uses cache for previously mapped genes."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mapper = GeneMapper()

        # Pre-populate cache
        mapper._gene_cache['BRAF'] = 'P15056'

        mock_result = pd.DataFrame({
            'From': ['TP53'],
            'To': ['P04637'],
        })
        mock_instance.get.return_value = (mock_result, [])

        results = mapper.map_genes_batch(['BRAF', 'TP53'])

        # Only TP53 should be queried
        mock_instance.get.assert_called_once_with(
            ids=['TP53'],
            from_db='Gene_Name',
            to_db='UniProtKB',
        )

        assert results['BRAF'] == 'P15056'
        assert results['TP53'] == 'P04637'

    def test_map_gene_to_uniprot_network_error(self, mock_protmapper):
        """Test handling of network errors."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance
        mock_instance.get.side_effect = ConnectionError("Network error")

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('BRAF')

        assert result is None

    def test_get_canonical_isoform(self, mock_protmapper):
        """Test canonical isoform getter."""
        mapper = GeneMapper()
        # Currently just returns the provided ID
        result = mapper.get_canonical_isoform('BRAF', 'P15056')
        assert result == 'P15056'

    def test_convenience_function_map_gene_to_uniprot(self, mock_protmapper):
        """Test the convenience function."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, [])

        result = map_gene_to_uniprot('BRAF')
        assert result == 'P15056'

    def test_multiple_isoforms_returns_first(self, mock_protmapper):
        """Test that first (canonical) isoform is returned for genes with multiple isoforms."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Multiple rows for different isoforms
        mock_result = pd.DataFrame({
            'From': ['BRAF', 'BRAF', 'BRAF'],
            'To': ['P15056', 'P15056-2', 'P15056-3'],
        })
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('BRAF')

        # Should return first (canonical) isoform
        assert result == 'P15056'


class TestRequestsFallbackMapper:
    """Cover the requests-based fallback when UniProtMapper is absent."""

    def test_build_mapper_uses_protmapper_when_available(self, monkeypatch):
        """When the package is available _build_mapper returns a ProtMapper instance."""
        sentinel = object()
        monkeypatch.setattr(
            'src.analysis.gene_mapper._HAS_UNIPROT_MAPPER', True
        )
        monkeypatch.setattr(
            'src.analysis.gene_mapper.ProtMapper',
            lambda: sentinel,
            raising=False,
        )
        assert _build_mapper() is sentinel

    def test_build_mapper_falls_back_when_package_missing(self, monkeypatch):
        """When the package is unavailable _build_mapper returns the requests mapper."""
        monkeypatch.setattr(
            'src.analysis.gene_mapper._HAS_UNIPROT_MAPPER', False
        )
        mapper = _build_mapper()
        assert isinstance(mapper, _RequestsUniProtMapper)

    def test_requests_mapper_empty_ids(self):
        mapper = _RequestsUniProtMapper()
        result, failed = mapper.get(ids=[])
        assert result.empty
        assert failed == []

    def test_requests_mapper_success(self):
        mapper = _RequestsUniProtMapper()
        submit_resp = Mock()
        submit_resp.text = "job123"
        submit_resp.raise_for_status = Mock()
        details_resp = Mock()
        details_resp.status_code = 200
        details_resp.json.return_value = {"jobStatus": "FINISHED"}
        results_resp = Mock()
        results_resp.raise_for_status = Mock()
        results_resp.json.return_value = {
            "results": [
                {"from": "BRAF", "to": {"primaryAccession": "P15056"}},
            ]
        }
        with patch('src.analysis.gene_mapper.requests.post', return_value=submit_resp), \
             patch('src.analysis.gene_mapper.requests.get', side_effect=[details_resp, results_resp]):
            result, failed = mapper.get(ids=["BRAF"])
        assert failed == []
        assert result.iloc[0]["From"] == "BRAF"
        assert result.iloc[0]["To"] == "P15056"

    def test_requests_mapper_submission_failure(self):
        import requests
        mapper = _RequestsUniProtMapper()
        with patch(
            'src.analysis.gene_mapper.requests.post',
            side_effect=requests.RequestException("boom"),
        ):
            result, failed = mapper.get(ids=["BRAF"])
        assert result.empty
        assert failed == ["BRAF"]
