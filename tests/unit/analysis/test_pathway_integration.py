"""Tests for pathway integration module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pickle

import networkx as nx

from src.analysis.pathway_integration import (
    PathwayDatabaseIntegration,
    load_kegg_pathways,
    load_reactome_pathways,
)


@pytest.fixture(autouse=True)
def _reset_sspa_mock(_sspa_mock_session):
    """Reset sspa mock state between tests to prevent leakage."""
    _sspa_mock_session.reset_mock()
    # reset_mock() does NOT clear side_effect or return_value on child mocks,
    # so clear them explicitly on the methods used by tests.
    _sspa_mock_session.process_kegg.side_effect = None
    _sspa_mock_session.process_kegg.return_value = MagicMock()
    _sspa_mock_session.process_reactome.side_effect = None
    _sspa_mock_session.process_reactome.return_value = MagicMock()
    return _sspa_mock_session


class TestPathwayDatabaseIntegration:
    """Test PathwayDatabaseIntegration class."""

    def test_init_creates_cache_dir(self):
        """Test initialization creates cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            integration = PathwayDatabaseIntegration(cache_dir=str(cache_dir))
            assert cache_dir.exists()

    def test_builtin_pathways_defined(self):
        """Test that 8 built-in pathways are defined."""
        integration = PathwayDatabaseIntegration()
        assert len(integration.BUILTIN_PATHWAYS) == 8

        expected_pathways = [
            'MAPK/ERK', 'PI3K/AKT', 'JAK/STAT', 'NF-kB',
            'Wnt/beta-catenin', 'Cell Cycle', 'Apoptosis', 'DNA Damage'
        ]
        for pathway in expected_pathways:
            assert pathway in integration.BUILTIN_PATHWAYS

    def test_load_kegg_pathways(self, _reset_sspa_mock):
        """Test loading KEGG pathways."""
        _reset_sspa_mock.process_kegg.return_value = {
            'hsa00010': ['GAPDH', 'PGK1', 'ENO1'],
            'hsa00020': ['CS', 'ACO2', 'IDH1'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)
            pathways = integration.load_kegg_pathways(organism="hsa")

            assert integration._kegg_loaded is True
            assert 'hsa00010' in pathways
            assert 'hsa00020' in pathways
            _reset_sspa_mock.process_kegg.assert_called_once_with(organism="hsa")

    def test_load_reactome_pathways(self, _reset_sspa_mock):
        """Test loading Reactome pathways."""
        _reset_sspa_mock.process_reactome.return_value = {
            'R-HSA-12345': ['BRAF', 'MAPK1', 'MAPK3'],
            'R-HSA-67890': ['AKT1', 'MTOR', 'PIK3CA'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)
            pathways = integration.load_reactome_pathways(organism="Homo sapiens")

            assert integration._reactome_loaded is True
            assert 'R-HSA-12345' in pathways
            assert 'R-HSA-67890' in pathways
            _reset_sspa_mock.process_reactome.assert_called_once_with(organism="Homo sapiens")

    def test_kegg_caching(self, _reset_sspa_mock):
        """Test that KEGG pathways are cached after first load."""
        _reset_sspa_mock.process_kegg.return_value = {
            'hsa00010': ['GAPDH', 'PGK1'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)

            # First load - should call sspa
            integration.load_kegg_pathways(organism="hsa")
            assert _reset_sspa_mock.process_kegg.call_count == 1

            # Second load - should use cache
            integration2 = PathwayDatabaseIntegration(cache_dir=tmpdir)
            integration2.load_kegg_pathways(organism="hsa")
            assert _reset_sspa_mock.process_kegg.call_count == 1  # No additional call

    def test_reactome_caching(self, _reset_sspa_mock):
        """Test that Reactome pathways are cached after first load."""
        _reset_sspa_mock.process_reactome.return_value = {
            'R-HSA-12345': ['BRAF', 'MAPK1'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)

            # First load - should call sspa
            integration.load_reactome_pathways(organism="Homo sapiens")
            assert _reset_sspa_mock.process_reactome.call_count == 1

            # Second load - should use cache
            integration2 = PathwayDatabaseIntegration(cache_dir=tmpdir)
            integration2.load_reactome_pathways(organism="Homo sapiens")
            assert _reset_sspa_mock.process_reactome.call_count == 1  # No additional call

    def test_kegg_network_error_handling(self, _reset_sspa_mock):
        """Test graceful handling of network errors."""
        _reset_sspa_mock.process_kegg.side_effect = Exception("Network error")

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)
            with pytest.raises(Exception, match="Network error"):
                integration.load_kegg_pathways(organism="hsa")

    def test_validate_builtin_pathways(self, _reset_sspa_mock):
        """Test validation of built-in pathways."""
        _reset_sspa_mock.process_reactome.return_value = {
            'R-HSA-MAPK': ['BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'EGFR'],
            'R-HSA-PI3K': ['PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'MTOR', 'PTEN'],
            'R-HSA-JAK': ['JAK1', 'JAK2', 'JAK3', 'TYK2', 'STAT1', 'STAT3'],
            'R-HSA-NFKB': ['IKBKB', 'IKBKA', 'CHUK', 'NFKBIA', 'RELA'],
            'R-HSA-WNT': ['GSK3B', 'CTNNB1', 'APC', 'AXIN1'],
            'R-HSA-CELL': ['CDK1', 'CDK2', 'CDK4', 'RB1', 'TP53'],
            'R-HSA-APOP': ['CASP3', 'CASP8', 'BCL2', 'BAX'],
            'R-HSA-DNA': ['ATM', 'ATR', 'TP53', 'BRCA1', 'BRCA2'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)
            report = integration.validate_builtin_pathways(min_coverage=0.5)

            # Check all 8 pathways are validated
            assert len(report) == 8

            # Check MAPK/ERK has good coverage
            mapk_report = report['MAPK/ERK']
            assert mapk_report['coverage'] > 0.5
            assert mapk_report['validated'] is True

    def test_build_pathway_graph(self, _reset_sspa_mock):
        """Test building NetworkX graph from pathway."""
        _reset_sspa_mock.process_reactome.return_value = {
            'R-HSA-12345': ['BRAF', 'MAPK1', 'MAPK3'],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)
            integration.load_reactome_pathways()

            G = integration.build_pathway_graph('R-HSA-12345')

            assert isinstance(G, nx.DiGraph)
            assert 'BRAF' in G.nodes
            assert 'MAPK1' in G.nodes
            assert 'MAPK3' in G.nodes

    def test_build_pathway_graph_invalid_id(self, _reset_sspa_mock):
        """Test building graph with invalid pathway ID."""
        _reset_sspa_mock.process_reactome.return_value = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)
            integration.load_reactome_pathways()

            with pytest.raises(ValueError, match="Pathway INVALID not found"):
                integration.build_pathway_graph('INVALID')

    def test_clear_cache(self):
        """Test clearing cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            integration = PathwayDatabaseIntegration(cache_dir=tmpdir)

            # Create a mock cache file
            cache_file = Path(tmpdir) / "kegg_hsa.pkl"
            cache_file.write_bytes(pickle.dumps({'test': ['data']}))

            assert cache_file.exists()
            integration.clear_cache()
            assert not cache_file.exists()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_kegg_pathways_function(self, _reset_sspa_mock):
        """Test load_kegg_pathways convenience function."""
        _reset_sspa_mock.process_kegg.return_value = {'hsa00010': ['GAPDH']}

        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                pathways = load_kegg_pathways(organism="hsa")
                assert 'hsa00010' in pathways
            finally:
                os.chdir(original_dir)

    def test_load_reactome_pathways_function(self, _reset_sspa_mock):
        """Test load_reactome_pathways convenience function."""
        _reset_sspa_mock.process_reactome.return_value = {'R-HSA-12345': ['BRAF']}

        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            original_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                pathways = load_reactome_pathways(organism="Homo sapiens")
                assert 'R-HSA-12345' in pathways
            finally:
                os.chdir(original_dir)
