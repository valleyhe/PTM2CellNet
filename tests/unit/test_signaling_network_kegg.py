"""Unit tests for KEGG/Reactome integration in src/models/signaling_network.py
(FEAT-02-FULL / CONF-03).

These tests verify that:
    1. ``SignalingNetworkMapper`` no longer raises ``NotImplementedError`` when
       given a pathway_db_path (CONF-03).
    2. KEGG/Reactome loading gracefully falls back to built-in pathways when
       sspa is unavailable (FEAT-02 graceful degradation).
    3. External pathway data is merged into the in-memory pathway map when
       available.
"""

from unittest.mock import MagicMock, patch


from src.models.signaling_network import SignalingNetworkMapper


class TestKEGGReactomeIntegration:
    """FEAT-02-FULL: signaling_network._load_pathway_db integration."""

    def test_load_pathway_db_no_longer_raises_not_implemented(self):
        """CONF-03: the NotImplementedError stub must be gone.

        Providing a pathway_db_path must not raise NotImplementedError even
        when the backend (sspa) is unavailable.
        """
        # Should complete without raising
        mapper = SignalingNetworkMapper(pathway_db_path="kegg")
        assert len(mapper.pathways) > 0  # built-in fallback

    def test_kegg_directive(self):
        mapper = SignalingNetworkMapper(pathway_db_path="kegg")
        # Falls back to 8 built-in pathways when sspa unavailable
        assert len(mapper.pathways) >= 8

    def test_reactome_directive(self):
        mapper = SignalingNetworkMapper(pathway_db_path="reactome")
        assert len(mapper.pathways) >= 8

    def test_both_directive(self):
        mapper = SignalingNetworkMapper(pathway_db_path="both")
        assert len(mapper.pathways) >= 8

    def test_unrecognized_db_path_falls_back_gracefully(self):
        """Unrecognized directive treated as cache dir; never raises."""
        mapper = SignalingNetworkMapper(pathway_db_path="/nonexistent/path/xyz")
        assert len(mapper.pathways) >= 8

    def test_protein_mapping_built_after_load(self):
        """protein_to_pathway is built regardless of DB load outcome."""
        mapper = SignalingNetworkMapper(pathway_db_path="kegg")
        assert isinstance(mapper.protein_to_pathway, dict)
        # Built-in MAPK kinases present
        assert "BRAF" in mapper.protein_to_pathway

    def test_external_pathways_merged_when_available(self):
        """When PathwayDatabaseIntegration returns data, it's merged in."""
        # Patch the integration object to return synthetic external pathways
        mapper = SignalingNetworkMapper()
        fake_integration = MagicMock()
        fake_integration.load_kegg_pathways.return_value = {
            "hsa00010": ["GENE_A", "GENE_B"],
        }
        fake_integration.load_reactome_pathways.return_value = {}
        mapper.pathway_integration = fake_integration

        mapper._load_pathway_db("both")

        assert "KEGG:hsa00010" in mapper.pathways
        entry = mapper.pathways["KEGG:hsa00010"]
        assert entry["key_substrates"] == ["GENE_A", "GENE_B"]

    def test_builtin_pathways_not_shadowed_by_external(self):
        """External pathways with built-in names don't overwrite built-ins."""
        mapper = SignalingNetworkMapper()
        fake_integration = MagicMock()
        # External entry uses a built-in pathway name
        fake_integration.load_kegg_pathways.return_value = {
            "MAPK/ERK": ["FAKE_GENE"],
        }
        fake_integration.load_reactome_pathways.return_value = {}
        mapper.pathway_integration = fake_integration

        original = dict(mapper.pathways["MAPK/ERK"])
        mapper._load_pathway_db("kegg")

        # Built-in MAPK/ERK entry unchanged
        assert mapper.pathways["MAPK/ERK"]["key_kinases"] == original["key_kinases"]


class TestSignalingNetworkMapperBehavior:
    """Sanity-check the existing mapping API still works after refactor."""

    def test_default_construction(self):
        mapper = SignalingNetworkMapper()
        # 12 from SIGNALING_PATHWAYS + 10 from _BUILTIN_PATHWAY_DATA
        assert len(mapper.pathways) == 22
        assert mapper.pathway_db_path is None

    def test_accepts_organism_kwargs(self):
        mapper = SignalingNetworkMapper(pathway_db_path="kegg", organism="mmu", organism_name="Mus musculus")
        assert mapper.organism == "mmu"
        assert mapper.organism_name == "Mus musculus"

    def test_pathway_integration_none_when_unavailable(self):
        """When pathway_integration module is unavailable, attribute is None."""
        with patch("src.models.signaling_network.PATHWAY_INTEGRATION_AVAILABLE", False):
            mapper = SignalingNetworkMapper(pathway_db_path="kegg")
            assert mapper.pathway_integration is None
            assert len(mapper.pathways) >= 8
