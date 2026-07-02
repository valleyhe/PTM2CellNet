"""KEGG/Reactome pathway database integration (FEAT-02)."""
# mypy: disable-error-code="arg-type,assignment,dict-item,operator,return-value,name-defined"
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Set

import networkx as nx

try:
    import sspa
    SSPA_AVAILABLE = True
except ImportError:
    SSPA_AVAILABLE = False
    sspa = None

logger = logging.getLogger(__name__)


class PathwayDatabaseIntegration:
    """Integrate external KEGG/Reactome pathway databases."""

    # 8 built-in pathways for validation
    BUILTIN_PATHWAYS = {
        'MAPK/ERK': {
            'description': 'RAS-RAF-MEK-ERK pathway',
            'key_kinases': ['BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3'],
            'key_substrates': ['EGFR', 'KRAS', 'NRAS', 'HRAS'],
        },
        'PI3K/AKT': {
            'description': 'PI3K-AKT-mTOR pathway',
            'key_kinases': ['PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'MTOR'],
            'key_substrates': ['PTEN', 'PDK1', 'TSC2', 'GSK3B'],
        },
        'JAK/STAT': {
            'description': 'JAK-STAT signaling pathway',
            'key_kinases': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
            'key_substrates': ['STAT1', 'STAT2', 'STAT3', 'STAT5A', 'STAT5B'],
        },
        'NF-kB': {
            'description': 'NF-kB inflammatory signaling',
            'key_kinases': ['IKBKB', 'IKBKA', 'CHUK'],
            'key_substrates': ['NFKBIA', 'NFKBIB', 'RELA', 'NFKB1'],
        },
        'Wnt/beta-catenin': {
            'description': 'Wnt signaling pathway',
            'key_kinases': ['GSK3B', 'CSNK1A1', 'CSNK2A1'],
            'key_substrates': ['CTNNB1', 'APC', 'AXIN1'],
        },
        'Cell Cycle': {
            'description': 'Cell cycle regulation',
            'key_kinases': ['CDK1', 'CDK2', 'CDK4', 'CDK6'],
            'key_substrates': ['RB1', 'TP53', 'CCND1', 'CCNE1'],
        },
        'Apoptosis': {
            'description': 'Apoptosis pathway',
            'key_kinases': ['CASP3', 'CASP8', 'CASP9'],
            'key_substrates': ['BCL2', 'BAX', 'PARP1', 'XIAP'],
        },
        'DNA Damage': {
            'description': 'DNA damage response',
            'key_kinases': ['ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53'],
            'key_substrates': ['H2AX', 'BRCA1', 'BRCA2', 'RAD51'],
        },
    }

    def __init__(self, cache_dir: str = "./pathway_cache"):
        """
        Initialize pathway database integration.

        Args:
            cache_dir: Directory to cache pathway data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.kegg_pathways: Dict[str, List[str]] = {}
        self.reactome_pathways: Dict[str, List[str]] = {}
        self._kegg_loaded = False
        self._reactome_loaded = False

    def load_kegg_pathways(
        self,
        organism: str = "hsa",
        use_cache: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Load KEGG pathways using sspa.

        Args:
            organism: KEGG organism code (hsa for human)
            use_cache: Whether to use cached data

        Returns:
            Dict mapping pathway ID to gene list
        """
        cache_file = self.cache_dir / f"kegg_{organism}.pkl"

        # Try cache first
        if use_cache and cache_file.exists():
            logger.info("Loading KEGG pathways from cache: %s", cache_file)
            with open(cache_file, 'rb') as f:
                self.kegg_pathways = pickle.load(f)  # nosec - cached experiment results from trusted source
            self._kegg_loaded = True
            return self.kegg_pathways

        # Load from KEGG via sspa
        logger.info("Loading KEGG pathways for organism %s", organism)
        try:
            if not SSPA_AVAILABLE or sspa is None:
                raise ImportError("sspa library not available (requires rpy2)")
            self.kegg_pathways = sspa.process_kegg(organism=organism)
            self._kegg_loaded = True

            # Cache results
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.kegg_pathways, f)
                    logger.info("Cached KEGG pathways to %s", cache_file)
                except (pickle.PicklingError, TypeError) as e:
                    logger.warning("Could not cache pathways: %s", e)

        except Exception as e:
            logger.error("Failed to load KEGG pathways: %s", e)
            raise

        return self.kegg_pathways

    def load_reactome_pathways(
        self,
        organism: str = "Homo sapiens",
        use_cache: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Load Reactome pathways using sspa.

        Args:
            organism: Organism name ("Homo sapiens" for human)
            use_cache: Whether to use cached data

        Returns:
            Dict mapping pathway ID to gene list
        """
        cache_file = self.cache_dir / f"reactome_{organism.replace(' ', '_')}.pkl"

        # Try cache first
        if use_cache and cache_file.exists():
            logger.info("Loading Reactome pathways from cache: %s", cache_file)
            with open(cache_file, 'rb') as f:
                self.reactome_pathways = pickle.load(f)  # nosec - cached experiment results from trusted source
            self._reactome_loaded = True
            return self.reactome_pathways

        # Load from Reactome via sspa
        logger.info("Loading Reactome pathways for %s", organism)
        try:
            if not SSPA_AVAILABLE or sspa is None:
                raise ImportError("sspa library not available (requires rpy2)")
            self.reactome_pathways = sspa.process_reactome(organism=organism)
            self._reactome_loaded = True

            # Cache results
            if use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.reactome_pathways, f)
                    logger.info("Cached Reactome pathways to %s", cache_file)
                except (pickle.PicklingError, TypeError) as e:
                    logger.warning("Could not cache pathways: %s", e)

        except Exception as e:
            logger.error("Failed to load Reactome pathways: %s", e)
            raise

        return self.reactome_pathways

    def validate_builtin_pathways(
        self,
        min_coverage: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Validate built-in pathways against external databases.

        Args:
            min_coverage: Minimum gene coverage threshold (0-1)

        Returns:
            Validation report with coverage statistics
        """
        if not self._reactome_loaded:
            self.load_reactome_pathways()

        validation_report = {}

        for pathway_name, pathway_info in self.BUILTIN_PATHWAYS.items():
            builtin_genes = set(
                pathway_info.get('key_kinases', []) +
                pathway_info.get('key_substrates', [])
            )

            # Find best matching Reactome pathway
            best_match = self._find_best_pathway_match(
                pathway_name, builtin_genes
            )

            coverage = 0.0
            if builtin_genes:
                coverage = len(builtin_genes & best_match['genes']) / len(builtin_genes)

            validation_report[pathway_name] = {
                'builtin_gene_count': len(builtin_genes),
                'reactome_match_id': best_match['pathway_id'],
                'reactome_match_name': best_match['pathway_name'],
                'reactome_gene_count': best_match['gene_count'],
                'overlap': len(builtin_genes & best_match['genes']),
                'coverage': coverage,
                'validated': coverage >= min_coverage,
            }

        return validation_report

    def _find_best_pathway_match(
        self,
        pathway_name: str,
        builtin_genes: Set[str],
    ) -> Dict:
        """Find pathway with maximum gene overlap."""
        best_match: Dict[str, Any] = {
            'pathway_id': None,
            'pathway_name': None,
            'genes': set(),
            'gene_count': 0,
        }
        best_overlap = 0

        for reactome_id, reactome_genes in self.reactome_pathways.items():
            reactome_gene_set = set(reactome_genes)
            overlap = len(builtin_genes & reactome_gene_set)

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = {
                    'pathway_id': reactome_id,
                    'pathway_name': reactome_id,  # Could extract name from metadata
                    'genes': reactome_gene_set,
                    'gene_count': len(reactome_gene_set),
                }

        return best_match

    def build_pathway_graph(self, pathway_id: str) -> nx.DiGraph:
        """
        Build NetworkX graph from pathway.

        Args:
            pathway_id: KEGG or Reactome pathway ID

        Returns:
            Directed graph of protein interactions
        """
        G = nx.DiGraph()

        # Get genes in pathway
        genes = []
        if pathway_id in self.reactome_pathways:
            genes = self.reactome_pathways[pathway_id]
        elif pathway_id in self.kegg_pathways:
            genes = self.kegg_pathways[pathway_id]
        else:
            raise ValueError(f"Pathway {pathway_id} not found")

        # Add nodes
        G.add_nodes_from(genes)

        # Add edges (simplified - would use actual interaction data)
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                G.add_edge(gene1, gene2, weight=1.0)

        return G

    def clear_cache(self):
        """Clear all cached pathway data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            logger.info("Removed cache file: %s", cache_file)


def load_kegg_pathways(organism: str = "hsa") -> Dict[str, List[str]]:
    """Convenience function to load KEGG pathways."""
    integration = PathwayDatabaseIntegration()
    return integration.load_kegg_pathways(organism)


def load_reactome_pathways(organism: str = "Homo sapiens") -> Dict[str, List[str]]:
    """Convenience function to load Reactome pathways."""
    integration = PathwayDatabaseIntegration()
    return integration.load_reactome_pathways(organism)
