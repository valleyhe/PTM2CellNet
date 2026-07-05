"""KEGG/Reactome pathway database integration (FEAT-02)."""
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from typing_extensions import TypedDict

from ..utils.io import safe_pickle_load

try:
    import sspa
    SSPA_AVAILABLE = True
except ImportError:
    SSPA_AVAILABLE = False
    sspa = None

logger = logging.getLogger(__name__)

# Cache schema version + TTL. Bump CACHE_VERSION when the cache format changes;
# entries older than CACHE_TTL_SECONDS are rebuilt on next load.
CACHE_VERSION = "v1"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# Default cache directory is anchored to the project root (two parents up from
# this module: src/analysis/ -> src/ -> <project root>) so it resolves
# identically regardless of the process cwd. Without an absolute default the
# relative './pathway_cache' would land in the cwd (e.g. /tmp or the home
# directory when services are launched from elsewhere), causing cache misses
# and repeated downloads.
_DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[2] / ".cache" / "pathway_cache")


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class PathwayMatchResult(TypedDict):
    """Shape of the dict returned by ``_find_best_pathway_match``."""

    pathway_id: Optional[str]
    pathway_name: Optional[str]
    genes: Set[str]
    gene_count: int


class InteractionNetworkData(TypedDict):
    """Shape of the dict returned by ``load_interaction_network`` / ``_load_edge_table``."""

    edges: List[Tuple[str, str, float, str]]
    genes: Set[str]
    source: Optional[str]


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

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR):
        """
        Initialize pathway database integration.

        Args:
            cache_dir: Directory to cache pathway data. Defaults to an
                absolute path under the project root (``.cache/pathway_cache``)
                so the cache is independent of the process working directory.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.kegg_pathways: Dict[str, List[str]] = {}
        self.reactome_pathways: Dict[str, List[str]] = {}
        self._kegg_loaded = False
        self._reactome_loaded = False

    def _cache_is_fresh(self, cache_file: Path) -> bool:
        """Return True if a versioned cache file exists and is not expired."""
        if not cache_file.exists():
            return False
        try:
            mtime = cache_file.stat().st_mtime
        except OSError:
            return False
        if (time.time() - mtime) > CACHE_TTL_SECONDS:
            logger.info("Pathway cache expired (TTL %ss): %s", CACHE_TTL_SECONDS, cache_file)
            return False
        try:
            with open(cache_file, "rb") as f:
                payload = safe_pickle_load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            logger.warning("Pathway cache unreadable, will rebuild: %s (%s)", cache_file, e)
            return False
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            logger.info("Pathway cache version mismatch, rebuilding: %s", cache_file)
            return False
        return True

    @staticmethod
    def _write_versioned_cache(cache_file: Path, data: Dict[str, List[str]]) -> None:
        """Persist pathway data with the current cache version."""
        payload = {
            "version": CACHE_VERSION,
            "created_at": time.time(),
            "pathways": data,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(payload, f)

    def load_kegg_pathways(
        self,
        organism: str = "hsa",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Load KEGG pathways using sspa.

        Args:
            organism: KEGG organism code (hsa for human)
            use_cache: Whether to use cached data
            force_refresh: If True, bypass cache and reload from source

        Returns:
            Dict mapping pathway ID to gene list
        """
        if force_refresh:
            use_cache = False
            logger.info("Force refresh requested, bypassing cache")

        cache_file = self.cache_dir / f"kegg_{organism}_{CACHE_VERSION}.pkl"

        # Try cache first
        if use_cache and self._cache_is_fresh(cache_file):
            logger.info("Loading KEGG pathways from cache: %s", cache_file)
            with open(cache_file, 'rb') as f:
                payload = safe_pickle_load(f)
            if isinstance(payload, dict):
                self.kegg_pathways = payload["pathways"]
            else:
                logger.warning("KEGG cache payload is not a dict; reloading from source")
                return self.load_kegg_pathways(organism, use_cache=False)
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
                    self._write_versioned_cache(cache_file, self.kegg_pathways)
                    logger.info("Cached KEGG pathways to %s", cache_file)
                except (pickle.PicklingError, TypeError) as e:
                    logger.warning("Could not cache pathways: %s", e)

        except (ImportError, RuntimeError, OSError) as e:
            # Try loading from local TSV/CSV cache file as network-independent fallback
            local_cache = self.cache_dir / f"kegg_{organism}.tsv"
            if local_cache.exists():
                logger.warning(
                    "sspa/network unavailable (%s). Loading KEGG pathways from local cache: %s",
                    e, local_cache,
                )
                try:
                    import pandas as pd
                    df = pd.read_csv(local_cache, sep="\t")
                    self.kegg_pathways = {
                        str(row.get("pathway_id", f"hsa_{i}")): str(row.get("genes", "")).split(";")
                        for i, row in df.iterrows()
                    }
                    self._kegg_loaded = True
                    return self.kegg_pathways
                except (OSError, ValueError) as cache_err:
                    logger.error("Local KEGG cache also failed: %s", cache_err)
            logger.error("Failed to load KEGG pathways: %s", e)
            raise

        return self.kegg_pathways

    def load_reactome_pathways(
        self,
        organism: str = "Homo sapiens",
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Load Reactome pathways using sspa.

        Args:
            organism: Organism name ("Homo sapiens" for human)
            use_cache: Whether to use cached data
            force_refresh: If True, bypass cache and reload from source

        Returns:
            Dict mapping pathway ID to gene list
        """
        if force_refresh:
            use_cache = False
            logger.info("Force refresh requested, bypassing cache")

        cache_file = self.cache_dir / f"reactome_{organism.replace(' ', '_')}_{CACHE_VERSION}.pkl"

        # Try cache first
        if use_cache and self._cache_is_fresh(cache_file):
            logger.info("Loading Reactome pathways from cache: %s", cache_file)
            with open(cache_file, 'rb') as f:
                payload = safe_pickle_load(f)
            if isinstance(payload, dict):
                self.reactome_pathways = payload["pathways"]
            else:
                logger.warning("Reactome cache payload is not a dict; reloading from source")
                return self.load_reactome_pathways(organism, use_cache=False)
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
                    self._write_versioned_cache(cache_file, self.reactome_pathways)
                    logger.info("Cached Reactome pathways to %s", cache_file)
                except (pickle.PicklingError, TypeError) as e:
                    logger.warning("Could not cache pathways: %s", e)

        except (ImportError, RuntimeError, OSError) as e:
            # Try loading from local TSV/CSV cache file as network-independent fallback
            local_cache = self.cache_dir / f"reactome_{organism.replace(' ', '_')}.tsv"
            if local_cache.exists():
                logger.warning(
                    "sspa/network unavailable (%s). Loading Reactome pathways from local cache: %s",
                    e, local_cache,
                )
                try:
                    import pandas as pd
                    df = pd.read_csv(local_cache, sep="\t")
                    self.reactome_pathways = {
                        str(row.get("pathway_id", f"R-HSA-{i}")): str(row.get("genes", "")).split(";")
                        for i, row in df.iterrows()
                    }
                    self._reactome_loaded = True
                    return self.reactome_pathways
                except (OSError, ValueError) as cache_err:
                    logger.error("Local Reactome cache also failed: %s", cache_err)
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
        best_match: PathwayMatchResult = {
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

    def load_interaction_network(self) -> InteractionNetworkData:
        """Load a protein-protein interaction (PPI) edge table from cache.

        Looks for a preprocessed edge table under ``cache_dir``. Supported
        file stems (tried in order): ``string_edges.tsv``,
        ``biogrid_edges.tsv``, ``ppi_edges.tsv``. The table must have columns
        ``gene_a``, ``gene_b``, ``score`` and optionally ``type``.

        Returns:
            A dict with keys ``edges`` (list of (gene_a, gene_b, score, type)),
            ``genes`` (set of all genes present in the table) and ``source``
            (the file path that was loaded). Returns an empty edge list when
            no interaction data file is available.
        """
        candidate_names = (
            "string_edges.tsv",
            "biogrid_edges.tsv",
            "ppi_edges.tsv",
        )
        for name in candidate_names:
            path = self.cache_dir / name
            if path.exists():
                try:
                    return self._load_edge_table(path)
                except (OSError, ValueError) as e:
                    logger.warning("Failed to read interaction table %s: %s", path, e)
                    continue
        return {"edges": [], "genes": set(), "source": None}

    @staticmethod
    def _load_edge_table(path: Path) -> InteractionNetworkData:
        """Parse a TSV edge table into an edge list + gene set."""
        edges: List[tuple] = []
        genes: Set[str] = set()
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            try:
                col_a = header.index("gene_a")
                col_b = header.index("gene_b")
                col_score = header.index("score")
            except ValueError as e:
                raise ValueError(
                    f"Interaction table {path} missing required column: {e}"
                ) from e
            col_type = header.index("type") if "type" in header else None
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(col_a, col_b, col_score):
                    continue
                gene_a = parts[col_a].strip()
                gene_b = parts[col_b].strip()
                if not gene_a or not gene_b:
                    continue
                try:
                    score = float(parts[col_score])
                except ValueError:
                    score = 1.0
                itype = parts[col_type].strip() if col_type is not None and col_type < len(parts) else "ppi"
                edges.append((gene_a, gene_b, score, itype))
                genes.update({gene_a, gene_b})
        return {"edges": edges, "genes": genes, "source": str(path)}

    def build_pathway_graph(self, pathway_id: str) -> nx.DiGraph:
        """
        Build NetworkX graph from pathway.

        When a preprocessed PPI edge table (STRING/BioGRID) is available in
        ``cache_dir``, the graph uses real interaction edges (weighted by the
        interaction score) restricted to the intersection of pathway genes and
        the interaction network. For built-in pathways, directed
        kinase -> substrate edges are added on top to reflect signaling
        directionality. When no interaction data file is present, the graph
        falls back to a complete (fully-connected) approximation and emits a
        warning.

        Args:
            pathway_id: KEGG or Reactome pathway ID, or a built-in pathway name

        Returns:
            Directed graph of protein interactions
        """
        G = nx.DiGraph()

        # Get genes in pathway
        if pathway_id in self.BUILTIN_PATHWAYS:
            pathway_info = self.BUILTIN_PATHWAYS[pathway_id]
            genes = list(
                pathway_info.get('key_kinases', []) +
                pathway_info.get('key_substrates', [])
            )
            # Deduplicate while preserving order
            seen: Set[str] = set()
            deduped_genes: List[str] = []
            for g in genes:
                if g not in seen:
                    seen.add(g)
                    deduped_genes.append(g)
            genes = deduped_genes
            kinases = set(pathway_info.get('key_kinases', []))
            substrates = set(pathway_info.get('key_substrates', []))
        elif pathway_id in self.reactome_pathways:
            genes = self.reactome_pathways[pathway_id]
            kinases, substrates = set(), set()
        elif pathway_id in self.kegg_pathways:
            genes = self.kegg_pathways[pathway_id]
            kinases, substrates = set(), set()
        else:
            raise ValueError(f"Pathway {pathway_id} not found")

        # Load real interaction network if available
        interaction_data = self.load_interaction_network()
        interaction_edges = interaction_data["edges"]
        interaction_genes = interaction_data["genes"]

        # Add nodes: intersection of pathway genes and interaction network when
        # real data exists, otherwise the full pathway gene set.
        if interaction_edges:
            node_set = set(genes) & interaction_genes
            # Always keep at least the pathway genes so isolated pathway
            # members are still represented as nodes.
            node_set = node_set | set(genes)
            G.add_nodes_from(node_set)
            for gene_a, gene_b, score, itype in interaction_edges:
                if gene_a in node_set and gene_b in node_set:
                    G.add_edge(gene_a, gene_b, weight=score, type=itype)
            logger.debug(
                "Built pathway graph for %s from %s with %d real interaction edges",
                pathway_id,
                interaction_data["source"],
                G.number_of_edges(),
            )
        else:
            logger.warning(
                "No interaction data file found for pathway %s; building a "
                "complete (fully-connected) graph as an approximation. Provide "
                "a STRING/BioGRID edge table under %s for real topology.",
                pathway_id,
                self.cache_dir,
            )
            G.add_nodes_from(genes)
            for i, gene1 in enumerate(genes):
                for gene2 in genes[i + 1:]:
                    G.add_edge(gene1, gene2, weight=1.0, type="approximation")

        # Add directed kinase -> substrate edges for built-in pathways to
        # capture signaling directionality on top of (undirected) PPI edges.
        if kinases and substrates:
            for kinase in kinases:
                for substrate in substrates:
                    if kinase == substrate:
                        continue
                    if kinase in G and substrate in G:
                        # Prefer not to overwrite a real directed edge; only
                        # add the kinase->substrate edge if absent.
                        if not G.has_edge(kinase, substrate):
                            G.add_edge(
                                kinase,
                                substrate,
                                weight=1.0,
                                type="kinase_substrate",
                            )

        return G

    def load_builtin_pathways(self) -> Dict[str, List[str]]:
        """Load built-in pathway definitions without requiring sspa.

        Returns a dict of pathway names to gene lists using the
        8 built-in pathway definitions. This always works regardless
        of whether sspa is installed.

        Returns:
            Dict mapping pathway name to gene list.
        """
        builtin = {}
        for name, info in self.BUILTIN_PATHWAYS.items():
            genes = list(set(
                info.get('key_kinases', []) + info.get('key_substrates', [])
            ))
            builtin[name] = genes
        self.kegg_pathways.update(builtin)
        self.reactome_pathways.update(builtin)
        return builtin

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
