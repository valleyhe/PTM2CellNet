"""Gene to UniProt mapping module (FEAT-01)."""
import logging
from typing import Dict, List, Optional

from UniProtMapper import ProtMapper

logger = logging.getLogger(__name__)


class GeneMapper:
    """Maps gene symbols to UniProt IDs and sequences."""

    def __init__(self):
        self._mapper = ProtMapper()
        # Cache for gene to UniProt mapping
        self._gene_cache: Dict[str, Optional[str]] = {}

    def map_gene_to_uniprot(
        self,
        gene_symbol: str,
        organism: str = "9606",  # Human
    ) -> Optional[str]:
        """
        Map gene symbol to primary UniProt ID.

        Args:
            gene_symbol: Gene symbol (e.g., "BRAF")
            organism: NCBI taxonomy ID (default 9606 for human)

        Returns:
            UniProt ID or None if not found
        """
        if gene_symbol in self._gene_cache:
            return self._gene_cache[gene_symbol]

        try:
            result, failed = self._mapper.get(
                ids=[gene_symbol],
                from_db="Gene_Name",
                to_db="UniProtKB",
            )

            if failed:
                logger.warning(f"Failed to map gene {gene_symbol}: {failed}")
                self._gene_cache[gene_symbol] = None
                return None

            if result.empty:
                logger.warning(f"No UniProt mapping found for gene: {gene_symbol}")
                self._gene_cache[gene_symbol] = None
                return None

            # Get primary (canonical) isoform
            # UniProt returns multiple rows for isoforms; take first (canonical)
            uniprot_id = result.iloc[0]["To"]
            self._gene_cache[gene_symbol] = uniprot_id

            logger.debug(f"Mapped {gene_symbol} -> {uniprot_id}")
            return uniprot_id

        except Exception as e:
            logger.error(f"Error mapping gene {gene_symbol}: {e}")
            return None

    def map_genes_batch(
        self,
        gene_symbols: List[str],
        organism: str = "9606",
    ) -> Dict[str, Optional[str]]:
        """
        Batch map gene symbols to UniProt IDs.

        Args:
            gene_symbols: List of gene symbols
            organism: NCBI taxonomy ID

        Returns:
            Dict mapping gene symbol to UniProt ID (or None)
        """
        if not gene_symbols:
            return {}

        # Check cache first
        uncached_genes = []
        results = {}

        for gene in gene_symbols:
            if gene in self._gene_cache:
                results[gene] = self._gene_cache[gene]
            else:
                uncached_genes.append(gene)

        if not uncached_genes:
            return results

        try:
            mapper_results, failed = self._mapper.get(
                ids=uncached_genes,
                from_db="Gene_Name",
                to_db="UniProtKB",
            )

            # Process successful mappings
            if not mapper_results.empty:
                for _, row in mapper_results.iterrows():
                    gene = row["From"]
                    uniprot_id = row["To"]
                    results[gene] = uniprot_id
                    self._gene_cache[gene] = uniprot_id

            # Mark failed as None
            for gene in failed:
                if gene not in results:
                    results[gene] = None
                    self._gene_cache[gene] = None

        except Exception as e:
            logger.error(f"Error in batch gene mapping: {e}")
            # Return what we have, mark rest as None
            for gene in uncached_genes:
                if gene not in results:
                    results[gene] = None

        return results

    def get_canonical_isoform(
        self,
        gene_symbol: str,
        uniprot_id: str,
    ) -> str:
        """
        Get canonical isoform for a gene.

        For now, returns the provided UniProt ID.
        Future: Could query UniProt for isoform info.
        """
        return uniprot_id


def map_gene_to_uniprot(gene_symbol: str) -> Optional[str]:
    """Convenience function for single gene mapping."""
    mapper = GeneMapper()
    return mapper.map_gene_to_uniprot(gene_symbol)
