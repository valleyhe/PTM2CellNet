"""Gene to UniProt mapping module (FEAT-01)."""
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

try:
    from UniProtMapper import ProtMapper

    _HAS_UNIPROT_MAPPER = True
except ImportError:  # pragma: no cover - exercised only when the package is absent
    ProtMapper = None
    _HAS_UNIPROT_MAPPER = False
    logger.info(
        "UniProtMapper package not available; falling back to requests-based "
        "UniProt ID mapping. Install UniProtMapper (>=0.1) for the optimized path."
    )


class _RequestsUniProtMapper:
    """Fallback UniProt ID mapper backed by the public REST ID-mapping service.

    Replicates the subset of :class:`UniProtMapper.ProtMapper` used by
    :class:`GeneMapper` (the ``get(ids, from_db, to_db)`` method returning
    ``(result_dataframe, failed_ids)``) so callers behave identically when the
    optional :mod:`UniProtMapper` package is not installed.
    """

    _ID_MAPPING_URL = "https://rest.uniprot.org/idmapping"

    def get(
        self,
        ids: List[str],
        from_db: str = "Gene_Name",
        to_db: str = "UniProtKB",
    ) -> "tuple[pd.DataFrame, List[str]]":
        if not ids:
            return pd.DataFrame(columns=["From", "To"]), []

        # Submit the ID-mapping job.
        try:
            submit_response = requests.post(
                f"{self._ID_MAPPING_URL}/run",
                data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
                timeout=30,
            )
            submit_response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("UniProt ID-mapping submission failed: %s", exc)
            return pd.DataFrame(columns=["From", "To"]), list(ids)

        job_id = submit_response.text.strip()
        if not job_id:
            return pd.DataFrame(columns=["From", "To"]), list(ids)

        # Poll for completion.
        import time

        details_url = f"{self._ID_MAPPING_URL}/details/{job_id}"
        for _ in range(60):  # up to ~60s
            try:
                details_response = requests.get(details_url, timeout=30)
            except requests.RequestException as exc:
                logger.warning("UniProt ID-mapping poll failed: %s", exc)
                time.sleep(1.0)
                continue
            if details_response.status_code != 200:
                time.sleep(1.0)
                continue
            try:
                payload = details_response.json()
            except ValueError:
                time.sleep(1.0)
                continue
            if payload.get("jobStatus") in ("FINISHED", "COMPLETED", "FAILED"):
                break
            time.sleep(1.0)

        if not payload or payload.get("jobStatus") not in ("FINISHED", "COMPLETED"):
            logger.warning("UniProt ID-mapping job %s did not complete", job_id)
            return pd.DataFrame(columns=["From", "To"]), list(ids)

        # Fetch the mapped results.
        try:
            results_response = requests.get(
                f"{self._ID_MAPPING_URL}/results/{job_id}",
                params={"size": 500},
                timeout=30,
            )
            results_response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("UniProt ID-mapping results fetch failed: %s", exc)
            return pd.DataFrame(columns=["From", "To"]), list(ids)

        try:
            results_payload = results_response.json()
        except ValueError as exc:
            logger.error("UniProt ID-mapping results are not JSON: %s", exc)
            return pd.DataFrame(columns=["From", "To"]), list(ids)

        rows: List[Dict[str, Any]] = []
        mapped_from: set = set()
        for entry in results_payload.get("results", []):
            from_id = entry.get("from")
            to_obj = entry.get("to")
            to_id = (
                to_obj.get("primaryAccession")
                if isinstance(to_obj, dict)
                else to_obj
            )
            if from_id and to_id:
                rows.append({"From": from_id, "To": to_id})
                mapped_from.add(from_id)

        failed = [gene for gene in ids if gene not in mapped_from]
        return pd.DataFrame(rows, columns=["From", "To"]), failed


def _build_mapper() -> Any:
    """Construct the UniProt mapper, preferring the optimized package.

    Reads the module-level :data:`ProtMapper` so tests that patch
    ``src.analysis.gene_mapper.ProtMapper`` remain effective.
    """
    if _HAS_UNIPROT_MAPPER and ProtMapper is not None:
        return ProtMapper()
    return _RequestsUniProtMapper()


class GeneMapper:
    """Maps gene symbols to UniProt IDs and sequences."""

    def __init__(self) -> None:
        self._mapper = _build_mapper()
        # Cache for gene to UniProt mapping
        self._gene_cache: Dict[str, Optional[str]] = {}
        # Cache for UniProt canonical isoform lookups
        self._isoform_cache: Dict[str, str] = {}

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
            return str(uniprot_id) if uniprot_id is not None else None

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
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

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
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

        Queries the UniProt REST API for the entry's isoform list and returns
        the canonical isoform accession (the sequence marked as canonical by
        UniProt, typically isoform 1). Falls back to the provided UniProt ID
        if the API is unavailable or the entry has a single isoform.

        Args:
            gene_symbol: Gene symbol (unused for the lookup, kept for API
                symmetry; UniProt is keyed by accession).
            uniprot_id: UniProt accession of the entry.

        Returns:
            Canonical isoform accession.
        """
        clean_id = uniprot_id.split(".")[0]
        if clean_id in self._isoform_cache:
            return self._isoform_cache[clean_id]

        url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(
                "Could not fetch isoform info for %s (%s); using provided ID",
                clean_id,
                e,
            )
            self._isoform_cache[clean_id] = clean_id
            return clean_id

        try:
            payload = response.json()
        except ValueError as e:
            logger.warning("Invalid JSON response from UniProt for %s: %s", clean_id, e)
            self._isoform_cache[clean_id] = clean_id
            return clean_id

        # UniProt JSON: top-level "canonicalIsoform" accession, or the first
        # entry of "uniProtKBCanonicalIsoform"/"alternativeProducts" sequence.
        canonical = None
        alternative_products = payload.get("alternativeProducts") or {}
        isoforms = alternative_products.get("isoforms") or []
        for isoform in isoforms:
            if isoform.get("isoformStatus") == "Canonical":
                canonical = isoform.get("isoformAccession")
                break
        if canonical is None and isoforms:
            # Fall back to the first listed isoform.
            canonical = isoforms[0].get("isoformAccession")

        resolved = canonical or clean_id
        self._isoform_cache[clean_id] = resolved
        return resolved


def map_gene_to_uniprot(gene_symbol: str) -> Optional[str]:
    """Convenience function for single gene mapping."""
    mapper = GeneMapper()
    return mapper.map_gene_to_uniprot(gene_symbol)
