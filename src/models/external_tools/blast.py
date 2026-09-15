"""BLAST client for sequence similarity search."""

from typing import List, Optional

from .base import (
    BIO_BLAST_AVAILABLE,
    NCBIWWW,
    NCBIXML,
    RequestsConnectionError,
    RequestsTimeout,
    RequestsHTTPError,
    ToolConfig,
    BLASTHit,
    logger,
)


class BLASTClient:
    """Client for NCBI BLAST sequence similarity search.

    Uses BioPython's ``NCBIWWW.qblast`` to query NCBI remotely.  Falls back
    to an empty result set on failure.
    """

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config: ToolConfig = config or ToolConfig()

    def check_available(self) -> bool:
        """Return whether BioPython BLAST modules are available."""
        return BIO_BLAST_AVAILABLE

    def search(
        self,
        sequence: str,
        database: str = "nr",
        e_value: float = 0.001,
    ) -> List[BLASTHit]:
        """Search for similar sequences using BLAST.

        Args:
            sequence: Query amino acid or nucleotide sequence.
            database: BLAST database name (default ``"nr"``).
            e_value: Expect-value threshold for reporting hits.

        Returns:
            List of BLASTHit dicts with keys ``accession``, ``description``,
            ``e_value``, ``score``, ``identity``.  Returns empty list on failure.
        """
        if not BIO_BLAST_AVAILABLE:
            logger.warning("BioPython BLAST not available; returning []")
            return []

        try:
            result_handle = NCBIWWW.qblast("blastp", database, sequence, expect=e_value)
            blast_records = NCBIXML.parse(result_handle)

            hits: List[BLASTHit] = []
            for record in blast_records:
                for alignment in record.alignments[:20]:
                    for hsp in alignment.hsps:
                        hits.append(
                            BLASTHit(
                                accession=alignment.accession,
                                description=alignment.title,
                                e_value=hsp.expect,
                                score=hsp.score,
                                identity=f"{hsp.identities}/{hsp.align_length}",
                            )
                        )
                    if len(hits) >= 50:
                        break
                if len(hits) >= 50:
                    break

            result_handle.close()
            return hits

        except RequestsConnectionError as exc:
            logger.warning("BLAST search connection failed: %s", exc)
            return []
        except RequestsTimeout:
            logger.warning("BLAST search request timed out")
            return []
        except RequestsHTTPError as exc:
            logger.warning("BLAST search HTTP error: %s", exc)
            return []
        except (ValueError, KeyError) as exc:
            logger.warning("BLAST search response parsing error: %s", exc)
            return []
        except (OSError, RuntimeError) as exc:
            logger.error("Unexpected BLAST search error: %s", exc)
            return []
