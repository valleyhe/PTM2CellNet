"""AlphaFold client for protein structure prediction."""

from typing import Optional

from .base import (
    REQUESTS_AVAILABLE,
    requests,
    RequestsConnectionError,
    RequestsTimeout,
    RequestsHTTPError,
    ToolConfig,
    StructurePrediction,
    _AA_THREE_LETTER,
    logger,
)


class AlphaFoldClient:
    """Client for AlphaFold protein structure prediction.

    Uses the public EBI AlphaFold Database API to fetch pre-computed structures.
    Falls back to a simple extended-chain PDB when the API is unreachable.
    """

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config: ToolConfig = config or ToolConfig()

    def check_available(self) -> bool:
        """Return whether the AlphaFold API is reachable."""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            resp = requests.get("https://alphafold.ebi.ac.uk/api", timeout=5)
            return bool(resp.ok)
        except RequestsConnectionError as e:
            logger.warning("AlphaFold API connection failed: %s", e)
            return False
        except RequestsTimeout:
            logger.warning("AlphaFold API request timed out")
            return False
        except RequestsHTTPError as e:
            logger.warning("AlphaFold API HTTP error: %s", e)
            return False
        except (OSError, RuntimeError) as e:
            logger.error("Unexpected error checking AlphaFold API: %s", e)
            return False

    def predict_structure(
        self, sequence: str, **kwargs: str
    ) -> StructurePrediction:
        """Predict the 3D structure of a protein.

        Tries the EBI AlphaFold Database API first (requires *uniprot_id* in
        *kwargs*).  Falls back to a simple extended-chain PDB string.

        Args:
            sequence: Amino acid sequence (single-letter codes).
            **kwargs: May include ``uniprot_id`` (str) for the EBI API lookup.

        Returns:
            StructurePrediction with keys ``pdb_string``, ``confidence``, and
            ``predicted_aligned_error``.
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests module not available; using fallback PDB")
            return self._fallback_pdb(sequence)

        uniprot_id = kwargs.get("uniprot_id", "")
        if uniprot_id:
            try:
                url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
                resp = requests.get(url, timeout=30)
                if resp.ok:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        entry = data[0]
                    else:
                        entry = data

                    pdb_url = entry.get("pdbUrl", "")
                    confidence = float(entry.get("confidence", 0.0))
                    pae = entry.get("predictedAlignedError", [])

                    if pdb_url:
                        pdb_resp = requests.get(pdb_url, timeout=60)
                        if pdb_resp.ok:
                            return StructurePrediction(
                                pdb_string=pdb_resp.text,
                                confidence=confidence / 100.0 if confidence > 1.0 else confidence,
                                predicted_aligned_error=pae,
                            )
            except RequestsConnectionError as exc:
                logger.warning("AlphaFold EBI API connection failed: %s", exc)
            except RequestsTimeout:
                logger.warning("AlphaFold EBI API request timed out")
            except RequestsHTTPError as exc:
                logger.warning("AlphaFold EBI API HTTP error: %s", exc)
            except (ValueError, KeyError) as exc:
                logger.warning("AlphaFold EBI API response parsing error: %s", exc)
            except (OSError, RuntimeError) as exc:
                logger.error("Unexpected AlphaFold EBI API error: %s", exc)

        logger.info("Using fallback PDB generation")
        return self._fallback_pdb(sequence)

    def _fallback_pdb(self, sequence: str) -> StructurePrediction:
        """Generate a simple extended-chain PDB string."""
        lines = []
        x, y, z = 0.0, 0.0, 0.0
        n = len(sequence)
        plddt = 50.0
        confidence = 0.0

        for i, aa in enumerate(sequence):
            res_name = _AA_THREE_LETTER.get(aa.upper(), "UNK")
            x += 3.8
            y = 0.5 * ((i % 4) - 1.5)
            z = 0.3 * ((i % 3) - 1)
            occ = 1.00
            b = plddt * (1.0 - 0.5 * (i / max(n, 1)))
            serial = i + 1
            resnum = i + 1
            lines.append(
                f"ATOM  {serial:5d}  CA  {res_name:3s} A{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          CA"
            )
        pdb_str = "\n".join(lines)

        return StructurePrediction(
            pdb_string=pdb_str,
            confidence=confidence,
            predicted_aligned_error=[],
            model_kind="synthetic_fallback",
        )
