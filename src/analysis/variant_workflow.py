# mypy: ignore-errors
"""Complete variant effect prediction workflow (FEAT-01)."""
import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import requests

from .variant_parser import HGVSVariantParser, VariantComponents
from .gene_mapper import GeneMapper
from ..models.variant_effect import VariantPTMEffectPredictor

logger = logging.getLogger(__name__)


def _create_network_analyzer() -> Any:
    """Import the network analyzer lazily to avoid package import cycles."""
    from ..models.signaling_network import PTMNetworkAnalyzer

    return PTMNetworkAnalyzer()


@dataclass
class VariantEffectResult:
    """Complete variant effect prediction result."""
    variant: Dict
    sequence_info: Dict
    ptm_effects: Dict[str, Dict]  # PTM type -> effect
    pathway_impacts: Optional[Dict] = None


class VariantEffectWorkflow:
    """
    Complete workflow: HGVS -> Gene -> UniProt -> Sequence -> Effect Prediction.
    """

    def __init__(
        self,
        model_path: str,
        ptm_types: Optional[List[str]] = None,
    ):
        """
        Initialize workflow.

        Args:
            model_path: Path to trained model
            ptm_types: List of PTM types to predict (default: all supported)
        """
        self.parser = HGVSVariantParser()
        self.gene_mapper = GeneMapper()
        self.network_analyzer = _create_network_analyzer()
        self.model_path = model_path

        # Initialize predictors for each PTM type
        self.ptm_types = ptm_types or [
            'Phosphorylation', 'Ubiquitination', 'Acetylation',
            'Methylation', 'Sumoylation', 'Succinylation',
        ]

        self.predictors: Dict[str, VariantPTMEffectPredictor] = {}
        for ptm_type in self.ptm_types:
            try:
                resolved_path = self._resolve_model_path(ptm_type)
                self.predictors[ptm_type] = VariantPTMEffectPredictor(
                    model_path=resolved_path,
                    ptm_type=ptm_type,
                )
            except Exception as e:
                logger.warning(f"Failed to load predictor for {ptm_type}: {e}")

    def _resolve_model_path(self, ptm_type: str) -> str:
        """Resolve model path specific to PTM type, with fallback to default."""
        type_specific_path = os.path.join(
            "outputs", "ptm_pretrain", ptm_type, "checkpoints", "best_model.pt"
        )
        if os.path.exists(type_specific_path):
            logger.info("Using PTM-type-specific model for %s: %s", ptm_type, type_specific_path)
            return type_specific_path
        type_specific_path_alt = os.path.join(
            "outputs", "ptm_pretrain", ptm_type, "checkpoints", "best.pt"
        )
        if os.path.exists(type_specific_path_alt):
            logger.info("Using PTM-type-specific model for %s: %s", ptm_type, type_specific_path_alt)
            return type_specific_path_alt
        logger.warning(
            "No PTM-type-specific checkpoint found for %s at %s, falling back to default: %s",
            ptm_type, type_specific_path, self.model_path,
        )
        return self.model_path

    def predict_from_hgvs(
        self,
        hgvs_string: str,
        sequence: Optional[str] = None,
        include_pathways: bool = True,
    ) -> VariantEffectResult:
        """
        Predict PTM effects from HGVS variant notation.

        Args:
            hgvs_string: HGVS notation (e.g., "BRAF:p.V600E" or "NP_004324.2:p.Val600Glu")
            sequence: Optional pre-fetched sequence
            include_pathways: Whether to run pathway impact analysis

        Returns:
            Complete effect prediction result
        """
        # Step 1: Parse HGVS
        variant = self.parser.parse(hgvs_string)
        resolved_uniprot_id: Optional[str] = None

        # Step 2: Get sequence if not provided
        if sequence is None:
            sequence, resolved_uniprot_id = self._fetch_sequence_for_variant(variant)

        # Step 3: Validate reference
        is_valid = self.parser.validate_reference(variant, sequence)
        if not is_valid:
            logger.warning(f"Reference validation failed for {hgvs_string}")

        # Step 4: Predict effects for all PTM types
        ptm_effects = {}
        for ptm_type, predictor in self.predictors.items():
            try:
                effect = predictor.predict_variant_effect(
                    sequence=sequence,
                    position=variant.position,  # 1-based
                    ref_aa=variant.ref_aa,
                    alt_aa=variant.alt_aa,
                )
                ptm_effects[ptm_type] = effect
            except Exception as e:
                logger.error(f"Prediction failed for {ptm_type}: {e}")
                ptm_effects[ptm_type] = {
                    'wildtype_prob': 0.0,
                    'mutant_prob': 0.0,
                    'delta_prob': 0.0,
                    'effect': 'error',
                }

        # Step 5: Compute pathway impacts
        pathway_impacts: Optional[Dict] = None
        if include_pathways:
            pathway_impacts = {}
            try:
                uniprot_id = resolved_uniprot_id or self._resolve_variant_uniprot_id(variant)
                if uniprot_id and variant.gene_symbol:
                    network_result = self.network_analyzer.analyze_variant(
                        gene_symbol=variant.gene_symbol,
                        uniprot_id=uniprot_id,
                        position=variant.position,
                        ref_aa=variant.ref_aa,
                        alt_aa=variant.alt_aa,
                        ptm_effects=ptm_effects,
                    )
                    network_effects = network_result.get("network_effects", {})
                    key_pathways = network_effects.get("key_pathways", [])
                    mapper = self.network_analyzer.mapper  # SignalingNetworkMapper instance
                    for pathway_name, activity in key_pathways:
                        output_genes = mapper.pathways.get(pathway_name, {}).get("output_genes", [])
                        confidence = "high" if abs(activity) > 0.5 else "medium"
                        pathway_impacts[pathway_name] = {
                            "activity": activity,
                            "genes": output_genes,
                            "confidence": confidence,
                        }
            except Exception as e:
                logger.warning(f"Pathway analysis failed: {e}")
                pathway_impacts = {}

        return VariantEffectResult(
            variant={
                'hgvs': hgvs_string,
                'gene_symbol': variant.gene_symbol,
                'accession': variant.accession,
                'position': variant.position,
                'ref_aa': variant.ref_aa,
                'alt_aa': variant.alt_aa,
            },
            sequence_info={
                'length': len(sequence),
                'validated': is_valid,
            },
            ptm_effects=ptm_effects,
            pathway_impacts=pathway_impacts,
        )

    def _fetch_sequence_for_variant(
        self,
        variant: VariantComponents,
    ) -> tuple[str, Optional[str]]:
        """
        Fetch protein sequence for variant.

        Args:
            variant: Parsed variant

        Returns:
            Tuple of protein sequence and resolved UniProt accession

        Raises:
            ValueError: If sequence cannot be fetched
        """
        if variant.accession:
            uniprot_id = self._resolve_variant_uniprot_id(variant)
            if uniprot_id is None:
                raise ValueError(f"Could not resolve accession {variant.accession} to UniProt ID")
            return self.fetch_sequence_from_uniprot(uniprot_id), uniprot_id
        if variant.gene_symbol:
            uniprot_id = self.gene_mapper.map_gene_to_uniprot(variant.gene_symbol)
            if uniprot_id is None:
                raise ValueError(f"Could not map gene {variant.gene_symbol} to UniProt ID")
            return self.fetch_sequence_from_uniprot(uniprot_id), uniprot_id
        raise ValueError("No UniProt accession or gene symbol available for sequence fetching")

    def _resolve_variant_uniprot_id(self, variant: VariantComponents) -> Optional[str]:
        """Resolve a parsed variant to a canonical UniProt accession."""
        if variant.gene_symbol and variant.accession == variant.gene_symbol:
            return self.gene_mapper.map_gene_to_uniprot(variant.gene_symbol)
        if variant.accession:
            return self.resolve_accession_to_uniprot(variant.accession)
        if variant.gene_symbol:
            return self.gene_mapper.map_gene_to_uniprot(variant.gene_symbol)
        return None

    def resolve_accession_to_uniprot(self, accession: str) -> str:
        """
        Resolve an external protein accession to a UniProtKB accession.

        Args:
            accession: Protein accession from HGVS input (e.g., RefSeq or UniProt)

        Returns:
            Canonical UniProt accession

        Raises:
            ValueError: If the accession cannot be resolved
        """
        clean_accession = accession.strip()
        if self._looks_like_uniprot_accession(clean_accession):
            return clean_accession.split(".")[0]

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f'xref:RefSeq:{clean_accession}',
            "fields": "accession",
            "format": "json",
            "size": 1,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to resolve accession {accession}: {e}") from e

        payload = response.json()
        results = payload.get("results", [])
        if not results:
            raise ValueError(f"Could not resolve accession {accession} to a UniProt ID")

        uniprot_id = results[0].get("primaryAccession")
        if not uniprot_id:
            raise ValueError(f"Could not resolve accession {accession} to a UniProt ID")
        return uniprot_id

    @staticmethod
    def _looks_like_uniprot_accession(accession: str) -> bool:
        """Heuristic check for canonical UniProtKB accessions."""
        clean_accession = accession.split(".")[0]
        if len(clean_accession) not in (6, 10):
            return False
        return clean_accession[0].isalnum() and clean_accession[-1].isalnum()

    def fetch_sequence_from_uniprot(self, uniprot_id: str) -> str:
        """
        Fetch protein sequence from UniProt by ID.

        Args:
            uniprot_id: UniProt accession (e.g., NP_004324.2 or P04637)

        Returns:
            Protein sequence string

        Raises:
            ValueError: If sequence cannot be fetched
        """
        if not self._looks_like_uniprot_accession(uniprot_id) or uniprot_id.upper().startswith("NP_"):
            raise ValueError(
                f"{uniprot_id} is not a UniProt accession; resolve it before sequence fetch"
            )
        # Normalize: strip version number if present (e.g., NP_004324.2 -> NP_004324)
        clean_id = uniprot_id.split(".")[0]
        url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.fasta"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch sequence from UniProt for {uniprot_id}: {e}") from e

        lines = response.text.split("\n")
        sequence = "".join(line for line in lines if not line.startswith(">"))
        if not sequence:
            raise ValueError(f"Empty sequence returned for {uniprot_id}")
        return sequence

    def predict_batch(
        self,
        variants: List[Dict],
        parallel: bool = False,
        max_workers: int = 4,
    ) -> List[VariantEffectResult]:
        """
        Batch predict variant effects.

        Args:
            variants: List of dicts with 'hgvs' and optional 'sequence'
            parallel: If True, predict variants concurrently using a thread
                pool (suitable for I/O-bound UniProt sequence fetches).
                Defaults to False (serial) for deterministic ordering.
            max_workers: Maximum worker threads when ``parallel`` is True.

        Returns:
            List of prediction results in the same order as ``variants``.
        """
        if not parallel:
            return [self._predict_one(var) for var in variants]

        import concurrent.futures

        results: List[Optional[VariantEffectResult]] = [None] * len(variants)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._predict_one, var): idx
                for idx, var in enumerate(variants)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
        return [r for r in results if r is not None]

    def _predict_one(self, var: Dict) -> VariantEffectResult:
        """Predict a single variant, returning an error result on failure."""
        try:
            result = self.predict_from_hgvs(
                hgvs_string=var['hgvs'],
                sequence=var.get('sequence'),
            )
            return result
        except Exception as e:
            logger.error(f"Batch prediction failed for {var}: {e}")
            # Add error result
            return VariantEffectResult(
                variant={'hgvs': var.get('hgvs'), 'error': str(e)},
                sequence_info={},
                ptm_effects={},
            )
