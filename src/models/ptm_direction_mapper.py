"""
PTM Direction Mapper Module

Maps PTM modification events (phosphorylation, ubiquitination, etc.) to
DAVF-compatible perturbation inputs: gene_ids, directions, and attention_mask tensors.

Direction codes (from BiPerturbEncoder):
    KO (Knockout): direction=0, effect=-1
    KD (Knockdown): direction=1, effect=-0.5
    OE (Overexpression): direction=2, effect=+1
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src.api.schemas import PTMSite

logger = logging.getLogger(__name__)


# PTM type → direction mapping (from CONTEXT.md decisions D-01 through D-07)
PTM_DIRECTION_MAP = {
    "phosphorylation": 2,   # OE - activates signaling pathways
    "ubiquitination": 0,    # KO - marks for degradation
    "acetylation": 2,       # OE - enhances activity
    "methylation": 2,       # OE - stabilizing/activating
    "sumoylation": 1,       # KD - partial inhibition
    "neddylation": 2,       # OE - activates ligases
}

# Default direction for unknown PTM types (D-07)
DEFAULT_DIRECTION = 2  # OE - conservative default

# Maximum targets per sample (D-13)
MAX_TARGETS = 32


@dataclass
class PTMDirectionMapperOutput:
    """
    Output from PTMDirectionMapper.

    Tensors are compatible with BiPerturbEncoder input signature.

    Attributes:
        gene_ids: [B, K] tensor of gene indices in Geneformer vocabulary (long)
        directions: [B, K] tensor of direction codes: 0=KO, 1=KD, 2=OE (long)
        attention_mask: [B, K] tensor: 1 for valid targets, 0 for padding/masked (float)
    """
    gene_ids: torch.Tensor
    directions: torch.Tensor
    attention_mask: torch.Tensor


class PTMDirectionMapper:
    """
    Maps PTM modification events to DAVF perturbation inputs.

    Converts PTM type strings to direction codes and resolves gene names
    to Geneformer vocabulary IDs. Unknown genes are masked with attention_mask=0.

    Usage:
        mapper = PTMDirectionMapper(geneformer_loader, gene_mapper)
        output = mapper.map_ptms(ptm_sites, gene_names)
        # output.gene_ids, output.directions, output.attention_mask ready for DAVF
    """

    def __init__(
        self,
        geneformer_loader=None,
        gene_mapper=None,
        max_targets: int = MAX_TARGETS,
    ):
        """
        Initialize PTMDirectionMapper.

        Args:
            geneformer_loader: GeneformerEmbeddingLoader for gene ID resolution.
                              If None, uses get_geneformer_loader() singleton.
            gene_mapper: GeneMapper for gene symbol → UniProt resolution.
                        If None, creates new GeneMapper instance.
            max_targets: Maximum targets per sample (K). Default 32.
        """
        if geneformer_loader is None:
            from src.models.geneformer_embedding import get_geneformer_loader
            geneformer_loader = get_geneformer_loader()

        if gene_mapper is None:
            from src.analysis.gene_mapper import GeneMapper
            gene_mapper = GeneMapper()

        self.geneformer_loader = geneformer_loader
        self.gene_mapper = gene_mapper
        self.max_targets = max_targets

    def _normalize_ptm_type(self, ptm_type: str) -> str:
        """
        Normalize PTM type string for lookup.

        Converts to lowercase and removes hyphens, underscores, and spaces.

        Args:
            ptm_type: Raw PTM type string (e.g., "Phosphorylation", "phospho-rylation")

        Returns:
            Normalized string (e.g., "phosphorylation")
        """
        return (
            ptm_type
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )

    def _get_direction(self, ptm_type: str) -> int:
        """
        Map PTM type to direction code.

        Args:
            ptm_type: PTM type string (will be normalized)

        Returns:
            Direction code: 0=KO, 1=KD, 2=OE
        """
        normalized = self._normalize_ptm_type(ptm_type)
        direction = PTM_DIRECTION_MAP.get(normalized, DEFAULT_DIRECTION)

        if normalized not in PTM_DIRECTION_MAP:
            logger.debug(
                f"Unknown PTM type '{ptm_type}', defaulting to OE (direction=2)"
            )

        return direction

    def _resolve_gene_id(self, gene_name: str) -> Tuple[int, int]:
        """
        Resolve gene name to Geneformer vocabulary ID.

        Resolution chain (D-09):
            1. Check Geneformer vocabulary directly
            2. Try UniProt resolution via GeneMapper
            3. If not found, return (0, 0) to mask out

        Args:
            gene_name: Gene symbol (e.g., "TP53", "BRAF")

        Returns:
            (gene_id, valid_flag): gene ID (0 if unknown) and validity flag (1 or 0)
        """
        # Check if directly in Geneformer vocabulary
        gene_to_idx = getattr(self.geneformer_loader, '_gene_to_idx', {})

        if gene_name in gene_to_idx:
            return gene_to_idx[gene_name], 1

        # Try UniProt resolution via GeneMapper
        try:
            uniprot_id = self.gene_mapper.map_gene_to_uniprot(gene_name)
            if uniprot_id and uniprot_id in gene_to_idx:
                return gene_to_idx[uniprot_id], 1
        except Exception as e:
            logger.debug(f"GeneMapper lookup failed for '{gene_name}': {e}")

        # Unknown gene - mask out (D-08, D-10)
        logger.debug(f"Gene '{gene_name}' not found in vocabulary, masking")
        return 0, 0

    def map_ptms(
        self,
        ptm_sites: List[PTMSite],
        gene_names: List[str],
    ) -> PTMDirectionMapperOutput:
        """
        Map PTM sites to DAVF input tensors.

        Args:
            ptm_sites: List of PTM modification events (from API schema)
            gene_names: Gene name for each PTM site (parallel list, D-11, D-17)

        Returns:
            PTMDirectionMapperOutput with gene_ids, directions, attention_mask tensors

        Raises:
            ValueError: If ptm_sites and gene_names have different lengths
        """
        # Validate parallel lists
        if len(ptm_sites) != len(gene_names):
            raise ValueError(
                f"ptm_sites ({len(ptm_sites)}) and gene_names ({len(gene_names)}) "
                "must have same length"
            )

        # Handle empty input (D-16)
        if not ptm_sites:
            return PTMDirectionMapperOutput(
                gene_ids=torch.zeros(1, self.max_targets, dtype=torch.long),
                directions=torch.zeros(1, self.max_targets, dtype=torch.long),
                attention_mask=torch.zeros(1, self.max_targets, dtype=torch.float),
            )

        # Process each PTM site
        gene_ids = []
        directions = []
        masks = []

        for ptm_site, gene_name in zip(ptm_sites, gene_names):
            # Resolve gene ID
            gene_id, valid = self._resolve_gene_id(gene_name)

            # Get direction code
            direction = self._get_direction(ptm_site.type)

            gene_ids.append(gene_id if valid else 0)
            directions.append(direction)
            masks.append(valid)

        # Truncate to max_targets (D-13)
        num_targets = min(len(gene_ids), self.max_targets)

        # Pad to max_targets with attention_mask=0 (D-16)
        padded_gene_ids = gene_ids[:num_targets] + [0] * (self.max_targets - num_targets)
        padded_directions = directions[:num_targets] + [0] * (self.max_targets - num_targets)
        padded_masks = masks[:num_targets] + [0.0] * (self.max_targets - num_targets)

        return PTMDirectionMapperOutput(
            gene_ids=torch.tensor([padded_gene_ids], dtype=torch.long),
            directions=torch.tensor([padded_directions], dtype=torch.long),
            attention_mask=torch.tensor([padded_masks], dtype=torch.float),
        )

    def map_batch(
        self,
        batch_ptm_sites: List[List[PTMSite]],
        batch_gene_names: List[List[str]],
    ) -> PTMDirectionMapperOutput:
        """
        Map batch of PTM sites to DAVF input tensors.

        Args:
            batch_ptm_sites: Batch of PTM site lists
            batch_gene_names: Batch of gene name lists (parallel to batch_ptm_sites)

        Returns:
            PTMDirectionMapperOutput with shape [B, K] where B=batch size, K=max_targets
        """
        B = len(batch_ptm_sites)
        K = self.max_targets

        # Initialize output tensors
        all_gene_ids = torch.zeros(B, K, dtype=torch.long)
        all_directions = torch.zeros(B, K, dtype=torch.long)
        all_masks = torch.zeros(B, K, dtype=torch.float)

        # Process each sample
        for i, (ptm_sites, gene_names) in enumerate(zip(batch_ptm_sites, batch_gene_names)):
            output = self.map_ptms(ptm_sites, gene_names)
            all_gene_ids[i] = output.gene_ids[0]
            all_directions[i] = output.directions[0]
            all_masks[i] = output.attention_mask[0]

        return PTMDirectionMapperOutput(
            gene_ids=all_gene_ids,
            directions=all_directions,
            attention_mask=all_masks,
        )
