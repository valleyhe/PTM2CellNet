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
from typing import Dict, List, Optional, Tuple, Type

import torch

from src.data.schemas import PTMSite

logger = logging.getLogger(__name__)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# Exception types to catch when gene mapper makes network calls.
_NetworkError: Type[Exception]  
if _REQUESTS_AVAILABLE:
    _NetworkError = _requests.RequestException
else:
    _NetworkError = OSError


# --------------------------------------------------------------------------
# Direction codes
# --------------------------------------------------------------------------
DIRECTION_KO = 0  # Knockout  — effect=-1   (e.g. ubiquitination → degradation)
DIRECTION_KD = 1  # Knockdown — effect=-0.5 (e.g. partial inhibition)
DIRECTION_OE = 2  # Overexpr  — effect=+1   (e.g. activating PTMs)


# --------------------------------------------------------------------------
# V22-05: Extensible PTM direction registry
# --------------------------------------------------------------------------
# Each entry records the *default* direction for a PTM type. New PTM types can
# be added at runtime via ``register_ptm_direction`` (or
# ``PTMDirectionMapper.register_ptm_type``) without editing this module.
#
# Defaults follow CONTEXT.md decisions D-01..D-07 and were extended (V22-05) to
# cover additional biologically relevant PTM types.
PTM_DIRECTION_MAP: Dict[str, int] = {
    # Core signaling PTMs (D-01..D-07)
    "phosphorylation": DIRECTION_OE,   # activates signaling pathways
    "ubiquitination":  DIRECTION_KO,   # marks for degradation
    "acetylation":     DIRECTION_OE,   # enhances activity
    "methylation":     DIRECTION_OE,   # stabilizing/activating (context-dependent, see below)
    "sumoylation":     DIRECTION_KD,   # partial inhibition
    "neddylation":     DIRECTION_OE,   # activates cullin-RING ligases
    # Extended PTM types (V22-05)
    "succinylation":   DIRECTION_OE,   # metabolic / mitochondrial activation
    "malonylation":    DIRECTION_KD,   # metabolic regulation, often repressive
    "glutarylation":   DIRECTION_KD,   # metabolic regulation, often repressive
    "glycosylation":   DIRECTION_OE,   # enhances stability / secretion
    "palmitoylation":  DIRECTION_OE,   # enhances membrane localization
    "hydroxylation":   DIRECTION_OE,   # stabilizing (e.g. HIF1A context)
    "oxidation":       DIRECTION_OE,   # redox signaling activation
    "nitrosylation":   DIRECTION_OE,   # activates signaling (e.g. SNO)
    "adpribosylation": DIRECTION_KD,   # modifies / inhibits target
    "deamidation":     DIRECTION_KD,   # alters function, often inactivating
    "citrullination":  DIRECTION_KD,   # alters charge, often inactivating
    "lactylation":     DIRECTION_OE,   # activates gene expression (recent)
    "crotonylation":   DIRECTION_OE,   # activates gene expression
    "propionylation":  DIRECTION_OE,   # activates gene expression
    "butyrylation":    DIRECTION_OE,   # activates gene expression
    "formylation":     DIRECTION_OE,   # regulatory
    "carbonylation":   DIRECTION_KD,   # oxidative damage marker, inactivating
    "sulfation":       DIRECTION_OE,   # enhances protein-protein interaction
    "myristoylation":  DIRECTION_OE,   # enhances membrane localization
    "prenylation":     DIRECTION_OE,   # enhances membrane localization
    "disulfidebond":   DIRECTION_OE,   # stabilizes structure
    "amidation":       DIRECTION_OE,   # stabilizes / activates peptide
}

# Default direction for unknown PTM types (D-07)
DEFAULT_DIRECTION = DIRECTION_OE  # OE - conservative default

# Maximum targets per sample (D-13)
MAX_TARGETS = 32


def register_ptm_direction(ptm_type: str, direction: int, overwrite: bool = False) -> None:
    """V22-05: Register or override a PTM type → direction mapping at runtime.

    Enables callers (plugins, user config) to extend the PTM registry without
    editing this module.

    Args:
        ptm_type: PTM type name (case-insensitive; normalized on lookup).
        direction: Direction code (0=KO, 1=KD, 2=OE).
        overwrite: If False (default), raise ValueError when the PTM type is
            already registered. Set True to override an existing entry.

    Raises:
        ValueError: If ``direction`` is not in {0, 1, 2}, or if ``overwrite``
            is False and ``ptm_type`` is already registered.
    """
    if direction not in (DIRECTION_KO, DIRECTION_KD, DIRECTION_OE):
        raise ValueError(
            f"direction must be 0 (KO), 1 (KD) or 2 (OE), got {direction}"
        )
    normalized = ptm_type.lower().replace("-", "").replace("_", "").replace(" ", "")
    if not overwrite and normalized in PTM_DIRECTION_MAP:
        raise ValueError(
            f"PTM type '{ptm_type}' already registered; "
            "pass overwrite=True to override"
        )
    PTM_DIRECTION_MAP[normalized] = direction
    logger.debug("Registered PTM direction: %s -> %d", ptm_type, direction)


# --------------------------------------------------------------------------
# V22-03: Pathway-context-aware direction overrides
# --------------------------------------------------------------------------
# Some PTMs are bidirectional depending on biological context. The canonical
# example is methylation: on histones / chromatin / DNA-damage contexts it is
# typically *repressive* (KD), whereas on signaling kinases it is stabilizing
# (OE). This table maps ``(normalized_ptm_type, pathway_keyword) -> direction``
# and takes precedence over the static default.
#
# Pathway keywords are matched case-insensitively against the supplied
# ``pathway_context`` string.
PTM_PATHWAY_CONTEXT_OVERRIDES: Dict[Tuple[str, str], int] = {
    # Methylation is repressive in chromatin / gene-regulation contexts
    ("methylation", "chromatin"):     DIRECTION_KD,
    ("methylation", "histone"):       DIRECTION_KD,
    ("methylation", "dna damage"):    DIRECTION_KD,
    ("methylation", "gene regulation"): DIRECTION_KD,
    ("methylation", "transcription"): DIRECTION_KD,
    # Ubiquitination can be activating in NF-kB context (degradation of IkB)
    ("ubiquitination", "nf-kb"):      DIRECTION_OE,
    ("ubiquitination", "inflammation"): DIRECTION_OE,
    # Sumoylation is activating in nuclear transcription contexts
    ("sumoylation", "transcription"): DIRECTION_OE,
    ("sumoylation", "nuclear"):       DIRECTION_OE,
}


def register_pathway_context_override(
    ptm_type: str, pathway_keyword: str, direction: int
) -> None:
    """V22-03: Register a context-aware direction override at runtime.

    Args:
        ptm_type: PTM type name (normalized on lookup).
        pathway_keyword: Substring to match against the pathway context.
        direction: Direction code (0=KO, 1=KD, 2=OE).

    Raises:
        ValueError: If ``direction`` is not in {0, 1, 2}.
    """
    if direction not in (DIRECTION_KO, DIRECTION_KD, DIRECTION_OE):
        raise ValueError(
            f"direction must be 0 (KO), 1 (KD) or 2 (OE), got {direction}"
        )
    normalized = ptm_type.lower().replace("-", "").replace("_", "").replace(" ", "")
    PTM_PATHWAY_CONTEXT_OVERRIDES[(normalized, pathway_keyword.lower())] = direction
    logger.debug(
        "Registered context override: (%s, %s) -> %d",
        ptm_type, pathway_keyword, direction,
    )


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

    def _get_direction(
        self,
        ptm_type: str,
        pathway_context: Optional[str] = None,
    ) -> int:
        """
        Map PTM type to a direction code (V22-03: pathway-context-aware).

        Resolution order:
            1. If ``pathway_context`` is provided and matches a registered
               ``(ptm_type, keyword)`` override, use the override (V22-03).
            2. Otherwise use the static ``PTM_DIRECTION_MAP`` default (V22-05).
            3. Fall back to ``DEFAULT_DIRECTION`` for unknown PTM types.

        Args:
            ptm_type: PTM type string (will be normalized).
            pathway_context: Optional pathway / biological context string.
                When supplied, context-aware overrides (e.g. repressive
                methylation in chromatin/DNA-damage contexts) take precedence.

        Returns:
            Direction code: 0=KO, 1=KD, 2=OE
        """
        normalized = self._normalize_ptm_type(ptm_type)

        # 1. Context-aware override (V22-03)
        if pathway_context:
            ctx_lower = pathway_context.lower()
            for (ctx_ptm, keyword), direction in PTM_PATHWAY_CONTEXT_OVERRIDES.items():
                if ctx_ptm == normalized and keyword in ctx_lower:
                    logger.debug(
                        "Context override for '%s' in context '%s' -> direction=%d",
                        ptm_type, pathway_context, direction,
                    )
                    return direction

        # 2. Static registry default (V22-05)
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
        except (_NetworkError, ValueError, KeyError) as e:
            logger.debug(f"GeneMapper lookup failed for '{gene_name}': {e}")

        # Unknown gene - mask out (D-08, D-10)
        logger.debug(f"Gene '{gene_name}' not found in vocabulary, masking")
        return 0, 0

    def map_ptms(
        self,
        ptm_sites: List[PTMSite],
        gene_names: List[str],
        pathway_context: Optional[str] = None,
    ) -> PTMDirectionMapperOutput:
        """
        Map PTM sites to DAVF input tensors.

        Args:
            ptm_sites: List of PTM modification events (from API schema)
            gene_names: Gene name for each PTM site (parallel list, D-11, D-17)
            pathway_context: Optional pathway / biological context used to
                resolve context-aware direction overrides (V22-03). When None,
                the static per-PTM defaults are used.

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

            # Get direction code (context-aware when pathway_context given, V22-03)
            direction = self._get_direction(ptm_site.type, pathway_context)

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
        pathway_contexts: Optional[List[Optional[str]]] = None,
    ) -> PTMDirectionMapperOutput:
        """
        Map batch of PTM sites to DAVF input tensors.

        Args:
            batch_ptm_sites: Batch of PTM site lists
            batch_gene_names: Batch of gene name lists (parallel to batch_ptm_sites)
            pathway_contexts: Optional per-sample pathway contexts (V22-03).
                If provided, must be parallel to ``batch_ptm_sites``; each entry
                may be None to use static defaults for that sample.

        Returns:
            PTMDirectionMapperOutput with shape [B, K] where B=batch size, K=max_targets

        Raises:
            ValueError: If ``pathway_contexts`` length differs from batch size.
        """
        B = len(batch_ptm_sites)
        K = self.max_targets

        if pathway_contexts is not None and len(pathway_contexts) != B:
            raise ValueError(
                f"pathway_contexts ({len(pathway_contexts)}) must match batch "
                f"size ({B})"
            )

        # Initialize output tensors
        all_gene_ids = torch.zeros(B, K, dtype=torch.long)
        all_directions = torch.zeros(B, K, dtype=torch.long)
        all_masks = torch.zeros(B, K, dtype=torch.float)

        # Process each sample
        for i, (ptm_sites, gene_names) in enumerate(zip(batch_ptm_sites, batch_gene_names)):
            ctx = pathway_contexts[i] if pathway_contexts else None
            output = self.map_ptms(ptm_sites, gene_names, pathway_context=ctx)
            all_gene_ids[i] = output.gene_ids[0]
            all_directions[i] = output.directions[0]
            all_masks[i] = output.attention_mask[0]

        return PTMDirectionMapperOutput(
            gene_ids=all_gene_ids,
            directions=all_directions,
            attention_mask=all_masks,
        )

    def apply_pathway_contexts_from_mapper(self, pathway_mapper) -> None:
        """Register pathway-context-aware direction overrides from a SignalingNetworkMapper.

        Inspects the pathway mapper's pathway definitions and registers
        context overrides for PTM types that have different effects in
        specific pathway contexts (V22-03).

        Args:
            pathway_mapper: A SignalingNetworkMapper instance whose pathways
                attribute contains pathway definitions with 'ptm_types' and
                'description' fields.
        """
        if pathway_mapper is None:
            return

        pathways = getattr(pathway_mapper, 'pathways', {})
        if not pathways:
            return

        # Context-aware overrides derived from pathway descriptions
        context_keywords = {
            'chromatin': ['methylation', 'acetylation'],
            'histone': ['methylation', 'acetylation', 'ubiquitination'],
            'dna damage': ['phosphorylation', 'ubiquitination', 'sumoylation'],
            'nf-kb': ['ubiquitination', 'phosphorylation'],
            'inflammation': ['ubiquitination', 'phosphorylation'],
            'transcription': ['methylation', 'acetylation', 'sumoylation'],
            'nuclear': ['sumoylation', 'methylation'],
        }

        registered = 0
        for _pathway_name, pathway_info in pathways.items():
            description = pathway_info.get('description', '').lower()
            pathway_keywords = [kw for kw in context_keywords if kw in description]

            for keyword in pathway_keywords:
                for ptm_type in context_keywords[keyword]:
                    # Only register if not already present
                    key = (ptm_type, keyword)
                    if key not in PTM_PATHWAY_CONTEXT_OVERRIDES:
                        register_pathway_context_override(ptm_type, keyword, DIRECTION_KD)
                        registered += 1

        if registered:
            logger.info("Applied %d pathway context overrides from SignalingNetworkMapper", registered)

    # Convenience class-level accessors for the runtime registry (V22-05)
    @staticmethod
    def register_ptm_type(ptm_type: str, direction: int, overwrite: bool = False) -> None:
        """Register a new PTM type → direction mapping (V22-05)."""
        register_ptm_direction(ptm_type, direction, overwrite=overwrite)

    @staticmethod
    def register_context_override(
        ptm_type: str, pathway_keyword: str, direction: int
    ) -> None:
        """Register a pathway-context-aware direction override (V22-03)."""
        register_pathway_context_override(ptm_type, pathway_keyword, direction)


# --------------------------------------------------------------------------
# V22-03 production integration: one-call entry point
# --------------------------------------------------------------------------


def register_pathway_contexts(signaling_mapper) -> int:
    """Register pathway-context-aware direction overrides from a SignalingNetworkMapper.

    This is the production entry point for V22-03 integration. Call this after
    constructing a SignalingNetworkMapper to apply context-aware direction
    overrides::

        from src.models.signaling_network import SignalingNetworkMapper
        from src.models.ptm_direction_mapper import register_pathway_contexts

        snm = SignalingNetworkMapper()
        count = register_pathway_contexts(snm)

    Args:
        signaling_mapper: A SignalingNetworkMapper instance.

    Returns:
        Number of new overrides registered.
    """
    temp_mapper = PTMDirectionMapper.__new__(PTMDirectionMapper)
    temp_mapper.geneformer_loader = None
    temp_mapper.gene_mapper = None
    temp_mapper.max_targets = MAX_TARGETS
    before = len(PTM_PATHWAY_CONTEXT_OVERRIDES)
    temp_mapper.apply_pathway_contexts_from_mapper(signaling_mapper)
    after = len(PTM_PATHWAY_CONTEXT_OVERRIDES)
    return after - before
