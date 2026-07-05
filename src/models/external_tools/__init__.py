"""External bioinformatics tool integration.

Provides interface classes for external bioinformatics tools (AlphaFold, BLAST,
ClustalW, PSIPRED) with real implementations that use public APIs where
available and fall back to built-in methods when external tools are not installed.
"""

from .alphafold import AlphaFoldClient
from .base import (
    AlignmentResult,
    BLASTHit,
    SecondaryStructurePrediction,
    StructurePrediction,
    ToolConfig,
    _needleman_wunsch,
    _nw_score,
)
from .blast import BLASTClient
from .clustalw import ClustalWClient
from .psipred import PSIPREDClient

__all__ = [
    "AlphaFoldClient",
    "AlignmentResult",
    "BLASTClient",
    "BLASTHit",
    "ClustalWClient",
    "PSIPREDClient",
    "SecondaryStructurePrediction",
    "StructurePrediction",
    "ToolConfig",
    "_needleman_wunsch",
    "_nw_score",
]
