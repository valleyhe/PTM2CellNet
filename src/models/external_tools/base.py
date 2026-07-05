"""Shared base types, constants, helpers, and optional-dependency checks
for the external_tools package."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency checks (module-level)
# ---------------------------------------------------------------------------

try:
    import requests
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout as RequestsTimeout
    from requests.exceptions import HTTPError as RequestsHTTPError

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    RequestsConnectionError = None
    RequestsTimeout = None
    RequestsHTTPError = None
    REQUESTS_AVAILABLE = False
    logger.info("requests library not installed; HTTP-based external tools will be unavailable.")

try:
    from Bio.Blast import NCBIWWW, NCBIXML

    BIO_BLAST_AVAILABLE = True
except ImportError:
    NCBIWWW = None  # type: ignore[assignment]
    NCBIXML = None  # type: ignore[assignment]
    BIO_BLAST_AVAILABLE = False
    logger.info("Bio.Blast not installed; BLAST web search will be unavailable.")

try:
    from Bio.Align.Applications import ClustalwCommandline, ClustalOmegaCommandline

    BIO_CLUSTAL_APP_AVAILABLE = True
except ImportError:
    ClustalwCommandline = None
    ClustalOmegaCommandline = None
    BIO_CLUSTAL_APP_AVAILABLE = False
    logger.info("Bio.Align.Applications not installed; ClustalW/ClustalOmega will be unavailable.")

try:
    from Bio import Align
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    BIO_ALIGN_AVAILABLE = True
except ImportError:
    Align = None  # type: ignore[assignment]
    SeqIO = None  # type: ignore[assignment]
    Seq = None  # type: ignore[misc, assignment]
    SeqRecord = None  # type: ignore[misc, assignment]
    BIO_ALIGN_AVAILABLE = False
    logger.info("BioPython not installed; alignment and sequence tools will be unavailable.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AA_THREE_LETTER: Dict[str, str] = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

# Chou-Fasman propensity tables (normalized)
_CHOU_FASMAN_HELIX: Dict[str, float] = {
    "A": 1.42,
    "C": 0.77,
    "D": 1.01,
    "E": 1.37,
    "F": 1.13,
    "G": 0.57,
    "H": 1.08,
    "I": 1.09,
    "K": 1.23,
    "L": 1.30,
    "M": 1.20,
    "N": 0.79,
    "P": 0.52,
    "Q": 1.17,
    "R": 1.15,
    "S": 0.79,
    "T": 0.82,
    "V": 1.02,
    "W": 1.14,
    "Y": 0.74,
}

_CHOU_FASMAN_SHEET: Dict[str, float] = {
    "A": 0.83,
    "C": 1.30,
    "D": 0.80,
    "E": 1.02,
    "F": 1.39,
    "G": 0.57,
    "H": 0.87,
    "I": 1.40,
    "K": 0.81,
    "L": 1.22,
    "M": 1.30,
    "N": 0.72,
    "P": 0.64,
    "Q": 1.12,
    "R": 0.99,
    "S": 0.79,
    "T": 1.05,
    "V": 1.43,
    "W": 1.27,
    "Y": 1.32,
}

_CHOU_FASMAN_COIL: Dict[str, float] = {
    "A": 0.75,
    "C": 0.93,
    "D": 1.19,
    "E": 0.61,
    "F": 0.48,
    "G": 1.86,
    "H": 1.05,
    "I": 0.47,
    "K": 1.02,
    "L": 0.48,
    "M": 0.60,
    "N": 1.49,
    "P": 1.84,
    "Q": 0.71,
    "R": 0.85,
    "S": 1.40,
    "T": 1.13,
    "V": 0.41,
    "W": 0.59,
    "Y": 0.94,
}


# ---------------------------------------------------------------------------
# Structured response types (replace Dict[str, Any] patterns)
# ---------------------------------------------------------------------------


@dataclass
class ToolConfig:
    """Configuration dictionary for external tool clients.

    Supports both ``ToolConfig(params={"timeout": 10})`` and
    ``ToolConfig({"timeout": 10})`` (positional dict) for backward
    compatibility with code that passed plain dicts.
    """

    params: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def __init__(
        self,
        params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        self.params = params if params is not None else {}

    def __getitem__(self, key: str) -> Union[str, int, float, bool]:
        return self.params[key]

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def get(
        self, key: str, default: Union[str, int, float, bool, None] = None
    ) -> Union[str, int, float, bool, None]:
        return self.params.get(key, default)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.params == other
        if isinstance(other, ToolConfig):
            return self.params == other.params
        return NotImplemented


@dataclass
class StructurePrediction:
    """Response from protein structure prediction (AlphaFold)."""

    pdb_string: str = ""
    confidence: float = 0.0
    predicted_aligned_error: List[object] = field(default_factory=list)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


@dataclass
class BLASTHit:
    """Single BLAST search hit."""

    accession: str = ""
    description: str = ""
    e_value: float = 0.0
    score: float = 0.0
    identity: str = ""

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class AlignmentResult:
    """Response from multiple sequence alignment (ClustalW)."""

    alignment: Dict[str, str] = field(default_factory=dict)
    phylogenetic_tree: str = ""
    consensus: str = ""

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class SecondaryStructurePrediction:
    """Response from secondary structure prediction (PSIPRED)."""

    ss_prediction: str = ""
    confidence_scores: List[float] = field(default_factory=list)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Helper: Needleman-Wunsch global alignment (fallback when ClustalW absent)
# ---------------------------------------------------------------------------


def _needleman_wunsch(
    seq1: str, seq2: str, match: int = 2, mismatch: int = -1, gap: int = -2
) -> Tuple[str, str]:
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
    i, j = n, m
    a1_chars, a2_chars = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            match if seq1[i - 1] == seq2[j - 1] else mismatch
        ):
            a1_chars.append(seq1[i - 1])
            a2_chars.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            a1_chars.append(seq1[i - 1])
            a2_chars.append("-")
            i -= 1
        else:
            a1_chars.append("-")
            a2_chars.append(seq2[j - 1])
            j -= 1
    return "".join(reversed(a1_chars)), "".join(reversed(a2_chars))


def _nw_score(
    seq1: str, seq2: str, match: int = 2, mismatch: int = -1, gap: int = -2
) -> int:
    """Return the Needleman-Wunsch optimal score (DP table value)."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(2)]
    for j in range(m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        dp[1][0] = i * gap
        for j in range(1, m + 1):
            score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[1][j] = max(
                dp[0][j - 1] + score,
                dp[0][j] + gap,
                dp[1][j - 1] + gap,
            )
        dp[0], dp[1] = dp[1], dp[0]
    return dp[0][m]
