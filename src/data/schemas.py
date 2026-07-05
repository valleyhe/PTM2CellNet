from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from src.api.schemas import PTMSite as ApiPTMSite


# ---------------------------------------------------------------------------
# TypedDict definitions for structured dicts used in this module
# ---------------------------------------------------------------------------

class PTMSiteDict(TypedDict, total=False):
    """Shape of a PTM site dict for serialization / deserialization."""

    position: int
    type: str
    amino_acid: Optional[str]


class PTMRecordDict(TypedDict, total=False):
    """Shape of a PTMRecord dict for serialization / deserialization."""

    protein_accession: str
    position: int
    ptm_type: str
    amino_acid: Optional[str]
    confidence: Optional[float]
    source: Optional[str]


@dataclass
class PTMSite:
    """Base PTM site data container (data layer).

    This is the canonical definition used by internal modules
    (models, data processing). The API layer defines its own
    :class:`src.api.schemas.PTMSite` (Pydantic) that extends this
    with request-validation fields like ``gene_symbol``.
    """
    position: int
    type: str
    amino_acid: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[int, str, Optional[str]]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: PTMSiteDict) -> "PTMSite":
        return cls(
            position=int(data["position"]),
            type=str(data["type"]),
            amino_acid=data.get("amino_acid"),
        )

    def to_api_model(self) -> "ApiPTMSite":
        """Convert to the API-layer Pydantic PTMSite model.

        Requires ``src.api.schemas`` to be importable (i.e. the API
        package and its dependencies are installed).  Only core fields
        shared between the two definitions are transferred; API-only
        fields (e.g. ``gene_symbol``) default to ``None``.
        """
        from src.api.schemas import PTMSite as ApiPTMSite
        return ApiPTMSite(
            position=self.position,
            type=self.type,
            amino_acid=self.amino_acid,
            gene_symbol=None,
        )


@dataclass
class ProteinData:
    accession: str
    sequence: str
    protein_name: Optional[str] = None
    organism: Optional[str] = None


@dataclass
class PTMRecord:
    protein_accession: str
    position: int
    ptm_type: str
    amino_acid: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[str, int, Optional[str], Optional[float]]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: PTMRecordDict) -> "PTMRecord":
        _conf = data.get("confidence")
        return cls(
            protein_accession=str(data["protein_accession"]),
            position=int(data["position"]),
            ptm_type=str(data["ptm_type"]),
            amino_acid=data.get("amino_acid"),
            confidence=float(_conf) if _conf is not None else None,
            source=data.get("source"),
        )


@dataclass
class CellStateLabel:
    label: str
    confidence: float = 1.0


@dataclass
class ModelPrediction:
    cell_state: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class DatasetMetadata:
    source: str
    num_proteins: int = 0
    num_ptm_sites: int = 0
    created_at: Optional[str] = None


def validate_protein_data(data: Dict[str, Union[str, int, List[PTMSiteDict]]]) -> bool:
    """验证基础蛋白质数据契约。"""
    accession = data.get("accession")
    sequence = data.get("sequence")

    if not isinstance(accession, str) or not accession.strip():
        return False
    if not isinstance(sequence, str) or not sequence.strip():
        return False

    ptm_sites = data.get("ptm_sites", [])
    if ptm_sites is None:
        return True
    if not isinstance(ptm_sites, list):
        return False

    try:
        for site in ptm_sites:
            parsed_site = PTMSite.from_dict(site)
            if parsed_site.position < 1 or not parsed_site.type:
                return False
    except (KeyError, TypeError, ValueError):
        return False

    return True


__all__ = [
    "PTMSite",
    "ProteinData",
    "PTMRecord",
    "CellStateLabel",
    "ModelPrediction",
    "DatasetMetadata",
    "validate_protein_data",
]
