from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class PTMSite:
    position: int
    type: str
    amino_acid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PTMSite":
        return cls(
            position=int(data["position"]),
            type=str(data["type"]),
            amino_acid=data.get("amino_acid"),
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PTMRecord":
        return cls(
            protein_accession=str(data["protein_accession"]),
            position=int(data["position"]),
            ptm_type=str(data["ptm_type"]),
            amino_acid=data.get("amino_acid"),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
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


def validate_protein_data(data: Dict[str, Any]) -> bool:
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
