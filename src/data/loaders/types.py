"""
类型定义 — UniProtRecord / UniProtRow / JsonRecord / _ReadCsvKwargs
从 loaders.py 拆分，供包内各模块共用。
"""

from typing import Dict, List, Optional, TypedDict, Union


class UniProtRecord(TypedDict, total=False):
    """Shape of a single record from the UniProt JSON API response."""
    primaryAccession: str
    sequence: Dict[str, str]
    genes: List[Dict[str, Dict[str, str]]]


class UniProtRow(TypedDict, total=False):
    """Row produced by load_from_uniprot before DataFrame construction."""
    sequence: str
    gene_symbol: str
    accession: str


class JsonRecord(TypedDict, total=False):
    """Record returned by load_from_json — keys depend on source file."""


# Pandas read_csv keyword-argument bag; values are str/int/bool/None
_ReadCsvKwargs = Dict[str, Union[str, int, bool, None]]
