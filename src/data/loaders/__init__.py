"""
loaders 包 — 保持 ``from src.data.loaders import DataLoader`` 的向后兼容性。
"""

from .loader import DataLoader
from .types import JsonRecord, UniProtRecord, UniProtRow

__all__ = [
    "DataLoader",
    "UniProtRecord",
    "UniProtRow",
    "JsonRecord",
]
