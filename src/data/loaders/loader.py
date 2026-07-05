"""
DataLoader 组装 — 通过多继承整合所有 mixin。
"""

from .base import DataLoaderBase
from .file_loaders import FileLoaderMixin
from .ptm_database_loaders import PTMDatabaseLoaderMixin
from .uniprot_loader import UniProtLoaderMixin


class DataLoader(
    FileLoaderMixin,
    PTMDatabaseLoaderMixin,
    UniProtLoaderMixin,
    DataLoaderBase,
):
    """
    统一数据加载器。

    通过多继承将文件加载、PTM 数据库加载和 UniProt API 加载整合为单一类，
    保持 ``from src.data.loaders import DataLoader`` 的向后兼容性。
    """
    pass
