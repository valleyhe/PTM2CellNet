"""
PTMDatabaseLoaderMixin — PhosphoSitePlus / dbPTM / CPLM / EPSD 数据库加载。
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import requests

from ...utils.logging import setup_logger

if TYPE_CHECKING:
    from .base import DataLoaderBase  # noqa: F401

logger = setup_logger(__name__)


class PTMDatabaseLoaderMixin:
    """PTM 数据库加载 mixin (PhosphoSitePlus / dbPTM / CPLM)。

    The attributes and private methods listed below are provided at runtime
    by DataLoaderBase via multiple inheritance.  They are declared here as
    TYPE_CHECKING-only stubs so that mypy can see them without requiring
    the mixin to inherit from DataLoaderBase directly.
    """

    if TYPE_CHECKING:
        data_raw_dir: str
        strict_load: bool
        config: Dict[str, Any]

        def _download_if_url(self, path_or_url: str) -> str: ...
        def _empty_or_raise(self, empty_df: pd.DataFrame, source: str) -> pd.DataFrame: ...
        @staticmethod
        def _empty_phosphositeplus_df() -> pd.DataFrame: ...
        @staticmethod
        def _empty_ptm_df() -> pd.DataFrame: ...
        def _read_tabular_file(self, file_path: str, sep: Optional[str] = None) -> pd.DataFrame: ...
        def _get_matching_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]: ...
        def _build_ptm_dataframe(
            self,
            df: pd.DataFrame,
            accession_candidates: List[str],
            position_candidates: List[str],
            amino_acid_candidates: List[str],
            source: str,
            ptm_type: str,
        ) -> pd.DataFrame: ...
        @staticmethod
        def _normalize_column_name(column_name: str) -> str: ...
        @staticmethod
        def _extract_amino_acid_and_position(
            value: object,
        ) -> Tuple[Optional[str], Optional[int]]: ...

    def load_from_phosphositeplus(
        self,
        file_path_or_url: str,
        organism: str = "human",
        ptm_type: str = "phosphorylation",
    ) -> pd.DataFrame:
        """
        Load PTM site annotations from a PhosphoSitePlus tabular export.

        Parameters:
            file_path_or_url: Local ``.gz``/``.txt`` file path or a PhosphoSitePlus downloads URL.
            organism: Organism filter applied when an organism/species column is present.
            ptm_type: PTM type label assigned to returned rows.

        Returns:
            DataFrame with columns ``protein_accession``, ``gene``, ``position``,
            ``ptm_type``, ``amino_acid``, ``confidence`` and ``source``.
        """
        try:
            local_path = self._download_if_url(file_path_or_url)
        except requests.RequestException as exc:
            logger.error("下载 PhosphoSitePlus 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_phosphositeplus_df(), "PhosphoSitePlus")

        try:
            df = self._read_tabular_file(local_path, sep="\t")
            organism_col = self._get_matching_column(df, ["ORGANISM", "species"])
            if organism_col is not None:
                organism_value = organism.strip().lower()
                df = df[df[organism_col].astype(str).str.strip().str.lower() == organism_value]

            accession_col = self._get_matching_column(df, ["ACC_ID", "accession", "UniProt Accession"])
            gene_col = self._get_matching_column(df, ["GENE", "gene symbol", "gene"])
            mod_rsd_col = self._get_matching_column(df, ["MOD_RSD", "modified residue"])
            confidence_col = self._get_matching_column(df, ["confidence", "CONFIDENCE", "site group id", "SITE_GRP_ID"])
            # 优先匹配真正的 confidence 列；若 PhosphoSitePlus 导出无该列，
            # 则回退到 site group id（非置信度）并在日志中说明。
            # 注意：经 _normalize_column_name 处理后 "site group id" -> "sitegroupid"，
            # 而 "SITE_GRP_ID" -> "sitegrpid"（下划线被移除但 g 与 grp 不同），两者均需覆盖。
            confidence_is_group_id = confidence_col is not None and (
                self._normalize_column_name(confidence_col) in {"sitegroupid", "sitegrpid"}
            )
            if confidence_is_group_id:
                logger.warning(
                    "PhosphoSitePlus 数据未找到真正的 confidence 列，"
                    "将使用 '%s'（site group id）填充 confidence 字段，"
                    "该值并非置信度，应按 pd.NA 处理。",
                    confidence_col,
                )
                confidence_col = None

            if accession_col is None or gene_col is None or mod_rsd_col is None:
                raise ValueError("PhosphoSitePlus 数据缺少必要列")

            parsed = df[mod_rsd_col].apply(self._extract_amino_acid_and_position)
            amino_acids = parsed.apply(lambda item: item[0])
            positions = parsed.apply(lambda item: item[1])

            result = pd.DataFrame(
                {
                    "protein_accession": df[accession_col].astype(str).str.strip(),
                    "gene": df[gene_col].astype(str).str.strip(),
                    "position": positions,
                    "ptm_type": ptm_type,
                    "amino_acid": amino_acids,
                    "confidence": (
                        pd.to_numeric(df[confidence_col], errors="coerce") if confidence_col is not None else pd.NA
                    ),
                    "source": "PhosphoSitePlus",
                }
            )
            result = result.dropna(subset=["protein_accession", "position", "amino_acid"])
            result = result[result["protein_accession"] != ""].copy()
            result["position"] = result["position"].astype(int)
            return cast(pd.DataFrame, result.reset_index(drop=True))
        except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("解析 PhosphoSitePlus 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_phosphositeplus_df(), "PhosphoSitePlus")

    def _load_generic_ptm_table(
        self,
        file_path_or_url: str,
        *,
        source: str,
        ptm_type: str,
        accession_candidates: List[str],
        position_candidates: List[str],
        amino_acid_candidates: List[str],
        sep: Optional[str] = None,
    ) -> pd.DataFrame:
        """Shared download/parse/fallback body of the external PTM loaders (TD-NEW-08)."""

        try:
            local_path = self._download_if_url(file_path_or_url)
        except requests.RequestException as exc:
            logger.error("下载 %s 数据失败: %s", source, exc)
            return self._empty_or_raise(self._empty_ptm_df(), source)

        try:
            if sep is None:
                df = self._read_tabular_file(local_path)
            else:
                df = self._read_tabular_file(local_path, sep=sep)
            return self._build_ptm_dataframe(
                df=df,
                accession_candidates=accession_candidates,
                position_candidates=position_candidates,
                amino_acid_candidates=amino_acid_candidates,
                source=source,
                ptm_type=ptm_type,
            )
        except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("解析 %s 数据失败: %s", source, exc)
            return self._empty_or_raise(self._empty_ptm_df(), source)

    def load_from_dbptm(self, file_path_or_url: str, ptm_type: str = "phosphorylation") -> pd.DataFrame:
        """
        Load PTM annotations from a dbPTM CSV/TSV export.

        Parameters:
            file_path_or_url: Local file path or downloadable URL.
            ptm_type: PTM type label assigned to returned rows.

        Returns:
            DataFrame with columns ``protein_accession``, ``position``, ``ptm_type``,
            ``amino_acid`` and ``source``.
        """
        return self._load_generic_ptm_table(
            file_path_or_url,
            source="dbPTM",
            ptm_type=ptm_type,
            accession_candidates=[
                "UniProtKB Accession",
                "UniProt Accession",
                "protein_accession",
                "ACC_ID",
                "accession",
            ],
            position_candidates=["Position", "Site Position", "site", "pos"],
            amino_acid_candidates=["Residue", "Amino Acid", "AA", "amino_acid"],
        )

    def load_from_epsd(self, file_path_or_url: str, ptm_type: str = "phosphorylation") -> pd.DataFrame:
        """
        Load PTM annotations from a tab-delimited EPSD export.

        EPSD (Eukaryotic Phosphorylation Sites Database) tabular release
        columns: ``EPSD ID``, ``UniProt ID``, ``AA``, ``Position``,
        ``Source``, ``Reference``.

        Parameters:
            file_path_or_url: Local file path or downloadable URL.
            ptm_type: PTM type label assigned to returned rows (default
                ``phosphorylation``, matching the EPSD scope).

        Returns:
            DataFrame with columns ``protein_accession``, ``position``, ``ptm_type``,
            ``amino_acid`` and ``source``.
        """
        return self._load_generic_ptm_table(
            file_path_or_url,
            source="EPSD",
            ptm_type=ptm_type,
            sep="\t",
            accession_candidates=[
                "UniProt ID",
                "UniProt Accession",
                "UniProtKB Accession",
                "protein_accession",
                "ACC_ID",
                "accession",
            ],
            position_candidates=["Position", "Site Position", "site", "pos"],
            amino_acid_candidates=["Amino Acid", "Residue", "AA", "amino_acid"],
        )

    def load_from_cplm(self, file_path_or_url: str, ptm_type: str = "cysteine") -> pd.DataFrame:
        """
        Load PTM annotations from a tab-delimited CPLM export.

        Parameters:
            file_path_or_url: Local file path or downloadable URL.
            ptm_type: PTM type label assigned to returned rows.

        Returns:
            DataFrame with columns ``protein_accession``, ``position``, ``ptm_type``,
            ``amino_acid`` and ``source``.
        """
        return self._load_generic_ptm_table(
            file_path_or_url,
            source="CPLM",
            ptm_type=ptm_type,
            sep="\t",
            accession_candidates=[
                "UniProt Accession",
                "UniProtKB Accession",
                "protein_accession",
                "ACC_ID",
                "accession",
            ],
            position_candidates=["Position", "Site Position", "site", "pos"],
            amino_acid_candidates=["Amino Acid", "Residue", "AA", "amino_acid"],
        )
