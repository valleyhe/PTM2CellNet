"""
数据加载器模块
功能概述: 从各种数据源加载蛋白质序列和PTM数据
设计思路: 提供统一的加载接口，支持CSV、FASTA、JSON等多种格式
"""

import os
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, cast
from urllib.parse import urlparse

import pandas as pd
import requests
from Bio import SeqIO

from ..utils.logging import setup_logger
from ..utils.helpers import validate_sequence, clean_sequence

logger = setup_logger(__name__)


class DataLoader:
    """
    数据加载器类
    负责从各种数据源加载原始数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据加载器

        参数:
            config: 配置字典，包含路径等设置
        """
        self.config = config or {}
        self.data_raw_dir = self.config.get("paths", {}).get("data_raw", "data/raw")
        self.valid_amino_acids = set(
            self.config.get("data", {}).get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY")
        )
        # strict 模式下，PTM 数据库下载/解析失败抛出 RuntimeError 而非静默返回空
        # DataFrame，避免下游在不知情时用空数据训练。
        self.strict_load = bool(self.config.get("data", {}).get("strict_load", False))

    def _empty_or_raise(self, empty_df: pd.DataFrame, source: str) -> pd.DataFrame:
        """strict 模式下抛错，否则返回空 DataFrame。

        参数:
            empty_df: 静默回退时返回的空 DataFrame
            source: 数据源名称，用于错误信息

        返回:
            空 DataFrame（非 strict 模式）

        抛出:
            RuntimeError: strict 模式下
        """
        if self.strict_load:
            raise RuntimeError(
                f"{source} 数据加载失败（strict_load=True）。"
                "请检查数据源路径/网络，或在 config 中设置 data.strict_load=False 以回退到空 DataFrame。"
            )
        return empty_df

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        从表格文件加载数据

        支持自动识别 HDF5 文件（.h5/.hdf5 后缀），通过 ``pd.read_hdf`` 读取，
        将 HDF5 接入主数据链路；其他后缀按 CSV 处理。

        参数:
            file_path: 数据文件路径（.csv/.h5/.hdf5）

        返回:
            DataFrame对象

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")

        # HDF5 分支：将 HDF5 接入主数据加载链路（配置 data.use_hdf5 开关或路径后缀）
        use_hdf5 = self.config.get("data", {}).get("use_hdf5", False)
        if use_hdf5 or file_path.lower().endswith((".h5", ".hdf5")):
            logger.info("从HDF5加载数据: %s", file_path)
            try:
                df = pd.read_hdf(file_path)
            except (ImportError, ValueError, KeyError) as exc:
                # read_hdf 需要 pytables；若不可用或 key 不匹配，回退到 src.utils.io.load_hdf5
                logger.warning(
                    "pd.read_hdf 失败 (%s)，尝试使用 src.utils.io.load_hdf5 回退", exc,
                )
                from ..utils.io import load_hdf5
                data = load_hdf5(file_path)
                if isinstance(data, pd.DataFrame):
                    df = data
                elif isinstance(data, dict) and len(data) == 1:
                    df = next(iter(data.values()))
                else:
                    raise
            logger.info("加载完成，共 %d 条记录", len(df))
            return df

        logger.info("从CSV加载数据: %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("加载完成，共 %d 条记录", len(df))
        return df

    def load_from_fasta(self, file_path: str) -> Dict[str, str]:
        """
        从FASTA文件加载蛋白质序列

        参数:
            file_path: FASTA文件路径

        返回:
            字典，键为序列ID，值为序列字符串

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FASTA文件不存在: {file_path}")

        logger.info("从FASTA加载序列: %s", file_path)
        sequences = {}

        for record in SeqIO.parse(file_path, "fasta"):
            seq_id = record.id
            sequence = clean_sequence(str(record.seq))
            is_valid, error_msg = validate_sequence(sequence, valid_amino_acids=self.valid_amino_acids)
            if is_valid:
                sequences[seq_id] = sequence
            else:
                logger.warning("跳过无效序列 %s: %s", seq_id, error_msg)

        logger.info("加载完成，共 %d 条有效序列", len(sequences))
        return sequences

    def load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载数据

        参数:
            file_path: JSON文件路径

        返回:
            数据列表

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON文件不存在: {file_path}")

        logger.info("从JSON加载数据: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise TypeError("JSON数组中的每一项都必须是对象")
            records = data
        else:
            raise TypeError("JSON内容必须是对象或对象数组")

        logger.info("加载完成，共 %d 条记录", len(records))
        return records

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        """Normalize a column name for fuzzy matching."""
        return re.sub(r"[^a-z0-9]+", "", str(column_name).strip().lower())

    @staticmethod
    def _is_url(path_or_url: str) -> bool:
        """Return True when a string looks like an HTTP(S) URL."""
        parsed = urlparse(path_or_url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _extract_amino_acid_and_position(value: Any) -> tuple[Optional[str], Optional[int]]:
        """
        Extract amino acid and position from a PTM residue string such as ``S473-p``.
        """
        if pd.isna(value):
            return None, None

        match = re.search(r"([A-Za-z])\s*(\d+)", str(value))
        if not match:
            return None, None

        amino_acid = match.group(1).upper()
        position = int(match.group(2))
        return amino_acid, position

    def _get_matching_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column using normalized names."""
        normalized_map = {
            self._normalize_column_name(column_name): column_name for column_name in df.columns
        }
        for candidate in candidates:
            matched = normalized_map.get(self._normalize_column_name(candidate))
            if matched is not None:
                return str(matched)
        return None

    def _download_if_url(self, path_or_url: str) -> str:
        """
        Download an HTTP(S) URL to a local file and return the local path.

        Local paths are returned unchanged.

        Safety:
            - Rejects URLs whose ``scheme`` is not http/https (defence-in-depth
              against ``file://`` / ``ftp://`` SSRF vectors).
            - Logs a loud warning when the scheme is ``http`` rather than
              ``https`` so plaintext downloads are visible in audit logs.
            - Caps total downloaded size at ``PTM2CELLNET_MAX_DOWNLOAD_BYTES``
              (default 256 MiB) to prevent accidental OOM from a hostile or
              misconfigured endpoint.
            - Optionally restricts the host via ``PTM2CELLNET_DOWNLOAD_ALLOWLIST``
              (comma-separated). When unset, any host is allowed (preserving
              existing behaviour); when set, only listed hosts may be fetched.
        """
        if not self._is_url(path_or_url):
            return path_or_url

        parsed = urlparse(path_or_url)
        # Scheme check: only http/https are ever fetched. _is_url already
        # enforces this, but we re-check here so the guard is local to the
        # network call site and survives refactors of _is_url.
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Refusing to download URL with scheme {parsed.scheme!r}; "
                "only http/https are allowed."
            )
        if parsed.scheme == "http":
            logger.warning(
                "Downloading over plaintext HTTP (host=%s). Configure the "
                "source to use HTTPS if possible.",
                parsed.netloc,
            )

        # Optional host allowlist (env-driven so the default behaviour is
        # unchanged but operators can lock down data sources in production).
        allowlist_env = os.environ.get("PTM2CELLNET_DOWNLOAD_ALLOWLIST", "")
        if allowlist_env:
            allowed_hosts = {h.strip().lower() for h in allowlist_env.split(",") if h.strip()}
            if parsed.netloc.split(":")[0].lower() not in allowed_hosts:
                raise ValueError(
                    f"Download host {parsed.netloc!r} is not in the "
                    "PTM2CELLNET_DOWNLOAD_ALLOWLIST allowlist."
                )

        # Max download size cap. Read the Content-Length header first; if the
        # server omits it, we still enforce the cap on the streamed bytes.
        max_bytes = int(os.environ.get("PTM2CELLNET_MAX_DOWNLOAD_BYTES", str(256 * 1024 * 1024)))

        logger.info("下载远程数据文件: %s", path_or_url)
        os.makedirs(self.data_raw_dir, exist_ok=True)

        filename = os.path.basename(parsed.path) or "downloaded_data"
        suffix = "".join(part for part in os.path.splitext(filename) if part)
        if not suffix:
            suffix = ".tmp"

        response = requests.get(path_or_url, stream=True, timeout=30)
        response.raise_for_status()

        # Pre-check declared size to fail fast on obviously-too-big payloads.
        # ``response.headers`` may be absent for mock responses in tests, so
        # guard with getattr to avoid AttributeError breaking the happy path.
        headers = getattr(response, "headers", None) or {}
        content_length = headers.get("content-length") if hasattr(headers, "get") else None
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared > max_bytes:
                raise ValueError(
                    f"Refusing to download {path_or_url}: declared size "
                    f"{declared} bytes exceeds limit {max_bytes} bytes "
                    "(set PTM2CELLNET_MAX_DOWNLOAD_BYTES to raise the cap)."
                )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="download_",
            dir=self.data_raw_dir,
            delete=False,
        ) as handle:
            bytes_written = 0
            # ``iter_content`` may not exist on mock responses; fall back to
            # writing ``response.content`` directly so test mocks keep working.
            iter_content = getattr(response, "iter_content", None)
            if callable(iter_content):
                for chunk in iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        handle.close()
                        try:
                            os.unlink(handle.name)
                        except OSError:
                            pass
                        raise ValueError(
                            f"Download exceeded {max_bytes} bytes (cap via "
                            "PTM2CELLNET_MAX_DOWNLOAD_BYTES); aborting."
                        )
                    handle.write(chunk)
            else:
                content = getattr(response, "content", b"") or b""
                if len(content) > max_bytes:
                    handle.close()
                    try:
                        os.unlink(handle.name)
                    except OSError:
                        pass
                    raise ValueError(
                        f"Download exceeded {max_bytes} bytes (cap via "
                        "PTM2CELLNET_MAX_DOWNLOAD_BYTES); aborting."
                    )
                handle.write(content)
            return handle.name

    @staticmethod
    def _empty_phosphositeplus_df() -> pd.DataFrame:
        """Return an empty DataFrame for the PhosphoSitePlus contract."""
        return pd.DataFrame(
            columns=[
                "protein_accession",
                "gene",
                "position",
                "ptm_type",
                "amino_acid",
                "confidence",
                "source",
            ]
        )

    @staticmethod
    def _empty_ptm_df() -> pd.DataFrame:
        """Return an empty DataFrame for generic PTM database contracts."""
        return pd.DataFrame(
            columns=["protein_accession", "position", "ptm_type", "amino_acid", "source"]
        )

    def _read_tabular_file(self, file_path: str, sep: Optional[str] = None) -> pd.DataFrame:
        """Read a delimited text file with optional separator inference."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        read_kwargs: Dict[str, Any] = {"comment": "#", "compression": "infer"}
        if sep is None:
            read_kwargs.update({"sep": None, "engine": "python"})
        else:
            read_kwargs["sep"] = sep

        return pd.read_csv(file_path, **read_kwargs)

    def _build_ptm_dataframe(
        self,
        df: pd.DataFrame,
        accession_candidates: List[str],
        position_candidates: List[str],
        amino_acid_candidates: List[str],
        source: str,
        ptm_type: str,
    ) -> pd.DataFrame:
        """Convert a raw PTM table into the shared PTM loader contract."""
        accession_col = self._get_matching_column(df, accession_candidates)
        position_col = self._get_matching_column(df, position_candidates)
        amino_acid_col = self._get_matching_column(df, amino_acid_candidates)
        mod_rsd_col = self._get_matching_column(
            df,
            [
                "MOD_RSD",
                "mod_rsd",
                "modified residue",
                "modification site",
                "site",
                "site position",
            ],
        )

        if accession_col is None:
            raise ValueError("缺少蛋白登录号列")

        if position_col is not None:
            positions = (
                df[position_col]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
        elif mod_rsd_col is not None:
            positions = df[mod_rsd_col].apply(lambda value: self._extract_amino_acid_and_position(value)[1])
        else:
            raise ValueError("缺少位点位置列")

        if amino_acid_col is not None:
            amino_acids = (
                df[amino_acid_col].astype(str).str.extract(r"([A-Za-z])", expand=False).str.upper()
            )
        elif mod_rsd_col is not None:
            amino_acids = df[mod_rsd_col].apply(
                lambda value: self._extract_amino_acid_and_position(value)[0]
            )
        else:
            raise ValueError("缺少氨基酸列")

        result = pd.DataFrame(
            {
                "protein_accession": df[accession_col].astype(str).str.strip(),
                "position": pd.Series(positions, index=df.index),
                "ptm_type": ptm_type,
                "amino_acid": amino_acids,
                "source": source,
            }
        )
        result = result.dropna(subset=["protein_accession", "position", "amino_acid"])
        result = result[result["protein_accession"] != ""].copy()
        result["position"] = result["position"].astype(int)
        result["amino_acid"] = result["amino_acid"].astype(str).str.upper()
        return result.reset_index(drop=True)

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
            confidence_col = self._get_matching_column(
                df, ["confidence", "CONFIDENCE", "site group id", "SITE_GRP_ID"]
            )
            # 优先匹配真正的 confidence 列；若 PhosphoSitePlus 导出无该列，
            # 则回退到 site group id（非置信度）并在日志中说明。
            # 注意：经 _normalize_column_name 处理后 "site group id" -> "sitegroupid"，
            # 而 "SITE_GRP_ID" -> "sitegrpid"（下划线被移除但 g 与 grp 不同），两者均需覆盖。
            confidence_is_group_id = confidence_col is not None and (
                self._normalize_column_name(confidence_col)
                in {"sitegroupid", "sitegrpid"}
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
                        pd.to_numeric(df[confidence_col], errors="coerce")
                        if confidence_col is not None
                        else pd.NA
                    ),
                    "source": "PhosphoSitePlus",
                }
            )
            result = result.dropna(subset=["protein_accession", "position", "amino_acid"])
            result = result[result["protein_accession"] != ""].copy()
            result["position"] = result["position"].astype(int)
            return result.reset_index(drop=True)
        except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("解析 PhosphoSitePlus 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_phosphositeplus_df(), "PhosphoSitePlus")

    def load_from_dbptm(
        self, file_path_or_url: str, ptm_type: str = "phosphorylation"
    ) -> pd.DataFrame:
        """
        Load PTM annotations from a dbPTM CSV/TSV export.

        Parameters:
            file_path_or_url: Local file path or downloadable URL.
            ptm_type: PTM type label assigned to returned rows.

        Returns:
            DataFrame with columns ``protein_accession``, ``position``, ``ptm_type``,
            ``amino_acid`` and ``source``.
        """
        try:
            local_path = self._download_if_url(file_path_or_url)
        except requests.RequestException as exc:
            logger.error("下载 dbPTM 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_ptm_df(), "dbPTM")

        try:
            df = self._read_tabular_file(local_path)
            return self._build_ptm_dataframe(
                df=df,
                accession_candidates=[
                    "UniProtKB Accession",
                    "UniProt Accession",
                    "protein_accession",
                    "ACC_ID",
                    "accession",
                ],
                position_candidates=["Position", "Site Position", "site", "pos"],
                amino_acid_candidates=["Residue", "Amino Acid", "AA", "amino_acid"],
                source="dbPTM",
                ptm_type=ptm_type,
            )
        except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("解析 dbPTM 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_ptm_df(), "dbPTM")

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
        try:
            local_path = self._download_if_url(file_path_or_url)
        except requests.RequestException as exc:
            logger.error("下载 CPLM 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_ptm_df(), "CPLM")

        try:
            df = self._read_tabular_file(local_path, sep="\t")
            return self._build_ptm_dataframe(
                df=df,
                accession_candidates=[
                    "UniProt Accession",
                    "UniProtKB Accession",
                    "protein_accession",
                    "ACC_ID",
                    "accession",
                ],
                position_candidates=["Position", "Site Position", "site", "pos"],
                amino_acid_candidates=["Amino Acid", "Residue", "AA", "amino_acid"],
                source="CPLM",
                ptm_type=ptm_type,
            )
        except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("解析 CPLM 数据失败: %s", exc)
            return self._empty_or_raise(self._empty_ptm_df(), "CPLM")

    def _fetch_uniprot_batch(self, accession_ids: List[str]) -> List[Dict[str, Any]]:
        """
        通过UniProt批量端点获取蛋白质数据

        参数:
            accession_ids: UniProt登录号列表

        返回:
            从UniProt API响应中解析的记录列表

        异常:
            requests.RequestException: 网络请求失败时抛出
        """
        # Guard against oversized URLs. UniProt's REST gateway (and many
        # upstream proxies / WAFs) reject URLs beyond ~8 KB with HTTP 414
        # "URI Too Long". Each accession contributes roughly
        # ``len("accession:")+12+4 = ~25`` chars to the query, so cap the
        # per-request accession count by estimating the resulting URL length
        # and chunking when it would exceed the safe limit.
        url = "https://rest.uniprot.org/uniprotkb/stream"
        # Conservative limit (bytes) — well below common proxy defaults
        # (nginx large_client_header_buffers default ~8k, AWS ALB 8192).
        max_url_bytes = 7000

        def _estimate_url_len(accs: List[str]) -> int:
            query = " OR ".join(f"accession:{a}" for a in accs)
            # +base url + other params (format=, fields=) ≈ +80 chars.
            return len(url) + len(query) + 100

        # If a single request would exceed the limit, split into chunks and
        # concatenate the results. We preserve order so callers see records
        # in the same order they passed accessions (best-effort: UniProt
        # does not guarantee ordering inside one chunk).
        if _estimate_url_len(accession_ids) > max_url_bytes:
            chunks: List[List[str]] = []
            current: List[str] = []
            for acc in accession_ids:
                current.append(acc)
                if _estimate_url_len(current) >= max_url_bytes:
                    # Push and start a new chunk; keep at least 1 element.
                    if len(current) > 1:
                        chunks.append(current[:-1])
                        current = [current[-1]]
                    else:
                        chunks.append(current)
                        current = []
            if current:
                chunks.append(current)

            logger.info(
                "UniProt batch URL 会超过 %d 字节，拆分为 %d 个子请求",
                max_url_bytes, len(chunks),
            )
            aggregated: List[Dict[str, Any]] = []
            for chunk in chunks:
                aggregated.extend(self._fetch_uniprot_batch(chunk))
            return aggregated

        query = " OR ".join(f"accession:{acc}" for acc in accession_ids)
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,sequence,gene",
        }
        # UniProt stream 端点接受查询参数通过 query string 传递（而非 form data）。
        # 使用 GET + params 更符合 REST 契约，且便于缓存与重试。
        session = self._uniprot_session()
        try:
            response = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            logger.error(
                "UniProt 批量请求失败 (query=%s): %s",
                query[:80], exc,
            )
            raise
        if response.status_code != 200:
            logger.error(
                "UniProt 批量请求返回非 200: status=%s, body=%s",
                response.status_code, response.text[:200],
            )
            response.raise_for_status()
        data = response.json()
        return cast(List[Dict[str, Any]], data.get("results", []))

    def _uniprot_session(self) -> requests.Session:
        """构建带指数退避重试的 UniProt 请求 Session（惰性创建）。"""
        if not hasattr(self, "_uniprot_http_session"):
            session = requests.Session()
            try:
                from requests.adapters import HTTPAdapter
                try:
                    from urllib3.util.retry import Retry
                except ImportError:  # pragma: no cover
                    from requests.packages.urllib3.util.retry import Retry  # type: ignore[no-redef]

                retry = Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(["GET", "POST"]),
                )
                adapter = HTTPAdapter(max_retries=retry)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
            except Exception as exc:  # pragma: no cover - 重试配置失败不阻塞基础功能
                logger.warning("配置 UniProt HTTP 重试失败，使用默认 Session: %s", exc)
            self._uniprot_http_session = session
        return self._uniprot_http_session

    def _uniprot_cache_dir(self) -> Path:
        """Return the directory used to cache UniProt JSON responses."""
        return Path(self.data_raw_dir).parent / "uniprot_cache"

    def _fetch_uniprot_single(self, accession: str) -> Optional[Dict[str, Any]]:
        """Fetch one UniProt entry, reading from cache when possible."""
        cache_dir = self._uniprot_cache_dir()
        cache_file = cache_dir / f"{accession}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return cast(Dict[str, Any], json.load(f))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("读取UniProt缓存 %s 失败: %s", cache_file, exc)

        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError as exc:
            logger.warning("写入UniProt缓存 %s 失败: %s", cache_file, exc)

        return cast(Dict[str, Any], data)

    def load_from_uniprot(self, accession_ids: List[str]) -> pd.DataFrame:
        """
        从UniProt数据库加载蛋白质数据

        参数:
            accession_ids: UniProt登录号列表

        返回:
            DataFrame对象，包含列: sequence, gene_symbol, accession
        """
        columns = ["sequence", "gene_symbol", "accession"]
        if not accession_ids:
            logger.warning("登录号列表为空，返回空DataFrame")
            return pd.DataFrame(columns=columns)

        logger.info("从UniProt加载 %d 个蛋白质数据", len(accession_ids))

        records: List[Dict[str, Any]] = []
        if len(accession_ids) == 1:
            try:
                data = self._fetch_uniprot_single(accession_ids[0])
                if data is not None:
                    records.append(data)
            except requests.RequestException as exc:
                logger.error("获取 %s 失败: %s", accession_ids[0], exc)
        else:
            try:
                records = self._fetch_uniprot_batch(accession_ids)
                logger.info("批量请求成功，获取 %d 条记录", len(records))
            except requests.RequestException as exc:
                logger.warning("批量请求失败，回退到逐个请求: %s", exc)
                for acc in accession_ids:
                    try:
                        data = self._fetch_uniprot_single(acc)
                        if data is not None:
                            records.append(data)
                    except requests.RequestException as inner_exc:
                        logger.error("获取 %s 失败: %s", acc, inner_exc)

        if not records:
            logger.warning("未获取到任何UniProt记录")
            return pd.DataFrame(columns=columns)

        rows: List[Dict[str, Any]] = []
        for entry in records:
            accession = entry.get("primaryAccession", "")
            seq_data = entry.get("sequence", {})
            raw_sequence = seq_data.get("value", "") if isinstance(seq_data, dict) else ""
            sequence = clean_sequence(raw_sequence)

            is_valid, error_msg = validate_sequence(sequence, valid_amino_acids=self.valid_amino_acids)
            if not is_valid:
                logger.warning("跳过无效序列 %s: %s", accession, error_msg)
                continue

            gene_symbol = ""
            genes = entry.get("genes", [])
            if isinstance(genes, list) and genes:
                first_gene = genes[0]
                if isinstance(first_gene, dict):
                    gene_name = first_gene.get("geneName", {})
                    if isinstance(gene_name, dict):
                        gene_symbol = gene_name.get("value", "")

            rows.append({
                "sequence": sequence,
                "gene_symbol": gene_symbol,
                "accession": accession,
            })

        df = pd.DataFrame(rows, columns=columns)
        logger.info("UniProt加载完成，共 %d 条有效记录", len(df))
        return df

    def load_sample_data(self, num_samples: int = 100) -> pd.DataFrame:
        """
        加载示例数据（用于测试）

        参数:
            num_samples: 样本数量

        返回:
            示例数据DataFrame
        """
        logger.info("生成示例数据，共 %d 个样本", num_samples)

        import random

        data_config = self.config.get("data", {})
        valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        ptm_types = data_config.get("ptm_types", ["phosphorylation", "acetylation", "methylation", "ubiquitination"])
        cell_states = data_config.get("cell_states", ["proliferation", "differentiation", "apoptosis", "quiescence"])
        min_sequence_length = data_config.get("min_sequence_length", 50)
        max_sequence_length = data_config.get("max_sequence_length", 500)

        data = []
        for i in range(num_samples):
            seq_length = random.randint(min_sequence_length, max_sequence_length)
            sequence = "".join(random.choice(valid_amino_acids) for _ in range(seq_length))

            num_ptms = random.randint(0, 5)
            ptm_sites = []
            positions = random.sample(range(1, seq_length + 1), min(num_ptms, seq_length))
            for pos in positions:
                ptm_sites.append({
                    "position": pos,
                    "type": random.choice(ptm_types),
                    "amino_acid": sequence[pos - 1]
                })

            data.append({
                "id": f"sample_{i:04d}",
                "sequence": sequence,
                "ptm_sites": json.dumps(ptm_sites),
                "cell_state": random.choice(cell_states)
            })

        df = pd.DataFrame(data)
        logger.info("示例数据生成完成")
        return df

    def load_combined_data(
        self,
        sequence_file: Optional[str] = None,
        ptm_file: Optional[str] = None,
        label_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        从多个文件加载并合并数据

        参数:
            sequence_file: 序列文件路径
            ptm_file: PTM数据文件路径
            label_file: 标签文件路径

        返回:
            合并后的DataFrame
        """
        data_frames = []

        if sequence_file:
            if sequence_file.endswith(".fasta") or sequence_file.endswith(".fa"):
                seq_dict = self.load_from_fasta(sequence_file)
                seq_df = pd.DataFrame(list(seq_dict.items()), columns=["id", "sequence"])
                data_frames.append(seq_df)
            elif sequence_file.endswith(".csv"):
                data_frames.append(self.load_from_csv(sequence_file))

        if ptm_file and ptm_file.endswith(".csv"):
            data_frames.append(self.load_from_csv(ptm_file))

        if label_file and label_file.endswith(".csv"):
            data_frames.append(self.load_from_csv(label_file))

        if not data_frames:
            logger.warning("未提供任何数据文件")
            return pd.DataFrame()

        if len(data_frames) == 1:
            return data_frames[0]

        result = data_frames[0]
        for df in data_frames[1:]:
            common_cols = [col for col in result.columns if col in df.columns]
            if common_cols:
                result = result.merge(df, on=common_cols, how="outer")
            else:
                result = pd.concat([result, df], axis=1)

        logger.info("数据合并完成，共 %d 条记录", len(result))
        return result
