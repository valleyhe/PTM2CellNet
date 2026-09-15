"""
DataLoaderBase — __init__ 以及被其它 mixin 共享的工具方法。
"""

import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, cast, runtime_checkable
from urllib.parse import urlparse

import pandas as pd
import requests

from ...utils.logging import setup_logger
from .types import _ReadCsvKwargs

logger = setup_logger(__name__)


@runtime_checkable
class _DataLoaderProtocol(Protocol):
    """Protocol defining the interface that DataLoader mixins depend on.

    This makes mixin method calls statically visible to mypy without
    requiring mixins to inherit from DataLoaderBase.
    """

    data_raw_dir: str
    strict_load: bool

    def _download_if_url(self, path_or_url: str) -> str: ...

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

    def _empty_or_raise(self, empty_df: pd.DataFrame, source: str) -> pd.DataFrame: ...

    @staticmethod
    def _empty_phosphositeplus_df() -> pd.DataFrame: ...

    @staticmethod
    def _empty_ptm_df() -> pd.DataFrame: ...

    @staticmethod
    def _normalize_column_name(column_name: str) -> str: ...

    @staticmethod
    def _extract_amino_acid_and_position(
        value: Union[str, float, int, None],
    ) -> Tuple[Optional[str], Optional[int]]: ...


# Default production download allowlist: official bioinformatics data sources.
# These are the hosts that the project's data loaders are known to fetch from.
# Operators can extend this via PTM2CELLNET_DOWNLOAD_ALLOWLIST.
_DEFAULT_DOWNLOAD_ALLOWLIST = frozenset(
    {
        "www.uniprot.org",
        "rest.uniprot.org",
        "alphafold.ebi.ac.uk",
        "ftp.ebi.ac.uk",
        "www.phosphosite.org",
    }
)


class DataLoaderBase:
    """
    DataLoader 的基类部分。包含 __init__、数据源验证、URL 下载、
    列名模糊匹配以及 PTM DataFrame 构建等所有公用方法。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据加载器

        参数:
            config: 配置字典，包含路径等设置
        """
        self.config = config or {}
        self.data_raw_dir = self.config.get("paths", {}).get("data_raw", "data/raw")
        self.valid_amino_acids = set(self.config.get("data", {}).get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY"))
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
    def _extract_amino_acid_and_position(value: Union[str, float, int, None]) -> Tuple[Optional[str], Optional[int]]:
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
        normalized_map = {self._normalize_column_name(column_name): column_name for column_name in df.columns}
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
            - Restricts the host via ``PTM2CELLNET_DOWNLOAD_ALLOWLIST``
              (comma-separated). When unset, the production default allowlist
              applies; set to empty string to allow any host (dev mode).
        """
        if not self._is_url(path_or_url):
            return path_or_url

        parsed = urlparse(path_or_url)
        # Scheme check: only http/https are ever fetched. _is_url already
        # enforces this, but we re-check here so the guard is local to the
        # network call site and survives refactors of _is_url.
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Refusing to download URL with scheme {parsed.scheme!r}; only http/https are allowed.")
        if parsed.scheme == "http":
            logger.warning(
                "Downloading over plaintext HTTP (host=%s). Configure the source to use HTTPS if possible.",
                parsed.netloc,
            )

        # Host allowlist: when PTM2CELLNET_DOWNLOAD_ALLOWLIST is unset, only
        # the default production allowlist (official bioinformatics hosts) is
        # permitted.  Set the env var to override with a custom list, or to
        # empty string to allow all hosts (dev/test back-compat).
        allowlist_env = os.environ.get("PTM2CELLNET_DOWNLOAD_ALLOWLIST", None)
        if allowlist_env is not None:
            # Explicit env var: use only the hosts listed (empty string = allow all,
            # preserving backward compat for dev/test).
            if allowlist_env.strip():
                allowed_hosts = {h.strip().lower() for h in allowlist_env.split(",") if h.strip()}
                if parsed.netloc.split(":")[0].lower() not in allowed_hosts:
                    raise ValueError(
                        f"Download host {parsed.netloc!r} is not in the PTM2CELLNET_DOWNLOAD_ALLOWLIST allowlist."
                    )
        else:
            # No env var set: use default production allowlist
            if parsed.netloc.split(":")[0].lower() not in _DEFAULT_DOWNLOAD_ALLOWLIST:
                logger.warning(
                    "Download host %r is not in the default production allowlist. "
                    "Set PTM2CELLNET_DOWNLOAD_ALLOWLIST to allow this host explicitly.",
                    parsed.netloc,
                )
                raise ValueError(
                    f"Download host {parsed.netloc!r} is not in the default "
                    "production allowlist. To allow this host, set "
                    "PTM2CELLNET_DOWNLOAD_ALLOWLIST environment variable with a "
                    "comma-separated list of allowed hosts, or set it to empty "
                    "string to allow all hosts (not recommended for production)."
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
                            f"Download exceeded {max_bytes} bytes (cap via PTM2CELLNET_MAX_DOWNLOAD_BYTES); aborting."
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
                        f"Download exceeded {max_bytes} bytes (cap via PTM2CELLNET_MAX_DOWNLOAD_BYTES); aborting."
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
        return pd.DataFrame(columns=["protein_accession", "position", "ptm_type", "amino_acid", "source"])

    def _read_tabular_file(self, file_path: str, sep: Optional[str] = None) -> pd.DataFrame:
        """Read a delimited text file with optional separator inference."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        read_kwargs: _ReadCsvKwargs = {"comment": "#", "compression": "infer"}
        if sep is None:
            read_kwargs.update({"sep": None, "engine": "python"})
        else:
            read_kwargs["sep"] = sep

        return cast(pd.DataFrame, pd.read_csv(file_path, **cast(Any, read_kwargs)))

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

        positions: "pd.Series"
        if position_col is not None:
            positions = (
                df[position_col].astype(str).str.extract(r"(\d+)", expand=False).pipe(pd.to_numeric, errors="coerce")
            )
        elif mod_rsd_col is not None:
            positions = df[mod_rsd_col].apply(lambda value: self._extract_amino_acid_and_position(value)[1])
        else:
            raise ValueError("缺少位点位置列")

        if amino_acid_col is not None:
            amino_acids = df[amino_acid_col].astype(str).str.extract(r"([A-Za-z])", expand=False).str.upper()
        elif mod_rsd_col is not None:
            amino_acids = df[mod_rsd_col].apply(lambda value: self._extract_amino_acid_and_position(value)[0])
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
        return cast(pd.DataFrame, result.reset_index(drop=True))
