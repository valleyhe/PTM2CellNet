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
from typing import Any, Dict, List, Optional
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

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        从CSV文件加载数据

        参数:
            file_path: CSV文件路径

        返回:
            DataFrame对象

        异常:
            FileNotFoundError: 文件不存在时抛出
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")

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
                return matched
        return None

    def _download_if_url(self, path_or_url: str) -> str:
        """
        Download an HTTP(S) URL to a local file and return the local path.

        Local paths are returned unchanged.
        """
        if not self._is_url(path_or_url):
            return path_or_url

        logger.info("下载远程数据文件: %s", path_or_url)
        os.makedirs(self.data_raw_dir, exist_ok=True)

        parsed = urlparse(path_or_url)
        filename = os.path.basename(parsed.path) or "downloaded_data"
        suffix = "".join(part for part in os.path.splitext(filename) if part)
        if not suffix:
            suffix = ".tmp"

        response = requests.get(path_or_url, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="download_",
            dir=self.data_raw_dir,
            delete=False,
        ) as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
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
            logger.warning("下载 PhosphoSitePlus 数据失败: %s", exc)
            return self._empty_phosphositeplus_df()

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
                df, ["SITE_GRP_ID", "site group id", "confidence"]
            )

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
            logger.warning("解析 PhosphoSitePlus 数据失败: %s", exc)
            return self._empty_phosphositeplus_df()

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
            logger.warning("下载 dbPTM 数据失败: %s", exc)
            return self._empty_ptm_df()

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
            logger.warning("解析 dbPTM 数据失败: %s", exc)
            return self._empty_ptm_df()

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
            logger.warning("下载 CPLM 数据失败: %s", exc)
            return self._empty_ptm_df()

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
            logger.warning("解析 CPLM 数据失败: %s", exc)
            return self._empty_ptm_df()

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
        url = "https://rest.uniprot.org/uniprotkb/stream"
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
        return data.get("results", [])

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
                    return json.load(f)
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

        return data

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
