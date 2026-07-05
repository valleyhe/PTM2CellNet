"""
UniProtLoaderMixin — UniProt REST API 批量/单条加载 + 缓存。
"""

import json
import os
from pathlib import Path
from typing import List, Optional, cast

import pandas as pd
import requests

from ...utils.logging import setup_logger
from ...utils.helpers import validate_sequence, clean_sequence
from .types import UniProtRecord, UniProtRow

logger = setup_logger(__name__)


class UniProtLoaderMixin:
    """UniProt REST API 加载 mixin，含批量 URL 分块、本地缓存与重试。"""

    def _fetch_uniprot_batch(self, accession_ids: List[str]) -> List[UniProtRecord]:
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
            aggregated: List[UniProtRecord] = []
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
        return cast(List[UniProtRecord], data.get("results", []))

    def _uniprot_session(self) -> requests.Session:
        """构建带指数退避重试的 UniProt 请求 Session（惰性创建）。"""
        if not hasattr(self, "_uniprot_http_session"):
            session = requests.Session()
            try:
                from requests.adapters import HTTPAdapter
                import importlib
                _retry_mod = "urllib3.util.retry"
                try:
                    _Retry = importlib.import_module(_retry_mod).Retry
                except ImportError:  # urllib3 不可用时回退到 vendored 副本；可用 monkeypatch sys.modules["urllib3"] 触发 ImportError 测试该分支
                    _Retry = importlib.import_module("requests.packages." + _retry_mod).Retry

                retry = _Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(["GET", "POST"]),
                )
                adapter = HTTPAdapter(max_retries=retry)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
            except (ValueError, TypeError, AttributeError) as exc:  # 重试配置失败不阻塞基础功能；可用 monkeypatch requests.adapters.HTTPAdapter 抛异常测试该分支
                logger.warning("配置 UniProt HTTP 重试失败，使用默认 Session: %s", exc)
            self._uniprot_http_session = session
        return self._uniprot_http_session

    def _uniprot_cache_dir(self) -> Path:
        """Return the directory used to cache UniProt JSON responses."""
        return Path(self.data_raw_dir).parent / "uniprot_cache"

    def _fetch_uniprot_single(self, accession: str) -> Optional[UniProtRecord]:
        """Fetch one UniProt entry, reading from cache when possible."""
        cache_dir = self._uniprot_cache_dir()
        cache_file = cache_dir / f"{accession}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return cast(UniProtRecord, json.load(f))
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

        return cast(UniProtRecord, data)

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

        records: List[UniProtRecord] = []
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

        rows: List[UniProtRow] = []
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
