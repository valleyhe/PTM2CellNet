"""Tests for gene to UniProt mapping."""
import threading
import time
import pytest
from unittest.mock import MagicMock, Mock, patch
import pandas as pd

import src.analysis.gene_mapper as gene_mapper
from src.analysis.gene_mapper import (
    GeneMapper,
    _RequestsUniProtMapper,
    _build_mapper,
    map_gene_to_uniprot,
)


class TestGeneMapper:
    """Test gene symbol to UniProt ID mapping functionality."""

    @pytest.fixture
    def mock_protmapper(self):
        """Create mock ProtMapper."""
        with patch('src.analysis.gene_mapper.ProtMapper') as mock:
            yield mock

    def test_map_gene_to_uniprot_success(self, mock_protmapper):
        """Test successful gene to UniProt mapping."""
        # Setup mock
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Create mock result DataFrame
        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, [])

        # Create mapper after mock is set up
        mapper = GeneMapper()

        # Test
        result = mapper.map_gene_to_uniprot('BRAF')

        assert result == 'P15056'
        mock_instance.get.assert_called_once_with(
            ids=['BRAF'],
            from_db='Gene_Name',
            to_db='UniProtKB',
        )

    def test_map_gene_to_uniprot_not_found(self, mock_protmapper):
        """Test mapping returns None for unknown gene."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Empty result for unknown gene
        mock_result = pd.DataFrame(columns=['From', 'To'])
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('UNKNOWN_GENE')

        assert result is None

    def test_map_gene_to_uniprot_failed_mapping(self, mock_protmapper):
        """Test mapping returns None when API reports failure."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Failed mapping
        mock_result = pd.DataFrame(columns=['From', 'To'])
        mock_instance.get.return_value = (mock_result, ['UNKNOWN_GENE'])

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('UNKNOWN_GENE')

        assert result is None

    def test_map_gene_to_uniprot_caching(self, mock_protmapper):
        """Test that gene mappings are cached."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()

        # First call
        result1 = mapper.map_gene_to_uniprot('BRAF')
        assert result1 == 'P15056'

        # Second call should use cache
        result2 = mapper.map_gene_to_uniprot('BRAF')
        assert result2 == 'P15056'

        # ProtMapper.get should only be called once
        mock_instance.get.assert_called_once()

    def test_map_genes_batch_success(self, mock_protmapper):
        """Test batch gene mapping."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF', 'TP53'],
            'To': ['P15056', 'P04637'],
        })
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()
        results = mapper.map_genes_batch(['BRAF', 'TP53'])

        assert results['BRAF'] == 'P15056'
        assert results['TP53'] == 'P04637'

    def test_map_genes_batch_partial_failure(self, mock_protmapper):
        """Test batch mapping with some failures."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, ['UNKNOWN_GENE'])

        mapper = GeneMapper()
        results = mapper.map_genes_batch(['BRAF', 'UNKNOWN_GENE'])

        assert results['BRAF'] == 'P15056'
        assert results['UNKNOWN_GENE'] is None

    def test_map_genes_batch_empty_list(self, mock_protmapper):
        """Test batch mapping with empty list returns empty dict."""
        mapper = GeneMapper()
        results = mapper.map_genes_batch([])
        assert results == {}

    def test_map_genes_batch_uses_cache(self, mock_protmapper):
        """Test batch mapping uses cache for previously mapped genes."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mapper = GeneMapper()

        # Pre-populate cache
        mapper._gene_cache['BRAF'] = 'P15056'

        mock_result = pd.DataFrame({
            'From': ['TP53'],
            'To': ['P04637'],
        })
        mock_instance.get.return_value = (mock_result, [])

        results = mapper.map_genes_batch(['BRAF', 'TP53'])

        # Only TP53 should be queried
        mock_instance.get.assert_called_once_with(
            ids=['TP53'],
            from_db='Gene_Name',
            to_db='UniProtKB',
        )

        assert results['BRAF'] == 'P15056'
        assert results['TP53'] == 'P04637'

    def test_map_gene_to_uniprot_network_error(self, mock_protmapper):
        """Test handling of network errors."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance
        mock_instance.get.side_effect = ConnectionError("Network error")

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('BRAF')

        assert result is None

    def test_map_gene_to_uniprot_timeout_returns_none_without_fallback(
        self, mock_protmapper, monkeypatch
    ):
        """A stalled optional mapper must not block or switch implementations."""
        monkeypatch.setattr(gene_mapper, '_EXTERNAL_MAPPER_TIMEOUT_S', 0.05)
        blocked = threading.Event()
        mock_instance = Mock()
        mock_instance.get.side_effect = lambda **kwargs: blocked.wait()
        mock_protmapper.return_value = mock_instance

        mapper = GeneMapper()
        started = time.monotonic()
        result = mapper.map_gene_to_uniprot('BRAF')

        assert result is None
        assert time.monotonic() - started < 0.5
        assert mapper._gene_cache['BRAF'] is None
        assert not isinstance(mapper._mapper, _RequestsUniProtMapper)
        mock_instance.get.assert_called_once_with(
            ids=['BRAF'],
            from_db='Gene_Name',
            to_db='UniProtKB',
        )

    def test_map_genes_batch_timeout_returns_none_without_fallback(
        self, mock_protmapper, monkeypatch
    ):
        """A stalled optional batch mapper must return bounded partial results."""
        monkeypatch.setattr(gene_mapper, '_EXTERNAL_MAPPER_TIMEOUT_S', 0.05)
        blocked = threading.Event()
        mock_instance = Mock()
        mock_instance.get.side_effect = lambda **kwargs: blocked.wait()
        mock_protmapper.return_value = mock_instance

        mapper = GeneMapper()
        started = time.monotonic()
        results = mapper.map_genes_batch(['BRAF', 'TP53'])

        assert results == {'BRAF': None, 'TP53': None}
        assert time.monotonic() - started < 0.5
        assert not isinstance(mapper._mapper, _RequestsUniProtMapper)
        mock_instance.get.assert_called_once_with(
            ids=['BRAF', 'TP53'],
            from_db='Gene_Name',
            to_db='UniProtKB',
        )

    def test_get_canonical_isoform(self, mock_protmapper):
        """Test canonical isoform getter."""
        mapper = GeneMapper()
        # Currently just returns the provided ID
        result = mapper.get_canonical_isoform('BRAF', 'P15056')
        assert result == 'P15056'

    def test_convenience_function_map_gene_to_uniprot(self, mock_protmapper):
        """Test the convenience function."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        mock_result = pd.DataFrame({
            'From': ['BRAF'],
            'To': ['P15056'],
        })
        mock_instance.get.return_value = (mock_result, [])

        result = map_gene_to_uniprot('BRAF')
        assert result == 'P15056'

    def test_multiple_isoforms_returns_first(self, mock_protmapper):
        """Test that first (canonical) isoform is returned for genes with multiple isoforms."""
        mock_instance = Mock()
        mock_protmapper.return_value = mock_instance

        # Multiple rows for different isoforms
        mock_result = pd.DataFrame({
            'From': ['BRAF', 'BRAF', 'BRAF'],
            'To': ['P15056', 'P15056-2', 'P15056-3'],
        })
        mock_instance.get.return_value = (mock_result, [])

        mapper = GeneMapper()
        result = mapper.map_gene_to_uniprot('BRAF')

        # Should return first (canonical) isoform
        assert result == 'P15056'


class TestRequestsFallbackMapper:
    """Cover the requests-based fallback when UniProtMapper is absent."""

    def test_build_mapper_uses_protmapper_when_available(self, monkeypatch):
        """When the package is available _build_mapper returns a ProtMapper instance."""
        sentinel = object()
        monkeypatch.setattr(
            'src.analysis.gene_mapper._HAS_UNIPROT_MAPPER', True
        )
        monkeypatch.setattr(
            'src.analysis.gene_mapper.ProtMapper',
            lambda: sentinel,
            raising=False,
        )
        assert _build_mapper() is sentinel

    def test_build_mapper_falls_back_when_package_missing(self, monkeypatch):
        """When the package is unavailable _build_mapper returns the requests mapper."""
        monkeypatch.setattr(
            'src.analysis.gene_mapper._HAS_UNIPROT_MAPPER', False
        )
        mapper = _build_mapper()
        assert isinstance(mapper, _RequestsUniProtMapper)

    def test_requests_mapper_empty_ids(self):
        mapper = _RequestsUniProtMapper()
        result, failed = mapper.get(ids=[])
        assert result.empty
        assert failed == []

    def test_requests_mapper_success(self):
        mapper = _RequestsUniProtMapper()
        submit_resp = Mock()
        submit_resp.text = "job123"
        submit_resp.raise_for_status = Mock()
        details_resp = Mock()
        details_resp.status_code = 200
        details_resp.json.return_value = {"jobStatus": "FINISHED"}
        results_resp = Mock()
        results_resp.raise_for_status = Mock()
        results_resp.links = {}
        results_resp.json.return_value = {
            "results": [
                {"from": "BRAF", "to": {"primaryAccession": "P15056"}},
            ]
        }
        with patch('src.analysis.gene_mapper.requests.post', return_value=submit_resp), \
             patch('src.analysis.gene_mapper.requests.get', side_effect=[details_resp, results_resp]):
            result, failed = mapper.get(ids=["BRAF"])
        assert failed == []
        assert result.iloc[0]["From"] == "BRAF"
        assert result.iloc[0]["To"] == "P15056"

    def test_requests_mapper_submission_failure(self):
        import requests
        mapper = _RequestsUniProtMapper()
        with patch(
            'src.analysis.gene_mapper.requests.post',
            side_effect=requests.RequestException("boom"),
        ):
            result, failed = mapper.get(ids=["BRAF"])
        assert result.empty
        assert failed == ["BRAF"]

    def test_requests_mapper_paginates_beyond_first_page(self):
        """TD-NEW-01: >500 ids must follow the Link rel="next" cursor."""
        mapper = _RequestsUniProtMapper()
        submit_resp = Mock()
        submit_resp.text = "job123"
        submit_resp.raise_for_status = Mock()
        details_resp = Mock()
        details_resp.status_code = 200
        details_resp.json.return_value = {"jobStatus": "FINISHED"}
        details_resp.links = {}

        page_one = Mock()
        page_one.raise_for_status = Mock()
        page_one.json.return_value = {
            "results": [{"from": f"G{i}", "to": {"primaryAccession": f"P{i:05d}"}} for i in range(500)]
        }
        page_one.links = {"next": {"url": "https://rest.uniprot.org/idmapping/results/job123?cursor=c1&size=500"}}
        page_two = Mock()
        page_two.raise_for_status = Mock()
        page_two.json.return_value = {
            "results": [{"from": "G500", "to": {"primaryAccession": "P50000"}}]
        }
        page_two.links = {}

        with patch('src.analysis.gene_mapper.requests.post', return_value=submit_resp), \
             patch('src.analysis.gene_mapper.requests.get', side_effect=[details_resp, page_one, page_two]):
            result, failed = mapper.get(ids=[f"G{i}" for i in range(501)])
        assert len(result) == 501
        assert failed == []
        assert result.iloc[500]["From"] == "G500"
        assert result.iloc[500]["To"] == "P50000"

    def test_requests_mapper_nonhuman_raises_not_implemented(self):
        mapper = _RequestsUniProtMapper()
        with pytest.raises(NotImplementedError, match="9606"):
            mapper.get(ids=["Braf"], organism="10090")

    def test_map_gene_to_uniprot_nonhuman_raises(self):
        """TD-NEW-03: a non-human organism must fail loudly, not return human IDs."""
        mapper = GeneMapper()
        with pytest.raises(NotImplementedError, match="not supported"):
            mapper.map_gene_to_uniprot("Braf", organism="10090")

    def test_map_genes_batch_nonhuman_raises(self):
        mapper = GeneMapper()
        with pytest.raises(NotImplementedError, match="not supported"):
            mapper.map_genes_batch(["Braf"], organism="10090")


class _FakeTime:
    """Deterministic clock: monotonic follows elapsed sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _poll_response(status_code: int = 200, payload=None, text: str = "job-1"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    # requests.Response.links is a plain dict; MagicMock auto-attributes would
    # otherwise look like a truthy "next" pagination cursor.
    resp.links = {}
    return resp


def _run_get(ids, fake_time, get_payloads, results_payload=None, post_payload_text="job-1"):
    """Drive mapper.get() with fake clock + scripted poll responses.

    ``get_payloads`` are the poll (/details) responses; after the job
    finishes, mapper.get() issues one more GET to /results/{job_id},
    answered by ``results_payload`` (default: empty results).
    """
    get_side_effect = [_poll_response(200, p) for p in get_payloads]
    get_side_effect.append(_poll_response(200, results_payload if results_payload is not None else {"results": []}))
    with (
        patch.object(gene_mapper.requests, "post", return_value=_poll_response(200, text=post_payload_text)) as post,
        patch.object(gene_mapper.requests, "get", side_effect=get_side_effect) as get,
        patch.object(gene_mapper.time, "monotonic", side_effect=fake_time.monotonic),
        patch.object(gene_mapper.time, "sleep", side_effect=fake_time.sleep),
    ):
        mapper = _RequestsUniProtMapper()
        result = mapper.get(ids, from_db="Gene_Name", to_db="UniProtKB")
    return result, post, get


class TestUniProtPollBackoff:
    """UniProt ID-mapping poll backoff (N08, 2026-08-16; merged 2026-08-24).

    Covers the exponential-backoff rewrite of ``_RequestsUniProtMapper.get``:
    * backoff interval sequence grows 0.2s x1.5 and caps at 3.0s;
    * a job that finishes immediately causes no polling sleep;
    * the 60s total budget is respected (no sleep past the deadline);
    * success/failure paths return the documented (DataFrame, failed_ids) shape.
    """

    def test_backoff_sequence_grows_and_caps(self) -> None:
        """N08: 连续 RUNNING 时 sleep 间隔按 0.2 x1.5 递增并封顶 3.0s。"""
        fake_time = _FakeTime()
        running = {"jobStatus": "RUNNING"}
        finished = {"jobStatus": "FINISHED"}
        # 4 次 RUNNING 后 FINISHED：sleep 序列 = 0.2, 0.3, 0.45, 0.675
        payloads = [running, running, running, running, finished]
        result, _, _ = _run_get(["BRAF"], fake_time, payloads)
        assert fake_time.sleeps == pytest.approx([0.2, 0.3, 0.45, 0.675])
        # 成功后取结果（results 为空 → failed 含原 id）
        _, failed = result
        assert failed == ["BRAF"]

    def test_backoff_caps_at_max_interval(self) -> None:
        """N08: 退避封顶 3.0s——连续失败 10 次后间隔保持 3.0。"""
        fake_time = _FakeTime()
        running = {"jobStatus": "RUNNING"}
        # 10 次 RUNNING 后 FINISHED
        payloads = [running] * 10 + [{"jobStatus": "FINISHED"}]
        _run_get(["BRAF"], fake_time, payloads)
        assert len(fake_time.sleeps) == 10
        assert fake_time.sleeps[0] == 0.2
        assert fake_time.sleeps[-1] == pytest.approx(3.0)
        # 递增且不超过 3.0（封顶后保持 3.0，允许相等）；同列表的 [:-1]/[1:] 切片长度恒相等
        for prev, nxt in zip(fake_time.sleeps[:-1], fake_time.sleeps[1:], strict=True):
            assert nxt <= 3.0 + 1e-9
            assert nxt >= prev

    def test_finished_immediately_no_poll_sleep(self) -> None:
        """N08: 首次轮询即 FINISHED → 无任何 sleep（快速路径）。"""
        fake_time = _FakeTime()
        _run_get(["BRAF"], fake_time, [{"jobStatus": "FINISHED"}])
        assert fake_time.sleeps == []

    def test_poll_respects_total_budget(self) -> None:
        """N08: 总预算 60s 内不越界——deadline 到达即放弃并返回 failed。"""
        fake_time = _FakeTime()
        running = {"jobStatus": "RUNNING"}
        # 大量 RUNNING：sleep 序列逼近但不超过 60s 预算
        payloads = [running] * 200
        result, _, _ = _run_get(["BRAF"], fake_time, payloads)
        assert fake_time.now <= 60.0 + 1e-9, f"polled past budget: {fake_time.now:.3f}s"
        _, failed = result
        assert failed == ["BRAF"]

    def test_submit_failure_returns_all_failed(self) -> None:
        """N08: 提交阶段网络异常 → 空结果 + 全部 id 标记 failed。"""
        with patch.object(gene_mapper.requests, "post", side_effect=gene_mapper.requests.RequestException("boom")):
            mapper = _RequestsUniProtMapper()
            result, failed = mapper.get(["BRAF"])
        assert result.empty
        assert failed == ["BRAF"]

    def test_submit_empty_job_id_returns_all_failed(self) -> None:
        """N08: 空 job_id → 空结果 + 全部 failed。"""
        with patch.object(gene_mapper.requests, "post", return_value=_poll_response(200, text="")):
            mapper = _RequestsUniProtMapper()
            result, failed = mapper.get(["BRAF"])
        assert result.empty
        assert failed == ["BRAF"]

    def test_success_path_returns_mapped_rows(self) -> None:
        """N08: FINISHED 后 results 解析出 From/To 行与 failed 清单。"""
        fake_time = _FakeTime()
        result_tuple, _, _ = _run_get(
            ["BRAF", "NONE"],
            fake_time,
            [{"jobStatus": "FINISHED"}],
            results_payload={
                "results": [
                    {"from": "BRAF", "to": {"primaryAccession": "P15056"}},
                ]
            },
        )
        result, failed = result_tuple
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["From", "To"]
        assert result["To"].tolist() == ["P15056"]
        assert failed == ["NONE"]

    def test_empty_ids_short_circuits(self) -> None:
        """N08: 空 id 列表不发起任何请求。"""
        with (
            patch.object(gene_mapper.requests, "post") as post,
            patch.object(gene_mapper.requests, "get") as get,
        ):
            mapper = _RequestsUniProtMapper()
            result, failed = mapper.get([])
        assert result.empty
        assert failed == []
        post.assert_not_called()
        get.assert_not_called()
