"""Unit tests for UniProt ID-mapping poll backoff (N08, 2026-08-16).

Covers the exponential-backoff rewrite of ``_RequestsUniProtMapper.get``:
* backoff interval sequence grows 0.2s x1.5 and caps at 3.0s;
* a job that finishes immediately causes no polling sleep;
* the 60s total budget is respected (no sleep past the deadline);
* success/failure paths return the documented (DataFrame, failed_ids) shape.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.analysis.gene_mapper as gm


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


def _response(status_code: int = 200, payload=None, text: str = "job-1"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = payload if payload is not None else {}
    return resp


def _run_get(ids, fake_time, get_payloads, results_payload=None, post_payload_text="job-1"):
    """Drive mapper.get() with fake clock + scripted poll responses.

    ``get_payloads`` are the poll (/details) responses; after the job
    finishes, mapper.get() issues one more GET to /results/{job_id},
    answered by ``results_payload`` (default: empty results).
    """
    get_side_effect = [_response(200, p) for p in get_payloads]
    get_side_effect.append(_response(200, results_payload if results_payload is not None else {"results": []}))
    with (
        patch.object(gm.requests, "post", return_value=_response(200, text=post_payload_text)) as post,
        patch.object(gm.requests, "get", side_effect=get_side_effect) as get,
        patch.object(gm.time, "monotonic", side_effect=fake_time.monotonic),
        patch.object(gm.time, "sleep", side_effect=fake_time.sleep),
    ):
        mapper = gm._RequestsUniProtMapper()
        result = mapper.get(ids, from_db="Gene_Name", to_db="UniProtKB")
    return result, post, get


def test_backoff_sequence_grows_and_caps() -> None:
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


def test_backoff_caps_at_max_interval() -> None:
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


def test_finished_immediately_no_poll_sleep() -> None:
    """N08: 首次轮询即 FINISHED → 无任何 sleep（快速路径）。"""
    fake_time = _FakeTime()
    _run_get(["BRAF"], fake_time, [{"jobStatus": "FINISHED"}])
    assert fake_time.sleeps == []


def test_poll_respects_total_budget() -> None:
    """N08: 总预算 60s 内不越界——deadline 到达即放弃并返回 failed。"""
    fake_time = _FakeTime()
    running = {"jobStatus": "RUNNING"}
    # 大量 RUNNING：sleep 序列逼近但不超过 60s 预算
    payloads = [running] * 200
    result, _, _ = _run_get(["BRAF"], fake_time, payloads)
    assert fake_time.now <= 60.0 + 1e-9, f"polled past budget: {fake_time.now:.3f}s"
    _, failed = result
    assert failed == ["BRAF"]


def test_submit_failure_returns_all_failed() -> None:
    """N08: 提交阶段网络异常 → 空结果 + 全部 id 标记 failed。"""
    with patch.object(gm.requests, "post", side_effect=gm.requests.RequestException("boom")):
        mapper = gm._RequestsUniProtMapper()
        result, failed = mapper.get(["BRAF"])
    assert result.empty
    assert failed == ["BRAF"]


def test_submit_empty_job_id_returns_all_failed() -> None:
    """N08: 空 job_id → 空结果 + 全部 failed。"""
    with patch.object(gm.requests, "post", return_value=_response(200, text="")):
        mapper = gm._RequestsUniProtMapper()
        result, failed = mapper.get(["BRAF"])
    assert result.empty
    assert failed == ["BRAF"]


def test_success_path_returns_mapped_rows() -> None:
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


def test_empty_ids_short_circuits() -> None:
    """N08: 空 id 列表不发起任何请求。"""
    with (
        patch.object(gm.requests, "post") as post,
        patch.object(gm.requests, "get") as get,
    ):
        mapper = gm._RequestsUniProtMapper()
        result, failed = mapper.get([])
    assert result.empty
    assert failed == []
    post.assert_not_called()
    get.assert_not_called()
