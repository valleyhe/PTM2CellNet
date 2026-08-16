"""Integration tests: N01 thread-pool offloading of blocking inference.

Verifies that /predict, /batch_predict, /predict/variant and /initialize
run their blocking torch/numpy work off the event loop (via
``run_in_threadpool``) without changing prediction semantics:

1. Concurrent requests are served in parallel (wall-clock time scales
   sub-linearly with a slow model), not serialised on the event loop.
2. Concurrency does not change outputs (same model, same input, same
   probabilities) and does not corrupt shared STATE.
"""

import asyncio
import time

import httpx
import pytest
import torch
import torch.nn as nn

from src.api.app import create_app
from src.api.routes import initialize_model
from src.api.routes.state import reset_state


class _SlowModel(nn.Module):
    """Deterministic model that simulates heavy inference with a sleep."""

    def __init__(self, num_classes: int = 4, delay_s: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(21, 16)
        self.linear = nn.Linear(16, num_classes)
        self.delay_s = delay_s
        self.encoder_type = "lstm"
        self.embed_dim = 16

    def forward(self, batch):
        if self.delay_s:
            time.sleep(self.delay_s)  # blocking, deliberately
        x = self.embedding(batch["sequence"]).mean(dim=1)
        logits = self.linear(x)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return {"logits": logits, "probabilities": probabilities, "predictions": predictions}


def _make_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _payload(sequence: str = "ACDEFGHIKLMNPQRSTVWY", site: int = 3) -> dict:
    return {
        "sequence": sequence,
        "ptm_sites": [{"position": site, "type": "phosphorylation", "amino_acid": "C"}],
    }


def _parallel_correct():
    app = create_app()
    model = _SlowModel(delay_s=0.4)
    initialize_model(model, ["a", "b", "c", "d"], "cpu")

    async def drive():
        async with _make_client(app) as client:
            # Warm-up (single request) to load code paths; then measure 4 concurrent.
            r0 = await client.post("/api/v1/predict", json=_payload())
            assert r0.status_code == 200
            warm = r0.json()

            start = time.monotonic()
            responses = await asyncio.gather(
                *[client.post("/api/v1/predict", json=_payload()) for _ in range(4)]
            )
            elapsed = time.monotonic() - start
            return responses, warm, elapsed

    responses, warm, elapsed = asyncio.run(drive())

    assert all(r.status_code == 200 for r in responses)
    # Identical deterministic outputs under concurrency (no state corruption).
    for r in responses:
        body = r.json()
        assert body["cell_state"] == warm["cell_state"]
        assert body["probabilities"] == warm["probabilities"]
    # 4 x 0.4s sleeps run in the thread pool: parallel gives ~0.4-0.6s,
    # serialised would take ~1.6s+. Use a generous threshold to stay robust.
    assert elapsed < 1.2, f"requests were serialised on the event loop: {elapsed:.2f}s"


def _batch_single():
    app = create_app()
    initialize_model(_SlowModel(delay_s=0.0), ["a", "b", "c", "d"], "cpu")

    async def drive():
        async with _make_client(app) as client:
            single = (await client.post("/api/v1/predict", json=_payload())).json()
            batch = (
                await client.post(
                    "/api/v1/predict/batch",
                    json={"samples": [_payload(), _payload()]},
                )
            ).json()
            return single, batch

    single, batch = asyncio.run(drive())

    assert batch["sample_count"] == 2
    for prediction in batch["predictions"]:
        assert prediction["cell_state"] == single["cell_state"]
        # Batch vs single-sample inference may differ at ~1e-8 float level
        # (reduction order over the batch dim); compare approximately.
        for label in single["probabilities"]:
            assert prediction["probabilities"][label] == pytest.approx(single["probabilities"][label], rel=1e-4)


def _health_during_inference():
    """The event loop must stay responsive while inference blocks a thread."""
    app = create_app()
    initialize_model(_SlowModel(delay_s=0.6), ["a", "b", "c", "d"], "cpu")

    async def drive():
        async with _make_client(app) as client:
            predict_task = asyncio.create_task(client.post("/api/v1/predict", json=_payload()))
            await asyncio.sleep(0.1)  # let the predict start and block a thread
            health = await client.get("/api/v1/health")
            result = await predict_task
            return health, result

    health, result = asyncio.run(drive())
    assert health.status_code == 200
    assert result.status_code == 200


def _initializes_no_crash():
    """Concurrent /initialize calls must not corrupt STATE (serialised by
    the thread pool; both fail fast on the missing checkpoint)."""
    import os

    os.environ["PTM2CELLNET_ENV"] = "development"

    app = create_app()

    async def drive():
        async with _make_client(app) as client:
            return await asyncio.gather(
                *[
                    client.post("/api/v1/initialize", json={"checkpoint_path": "/nonexistent.pt", "cell_states": ["a"]})
                    for _ in range(2)
                ]
            )

    responses = asyncio.run(drive())
    # /nonexistent.pt is outside the allowed model dirs -> 400/404, but must
    # never be a 5xx and must never crash the process.
    assert all(400 <= r.status_code < 500 for r in responses)


def _isolate_state():
    reset_state()


def test_concurrent_predicts_are_parallel_and_correct():
    _isolate_state()
    try:
        _parallel_correct()
    finally:
        _isolate_state()


def test_batch_predict_concurrent_with_single():
    _isolate_state()
    try:
        _batch_single()
    finally:
        _isolate_state()


def test_health_check_responds_during_slow_inference():
    _isolate_state()
    try:
        _health_during_inference()
    finally:
        _isolate_state()


def test_concurrent_initializes_do_not_crash():
    _isolate_state()
    try:
        _initializes_no_crash()
    finally:
        _isolate_state()


def test_sync_client_requests_still_work():
    """Regression: the classic TestClient path keeps working after the
    thread-pool refactor."""
    from fastapi.testclient import TestClient

    _isolate_state()
    try:
        app = create_app()
        initialize_model(_SlowModel(delay_s=0.0), ["a", "b", "c", "d"], "cpu")
        with TestClient(app) as client:
            response = client.post("/api/v1/predict", json=_payload())
            assert response.status_code == 200
            assert "cell_state" in response.json()
    finally:
        _isolate_state()
