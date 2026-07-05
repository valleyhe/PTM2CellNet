"""Unit tests for Round 2 coverage gaps identified in the v2 audit report.

Covers:
- ``cross_validate`` standalone function (was 0 dedicated tests).
- ``calculate_ranking_metrics`` + ``calculate_metric_ci`` (only indirectly
  exercised via ``evaluate()``).
- ``SignalingNetworkMapper`` core methods
  (``predict_pathway_activity`` / ``generate_network_report`` / downstream).
- API ``/metrics`` endpoint in both Prometheus text and legacy JSON formats.
- API rate-limit middleware (429 path).
- DAVF-aware ``preprocess_request`` / ``batch_preprocess`` producing
  ``davf_sites`` / ``davf_gene_names`` batch keys when ``use_davf`` is set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------


class _TinyDataset(Dataset):
    """Deterministic toy dataset for cross-validate tests."""

    def __init__(self, n: int = 20, in_dim: int = 4, n_classes: int = 3) -> None:
        rng = np.random.RandomState(0)
        self.x = torch.from_numpy(rng.randn(n, in_dim)).float()
        # Labels that correlate with the first feature so a model can learn
        # *something* (otherwise mean/std tests are uninformative).
        self.y = torch.from_numpy(
            (self.x[:, 0] > 0).long().numpy()
        ).long() % n_classes

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return {"features": self.x[idx], "label": self.y[idx]}


class _TinyLinearClassifier(nn.Module):
    """Minimal model that consumes ``features`` and emits a prediction dict.

    Returns the ``logits`` / ``probabilities`` / ``predictions`` keys that
    :func:`src.evaluation.evaluators.evaluate` expects from a model output
    dict (see ``Evaluator.evaluate`` contract).
    """

    def __init__(self, in_dim: int = 4, n_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, batch):
        if isinstance(batch, dict):
            logits = self.linear(batch["features"])
        else:
            logits = self.linear(batch)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }


def _tiny_train_fn(model, loader, **_kw):
    """One-pass SGD training callback for cross_validate."""
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        opt.zero_grad()
        out = model(batch)
        loss = loss_fn(out["logits"], batch["label"])
        loss.backward()
        opt.step()


def test_cross_validate_returns_aggregated_metrics(monkeypatch):
    """cross_validate should produce fold_metrics + mean/std/CI structure."""
    # Force CPU so the test is deterministic even when earlier tests in the
    # full suite have left CUDA state warm (e.g. models moved to cuda). The
    # Evaluator auto-selects CUDA when available, but our toy data is CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from src.evaluation.evaluators import Evaluator

    # n_classes=2 matches the binary labels produced by _TinyDataset, so every
    # fold contains both classes and multiclass AUC doesn't raise.
    ds = _TinyDataset(n=20, in_dim=4, n_classes=2)
    # Force CPU so the test is deterministic on machines with CUDA.
    evaluator = Evaluator(model=_TinyLinearClassifier(n_classes=2), device="cpu")
    results = evaluator.cross_validate(
        model_class=_TinyLinearClassifier,
        dataset=ds,
        n_splits=4,
        batch_size=4,
        train_fn=_tiny_train_fn,
        model_kwargs={"n_classes": 2},
    )

    assert results["n_splits"] == 4
    assert len(results["fold_metrics"]) == 4
    # mean_metrics should contain the same scalar keys present in fold_metrics.
    fold_keys = set(results["fold_metrics"][0].keys())
    mean_keys = set(results["mean_metrics"].keys())
    assert mean_keys.issubset(fold_keys)
    # Every mean value must be a float. NaN is acceptable here because some
    # fold-level metrics (e.g. AUC) are undefined when a fold happens to
    # contain a single class on this tiny dataset; aggregating across folds
    # then yields NaN. The contract we care about is that the aggregation
    # produced a scalar — not that every metric is finite on toy data.
    for v in results["mean_metrics"].values():
        assert isinstance(v, float)


def test_cross_validate_rejects_invalid_args():
    """cross_validate must validate n_splits and dataset size up front."""
    from src.evaluation.evaluators import Evaluator

    evaluator = Evaluator(model=_TinyLinearClassifier(), device="cpu")
    ds = _TinyDataset(n=5)
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        evaluator.cross_validate(_TinyLinearClassifier, ds, n_splits=1)
    with pytest.raises(ValueError, match="fewer than"):
        evaluator.cross_validate(_TinyLinearClassifier, ds, n_splits=10)


def test_cross_validate_standalone_function_returns_scalars(monkeypatch):
    """The module-level cross_validate helper must return scalar-only metrics."""
    # Force CPU so the test is deterministic regardless of host CUDA. The
    # standalone helper builds its own Evaluator and auto-selects CUDA when
    # available, but the toy dataset/model here are CPU-only.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from src.evaluation.evaluators import cross_validate

    # n_classes=2 matches _TinyDataset labels so every fold has both classes.
    ds = _TinyDataset(n=12, in_dim=4, n_classes=2)
    results = cross_validate(
        _TinyLinearClassifier,
        ds,
        n_splits=3,
        batch_size=4,
        train_fn=_tiny_train_fn,
        model_kwargs={"n_classes": 2},
    )
    # Standalone variant filters to scalar metrics only (per AGENTS.md).
    assert isinstance(results, dict)
    assert all(isinstance(v, float) for v in results.values())


# ---------------------------------------------------------------------------
# Ranking metrics + bootstrap CI
# ---------------------------------------------------------------------------


def test_calculate_ranking_metrics_returns_ndcg_and_map():
    """Perfect ranking -> NDCG=1; random has lower values; both >= 0.

    NDCG handles graded relevance; MAP is defined over binary labels, so we
    use binary labels here (the conventional MAP setting).
    """
    from src.evaluation.metrics import calculate_ranking_metrics

    # Binary labels: three relevant items out of four.
    y_true = np.array([1, 1, 1, 0])
    # Perfect ranking: predicted score preserves true relevance order.
    y_score = np.array([0.9, 0.8, 0.7, 0.1])
    perfect = calculate_ranking_metrics(y_true, y_score)
    assert "ndcg" in perfect and "map" in perfect
    assert perfect["ndcg"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["map"] == pytest.approx(1.0, abs=1e-6)

    # Worst ranking: inverted — all relevant items sink to the bottom.
    inverted = calculate_ranking_metrics(y_true, y_score[::-1])
    assert 0.0 <= inverted["ndcg"] <= 1.0
    assert 0.0 <= inverted["map"] <= 1.0
    assert inverted["ndcg"] < perfect["ndcg"]
    assert inverted["map"] < perfect["map"]


def test_calculate_metric_ci_brackets_point_estimate():
    """CI must bound the point estimate and be ordered lower <= upper."""
    from src.evaluation.metrics import calculate_accuracy, calculate_metric_ci

    rng = np.random.RandomState(7)
    n = 200
    y_true = rng.randint(0, 2, n)
    # A fairly accurate prediction so the CI is non-trivial.
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.2
    y_pred[flip] = 1 - y_pred[flip]

    ci = calculate_metric_ci(y_true, y_pred, calculate_accuracy, n_bootstrap=200)
    assert {"point_estimate", "ci_lower", "ci_upper", "confidence"} == set(ci.keys())
    assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]
    # 95% CI for n=200 with ~0.8 accuracy should be reasonably tight.
    assert ci["ci_upper"] - ci["ci_lower"] < 0.2

# ---------------------------------------------------------------------------
# SignalingNetworkMapper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signaling_mapper():
    from src.models.signaling_network import SignalingNetworkMapper

    # No external DB → built-in pathways only. Stable for assertions.
    return SignalingNetworkMapper()


def test_signaling_mapper_predict_pathway_activity_basic(signaling_mapper):
    """BRAF phosphorylation gain should activate MAPK/ERK (positive score)."""
    df = pd.DataFrame(
        [
            {
                "gene_symbol": "BRAF",
                "ptm_type": "Phosphorylation",
                "effect": "gain",
                "delta_prob": 0.5,
            }
        ]
    )
    activities = signaling_mapper.predict_pathway_activity(df)
    assert "MAPK/ERK" in activities
    # gain effect + positive delta_prob => positive activity.
    assert activities["MAPK/ERK"] > 0


def test_signaling_mapper_predict_pathway_activity_loss_is_negative(signaling_mapper):
    """Loss effect on a substrate should produce a negative activity."""
    df = pd.DataFrame(
        [
            {
                "gene_symbol": "PTEN",  # PI3K/AKT substrate
                "ptm_type": "Phosphorylation",
                "effect": "loss",
                "delta_prob": 0.5,
            }
        ]
    )
    activities = signaling_mapper.predict_pathway_activity(df)
    assert "PI3K/AKT" in activities
    assert activities["PI3K/AKT"] < 0


def test_signaling_mapper_predict_pathway_activity_unknown_gene(signaling_mapper):
    """Unknown gene / unrelated PTM type should yield empty dict."""
    df = pd.DataFrame(
        [{"gene_symbol": "GENE_NOT_IN_DB", "ptm_type": "Phosphorylation", "effect": "gain"}]
    )
    assert signaling_mapper.predict_pathway_activity(df) == {}


def test_signaling_mapper_generate_network_report_shape(signaling_mapper):
    """generate_network_report must return the documented top-level keys."""
    df = pd.DataFrame(
        [
            {
                "gene_symbol": "BRAF",
                "ptm_type": "Phosphorylation",
                "effect": "gain",
                "delta_prob": 0.5,
            },
            {
                "gene_symbol": "AKT1",
                "ptm_type": "Phosphorylation",
                "effect": "gain",
                "delta_prob": 0.4,
            },
        ]
    )
    report = signaling_mapper.generate_network_report(df)
    for key in ("summary", "pathway_activities", "key_pathways", "downstream_genes", "interpretation"):
        assert key in report
    assert report["summary"]["total_ptm_changes"] == 2


# ---------------------------------------------------------------------------
# API /metrics endpoint (Prometheus + JSON)
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_app():
    """Build a fresh app with metrics_enabled turned on."""
    import importlib
    import os

    # Make _setup_monitoring believe the config enables metrics. We patch the
    # config file path env var to point at a temporary YAML.
    import tempfile

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write("monitoring:\n  metrics_enabled: true\n  health_check_interval: 30\n")
    tmp.close()
    prev = os.environ.get("PTM2CELLNET_CONFIG")
    # Disable rate limiting during metrics endpoint tests so the TestClient
    # is never throttled by the protective middleware.
    prev_rpm = os.environ.get("PTM2CELLNET_RATE_LIMIT_RPM")
    os.environ["PTM2CELLNET_CONFIG"] = tmp.name
    os.environ["PTM2CELLNET_RATE_LIMIT_RPM"] = "0"
    try:
        from src.api import app as app_module

        importlib.reload(app_module)
        yield app_module.app
    finally:
        if prev is None:
            os.environ.pop("PTM2CELLNET_CONFIG", None)
        else:
            os.environ["PTM2CELLNET_CONFIG"] = prev
        if prev_rpm is None:
            os.environ.pop("PTM2CELLNET_RATE_LIMIT_RPM", None)
        else:
            os.environ["PTM2CELLNET_RATE_LIMIT_RPM"] = prev_rpm
        import os as _os

        _os.unlink(tmp.name)


def test_metrics_endpoint_returns_prometheus_text(metrics_app):
    """Default /metrics should be Prometheus exposition text."""
    from fastapi.testclient import TestClient

    client = TestClient(metrics_app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    # Prometheus exposition must include HELP/TYPE lines for our metrics.
    assert "# HELP ptm2cellnet_requests_total" in body
    assert "# TYPE ptm2cellnet_requests_total counter" in body
    assert "ptm2cellnet_requests_total" in body
    # Latency histogram bucket line with label.
    assert "ptm2cellnet_latency_bucket_ms{" in body


def test_metrics_endpoint_returns_json_when_requested(metrics_app):
    """?format=json (or Accept: application/json) returns the legacy JSON."""
    import json as _json

    from fastapi.testclient import TestClient

    client = TestClient(metrics_app)
    # JSON via Accept header
    resp = client.get("/metrics", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["service"] == "PTM2CellNet"
    assert "runtime" in payload

    # JSON via query param
    resp2 = client.get("/metrics?format=json")
    assert resp2.status_code == 200
    assert _json.loads(resp2.text)["service"] == "PTM2CellNet"


# ---------------------------------------------------------------------------
# Rate-limit middleware (429 path)
# ---------------------------------------------------------------------------


def test_rate_limit_middleware_returns_429_when_exceeded():
    """Burst over the configured limit must return HTTP 429 + Retry-After."""
    import os
    from fastapi.testclient import TestClient

    # Build a fresh app with a tiny rate limit so we can exhaust it in-test.
    prev_rpm = os.environ.get("PTM2CELLNET_RATE_LIMIT_RPM")
    prev_burst = os.environ.get("PTM2CELLNET_RATE_LIMIT_BURST")
    os.environ["PTM2CELLNET_RATE_LIMIT_RPM"] = "3"
    os.environ["PTM2CELLNET_RATE_LIMIT_BURST"] = "3"
    try:
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)
        # / is exempt from rate limiting (health), so hit /docs (always 200)
        # or any non-health path. We use a deliberately unknown path: it 404s
        # but still counts against the limiter until exhausted.
        codes = []
        for _ in range(6):
            r = client.get("/api/v1/some-predict-path-that-does-not-exist")
            codes.append(r.status_code)
        # Once the bucket is exhausted we must see at least one 429.
        assert 429 in codes, f"Expected a 429 in {codes}"
    finally:
        if prev_rpm is None:
            os.environ.pop("PTM2CELLNET_RATE_LIMIT_RPM", None)
        else:
            os.environ["PTM2CELLNET_RATE_LIMIT_RPM"] = prev_rpm
        if prev_burst is None:
            os.environ.pop("PTM2CELLNET_RATE_LIMIT_BURST", None)
        else:
            os.environ["PTM2CELLNET_RATE_LIMIT_BURST"] = prev_burst


# ---------------------------------------------------------------------------
# DAVF-aware preprocess_request / batch_preprocess
# ---------------------------------------------------------------------------


class _FakeDavfModel:
    """Stand-in model with use_davf=True so preprocess emits DAVF batch keys."""

    use_davf = True


def test_preprocess_request_emits_davf_keys_when_model_supports_davf():
    """When STATE.model.use_davf, preprocess_request adds davf_* keys."""
    from src.api.routes import state as state_mod
    from src.api.routes import predictions as preds
    from src.api.schemas import PTMSite, PredictionRequest

    # Save & patch STATE.
    saved_model = state_mod.STATE.model
    state_mod.STATE.model = _FakeDavfModel()
    try:
        req = PredictionRequest(
            sequence="ACDEFGHIK",
            ptm_sites=[
                PTMSite(position=3, type="phosphorylation", gene_symbol="BRAF"),
                PTMSite(position=5, type="phosphorylation"),  # no gene → skipped
            ],
            use_davf=True,
        )

        batch = preds.preprocess_request(req)
        assert "davf_sites" in batch
        assert "davf_gene_names" in batch
        # Only the gene_symbol-bearing site survives.
        assert len(batch["davf_sites"]) == 1
        assert batch["davf_gene_names"] == ["BRAF"]
    finally:
        state_mod.STATE.model = saved_model


def test_batch_preprocess_emits_davf_keys_per_sample():
    """batch_preprocess should attach per-sample DAVF lists when enabled."""
    from src.api.routes import state as state_mod
    from src.api.routes import predictions as preds
    from src.api.schemas import PTMSite, PredictionRequest

    saved_model = state_mod.STATE.model
    state_mod.STATE.model = _FakeDavfModel()
    try:
        samples = [
            PredictionRequest(
                sequence="ACDEFGHIK",
                ptm_sites=[
                    PTMSite(position=3, type="phosphorylation", gene_symbol="BRAF"),
                ],
                use_davf=True,
            ),
            PredictionRequest(sequence="LMNPQRSTVWY", use_davf=True),
        ]
        batch = preds.batch_preprocess(samples)
        assert "davf_sites" in batch
        assert "davf_gene_names" in batch
        # Two samples -> two entries (second is empty list).
        assert len(batch["davf_sites"]) == 2
        assert batch["davf_sites"][0] == [{"position": 3, "ptm_type": "phosphorylation"}]
        assert batch["davf_sites"][1] == []
        assert batch["davf_gene_names"][0] == ["BRAF"]
        assert batch["davf_gene_names"][1] == []
    finally:
        state_mod.STATE.model = saved_model
