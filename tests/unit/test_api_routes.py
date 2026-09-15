"""Tests for API route handlers in src/api/routes/.

Covers model_info, predictions, and state routes using FastAPI TestClient
with mocked model initialization. TestEndToEndPrediction additionally
exercises the full chain with a real in-test model (no mock, no external
weights required).
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.state import STATE, initialize_model


# ---------------------------------------------------------------------------
# Minimal test model (matches the SimpleModel pattern from test_api.py)
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Minimal nn.Module that satisfies the route handler's expectations."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(21, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)
        self.encoder_type = "lstm"
        self.embed_dim = 64

    def forward(self, batch):
        sequences = batch["sequence"]
        x = self.embedding(sequences)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    """Ensure STATE is clean before and after each test."""
    # Save original state
    orig = {
        "model": STATE.model,
        "cell_states": STATE.cell_states,
        "label_to_idx": STATE.label_to_idx,
        "idx_to_label": STATE.idx_to_label,
        "device": STATE.device,
        "feature_extractor": STATE.feature_extractor,
        "max_sequence_length": STATE.max_sequence_length,
        "ptm_type_to_idx": STATE.ptm_type_to_idx,
        "variant_workflow": STATE.variant_workflow,
        "pathway_mapper": STATE.pathway_mapper,
    }
    yield
    # Restore original state
    for k, v in orig.items():
        setattr(STATE, k, v)


@pytest.fixture()
def app_client():
    """TestClient with an initialized model."""
    app = create_app()
    model = _SimpleModel(num_classes=4)
    cell_states = ["proliferation", "differentiation", "apoptosis", "quiescence"]
    initialize_model(model, cell_states, "cpu")
    return TestClient(app)


@pytest.fixture()
def app_client_no_model():
    """TestClient without model initialization (unhealthy state)."""
    STATE.model = None
    STATE.cell_states = []
    STATE.idx_to_label = {}
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# model_info route tests
# ---------------------------------------------------------------------------


class TestHealthRoute:
    """Tests for GET /api/v1/health."""

    def test_health_with_model_loaded(self, app_client):
        """Health endpoint returns 'healthy' when model is loaded."""
        resp = app_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_health_without_model(self, app_client_no_model):
        """Health endpoint returns 'unhealthy' when no model is loaded."""
        resp = app_client_no_model.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestReadinessRoute:
    """Tests for GET /api/v1/live and GET /api/v1/ready.

    These follow the liveness/readiness split (P1-3): ``/live`` reports process
    liveness (always 200 once the process is up), while ``/ready`` reports
    inference readiness and returns 503 until a model is loaded.
    """

    def test_live_is_200_without_model(self, app_client_no_model):
        """Liveness probe succeeds even when no model is loaded."""
        resp = app_client_no_model.get("/api/v1/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"

    def test_live_is_200_with_model(self, app_client):
        """Liveness probe succeeds when model is loaded too."""
        resp = app_client.get("/api/v1/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_ready_returns_503_without_model(self, app_client_no_model):
        """Readiness probe fails (503) when the model is not loaded."""
        resp = app_client_no_model.get("/api/v1/ready")
        assert resp.status_code == 503

    def test_ready_returns_200_with_model(self, app_client):
        """Readiness probe succeeds when the model is loaded."""
        resp = app_client.get("/api/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True


class TestVariantReadinessRoute:
    """Tests for GET /api/v1/ready/variant (P1-1)."""

    def test_variant_ready_503_without_core_model(self, app_client_no_model):
        """Without the core model the variant capability is not ready."""
        resp = app_client_no_model.get("/api/v1/ready/variant")
        assert resp.status_code == 503

    def test_variant_ready_503_without_workflow(self, app_client):
        """Core model ready but variant workflow absent -> 503 (per-capability)."""
        assert STATE.variant_workflow is None
        resp = app_client.get("/api/v1/ready/variant")
        assert resp.status_code == 503

    def test_variant_ready_200_with_workflow(self, app_client):
        """When the variant workflow is loaded the probe returns 200."""
        from unittest.mock import MagicMock

        mock_workflow = MagicMock()
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.get("/api/v1/ready/variant")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["variant_workflow_loaded"] is True
        assert data["model_loaded"] is True


class TestModelInfoRoute:
    """Tests for GET /api/v1/model_info."""

    def test_model_info_with_model(self, app_client):
        """Model info returns correct metadata when model is loaded."""
        resp = app_client.get("/api/v1/model_info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "PTM2CellNet"
        assert data["model_version"] == "1.0.0"
        assert data["encoder_type"] == "lstm"
        assert data["embed_dim"] == 64
        assert data["num_classes"] == 4
        assert data["cell_states"] == [
            "proliferation",
            "differentiation",
            "apoptosis",
            "quiescence",
        ]
        assert isinstance(data["supported_ptm_types"], list)

    def test_model_info_alt_path(self, app_client):
        """Model info is also available at /api/v1/model/info."""
        resp = app_client.get("/api/v1/model/info")
        assert resp.status_code == 200

    def test_model_info_without_model_returns_503(self, app_client_no_model):
        """Model info returns 503 when model is not initialized."""
        resp = app_client_no_model.get("/api/v1/model_info")
        assert resp.status_code == 503

    def test_model_info_reports_capability_and_provenance(self, app_client):
        """Model info surfaces capability flags and provenance (P1-1/P1-3)."""
        # Default state: no variant workflow / pathway mapper / provenance.
        resp = app_client.get("/api/v1/model_info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["variant_workflow_loaded"] is False
        assert data["pathway_mapper_loaded"] is False
        assert data["is_demo_model"] is False
        assert "model_kind" in data

        # When STATE carries provenance / a loaded variant workflow, it surfaces.
        from unittest.mock import MagicMock

        with (
            patch.object(STATE, "variant_workflow", MagicMock()),
            patch.object(STATE, "model_kind", "demo"),
            patch.object(STATE, "is_demo_model", True),
            patch.object(STATE, "checkpoint_path", "/x/best_model.pt"),
        ):
            resp = app_client.get("/api/v1/model_info")
        data = resp.json()
        assert data["variant_workflow_loaded"] is True
        assert data["model_kind"] == "demo"
        assert data["is_demo_model"] is True
        # SEC-03: route returns basename only to avoid leaking server paths.
        assert data["checkpoint_path"] == "best_model.pt"


# ---------------------------------------------------------------------------
# predictions route tests
# ---------------------------------------------------------------------------


class TestPredictRoute:
    """Tests for POST /api/v1/predict."""

    def test_predict_success(self, app_client):
        """Single prediction returns cell_state, confidence, probabilities."""
        payload = {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_sites": [{"position": 5, "type": "phosphorylation"}],
        }
        resp = app_client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "cell_state" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["probabilities"], dict)

    def test_predict_without_ptm_sites(self, app_client):
        """Prediction works with no PTM sites."""
        payload = {"sequence": "ACDEFGHIKLMNPQRSTVWY", "ptm_sites": []}
        resp = app_client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_invalid_sequence_returns_400(self, app_client):
        """Invalid amino acid characters result in 400."""
        payload = {"sequence": "ACXDEFGHIKL", "ptm_sites": []}
        resp = app_client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 400

    def test_predict_without_model_returns_503(self, app_client_no_model):
        """Prediction returns 503 when model is not loaded."""
        payload = {"sequence": "ACDEFGHIKLMNPQRSTVWY", "ptm_sites": []}
        resp = app_client_no_model.post("/api/v1/predict", json=payload)
        assert resp.status_code == 503


class TestBatchPredictRoute:
    """Tests for POST /api/v1/batch_predict."""

    def test_batch_predict_success(self, app_client):
        """Batch prediction returns results for each sample."""
        payload = {
            "samples": [
                {"sequence": "ACDEFGHIKL", "ptm_sites": []},
                {"sequence": "LMNPQRSTVWY", "ptm_sites": [{"position": 3, "type": "acetylation"}]},
            ]
        }
        resp = app_client.post("/api/v1/batch_predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_without_model_returns_503(self, app_client_no_model):
        """Batch prediction returns 503 when model is not loaded."""
        payload = {"samples": [{"sequence": "ACDEFGHIKL", "ptm_sites": []}]}
        resp = app_client_no_model.post("/api/v1/batch_predict", json=payload)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# state route tests
# ---------------------------------------------------------------------------


class TestStateModule:
    """Tests for src/api/routes/state.py initialization helpers."""

    def test_initialize_model_sets_state(self):
        """initialize_model populates STATE fields correctly."""
        model = _SimpleModel(num_classes=3)
        cell_states = ["A", "B", "C"]
        initialize_model(model, cell_states, "cpu")
        assert STATE.model is model
        assert STATE.cell_states == cell_states
        assert STATE.label_to_idx == {"A": 0, "B": 1, "C": 2}
        assert STATE.idx_to_label == {0: "A", 1: "B", 2: "C"}
        assert STATE.device == "cpu"
        assert STATE.feature_extractor is not None
        assert isinstance(STATE.ptm_type_to_idx, dict)

    def test_initialize_model_with_config(self):
        """initialize_model respects max_sequence_length from config."""
        model = _SimpleModel()
        config = {"data": {"max_sequence_length": 500}}
        initialize_model(model, ["X"], "cpu", config=config)
        assert STATE.max_sequence_length == 500

    def test_initialize_variant_workflow_unavailable(self):
        """When VARIANT_WORKFLOW_AVAILABLE is False, workflow stays None."""
        with patch("src.api.routes.state.VARIANT_WORKFLOW_AVAILABLE", False):
            from src.api.routes.state import initialize_variant_workflow

            initialize_variant_workflow("/fake/path")
            assert STATE.variant_workflow is None

    def test_initialize_variant_workflow_logs_failure_reason(self, caplog):
        """Initialization failures log the exception message before clearing state."""
        with patch("src.api.routes.state.VARIANT_WORKFLOW_AVAILABLE", True):
            with patch("src.api.routes.state.VariantEffectWorkflow", side_effect=RuntimeError("boom")):
                from src.api.routes.state import initialize_variant_workflow

                with caplog.at_level("WARNING"):
                    initialize_variant_workflow("/fake/path")

        assert STATE.variant_workflow is None
        assert "Variant workflow initialization failed: boom" in caplog.text

    def test_initialize_pathway_mapper_unavailable(self):
        """When SIGNALING_NETWORK_AVAILABLE is False, mapper stays None."""
        with patch("src.api.routes.state.SIGNALING_NETWORK_AVAILABLE", False):
            from src.api.routes.state import initialize_pathway_mapper

            initialize_pathway_mapper()
            assert STATE.pathway_mapper is None

    def test_reset_state_clears_all_lifecycle_fields(self):
        """reset_state wipes model + variant workflow + provenance (P0-1)."""
        from src.api.routes.state import reset_state, record_model_provenance

        # Populate everything, including provenance-only fields.
        initialize_model(_SimpleModel(), ["A", "B"], "cpu")
        STATE.variant_workflow = object()
        STATE.pathway_mapper = object()
        record_model_provenance("/x/m.pt", "/x/m.config.yaml", {"model": {"model_kind": "demo"}})
        assert STATE.model is not None
        assert STATE.variant_workflow is not None
        assert STATE.is_demo_model is True

        reset_state()
        assert STATE.model is None
        assert STATE.cell_states == []
        assert STATE.variant_workflow is None
        assert STATE.pathway_mapper is None
        assert STATE.checkpoint_path is None
        assert STATE.config_path is None
        assert STATE.model_kind is None
        assert STATE.is_demo_model is False

    def test_initialize_model_clears_stale_variant_workflow(self):
        """Loading a new model must clear a stale variant workflow (P0-1/P1-1)."""
        initialize_model(_SimpleModel(), ["A"], "cpu")
        STATE.variant_workflow = object()  # simulate leftover from prior model
        STATE.pathway_mapper = object()

        initialize_model(_SimpleModel(), ["A", "B"], "cpu")
        # New model lifecycle must not inherit the prior variant workflow.
        assert STATE.variant_workflow is None
        assert STATE.pathway_mapper is None
        assert STATE.model is not None

    def test_record_model_provenance_detects_demo(self):
        """record_model_provenance flags demo models from config (P1-3)."""
        from src.api.routes.state import record_model_provenance

        record_model_provenance("/x/m.pt", "/x/m.config.yaml", {"model": {"model_kind": "demo"}})
        assert STATE.model_kind == "demo"
        assert STATE.is_demo_model is True
        assert STATE.checkpoint_path == "/x/m.pt"

        record_model_provenance(
            "/x/real.pt",
            "/x/real.config.yaml",
            {"data_provenance": {"training_data": "synthetic_random"}},
        )
        assert STATE.model_kind == "demo"
        assert STATE.is_demo_model is True

        record_model_provenance("/x/real.pt", "/x/real.config.yaml", {"model": {"model_kind": "real"}})
        assert STATE.model_kind == "real"
        assert STATE.is_demo_model is False


# ---------------------------------------------------------------------------
# Variant prediction route tests
# ---------------------------------------------------------------------------


class TestVariantPredictRoute:
    """Tests for POST /api/v1/predict/variant."""

    def test_variant_not_initialized_returns_503(self, app_client):
        """Variant prediction returns 503 when workflow is not initialized."""
        payload = {"hgvs": "BRAF:p.V600E", "sequence": "MMMM"}
        resp = app_client.post("/api/v1/predict/variant", json=payload)
        assert resp.status_code == 503

    def test_variant_with_mock_workflow(self, app_client):
        """Variant prediction works with a mocked workflow."""
        mock_workflow = MagicMock()
        mock_workflow.predict_from_hgvs.return_value = MagicMock(
            variant={
                "hgvs": "BRAF:p.V600E",
                "gene_symbol": "BRAF",
                "position": 600,
                "ref_aa": "V",
                "alt_aa": "E",
            },
            ptm_effects={
                "Phosphorylation": {
                    "wildtype_prob": 0.8,
                    "mutant_prob": 0.2,
                    "delta_prob": -0.6,
                    "effect": "loss",
                }
            },
            pathway_impacts=None,
        )
        with patch.object(STATE, "variant_workflow", mock_workflow):
            payload = {
                "hgvs": "BRAF:p.V600E",
                "sequence": "M" * 100,
                "include_pathways": False,
            }
            resp = app_client.post("/api/v1/predict/variant", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "variant" in data
        assert "ptm_effects" in data

    def test_variant_confidence_from_max_abs_delta(self, app_client):
        """confidence = min(max|delta_prob| * 2, 1.0)."""
        mock_workflow = MagicMock()
        mock_workflow.predict_from_hgvs.return_value = MagicMock(
            variant={"hgvs": "X:p.A1B", "gene_symbol": "X", "position": 1, "ref_aa": "A", "alt_aa": "B"},
            ptm_effects={
                "Phosphorylation": {"wildtype_prob": 0.8, "mutant_prob": 0.2, "delta_prob": -0.6, "effect": "loss"},
                "Ubiquitination": {"wildtype_prob": 0.5, "mutant_prob": 0.9, "delta_prob": 0.4, "effect": "gain"},
            },
            pathway_impacts=None,
        )
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B", "sequence": "M" * 100},
            )
        assert resp.status_code == 200
        # max |delta| = 0.6 -> 0.6 * 2 = 1.2 -> capped at 1.0
        assert resp.json()["confidence"] == 1.0

    def test_variant_fetches_sequence_from_uniprot(self, app_client):
        """uniprot_id-only request resolves the sequence via the workflow."""
        mock_workflow = MagicMock()
        mock_workflow.fetch_sequence_from_uniprot.return_value = "M" * 80
        mock_workflow.predict_from_hgvs.return_value = MagicMock(
            variant={"hgvs": "X:p.A1B", "gene_symbol": "X", "position": 1, "ref_aa": "A", "alt_aa": "B"},
            ptm_effects={},
            pathway_impacts=None,
        )
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B", "uniprot_id": "P12345"},
            )
        assert resp.status_code == 200
        mock_workflow.fetch_sequence_from_uniprot.assert_called_once_with("P12345")

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (ConnectionError("down"), 504),
            (TimeoutError("slow"), 504),
            (ValueError("bad id"), 400),
            (KeyError("missing"), 400),
            (TypeError("wrong type"), 400),
            (OSError("io"), 400),
        ],
    )
    def test_variant_uniprot_fetch_error_mapping(self, app_client, exc, expected):
        """UniProt fetch failures map to 504 (network) / 400 (parse, unexpected)."""
        mock_workflow = MagicMock()
        mock_workflow.fetch_sequence_from_uniprot.side_effect = exc
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B", "uniprot_id": "P12345"},
            )
        assert resp.status_code == expected

    def test_variant_missing_sequence_returns_400(self, app_client):
        """Neither sequence nor uniprot_id -> 400 with explicit detail."""
        mock_workflow = MagicMock()
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B"},
            )
        assert resp.status_code == 400
        assert "Sequence required" in resp.json()["detail"]

    def test_variant_cell_state_prediction_populated(self, app_client):
        """With STATE.model initialized, cell_state_prediction is a known label."""
        mock_workflow = MagicMock()
        mock_workflow.predict_from_hgvs.return_value = MagicMock(
            variant={"hgvs": "X:p.A1B", "gene_symbol": "X", "position": 1, "ref_aa": "A", "alt_aa": "B"},
            ptm_effects={
                "Phosphorylation": {"wildtype_prob": 0.8, "mutant_prob": 0.2, "delta_prob": -0.6, "effect": "loss"},
            },
            pathway_impacts=None,
        )
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B", "sequence": "M" * 100},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cell_state_prediction"] in {
            "proliferation",
            "differentiation",
            "apoptosis",
            "quiescence",
        }
        assert not data["warnings"]

    def test_variant_cell_state_failure_degrades_to_warning(self, app_client):
        """Cell-state model failure yields 200 + warning, never a 500."""
        mock_workflow = MagicMock()
        mock_workflow.predict_from_hgvs.return_value = MagicMock(
            variant={"hgvs": "X:p.A1B", "gene_symbol": "X", "position": 1, "ref_aa": "A", "alt_aa": "B"},
            ptm_effects={},
            pathway_impacts=None,
        )
        broken_model = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(STATE, "variant_workflow", mock_workflow), patch.object(STATE, "model", broken_model):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B", "sequence": "M" * 100},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cell_state_prediction"] is None
        assert any("Cell-state prediction unavailable" in w for w in data["warnings"])

    def test_variant_pathway_impacts_mapped_with_confidence(self, app_client):
        """include_pathways maps workflow dicts to PathwayImpact with thresholds."""
        mock_workflow = MagicMock()
        mock_workflow.predict_from_hgvs.return_value = MagicMock(
            variant={"hgvs": "X:p.A1B", "gene_symbol": "X", "position": 1, "ref_aa": "A", "alt_aa": "B"},
            ptm_effects={},
            pathway_impacts={
                "MAPK cascade": {"activity": 0.8, "genes": ["BRAF", "MAP2K1"]},
                "Apoptosis": {"activity": -0.2, "genes": ["BAX"]},
            },
        )
        with patch.object(STATE, "variant_workflow", mock_workflow):
            resp = app_client.post(
                "/api/v1/predict/variant",
                json={"hgvs": "X:p.A1B", "sequence": "M" * 100, "include_pathways": True},
            )
        assert resp.status_code == 200
        impacts = {p["pathway_name"]: p for p in resp.json()["pathway_impacts"]}
        assert impacts["MAPK cascade"]["confidence"] == "high"
        assert impacts["Apoptosis"]["confidence"] == "medium"
        assert impacts["MAPK cascade"]["key_genes"] == ["BRAF", "MAP2K1"]


# ---------------------------------------------------------------------------
# End-to-end prediction with a real (in-test) model — replaces the former
# "TestWithRealWeights" placeholder whose body was `pass` and which was
# permanently skipped via a hard-coded has_real_weights=False flag.
# ---------------------------------------------------------------------------


class TestEndToEndPrediction:
    """Full predict chain: real nn.Module -> initialize_model -> route -> JSON.

    Uses a real model instance (no mock), so the assertions reflect actual
    route/model behavior: label set, probability normalization, PTM-site
    parsing, and batch responses.
    """

    def test_single_predict_returns_normalized_probs(self, app_client):
        resp = app_client.post(
            "/api/v1/predict",
            json={
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": [{"position": 5, "type": "phosphorylation", "amino_acid": "F"}],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "probabilities" in data
        probs = data["probabilities"]
        # 概率分布键必须与初始化标签一致（避免标签语义错位）
        assert set(probs.keys()) == set(STATE.cell_states)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-5

    def test_single_predict_with_ptm_sites(self, app_client):
        resp = app_client.post(
            "/api/v1/predict",
            json={
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm_sites": [{"position": 5, "type": "phosphorylation", "amino_acid": "F"}],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["cell_state"] in STATE.cell_states

    def test_batch_predict_returns_one_row_per_input(self, app_client):
        resp = app_client.post(
            "/api/v1/batch_predict",
            json={
                "samples": [
                    {"sequence": "ACDEFGHIKLMNPQRSTVWY", "ptm_sites": []},
                    {"sequence": "M" * 40, "ptm_sites": []},
                ]
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["sample_count"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["cell_state"] in STATE.cell_states
