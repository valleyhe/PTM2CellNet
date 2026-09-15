"""
API模块单元测试
"""

import pydantic
import pytest
from fastapi.testclient import TestClient
import torch
import torch.nn as nn

from src.api.schemas import (
    PTMSite,
    PredictionRequest,
    BatchPredictionRequest,
    VariantPredictionRequest,
    PTMEffect,
    PathwayImpact,
    VariantInfo,
)
from src.api.app import create_app
from src.api.routes import initialize_model
from src.api.routes.state import reset_state


class SimpleModel(nn.Module):
    """简单的测试模型"""

    def __init__(self, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(21, 64)  # 0=padding, 1-20=amino acids
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
        predictions = torch.argmax(probabilities, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }


@pytest.fixture(autouse=True)
def _isolate_state():
    """Reset global STATE before and after each test (P0-1).

    Without this, tests that call ``/api/v1/initialize`` (or production code
    paths that initialize the variant workflow) leak state into later tests
    that assert "workflow not initialized". ``reset_state`` is the single
    source of truth for clearing every lifecycle field.
    """
    reset_state()
    yield
    reset_state()


@pytest.fixture
def client():
    """测试客户端fixture"""
    app = create_app()
    model = SimpleModel()
    cell_states = ["proliferation", "differentiation", "apoptosis", "quiescence"]
    initialize_model(model, cell_states, "cpu")
    return TestClient(app)


class TestSchemas:
    """数据模型测试"""

    def test_ptm_site(self):
        """测试PTM位点模型"""
        site = PTMSite(position=5, type="phosphorylation", amino_acid="S")
        assert site.position == 5
        assert site.type == "phosphorylation"
        assert site.amino_acid == "S"

    def test_prediction_request(self):
        """测试预测请求模型"""
        request = PredictionRequest(
            sequence="ACDEFGHIKLMNPQRSTVWY", ptm_sites=[PTMSite(position=5, type="phosphorylation")]
        )
        assert request.sequence == "ACDEFGHIKLMNPQRSTVWY"
        assert len(request.ptm_sites) == 1

    def test_batch_prediction_request(self):
        """测试批量预测请求模型"""
        request = BatchPredictionRequest(
            samples=[
                PredictionRequest(sequence="ACDEFGHIKL"),
                PredictionRequest(sequence="LMNPQRSTVWY"),
            ]
        )
        assert len(request.samples) == 2


class TestAPI:
    """API测试"""

    def test_root(self, client):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_model_info(self, client):
        """测试模型信息"""
        response = client.get("/api/v1/model_info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "PTM2CellNet"
        assert "cell_states" in data
        assert len(data["cell_states"]) == 4

    def test_predict(self, client):
        """测试单样本预测"""
        request_data = {
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_sites": [{"position": 5, "type": "phosphorylation", "amino_acid": "F"}],
        }
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "cell_state" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_invalid_sequence(self, client):
        """测试无效序列预测"""
        request_data = {"sequence": "ACXDEFGHIKL", "ptm_sites": []}
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 400

    def test_batch_predict(self, client):
        """测试批量预测"""
        request_data = {
            "samples": [
                {"sequence": "ACDEFGHIKL", "ptm_sites": []},
                {"sequence": "LMNPQRSTVWY", "ptm_sites": [{"position": 3, "type": "acetylation"}]},
            ]
        }
        response = client.post("/api/v1/batch_predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["sample_count"] == 2


class TestVariantSchemas:
    """Variant prediction schema tests."""

    def test_ptm_effect_schema(self):
        """Test PTMEffect schema validation."""
        effect = PTMEffect(
            ptm_type="Phosphorylation", wildtype_prob=0.8, mutant_prob=0.3, delta_prob=-0.5, effect="loss"
        )
        assert effect.ptm_type == "Phosphorylation"
        assert effect.wildtype_prob == 0.8
        assert effect.mutant_prob == 0.3
        assert effect.delta_prob == -0.5
        assert effect.effect == "loss"

    def test_ptm_effect_probabilities_validated(self):
        """Test PTMEffect validates probability ranges."""
        with pytest.raises(pydantic.ValidationError):
            PTMEffect(
                ptm_type="Phosphorylation",
                wildtype_prob=1.5,  # Invalid: > 1.0
                mutant_prob=0.3,
                delta_prob=-0.5,
                effect="loss",
            )

    def test_pathway_impact_schema(self):
        """Test PathwayImpact schema."""
        impact = PathwayImpact(
            pathway_name="MAPK/ERK", activity_change=0.5, confidence="high", key_genes=["BRAF", "MAPK1"]
        )
        assert impact.pathway_name == "MAPK/ERK"
        assert impact.activity_change == 0.5
        assert impact.confidence == "high"
        assert "BRAF" in impact.key_genes

    def test_variant_info_schema(self):
        """Test VariantInfo schema."""
        variant = VariantInfo(
            hgvs="BRAF:p.V600E", gene_symbol="BRAF", uniprot_id="P15056", position=600, ref_aa="V", alt_aa="E"
        )
        assert variant.hgvs == "BRAF:p.V600E"
        assert variant.position == 600
        assert variant.ref_aa == "V"
        assert variant.alt_aa == "E"

    def test_variant_info_position_validation(self):
        """Test VariantInfo validates position >= 1."""
        with pytest.raises(pydantic.ValidationError):
            VariantInfo(
                hgvs="BRAF:p.V600E",
                position=0,  # Invalid: < 1
                ref_aa="V",
                alt_aa="E",
            )

    def test_variant_prediction_request_schema(self):
        """Test VariantPredictionRequest schema."""
        request = VariantPredictionRequest(
            hgvs="BRAF:p.V600E", sequence="MNT...", uniprot_id="P15056", include_pathways=True
        )
        assert request.hgvs == "BRAF:p.V600E"
        assert request.sequence == "MNT..."
        assert request.uniprot_id == "P15056"
        assert request.include_pathways is True

    def test_variant_prediction_request_defaults(self):
        """Test VariantPredictionRequest default values."""
        request = VariantPredictionRequest(hgvs="BRAF:p.V600E")
        assert request.sequence is None
        assert request.uniprot_id is None
        assert request.include_pathways is True  # Default


class TestVariantAPI:
    """Variant prediction API tests."""

    def test_predict_variant_not_initialized(self, client):
        """Test variant prediction when workflow not initialized."""
        request_data = {"hgvs": "BRAF:p.V600E", "sequence": "MNT..."}
        response = client.post("/api/v1/predict/variant", json=request_data)
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()

    def test_predict_variant_missing_sequence(self):
        """Test variant prediction without sequence raises error."""
        # This tests the request validation, not the endpoint
        # The endpoint requires sequence or uniprot_id
        request_data = {
            "hgvs": "BRAF:p.V600E"
            # No sequence provided
        }
        # When workflow is initialized but no sequence, should return 400
        # Note: This test would need workflow initialization to fully test

    def test_predict_variant_with_pathway_impacts(self, client):
        """Test variant prediction returns pathway_impacts when include_pathways=True."""
        from unittest.mock import patch, Mock
        from src.analysis.variant_workflow import VariantEffectResult

        # Initialize variant workflow with a mock
        mock_workflow = Mock()
        mock_workflow.predict_from_hgvs.return_value = VariantEffectResult(
            variant={
                "hgvs": "BRAF:p.V600E",
                "gene_symbol": "BRAF",
                "accession": "P15056",
                "position": 600,
                "ref_aa": "V",
                "alt_aa": "E",
            },
            sequence_info={"length": 766, "validated": True},
            ptm_effects={
                "Phosphorylation": {
                    "wildtype_prob": 0.8,
                    "mutant_prob": 0.2,
                    "delta_prob": -0.6,
                    "effect": "loss",
                }
            },
            pathway_impacts={
                "MAPK/ERK": {"activity": 0.6, "genes": ["FOS", "JUN"], "confidence": "high"},
                "PI3K/AKT": {"activity": 0.3, "genes": ["FOXO1"], "confidence": "medium"},
            },
        )

        with patch("src.api.routes.state.STATE.variant_workflow", mock_workflow):
            request_data = {
                "hgvs": "BRAF:p.V600E",
                "sequence": "M" * 599 + "V" + "A" * 166,
                "include_pathways": True,
            }
            response = client.post("/api/v1/predict/variant", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "pathway_impacts" in data
        assert data["pathway_impacts"] is not None
        impacts = data["pathway_impacts"]
        assert len(impacts) == 2
        # Check first pathway
        mapk_impact = next(i for i in impacts if i["pathway_name"] == "MAPK/ERK")
        assert mapk_impact["confidence"] == "high"
        assert mapk_impact["key_genes"] == ["FOS", "JUN"]


class TestApp:
    """应用工厂测试"""

    def test_create_app_default(self):
        """测试默认应用创建"""
        app = create_app()
        assert app.title == "PTM2CellNet API"
        assert app.version == "1.0.0"

    def test_create_app_custom(self):
        """测试自定义应用创建"""
        app = create_app(
            title="Custom API",
            description="Custom description",
            version="2.0.0",
        )
        assert app.title == "Custom API"
        assert app.version == "2.0.0"
