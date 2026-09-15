"""
PTM预测器模型和Lightning模块单元测试
"""

import importlib
import sys

import pytest
import torch

from src.models.multitask_ptm import (
    MultiTaskPTMPredictor,
    CNNEncoder as MultiTaskCNNEncoder,
    TransformerEncoder as MultiTaskTransformerEncoder,
    LSTMEncoder as MultiTaskLSTMEncoder,
    MultiTaskLoss,
)
from src.models.ptm_site_predictor import PTMSitePredictor, create_model
from src.training.ptm_site_lightning import PTMSiteLightning


class TestMultiTaskPTMPredictor:
    """MultiTaskPTMPredictor测试"""

    @pytest.fixture
    def sample_model(self):
        return MultiTaskPTMPredictor(
            vocab_size=21,
            embed_dim=8,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            ptm_types=["Phosphorylation", "Acetylation"],
        )

    @pytest.fixture
    def sample_batch(self):
        return {
            "sequence_indices": torch.randint(0, 21, (2, 10)),
            "labels": {
                "Phosphorylation": torch.tensor([0, 1]),
                "Acetylation": torch.tensor([1, 0]),
            },
        }

    def test_forward_returns_logits_and_probs(self, sample_model):
        batch = {
            "sequence_indices": torch.randint(0, 21, (2, 10)),
        }
        output = sample_model.forward(batch["sequence_indices"], ptm_type="Phosphorylation")
        assert "logits" in output
        assert "probs" in output
        assert output["logits"].shape == (2, 2)
        assert output["probs"].shape == (2, 2)

    def test_compute_loss(self, sample_model, sample_batch):
        output = sample_model.forward(sample_batch["sequence_indices"], ptm_type="Phosphorylation")
        # Use MultiTaskLoss to compute loss
        criterion = MultiTaskLoss(ptm_types=["Phosphorylation", "Acetylation"])
        outputs = {
            "Phosphorylation": output,
        }
        losses = criterion(outputs, sample_batch["labels"])
        assert "total_loss" in losses
        assert losses["total_loss"].dim() == 0


class TestMultiTaskPTMEncoders:
    """MultiTaskPTM编码器测试"""

    def test_cnn_encoder_forward(self):
        encoder = MultiTaskCNNEncoder(embed_dim=8, hidden_dim=8, num_layers=1)
        x = torch.randn(1, 5, 8)
        output = encoder(x)
        assert output.shape == (1, 8)

    def test_transformer_encoder_forward(self):
        encoder = MultiTaskTransformerEncoder(embed_dim=8, hidden_dim=8, num_layers=1, num_heads=2)
        x = torch.randn(1, 5, 8)
        output = encoder(x)
        assert output.shape == (1, 8)

    def test_lstm_encoder_forward(self):
        encoder = MultiTaskLSTMEncoder(embed_dim=8, hidden_dim=8, num_layers=1)
        x = torch.randn(1, 5, 8)
        output = encoder(x)
        assert output.shape == (1, 8)


class TestPTMSitePredictor:
    """PTMSitePredictor测试"""

    def test_site_predictor_forward_cnn(self):
        model = PTMSitePredictor(encoder_type="cnn", window_size=31)
        sequence_indices = torch.randint(0, 21, (1, 31))
        output = model(sequence_indices)
        assert output["probs"].shape == (1, 2)

    def test_site_predictor_forward_transformer(self):
        model = PTMSitePredictor(encoder_type="transformer", window_size=31)
        sequence_indices = torch.randint(0, 21, (1, 31))
        output = model(sequence_indices)
        assert output["probs"].shape == (1, 2)

    def test_site_predictor_forward_lstm(self):
        model = PTMSitePredictor(encoder_type="lstm", window_size=31)
        sequence_indices = torch.randint(0, 21, (1, 31))
        output = model(sequence_indices)
        assert output["probs"].shape == (1, 2)

    def test_site_predictor_invalid_encoder_raises(self):
        with pytest.raises(ValueError, match="未知编码器类型"):
            PTMSitePredictor(encoder_type="invalid")

    def test_create_model(self):
        model = create_model({"encoder_type": "cnn", "num_classes": 3})
        assert isinstance(model, PTMSitePredictor)
        assert model.classifier[-1].out_features == 3


class TestPTMSiteLightning:
    """PTMSiteLightning测试"""

    def test_lightning_module_creation(self):
        module = PTMSiteLightning(config={"model": {"encoder_type": "cnn"}})
        assert hasattr(module, "model")
        assert isinstance(module.model, PTMSitePredictor)

    def test_training_step_returns_loss(self):
        module = PTMSiteLightning(config={"model": {"encoder_type": "cnn"}})
        batch = {
            "sequence_indices": torch.randint(0, 21, (2, 31)),
            "label": torch.tensor([0, 1]),
        }
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_validation_step_returns_loss(self):
        module = PTMSiteLightning(config={"model": {"encoder_type": "cnn"}})
        batch = {
            "sequence_indices": torch.randint(0, 21, (2, 31)),
            "label": torch.tensor([0, 1]),
        }
        loss = module.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_configure_optimizers(self):
        module = PTMSiteLightning(config={"optimizer": "adamw", "learning_rate": 1e-3})
        result = module.configure_optimizers()
        assert isinstance(result, dict) or hasattr(result, "step")
        if isinstance(result, dict):
            assert "optimizer" in result


class TestEvaluationInitFallback:
    """src.evaluation导入回退测试"""

    def test_evaluation_init_import_fallback(self):
        import types

        # Create a synthetic evaluation module with try/except fallback
        # to verify the pattern works as intended
        fake_module = types.ModuleType("src.evaluation.fake_test")
        code = """
try:
    from sys import version_info as _vi
except ImportError:
    _vi = None
"""
        exec(code, fake_module.__dict__)
        assert fake_module._vi is not None

        # Now test the actual fallback behavior by creating a module where import fails
        fake_module2 = types.ModuleType("src.evaluation.fake_test2")
        code2 = """
try:
    from nonexistent_module import NonexistentClass
except ImportError:
    NonexistentClass = None
"""
        exec(code2, fake_module2.__dict__)
        assert fake_module2.NonexistentClass is None

        # To exercise the actual src.evaluation code path, temporarily
        # remove explainers from sys.modules and insert a broken stub
        explainers_key = "src.evaluation.explainers"
        original_explainers = sys.modules.get(explainers_key)

        try:
            broken = types.ModuleType(explainers_key)
            # Insert a broken stub that will fail on attribute access during from-import
            # Actually from X import Y just needs the module to exist and have the attribute
            # If the attribute is missing, ImportError is raised automatically
            sys.modules[explainers_key] = broken
            # Remove the already-loaded src.evaluation to force reload
            sys.modules.pop("src.evaluation", None)
            # Now the real import will try to import from our broken stub
            import src.evaluation as eval_module

            importlib.reload(eval_module)
            # Because our stub is missing LeaveOnePTMOutScorer, the fallback
            # should be triggered — either None (direct try/except) or a
            # LazyImport that raises AttributeError on access.
            scorer = eval_module.LeaveOnePTMOutScorer
            if hasattr(scorer, "get"):
                with pytest.raises(AttributeError):
                    scorer.get()
            else:
                assert scorer is None
        finally:
            if original_explainers is not None:
                sys.modules[explainers_key] = original_explainers
            else:
                sys.modules.pop(explainers_key, None)
            # Restore the real src.evaluation
            sys.modules.pop("src.evaluation", None)
            import src.evaluation as eval_module

            importlib.reload(eval_module)
