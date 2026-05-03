"""
变异效应预测模块单元测试
覆盖: VariantPTMEffectPredictor、predict_ptm_effects_for_variant
"""

from unittest import mock

import pandas as pd
import pytest
import torch

from src.models.variant_effect import VariantPTMEffectPredictor, predict_ptm_effects_for_variant


class FakeModel:
    """用于mock PTMSitePredictor的伪模型"""

    def __init__(self):
        self.eval_called = False
        self.to_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_called = True
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.state_dict = state_dict

    def __call__(self, indices):
        batch_size = indices.shape[0]
        return {
            "logits": torch.zeros(batch_size, 2),
            "probs": torch.tensor([[0.3, 0.7]] * batch_size),
        }


class TestVariantPTMEffectPredictorInit:
    def test_init_loads_model(self, monkeypatch):
        load_mock = mock.Mock(return_value={"state_dict": {"model.fc.weight": torch.ones(2, 2)}})
        monkeypatch.setattr(torch, "load", load_mock)

        with mock.patch("src.models.ptm_site_predictor.PTMSitePredictor", return_value=FakeModel()):
            predictor = VariantPTMEffectPredictor("fake.pt", ptm_type="Phosphorylation")
            assert isinstance(predictor.model, FakeModel)
            assert predictor.ptm_type == "Phosphorylation"
            load_mock.assert_called_once_with("fake.pt", map_location=predictor.device, weights_only=True)

    def test_init_checkpoint_without_state_dict(self, monkeypatch):
        plain_dict = {"fc.weight": torch.ones(2, 2)}
        load_mock = mock.Mock(return_value=plain_dict)
        monkeypatch.setattr(torch, "load", load_mock)

        with mock.patch("src.models.ptm_site_predictor.PTMSitePredictor", return_value=FakeModel()):
            predictor = VariantPTMEffectPredictor("fake.pt")
            assert predictor.model.state_dict is plain_dict
            load_mock.assert_called_once_with("fake.pt", map_location=predictor.device, weights_only=True)


class TestVariantPTMEffectPredictorMethods:
    @pytest.fixture
    def predictor(self, monkeypatch):
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location, **kwargs: {"state_dict": {"model.fc.weight": torch.ones(2, 2)}},
        )
        with mock.patch("src.models.ptm_site_predictor.PTMSitePredictor", return_value=FakeModel()):
            return VariantPTMEffectPredictor("fake.pt")

    def test_encode_sequence_maps_aa(self, predictor):
        encoded = predictor.encode_sequence("ACDEFG")
        expected = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        assert torch.equal(encoded, expected)

    def test_extract_window_valid_position(self, predictor):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        window, center = predictor.extract_window(seq, position=5)
        assert window is not None
        assert len(window) == 2 * predictor.window_size + 1
        assert center == predictor.window_size

    def test_extract_window_invalid_position(self, predictor):
        window, center = predictor.extract_window("ACDEFG", position=0)
        assert window is None
        assert center is None

    def test_predict_site_valid(self, monkeypatch):
        monkeypatch.setattr(
            torch,
            "load",
            lambda path, map_location, **kwargs: {"state_dict": {"model.fc.weight": torch.ones(2, 2)}},
        )
        with mock.patch("src.models.ptm_site_predictor.PTMSitePredictor", return_value=FakeModel()):
            predictor = VariantPTMEffectPredictor("fake.pt")
            prob = predictor.predict_site("ACDEFGHIKLMNPQRSTVWY", 5)
            assert prob == pytest.approx(0.7, rel=1e-6)

    def test_predict_site_invalid_window(self, predictor):
        prob = predictor.predict_site("ACDEFG", position=0)
        assert prob == 0.0

    def test_predict_variant_effect_gain(self, predictor):
        with mock.patch.object(predictor, "predict_site", side_effect=[0.2, 0.4]):
            result = predictor.predict_variant_effect("ACDEFG", 1, "A", "V")
            assert result["effect"] == "gain"
            assert result["delta_prob"] == pytest.approx(0.2, rel=1e-6)

    def test_predict_variant_effect_loss(self, predictor):
        with mock.patch.object(predictor, "predict_site", side_effect=[0.5, 0.3]):
            result = predictor.predict_variant_effect("ACDEFG", 1, "A", "V")
            assert result["effect"] == "loss"

    def test_predict_variant_effect_neutral(self, predictor):
        with mock.patch.object(predictor, "predict_site", side_effect=[0.3, 0.35]):
            result = predictor.predict_variant_effect("ACDEFG", 1, "A", "V")
            assert result["effect"] == "neutral"

    def test_predict_variants_batch(self, predictor):
        variants = [
            {"uniprot_id": "P1", "position": 1, "ref_aa": "A", "alt_aa": "V"},
            {"uniprot_id": "P1", "position": 2, "ref_aa": "C", "alt_aa": "D"},
        ]
        sequences = {"P1": "ACDEFG"}
        with mock.patch.object(predictor, "predict_variant_effect", return_value={"effect": "gain", "delta_prob": 0.2}):
            df = predictor.predict_variants_batch(variants, sequences)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "uniprot_id" in df.columns
            assert "ptm_type" in df.columns

    def test_predict_variants_batch_missing_sequence(self, predictor):
        variants = [
            {"uniprot_id": "P1", "position": 1, "ref_aa": "A", "alt_aa": "V"},
            {"uniprot_id": "P2", "position": 1, "ref_aa": "A", "alt_aa": "V"},
        ]
        sequences = {"P1": "ACDEFG"}
        with mock.patch.object(predictor, "predict_variant_effect", return_value={"effect": "gain", "delta_prob": 0.2}):
            df = predictor.predict_variants_batch(variants, sequences)
            assert len(df) == 1


class TestPredictPTMEffectsForVariant:
    def test_multi_model_prediction(self, monkeypatch):
        p1 = mock.Mock(spec=VariantPTMEffectPredictor)
        p1.predict_variant_effect.return_value = {"effect": "gain", "delta_prob": 0.2}
        p2 = mock.Mock(spec=VariantPTMEffectPredictor)
        p2.predict_variant_effect.return_value = {"effect": "loss", "delta_prob": -0.2}

        models = {"Phosphorylation": p1, "Acetylation": p2}
        results = predict_ptm_effects_for_variant("P1", 1, "A", "V", "ACDEFG", models)
        assert "Phosphorylation" in results
        assert "Acetylation" in results

    def test_multi_model_failure_handling(self, monkeypatch):
        p_good = mock.Mock(spec=VariantPTMEffectPredictor)
        p_good.predict_variant_effect.return_value = {"effect": "neutral", "delta_prob": 0.0}
        p_bad = mock.Mock(spec=VariantPTMEffectPredictor)
        p_bad.predict_variant_effect.side_effect = Exception("boom")

        models = {"Good": p_good, "Bad": p_bad}
        results = predict_ptm_effects_for_variant("P1", 1, "A", "V", "ACDEFG", models)
        assert "Good" in results
        assert "Bad" not in results
