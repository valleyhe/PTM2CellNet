"""
PTM2CellNet模型边界测试
测试ESM2路径和分类输出的边界情况
"""
import os
import pytest
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@pytest.fixture(scope="module")
def esm2_encoder():
    from src.models.pretrained_encoders import ESM2Encoder
    return ESM2Encoder(model_size="8M", freeze=True)


class TestPTM2CellNetESMPath:
    """测试PTM2CellNet使用ESM tokenizer的input_ids路径"""

    def test_esm_input_ids_path(self, esm2_encoder):
        """测试使用input_ids而不是sequence作为输入"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=4,
            freeze_encoder=True,
        )
        model.eval()

        # Tokenize序列
        encoded = esm2_encoder.tokenize(["ACDEFGHIKL"])
        L = encoded["input_ids"].shape[1]

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": torch.zeros(1, L),
            "ptm_types": torch.zeros(1, L, dtype=torch.long),
        }

        with torch.no_grad():
            output = model(batch)

        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (1, 4)
        assert output["probabilities"].shape == (1, 4)
        assert output["predictions"].shape == (1,)
        assert output["predictions"].dtype == torch.long

    def test_esm_with_ptm_sites(self, esm2_encoder):
        """测试ESM路径下PTM位点正确处理（位置对齐）"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=2,
            num_ptm_types=3,
            freeze_encoder=True,
        )
        model.eval()

        # 序列: A C D E F (位置1-5)
        # 在位置3(D)有磷酸化修饰
        encoded = esm2_encoder.tokenize(["ACDEF"])
        L = encoded["input_ids"].shape[1]  # 应该是 5 + 2 = 7 (<cls> + seq + <eos>)

        # PTM在序列位置3，对应input_ids位置4（+1因为<cls>）
        ptm_mask = torch.zeros(1, L)
        ptm_mask[0, 4] = 1.0  # 位置4是D

        ptm_types = torch.zeros(1, L, dtype=torch.long)
        ptm_types[0, 4] = 1  # 磷酸化类型

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": ptm_mask,
            "ptm_types": ptm_types,
        }

        with torch.no_grad():
            output = model(batch)

        assert output["logits"].shape == (1, 2)
        assert torch.isfinite(output["logits"]).all()

    def test_esm_batch_multiple_lengths(self, esm2_encoder):
        """测试ESM路径下不同长度序列的batch处理"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=3,
            freeze_encoder=True,
        )
        model.eval()

        sequences = ["ACD", "MNPQRST", "WY"]
        encoded = esm2_encoder.tokenize(sequences)

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": torch.zeros_like(encoded["attention_mask"]),
            "ptm_types": torch.zeros_like(encoded["input_ids"]),
        }

        with torch.no_grad():
            output = model(batch)

        assert output["logits"].shape == (3, 3)
        assert output["predictions"].shape == (3,)


class TestRegressionPredictor:
    """测试回归预测器的输出格式"""

    def test_regression_output_format(self):
        """测试回归预测器的输出格式"""
        from src.models.predictors import RegressionPredictor

        batch_size = 4
        input_dim = 256

        predictor = RegressionPredictor(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=[512, 256]
        )
        predictor.eval()

        features = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            output = predictor(features)

        # 回归任务应该只有predictions
        assert "predictions" in output
        assert output["predictions"].shape == (batch_size, 1)
        assert output["predictions"].dtype == torch.float32

        # 回归任务不应该有logits和probabilities
        assert "logits" not in output
        assert "probabilities" not in output

    def test_regression_multi_output(self):
        """测试多目标回归的输出格式"""
        from src.models.predictors import RegressionPredictor

        batch_size = 4
        input_dim = 256
        output_dim = 3  # 3个回归目标

        predictor = RegressionPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[512, 256]
        )
        predictor.eval()

        features = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            output = predictor(features)

        assert output["predictions"].shape == (batch_size, output_dim)


class TestPTM2CellNetOutputConsistency:
    """测试模型输出的一致性和数值有效性"""

    def test_classification_probabilities_sum_to_one(self):
        """测试分类概率之和为1"""
        from src.models.architectures import PTM2CellNet

        batch = {
            "sequence": torch.randint(0, 20, (4, 50)),
            "ptm_mask": torch.zeros(4, 50),
            "ptm_types": torch.zeros(4, 50, dtype=torch.long),
        }

        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            num_classes=5,
        )
        model.eval()

        with torch.no_grad():
            output = model(batch)

        probs = output["probabilities"]
        probs_sum = probs.sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    def test_predictions_match_argmax_logits(self):
        """测试predictions等于logits的argmax"""
        from src.models.architectures import PTM2CellNet

        batch = {
            "sequence": torch.randint(0, 20, (4, 50)),
            "ptm_mask": torch.zeros(4, 50),
            "ptm_types": torch.zeros(4, 50, dtype=torch.long),
        }

        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            num_classes=5,
        )
        model.eval()

        with torch.no_grad():
            output = model(batch)

        expected_predictions = output["logits"].argmax(dim=1)
        assert torch.equal(output["predictions"], expected_predictions)

    def test_output_values_are_finite(self, esm2_encoder):
        """测试输出值都是有限数（无nan或inf）"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=4,
            freeze_encoder=True,
        )
        model.eval()

        encoded = esm2_encoder.tokenize(["ACDEFGHIKL", "MNPQRSTVWY"])
        L = encoded["input_ids"].shape[1]

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": torch.zeros(2, L),
            "ptm_types": torch.zeros(2, L, dtype=torch.long),
        }

        with torch.no_grad():
            output = model(batch)

        for key, value in output.items():
            assert torch.isfinite(value).all(), f"{key} contains non-finite values"

    def test_model_deterministic(self, esm2_encoder):
        """测试模型在eval模式下是确定性的（相同输入->相同输出）"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=3,
            freeze_encoder=True,
        )
        model.eval()

        encoded = esm2_encoder.tokenize(["ACDEFGHIKL"])
        L = encoded["input_ids"].shape[1]

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": torch.zeros(1, L),
            "ptm_types": torch.zeros(1, L, dtype=torch.long),
        }

        with torch.no_grad():
            output1 = model(batch)
            output2 = model(batch)

        for key in output1.keys():
            assert torch.equal(output1[key], output2[key]), f"{key} is not deterministic"

    def test_empty_ptm_mask(self, esm2_encoder):
        """测试空PTM mask时的输出"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=3,
            freeze_encoder=True,
        )
        model.eval()

        encoded = esm2_encoder.tokenize(["ACDEFGHIKL"])
        L = encoded["input_ids"].shape[1]

        # 全零ptm_mask
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": torch.zeros(1, L),
            "ptm_types": torch.zeros(1, L, dtype=torch.long),
        }

        with torch.no_grad():
            output = model(batch)

        assert "logits" in output
        assert output["logits"].shape == (1, 3)
        # 没有PTM时模型仍应输出有效结果
        assert torch.isfinite(output["logits"]).all()

    def test_full_ptm_mask(self, esm2_encoder):
        """测试所有位置都是PTM时的输出"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm2_8M",
            num_classes=3,
            num_ptm_types=5,
            freeze_encoder=True,
        )
        model.eval()

        encoded = esm2_encoder.tokenize(["ACDEF"])
        L = encoded["input_ids"].shape[1]

        # 所有位置都是PTM（除了特殊token）
        ptm_mask = torch.ones(1, L)
        # <cls>和<eos>位置设为0（不是PTM）
        ptm_mask[0, 0] = 0
        ptm_mask[0, L-1] = 0

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ptm_mask": ptm_mask,
            "ptm_types": torch.randint(1, 5, (1, L)),
        }

        with torch.no_grad():
            output = model(batch)

        assert output["logits"].shape == (1, 3)
        assert torch.isfinite(output["logits"]).all()
