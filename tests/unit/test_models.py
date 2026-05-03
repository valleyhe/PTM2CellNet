"""
模型模块单元测试
"""

import pytest
import torch
from src.models.encoders import (
    CNNEncoder,
    TransformerEncoder,
    LSTMEncoder,
    PositionalEncoding,
)
from src.models.ptm_modules import PTMEmbedding, PTMAttention, PTMModule
from src.models.predictors import ClassificationPredictor, RegressionPredictor
from src.models.architectures import PTM2CellNet


class TestEncoders:
    """编码器测试"""

    @pytest.fixture
    def sample_sequences(self):
        """示例序列fixture"""
        batch_size = 4
        seq_len = 50
        vocab_size = 20
        return torch.randint(0, vocab_size, (batch_size, seq_len))

    def test_cnn_encoder(self, sample_sequences):
        """测试CNN编码器"""
        encoder = CNNEncoder(vocab_size=20, embed_dim=128, max_len=1000)
        output = encoder(sample_sequences)
        assert output.shape == (4, 50, 128)

    def test_transformer_encoder(self, sample_sequences):
        """测试Transformer编码器"""
        encoder = TransformerEncoder(
            vocab_size=20, embed_dim=128, max_len=1000,
            num_layers=2, num_heads=4
        )
        output = encoder(sample_sequences)
        assert output.shape == (4, 50, 128)

    def test_lstm_encoder(self, sample_sequences):
        """测试LSTM编码器"""
        encoder = LSTMEncoder(
            vocab_size=20, embed_dim=128, max_len=1000,
            hidden_dim=256, num_layers=2
        )
        output = encoder(sample_sequences)
        assert output.shape == (4, 50, 128)

    def test_positional_encoding(self):
        """测试位置编码"""
        pos_enc = PositionalEncoding(d_model=128, max_len=5000, dropout=0.1)
        x = torch.randn(50, 4, 128)
        output = pos_enc(x)
        assert output.shape == (50, 4, 128)


class TestPTMModules:
    """PTM模块测试"""

    @pytest.fixture
    def sample_ptm_data(self):
        """示例PTM数据fixture"""
        batch_size = 4
        seq_len = 50
        sequence_emb = torch.randn(batch_size, seq_len, 128)
        ptm_types = torch.randint(0, 10, (batch_size, seq_len))
        ptm_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        ptm_mask = torch.randint(0, 2, (batch_size, seq_len)).float()
        return sequence_emb, ptm_types, ptm_positions, ptm_mask

    def test_ptm_embedding(self, sample_ptm_data):
        """测试PTM嵌入"""
        sequence_emb, ptm_types, ptm_positions, ptm_mask = sample_ptm_data
        ptm_embedding = PTMEmbedding(num_ptm_types=10, embed_dim=128, max_position=1000)
        output = ptm_embedding(ptm_types, ptm_positions)
        assert output.shape == (4, 50, 128)

    def test_ptm_attention(self, sample_ptm_data):
        """测试PTM注意力"""
        sequence_emb, ptm_types, ptm_positions, ptm_mask = sample_ptm_data
        ptm_embedding = PTMEmbedding(num_ptm_types=10, embed_dim=128, max_position=1000)
        ptm_emb = ptm_embedding(ptm_types, ptm_positions)

        attention = PTMAttention(embed_dim=128, num_heads=8)
        output = attention(sequence_emb, ptm_emb, ptm_mask)
        assert output.shape == (4, 50, 128)

    def test_ptm_module(self, sample_ptm_data):
        """测试PTM完整模块"""
        sequence_emb, ptm_types, ptm_positions, ptm_mask = sample_ptm_data
        ptm_module = PTMModule(
            num_ptm_types=10, embed_dim=128, max_position=1000,
            num_attention_heads=8, num_layers=2
        )
        output = ptm_module(sequence_emb, ptm_types, ptm_positions, ptm_mask)
        assert output.shape == (4, 50, 128)


class TestPredictors:
    """预测器测试"""

    @pytest.fixture
    def sample_features(self):
        """示例特征fixture"""
        batch_size = 4
        input_dim = 256
        return torch.randn(batch_size, input_dim)

    def test_classification_predictor(self, sample_features):
        """测试分类预测器"""
        predictor = ClassificationPredictor(
            input_dim=256, num_classes=4, hidden_dims=[512, 256]
        )
        output = predictor(sample_features)
        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (4, 4)
        assert output["probabilities"].shape == (4, 4)
        assert output["predictions"].shape == (4,)

    def test_regression_predictor(self, sample_features):
        """测试回归预测器"""
        predictor = RegressionPredictor(
            input_dim=256, output_dim=1, hidden_dims=[512, 256]
        )
        output = predictor(sample_features)
        assert "predictions" in output
        assert output["predictions"].shape == (4, 1)


class TestPTM2CellNet:
    """PTM2CellNet完整模型测试"""

    @pytest.fixture
    def sample_batch(self):
        """示例批次数据fixture"""
        batch_size = 4
        seq_len = 50
        vocab_size = 20
        return {
            "sequence": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "ptm_mask": torch.randint(0, 2, (batch_size, seq_len)).float(),
            "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
        }

    def test_ptm2cellnet_cnn(self, sample_batch):
        """测试PTM2CellNet CNN版本"""
        model = PTM2CellNet(
            encoder_type="cnn",
            vocab_size=20,
            embed_dim=128,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )
        output = model(sample_batch)
        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (4, 4)

    def test_ptm2cellnet_transformer(self, sample_batch):
        """测试PTM2CellNet Transformer版本"""
        model = PTM2CellNet(
            encoder_type="transformer",
            vocab_size=20,
            embed_dim=128,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )
        output = model(sample_batch)
        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (4, 4)

    def test_ptm2cellnet_lstm(self, sample_batch):
        """测试PTM2CellNet LSTM版本"""
        model = PTM2CellNet(
            encoder_type="lstm",
            vocab_size=20,
            embed_dim=128,
            max_seq_len=1000,
            num_ptm_types=10,
            num_classes=4,
        )
        output = model(sample_batch)
        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (4, 4)

    def test_ptm2cellnet_from_config(self, sample_batch):
        """测试从配置创建模型"""
        config = {
            "model": {
                "encoder_type": "transformer",
                "hidden_dim": 128,
            }
        }
        model = PTM2CellNet.from_config(config)
        output = model(sample_batch)
        assert "logits" in output
        assert output["logits"].shape == (4, 4)

    def test_ptm2cellnet_invalid_encoder(self):
        """测试无效编码器类型"""
        with pytest.raises(ValueError):
            PTM2CellNet(encoder_type="invalid")


class TestPretrainedEncoders:
    """预训练编码器测试"""

    @pytest.fixture
    def sample_sequences(self):
        """示例序列fixture"""
        batch_size = 2
        seq_len = 50
        # ESM-2词汇表大小为33
        return torch.randint(0, 33, (batch_size, seq_len))

    @pytest.fixture(autouse=True)
    def mock_transformers(self, monkeypatch):
        """Mock transformers from_pretrained to avoid downloading models."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import transformers

        class MockHFModel(torch.nn.Module):
            def __init__(self, hidden_size: int, num_hidden_layers: int):
                super().__init__()
                self.config = MagicMock(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
                self.layer = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(num_hidden_layers)])
                self.encoder = torch.nn.Module()
                self.encoder.layer = self.layer
                self.layers = self.layer

            def forward(self, input_ids=None, attention_mask=None, output_attentions=False):
                if input_ids is None:
                    batch_size, seq_len = 1, 1
                    device = torch.device("cpu")
                else:
                    batch_size, seq_len = input_ids.shape[:2]
                    device = input_ids.device
                hidden = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
                return SimpleNamespace(last_hidden_state=hidden)

        mock_model = MockHFModel(hidden_size=320, num_hidden_layers=6)

        mock_bert = MockHFModel(hidden_size=1024, num_hidden_layers=30)

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: mock_model)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: MagicMock(hidden_size=320, num_hidden_layers=6),
        )
        monkeypatch.setattr(transformers.BertModel, "from_pretrained", lambda *args, **kwargs: mock_bert)
        monkeypatch.setattr(
            transformers.BertConfig,
            "from_pretrained",
            lambda *args, **kwargs: MagicMock(hidden_size=1024, num_hidden_layers=30),
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_esm2_encoder_init(self):
        """测试ESM-2编码器初始化"""
        from src.models.pretrained_encoders import ESM2Encoder

        encoder = ESM2Encoder(model_size="8M", freeze=True)
        assert encoder.hidden_dim > 0
        assert encoder.model_size == "8M"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_esm2_encoder_forward(self, sample_sequences):
        """测试ESM-2编码器前向传播"""
        from src.models.pretrained_encoders import ESM2Encoder

        encoder = ESM2Encoder(model_size="8M", freeze=True)
        output = encoder(sample_sequences)

        # ESM-2 8M的隐藏维度是320
        assert output.shape[0] == 2
        assert output.shape[1] == 50
        assert output.shape[2] == 320

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_protbert_encoder_init(self):
        """测试ProtBERT编码器初始化"""
        from src.models.pretrained_encoders import ProtBERTEncoder

        encoder = ProtBERTEncoder(freeze=True)
        assert encoder.hidden_dim == 1024

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_freeze_parameters(self):
        """测试参数冻结"""
        from src.models.pretrained_encoders import ESM2Encoder

        encoder = ESM2Encoder(model_size="8M", freeze=True)

        # 检查所有参数都被冻结
        for param in encoder.parameters():
            assert not param.requires_grad

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_unfreeze_layers(self):
        """测试逐层解冻"""
        from src.models.pretrained_encoders import ESM2Encoder

        encoder = ESM2Encoder(model_size="8M", freeze=True)

        # 解冻最后2层
        encoder.unfreeze_layers(2)

        # 检查是否有可训练参数
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        assert trainable_params > 0


class TestPTM2CellNetWithPretrained:
    """PTM2CellNet预训练模型测试"""

    @pytest.fixture
    def sample_batch(self):
        """示例批次数据fixture"""
        batch_size = 2
        seq_len = 50
        # ESM-2词汇表大小为33
        return {
            "sequence": torch.randint(0, 33, (batch_size, seq_len)),
            "ptm_mask": torch.randint(0, 2, (batch_size, seq_len)).float(),
            "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
        }

    @pytest.fixture(autouse=True)
    def mock_transformers(self, monkeypatch):
        """Mock transformers from_pretrained to avoid downloading models."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import transformers

        class MockHFModel(torch.nn.Module):
            def __init__(self, hidden_size: int, num_hidden_layers: int):
                super().__init__()
                self.config = MagicMock(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
                self.layer = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(num_hidden_layers)])
                self.encoder = torch.nn.Module()
                self.encoder.layer = self.layer
                self.layers = self.layer

            def forward(self, input_ids=None, attention_mask=None, output_attentions=False):
                if input_ids is None:
                    batch_size, seq_len = 1, 1
                    device = torch.device("cpu")
                else:
                    batch_size, seq_len = input_ids.shape[:2]
                    device = input_ids.device
                hidden = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
                return SimpleNamespace(last_hidden_state=hidden)

        mock_model = MockHFModel(hidden_size=320, num_hidden_layers=6)

        mock_bert = MockHFModel(hidden_size=1024, num_hidden_layers=30)

        monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: mock_model)
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: MagicMock(hidden_size=320, num_hidden_layers=6),
        )
        monkeypatch.setattr(transformers.BertModel, "from_pretrained", lambda *args, **kwargs: mock_bert)
        monkeypatch.setattr(
            transformers.BertConfig,
            "from_pretrained",
            lambda *args, **kwargs: MagicMock(hidden_size=1024, num_hidden_layers=30),
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_ptm2cellnet_esm2_8m(self, sample_batch):
        """测试PTM2CellNet ESM-2 8M版本"""
        model = PTM2CellNet(
            encoder_type="esm2_8M",
            freeze_encoder=True,
            num_ptm_types=10,
            num_classes=4,
        )
        output = model(sample_batch)

        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (2, 4)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_ptm2cellnet_protbert(self):
        """测试PTM2CellNet ProtBERT版本"""
        # ProtBERT使用不同的词表大小(30)，需要单独创建batch
        batch_size = 2
        seq_len = 50
        sample_batch = {
            "sequence": torch.randint(0, 30, (batch_size, seq_len)),  # ProtBERT vocab_size=30
            "ptm_mask": torch.randint(0, 2, (batch_size, seq_len)).float(),
            "ptm_types": torch.randint(0, 10, (batch_size, seq_len)),
        }
        model = PTM2CellNet(
            encoder_type="protbert",
            freeze_encoder=True,
            num_ptm_types=10,
            num_classes=4,
        )
        output = model(sample_batch)

        assert "logits" in output
        assert "probabilities" in output
        assert "predictions" in output
        assert output["logits"].shape == (2, 4)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="预训练模型测试需要GPU"
    )
    def test_ptm2cellnet_pretrained_from_config(self, sample_batch):
        """测试从配置创建预训练模型"""
        config = {
            "model": {
                "encoder_type": "esm2_8M",
                "freeze_encoder": True,
                "num_classes": 4,
            }
        }
        model = PTM2CellNet.from_config(config)
        output = model(sample_batch)

        assert "logits" in output
        assert output["logits"].shape == (2, 4)
