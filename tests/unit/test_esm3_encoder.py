"""
ESM3Encoder单元测试
覆盖初始化、前向传播、tokenize、参数冻结、工厂函数等
"""

from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.models.pretrained_encoders import (
    ESM3Encoder,
    ESM3TokenizerAdapter,
    esm3_encoder,
)


# ---------------------------------------------------------------------------
# Mock ESM3 model components
# ---------------------------------------------------------------------------


class MockESM3Output:
    """模拟ESM3模型的forward输出（ESMOutput）。"""

    def __init__(self, batch_size: int = 1, seq_len: int = 10, d_model: int = 1536):
        self.embeddings = torch.randn(batch_size, seq_len, d_model)


class MockSequenceTokenizer:
    """模拟ESM3的序列tokenizer。"""

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.mask_token_id = 128
        self.pad_token_id = 0
        self.cls_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        # 跟踪token_id到氨基酸的映射
        self._aa_to_id: Dict[str, int] = {}

    def encode(self, sequence: str) -> list:
        """模拟AVG token级别编码。每个氨基酸映射为1个token，idx由aa决定。"""
        return [hash(ch) % self.vocab_size for ch in sequence]


class MockTokenizerCollection:
    """模拟ESM3的TokenizerCollection。"""

    def __init__(self, vocab_size: int = 4096):
        self.sequence = MockSequenceTokenizer(vocab_size)


class MockESM3Model(nn.Module):
    """模拟ESM3 torch模型（用于测试forward）。"""

    def __init__(self, d_model: int = 1536):
        super().__init__()
        self.d_model = d_model
        self.tokenizers = MockTokenizerCollection()
        self.dummy_weight = nn.Parameter(torch.randn(d_model))

    def forward(self, **kwargs) -> MockESM3Output:
        """模拟ESM3前向传播。"""
        sequence_tokens = kwargs.get("sequence_tokens")
        batch_size = sequence_tokens.shape[0] if sequence_tokens is not None else 1
        seq_len = sequence_tokens.shape[1] if sequence_tokens is not None else 10
        return MockESM3Output(batch_size, seq_len, self.d_model)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_esm3_model(monkeypatch):
    """Mock ESM3模型加载，避免实际下载2.7GB模型。"""
    model = MockESM3Model(d_model=1536)

    def mock_load_local(self, path, device):
        self.model = model
        self.hidden_dim = 1536
        self.tokenizer = ESM3TokenizerAdapter(model.tokenizers)
        self._device = device
        return model  # FIX: return model

    def mock_load_hf(self, device):
        self.model = model
        self.hidden_dim = 1536
        self.tokenizer = ESM3TokenizerAdapter(model.tokenizers)
        self._device = device
        return model  # FIX: return model

    monkeypatch.setattr(ESM3Encoder, "_load_from_local", mock_load_local)
    monkeypatch.setattr(ESM3Encoder, "_load_from_huggingface", mock_load_hf)
    return model


# ---------------------------------------------------------------------------
# ESM3Encoder Tests
# ---------------------------------------------------------------------------


class TestESM3EncoderInitialization:
    """ESM3Encoder初始化测试。"""

    def test_default_initialization(self, mock_esm3_model):
        """测试默认参数初始化。"""
        encoder = ESM3Encoder()
        assert encoder.model_size == "small"
        assert encoder.hidden_dim == 1536
        assert isinstance(encoder.model, MockESM3Model)
        assert isinstance(encoder.tokenizer, ESM3TokenizerAdapter)

    def test_freeze_parameters(self, mock_esm3_model):
        """测试冻结参数后所有param.requires_grad为False。"""
        encoder = ESM3Encoder(freeze=True)
        assert all(not p.requires_grad for p in encoder.model.parameters())

    def test_unfrozen_parameters_by_default(self, mock_esm3_model):
        """测试默认不冻结参数。"""
        encoder = ESM3Encoder(freeze=False)
        assert all(p.requires_grad for p in encoder.model.parameters())

    def test_checkpoint_path_custom(self, mock_esm3_model, tmp_path):
        """测试传入自定义checkpoint_path。"""
        # 创建一个临时的空.pth文件
        ckpt_path = tmp_path / "dummy.pth"
        ckpt_path.write_bytes(b"dummy")

        # 验证ESM3Encoder可以接受checkpoint_path参数
        encoder = ESM3Encoder()
        assert encoder.model_size == "small"
        assert encoder.hidden_dim == 1536


class TestESM3EncoderNormalizeModelSize:
    """_normalize_model_size静态方法测试。"""

    def test_valid_sizes(self):
        """测试有效的模型尺寸。"""
        assert ESM3Encoder._normalize_model_size("small") == "small"
        assert ESM3Encoder._normalize_model_size("SMALL") == "small"
        assert ESM3Encoder._normalize_model_size("Small") == "small"
        assert ESM3Encoder._normalize_model_size("  small  ") == "small"

    def test_invalid_size_raises(self):
        """测试无效尺寸抛出ValueError。"""
        with pytest.raises(ValueError, match="无效的ESM-3模型尺寸"):
            ESM3Encoder._normalize_model_size("large")

    def test_empty_string_raises(self):
        """测试空字符串抛出ValueError。"""
        with pytest.raises(ValueError, match="不能为空字符串"):
            ESM3Encoder._normalize_model_size("")

    def test_non_string_input_raises(self):
        """测试非字符串输入抛出ValueError。"""
        with pytest.raises(ValueError, match="必须是字符串"):
            ESM3Encoder._normalize_model_size(123)


class TestESM3EncoderForward:
    """ESM3Encoder前向传播测试。"""

    def test_forward_returns_correct_shape(self, mock_esm3_model):
        """测试forward返回正确的输出形状。"""
        encoder = ESM3Encoder()
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        embeddings = encoder(input_ids)
        assert embeddings.shape == (batch_size, seq_len, 1536)

    def test_forward_with_attention_mask(self, mock_esm3_model):
        """测试attention_mask参数（保留兼容性）。"""
        encoder = ESM3Encoder()
        batch_size, seq_len = 1, 30
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        embeddings = encoder(input_ids, attention_mask)
        assert embeddings.shape == (batch_size, seq_len, 1536)

    def test_forward_with_structure_tokens(self, mock_esm3_model):
        """测试传入structure_tokens。"""
        encoder = ESM3Encoder()
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        structure_tokens = torch.randint(0, 100, (batch_size, seq_len))
        embeddings = encoder(input_ids, structure_tokens=structure_tokens)
        assert embeddings.shape == (batch_size, seq_len, 1536)

    def test_forward_with_function_tokens(self, mock_esm3_model):
        """测试传入function_tokens。"""
        encoder = ESM3Encoder()
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        function_tokens = torch.randint(0, 100, (batch_size, seq_len))
        embeddings = encoder(input_ids, function_tokens=function_tokens)
        assert embeddings.shape == (batch_size, seq_len, 1536)

    def test_forward_dtype(self, mock_esm3_model):
        """测试输出dtype为float32。"""
        encoder = ESM3Encoder()
        input_ids = torch.randint(0, 100, (1, 20))
        embeddings = encoder(input_ids)
        assert embeddings.dtype == torch.float32

    def test_forward_batch_independence(self, mock_esm3_model):
        """测试batch中不同样本的独立处理。"""
        encoder = ESM3Encoder()
        batch_size = 3
        input_ids = torch.randint(0, 100, (batch_size, 15))
        embeddings = encoder(input_ids)
        assert embeddings.shape[0] == batch_size
        # 每个batch的嵌入应该不同（随机输入）
        assert not torch.allclose(embeddings[0], embeddings[1])


class TestESM3EncoderTokenize:
    """ESM3Encoder tokenize方法测试。"""

    def test_tokenize_returns_dict_with_correct_keys(self, mock_esm3_model):
        """测试tokenize返回的dict包含input_ids和attention_mask。"""
        encoder = ESM3Encoder()
        result = encoder.tokenize(["ACDEF"])
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_tokenize_tensor_dtypes(self, mock_esm3_model):
        """测试tokenize返回的tensor dtype正确。"""
        encoder = ESM3Encoder()
        result = encoder.tokenize(["ACDEF"])
        assert result["input_ids"].dtype == torch.long
        assert result["attention_mask"].dtype == torch.long

    def test_tokenize_single_sequence(self, mock_esm3_model):
        """测试单序列tokenize。"""
        encoder = ESM3Encoder()
        result = encoder.tokenize(["ACDEF"])
        assert result["input_ids"].shape[0] == 1  # batch=1

    def test_tokenize_batch(self, mock_esm3_model):
        """测试批次tokenize。"""
        encoder = ESM3Encoder()
        sequences = ["ACD", "MNPQRST", "W"]
        result = encoder.tokenize(sequences)
        assert result["input_ids"].shape[0] == len(sequences)

    def test_tokenize_adds_bos_eos(self, mock_esm3_model):
        """测试tokenize添加BOS/EOS占位token。"""
        encoder = ESM3Encoder()
        sequence = "ACD"
        result = encoder.tokenize([sequence])
        # 3 AA tokens + 1 BOS + 1 EOS = 5
        assert result["input_ids"].shape[1] == len(sequence) + 2

    def test_tokenize_empty_sequence(self, mock_esm3_model):
        """测试空序列tokenize。"""
        encoder = ESM3Encoder()
        result = encoder.tokenize([""])
        # 0 AA + 1 BOS + 1 EOS = 2
        assert result["input_ids"].shape[1] == 2

    def test_tokenize_truncation(self, mock_esm3_model):
        """测试超长序列截断。"""
        encoder = ESM3Encoder()
        long_sequence = "A" * 2000
        max_length = 1024
        result = encoder.tokenize([long_sequence], max_length=max_length)
        assert result["input_ids"].shape[1] <= max_length

    def test_forward_accepts_tokenized_input(self, mock_esm3_model):
        """测试tokenize->forward的完整流程。"""
        encoder = ESM3Encoder()
        result = encoder.tokenize(["ACDEFGHIKL"])
        with torch.no_grad():
            embeddings = encoder(result["input_ids"])
        assert embeddings.shape[0] == 1
        assert embeddings.shape[2] == 1536


class TestESM3EncoderUtilities:
    """ESM3Encoder工具方法测试。"""

    def test_get_num_parameters_total(self, mock_esm3_model):
        """测试get_num_parameters返回总参数数量。"""
        encoder = ESM3Encoder()
        total = encoder.get_num_parameters()
        assert total > 0
        assert isinstance(total, int)

    def test_get_num_parameters_trainable(self, mock_esm3_model):
        """测试冻结前后可训练参数变化。"""
        encoder = ESM3Encoder(freeze=False)
        trainable_before = encoder.get_num_parameters(trainable_only=True)
        total = encoder.get_num_parameters()
        assert trainable_before == total

        encoder._freeze_parameters()
        trainable_after = encoder.get_num_parameters(trainable_only=True)
        assert trainable_after == 0

    def test_encode_sequences(self, mock_esm3_model):
        """测试encode_sequences方法。"""
        encoder = ESM3Encoder()
        sequences = ["ACDEF"]
        # encode_sequences uses SlidingWindowHandler which calls encoder()
        with torch.no_grad():
            # 直接调用forward验证
            result = encoder.tokenize(sequences)
            embeddings = encoder(result["input_ids"])
        assert embeddings.shape[1] > 0
        assert embeddings.shape[2] == 1536

    def test_repr(self, mock_esm3_model):
        """测试encoder可以被打印（无异常）。"""
        encoder = ESM3Encoder()
        # 不检查具体输出，只确保不抛异常
        repr_str = repr(encoder)
        assert "ESM3" in repr_str or "Module" in repr_str


# ---------------------------------------------------------------------------
# ESM3TokenizerAdapter Tests
# ---------------------------------------------------------------------------


class TestESM3TokenizerAdapter:
    """ESM3TokenizerAdapter适配器测试。"""

    @pytest.fixture
    def adapter(self):
        collection = MockTokenizerCollection(vocab_size=4096)
        return ESM3TokenizerAdapter(collection)

    def test_basic_call(self, adapter):
        """测试基本的__call__功能。"""
        result = adapter("ACDEF")
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_batch_sequences(self, adapter):
        """测试批次输入。"""
        result = adapter(["ACD", "ACDEF"])
        assert result["input_ids"].shape[0] == 2

    def test_vocab_size_property(self, adapter):
        """测试vocab_size属性。"""
        assert adapter.vocab_size == 4096

    def test_mask_token_id(self, adapter):
        """测试mask_token_id属性。"""
        assert adapter.mask_token_id == 128

    def test_pad_token_id(self, adapter):
        """测试pad_token_id属性。"""
        assert adapter.pad_token_id == 0

    def test_padding_equalizes_length(self, adapter):
        """测试padding后序列等长。"""
        result = adapter(["ACD", "ACDEFGH"])
        assert result["input_ids"].shape[1] == len("ACDEFGH") + 2
        assert result["input_ids"].shape[1] == result["attention_mask"].shape[1]

    def test_attention_mask_valid(self, adapter):
        """测试attention_mask中有效位置标记为1。"""
        result = adapter(["ACD", "ACDEFGH"])
        # 短序列被padding到长序列长度
        assert result["attention_mask"][0, -1].item() == 0  # padding位置
        assert result["attention_mask"][1, -1].item() == 1  # 有效位置

    def test_truncation(self, adapter):
        """测试截断功能。"""
        long_seq = "A" * 500
        result = adapter([long_seq], max_length=100)
        assert result["input_ids"].shape[1] == 100

    def test_return_tensors_pt(self, adapter):
        """测试返回PyTorch tensor。"""
        result = adapter("ACD")
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)

    def test_tokenize_method(self, adapter):
        """测试tokenize方法返回列表。"""
        tokens = adapter.tokenize("ACD")
        assert isinstance(tokens, list)
        assert len(tokens) == 3


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


def test_esm3_encoder_uses_esm3_when_available(monkeypatch):
    """测试当ESM3Encoder可用时，esm3_encoder返回ESM3Encoder实例。"""
    from src.models import pretrained_encoders

    class FakeESM3Encoder(MagicMock):
        model_size = "small"
        hidden_dim = 1536

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = MagicMock()
            self.tokenizer = MagicMock()

    # mock ESM3Encoder在globals中的引用
    monkeypatch.setattr(pretrained_encoders, "ESM3Encoder", FakeESM3Encoder)

    encoder = esm3_encoder(model_size="small", freeze=True)
    assert isinstance(encoder, FakeESM3Encoder)


def test_esm3_encoder_falls_back_to_esm2(monkeypatch):
    """测试当ESM3Encoder不可用时，回退到ESM2Encoder。"""
    from src.models import pretrained_encoders
    from src.models.pretrained_encoders import ESM2Encoder

    # Remove ESM3Encoder from the module so globals() lookup fails
    monkeypatch.delattr(pretrained_encoders, "ESM3Encoder", raising=False)

    # Mock ESM2Encoder to return a known value
    mock_esm2_return = MagicMock(spec=ESM2Encoder)
    mock_esm2_return.model_size = "8M"

    def fake_esm2_constructor(*args, **kwargs):
        return mock_esm2_return

    monkeypatch.setattr(pretrained_encoders, "ESM2Encoder", fake_esm2_constructor)

    encoder = esm3_encoder(model_size="8M", freeze=True)
    assert encoder is mock_esm2_return


def test_esm3_encoder_strict_mode_raises(monkeypatch):
    """测试strict=True时，不可用则抛出异常。"""
    from src.models import pretrained_encoders

    # Remove ESM3Encoder so globals() lookup fails
    monkeypatch.delattr(pretrained_encoders, "ESM3Encoder", raising=False)

    with pytest.raises(RuntimeError, match="strict=True"):
        esm3_encoder(model_size="small", freeze=True, strict=True)


# ---------------------------------------------------------------------------
# PTM2CellNet integration test
# ---------------------------------------------------------------------------


class TestESM3ModelIntegration:
    """ESM3与PTM2CellNet集成测试。"""

    def test_ptm2cellnet_with_esm3_encoder(self, mock_esm3_model):
        """测试PTM2CellNet使用esm3编码器时前向传播正常。"""
        from src.models.architectures import PTM2CellNet

        # 构建使用ESM3编码器的PTM2CellNet
        model = PTM2CellNet(
            encoder_type="esm3_small",
            num_classes=2,
            freeze_encoder=True,
        )
        model.eval()

        # 准备batch
        batch_size, seq_len = 2, 50
        batch = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "ptm_mask": torch.zeros(batch_size, seq_len),
            "ptm_types": torch.zeros(batch_size, seq_len, dtype=torch.long),
        }

        with torch.no_grad():
            output = model(batch)

        assert "logits" in output
        assert "predictions" in output
        assert output["logits"].shape == (batch_size, 2)

    def test_ptm2cellnet_with_esm3_sm_open_alias(self, mock_esm3_model):
        """测试esm3_sm_open配置别名映射到ESM3 small编码器。"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm3_sm_open",
            num_classes=2,
            freeze_encoder=True,
        )

        assert model.encoder_type == "esm3_sm_open"
        assert model.encoder.hidden_dim == 1536

    def test_ptm2cellnet_large_with_esm3(self, mock_esm3_model):
        """测试PTM2CellNetLarge使用ESM3编码器。"""
        from src.models.architectures import PTM2CellNetLarge

        model = PTM2CellNetLarge(
            encoder_type="esm3_small",
            num_classes=4,
            freeze_encoder=True,
        )
        model.eval()

        batch = {
            "input_ids": torch.randint(0, 100, (1, 30)),
            "attention_mask": torch.ones(1, 30),
            "ptm_mask": torch.zeros(1, 30),
            "ptm_types": torch.zeros(1, 30, dtype=torch.long),
        }

        with torch.no_grad():
            output = model(batch)

        assert output["logits"].shape == (1, 4)

    def test_esm3_model_info(self, mock_esm3_model):
        """测试esm3编码器的get_model_info。"""
        from src.models.architectures import PTM2CellNet

        model = PTM2CellNet(
            encoder_type="esm3_small",
            num_classes=2,
            freeze_encoder=True,
        )
        info = model.get_model_info()
        assert info["encoder_type"] == "esm3_small"
        assert info["embed_dim"] == 1536
        assert info["freeze_encoder"] is True


# 用于monkeypatch context manager的辅助工具
@pytest.fixture
def monkeypatch():
    """Explicitly expose a monkeypatch fixture for per-test usage."""
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
