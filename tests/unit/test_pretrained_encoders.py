"""
预训练编码器单元测试
覆盖ESM2、ProtBERT、ProtT5初始化及冻结/解冻行为
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.models.pretrained_encoders import (
    ESM2Encoder,
    PretrainedEncoder,
    ProtBERTEncoder,
    ProtT5Encoder,
)


class _DummyTokenizer:
    """用于替代HF tokenizer的轻量实现。"""

    def __call__(
        self,
        sequences,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 1024,
    ):
        del return_tensors, padding, truncation
        if isinstance(sequences, str):
            sequences = [sequences]

        token_lengths = [min(len(seq) + 2, max_length) for seq in sequences]
        max_len = max(token_lengths) if token_lengths else 2

        input_ids = torch.zeros(len(sequences), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)

        for i, length in enumerate(token_lengths):
            input_ids[i, :length] = torch.arange(length, dtype=torch.long)
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _DummyHFModel(nn.Module):
    """用于替代HF AutoModel的轻量模型。"""

    def __init__(self, hidden_size: int = 16, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(64, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        del attention_mask
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)

        attentions = None
        if output_attentions:
            batch_size, seq_len = input_ids.shape
            attentions = tuple(
                torch.ones(batch_size, 1, seq_len, seq_len) for _ in self.layers
            )

        return SimpleNamespace(last_hidden_state=hidden, attentions=attentions)


@pytest.fixture(autouse=True)
def _set_hf_mirror(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")


@pytest.fixture
def _mock_hf_components(monkeypatch):
    calls = {
        "config": [],
        "model": [],
        "tokenizer": [],
    }

    def fake_config_from_pretrained(model_name, cache_dir=None):
        calls["config"].append((model_name, cache_dir))
        return SimpleNamespace(hidden_size=16)

    def fake_model_from_pretrained(model_name, cache_dir=None):
        calls["model"].append((model_name, cache_dir))
        return _DummyHFModel(hidden_size=16, num_layers=4)

    def fake_tokenizer_from_pretrained(model_name, cache_dir=None):
        calls["tokenizer"].append((model_name, cache_dir))
        return _DummyTokenizer()

    monkeypatch.setattr(
        "src.models.pretrained_encoders.AutoConfig.from_pretrained",
        fake_config_from_pretrained,
    )
    monkeypatch.setattr(
        "src.models.pretrained_encoders.AutoModel.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )

    return calls


def test_esm2_all_model_sizes(_mock_hf_components, monkeypatch):
    """测试ESM2不同模型尺寸（8M/35M/150M/650M）。"""
    small_model_name = "facebook/esm2_t6_8M_UR50D"
    monkeypatch.setattr(
        ESM2Encoder,
        "MODEL_NAMES",
        {
            **ESM2Encoder.MODEL_NAMES,
            "8M": small_model_name,
            "35M": small_model_name,
            "150M": small_model_name,
            "650M": small_model_name,
        },
    )

    for model_size in ("8M", "35M", "150M", "650M"):
        encoder = ESM2Encoder(model_size=model_size, freeze=True)
        assert encoder.model_size == model_size
        assert encoder.hidden_dim == 16
        assert all(not param.requires_grad for param in encoder.model.parameters())

    assert len(_mock_hf_components["model"]) == 4
    assert len(_mock_hf_components["tokenizer"]) == 4
    assert {
        model_name for model_name, _ in _mock_hf_components["model"]
    } == {small_model_name}


def test_protbert_encoder_initialization(_mock_hf_components, monkeypatch):
    """测试ProtBERTEncoder初始化。"""
    # ProtBERTEncoder 使用 BertConfig/BertModel 而非 AutoConfig/AutoModel
    from transformers import BertConfig, BertModel

    def fake_bert_config_from_pretrained(model_name, cache_dir=None, **kwargs):
        _mock_hf_components["config"].append((model_name, cache_dir))
        return SimpleNamespace(hidden_size=16)

    def fake_bert_model_from_pretrained(model_name, cache_dir=None, **kwargs):
        _mock_hf_components["model"].append((model_name, cache_dir))
        return _DummyHFModel(hidden_size=16, num_layers=4)

    monkeypatch.setattr(
        BertConfig, "from_pretrained", staticmethod(fake_bert_config_from_pretrained)
    )
    monkeypatch.setattr(
        BertModel, "from_pretrained", staticmethod(fake_bert_model_from_pretrained)
    )

    encoder = ProtBERTEncoder(freeze=False)
    assert encoder.model_name == "Rostlab/prot_bert"
    assert encoder.hidden_dim == 16
    assert any(param.requires_grad for param in encoder.model.parameters())
    assert _mock_hf_components["model"][0][0] == "Rostlab/prot_bert"


def test_prott5_encoder_initialization(_mock_hf_components):
    """测试ProtT5Encoder初始化。"""
    encoder = ProtT5Encoder(freeze=False)
    assert encoder.model_name == "Rostlab/prot_t5_xl_uniref50"
    assert encoder.hidden_dim == 16
    assert any(param.requires_grad for param in encoder.model.parameters())
    assert _mock_hf_components["model"][0][0] == "Rostlab/prot_t5_xl_uniref50"


def test_pretrain_encoder_freeze_unfreeze(_mock_hf_components):
    """测试预训练编码器参数冻结/解冻。"""
    encoder = PretrainedEncoder(model_name="dummy/model", freeze=True)
    assert all(not param.requires_grad for param in encoder.model.parameters())

    for param in encoder.model.parameters():
        param.requires_grad = True

    assert all(param.requires_grad for param in encoder.model.parameters())


def test_encoder_get_num_parameters(_mock_hf_components):
    """测试参数计数逻辑。"""
    encoder = PretrainedEncoder(model_name="dummy/model", freeze=False)
    total_params = encoder.get_num_parameters()
    trainable_params = encoder.get_num_parameters(trainable_only=True)

    assert total_params > 0
    assert trainable_params == total_params

    encoder._freeze_parameters()
    assert encoder.get_num_parameters(trainable_only=True) == 0
    assert encoder.get_num_parameters() == total_params


def test_encoder_unfreeze_layers(_mock_hf_components):
    """测试逐层解冻最后N层。"""
    encoder = PretrainedEncoder(model_name="dummy/model", freeze=True)
    encoder.unfreeze_layers(num_layers=2)

    frozen_layers = encoder.model.layers[:-2]
    unfrozen_layers = encoder.model.layers[-2:]

    assert all(
        not param.requires_grad
        for layer in frozen_layers
        for param in layer.parameters()
    )
    assert all(
        param.requires_grad
        for layer in unfrozen_layers
        for param in layer.parameters()
    )

    # 非layers参数仍保持冻结（例如embedding）
    assert all(not param.requires_grad for param in encoder.model.embed.parameters())
