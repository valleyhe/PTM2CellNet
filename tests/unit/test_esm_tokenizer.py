"""
ESM2Encoder tokenizer功能单元测试
验证tokenizer正确加载和tokenize方法输出格式
"""
import os
import pytest
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@pytest.fixture(scope="module")
def esm2_encoder():
    from src.models.pretrained_encoders import ESM2Encoder
    return ESM2Encoder(model_size="8M", freeze=True)


def test_tokenizer_exists(esm2_encoder):
    assert esm2_encoder.tokenizer is not None


def test_tokenizer_tokenize_returns_tokens(esm2_encoder):
    tokens = esm2_encoder.tokenizer.tokenize("ACDEF")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_tokenize_method_returns_dict(esm2_encoder):
    result = esm2_encoder.tokenize(["ACDEF"])
    assert "input_ids" in result
    assert "attention_mask" in result


def test_tokenize_adds_special_tokens(esm2_encoder):
    seq = "ACDEF"
    result = esm2_encoder.tokenize([seq])
    # ESM tokenizer adds <cls> at start and <eos> at end
    assert result["input_ids"].shape[1] == len(seq) + 2


def test_forward_accepts_tokenized_input(esm2_encoder):
    seq = "ACDEF"
    result = esm2_encoder.tokenize([seq])
    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    with torch.no_grad():
        output = esm2_encoder.forward(input_ids, attention_mask)
    assert output.shape == (1, len(seq) + 2, esm2_encoder.hidden_dim)


def test_input_ids_in_vocab_range(esm2_encoder):
    result = esm2_encoder.tokenize(["ACDEFGHIKLMNPQRSTVWY"])
    input_ids = result["input_ids"]
    assert input_ids.min().item() >= 0
    assert input_ids.max().item() < esm2_encoder.tokenizer.vocab_size

def test_forward_with_input_ids_batch(esm2_encoder):
    """测试PTM2CellNet.forward()能正确处理含input_ids的batch"""
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from src.models.architectures import PTM2CellNet
    import torch
    model = PTM2CellNet(
        encoder_type="esm2_8M",
        num_classes=2,
        freeze_encoder=True,
    )
    model.eval()
    seq = "ACDEFGHIKL"
    encoded = esm2_encoder.tokenize([seq])
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
    assert output["logits"].shape == (1, 2)
    assert "predictions" in output


def test_tokenize_empty_sequence(esm2_encoder):
    """测试空序列tokenize后仅包含特殊token。"""
    result = esm2_encoder.tokenize([""])
    assert result["input_ids"].shape == (1, 2)
    assert result["attention_mask"].shape == (1, 2)
    assert result["attention_mask"].sum().item() == 2


def test_tokenize_invalid_amino_acids(esm2_encoder):
    """测试无效氨基酸字符会被安全处理，不会导致tokenize失败。"""
    sequence = "ACD*1?"
    result = esm2_encoder.tokenize([sequence])
    input_ids = result["input_ids"]
    assert input_ids.shape[0] == 1
    assert 2 <= input_ids.shape[1] <= len(sequence) + 2
    assert input_ids.min().item() >= 0
    assert input_ids.max().item() < esm2_encoder.tokenizer.vocab_size

    unk_token_id = esm2_encoder.tokenizer.unk_token_id
    if unk_token_id is not None:
        assert (input_ids == unk_token_id).any().item()


def test_tokenize_batch(esm2_encoder):
    """测试不同长度序列的批量tokenize与padding行为。"""
    sequences = ["ACD", "MNPQRST", "W"]
    result = esm2_encoder.tokenize(sequences)

    assert result["input_ids"].shape[0] == len(sequences)
    assert result["attention_mask"].shape[0] == len(sequences)
    assert result["input_ids"].shape[1] == max(len(seq) for seq in sequences) + 2

    expected_lengths = torch.tensor([len(seq) + 2 for seq in sequences], dtype=torch.long)
    actual_lengths = result["attention_mask"].sum(dim=1).to(torch.long)
    assert torch.equal(actual_lengths, expected_lengths)


def test_tokenize_long_sequence(esm2_encoder):
    """测试超长序列会按max_length=1024进行截断。"""
    long_sequence = "A" * 1200
    result = esm2_encoder.tokenize([long_sequence])
    assert result["input_ids"].shape == (1, 1024)
    assert result["attention_mask"].shape == (1, 1024)
    assert result["attention_mask"].sum().item() == 1024


def test_tokenize_single_aa(esm2_encoder):
    """测试单氨基酸序列tokenize后长度正确。"""
    result = esm2_encoder.tokenize(["A"])
    assert result["input_ids"].shape == (1, 3)
    assert result["attention_mask"].shape == (1, 3)
    assert result["attention_mask"].sum().item() == 3


def test_tokenizer_vocab_consistency(esm2_encoder):
    """测试同一序列多次tokenize结果一致。"""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    first = esm2_encoder.tokenize([sequence])
    second = esm2_encoder.tokenize([sequence])

    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["attention_mask"], second["attention_mask"])


def test_forward_without_attention_mask(esm2_encoder):
    """测试forward不传attention_mask时可正常执行。"""
    result = esm2_encoder.tokenize(["ACDEFGHIKL"])
    with torch.no_grad():
        output = esm2_encoder.forward(result["input_ids"])
    assert output.shape[0] == 1
    assert output.shape[1] == result["input_ids"].shape[1]
    assert output.shape[2] == esm2_encoder.hidden_dim
