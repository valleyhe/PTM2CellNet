"""
ESMTokenizedDataset单元测试
验证tokenizer编码、PTM位置对齐、标签编码
"""

import os
import json
import pytest
import torch
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@pytest.fixture(scope="module")
def tokenizer():
    from src.models.pretrained_encoders import ESM2Encoder

    enc = ESM2Encoder(model_size="8M", freeze=True)
    return enc.tokenizer


@pytest.fixture(scope="module")
def sample_df():
    return pd.DataFrame(
        [
            {
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps([{"position": 3, "type": "Phosphorylation", "amino_acid": "D"}]),
                "cell_state": "Activated",
            },
            {
                "sequence": "MNPQRSTVWY",
                "ptm_sites": json.dumps([]),
                "cell_state": "Quiescent",
            },
        ]
    )


@pytest.fixture(scope="module")
def dataset(sample_df, tokenizer):
    from src.data.datasets import ESMTokenizedDataset

    return ESMTokenizedDataset(df=sample_df, tokenizer=tokenizer)


def test_dataset_len(dataset, sample_df):
    assert len(dataset) == len(sample_df)


def test_sample_keys(dataset):
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "ptm_mask" in sample
    assert "ptm_types" in sample
    assert "label" in sample


def test_shapes_consistent(dataset):
    sample = dataset[0]
    L = sample["input_ids"].shape[0]
    assert sample["attention_mask"].shape[0] == L
    assert sample["ptm_mask"].shape[0] == L
    assert sample["ptm_types"].shape[0] == L


def test_ptm_position_alignment(dataset):
    # sequence "ACDEFGHIKL", PTM at position=3 (1-based) → amino acid 'D' (0-based index 2)
    # tokenized_pos = 2 + 1 = 3 (skip <cls>)
    sample = dataset[0]
    assert sample["ptm_mask"][3].item() == 1.0


def test_no_ptm_mask_is_zero(dataset):
    sample = dataset[1]
    assert sample["ptm_mask"].sum().item() == 0.0


def test_label_dtype(dataset):
    sample = dataset[0]
    assert sample["label"].dtype == torch.long


def test_input_ids_dtype(dataset):
    sample = dataset[0]
    assert sample["input_ids"].dtype == torch.long


def test_special_tokens_not_ptm(dataset):
    # <cls> at index 0 and <eos> at last index should never have ptm_mask=1
    sample = dataset[0]
    assert sample["ptm_mask"][0].item() == 0.0
    assert sample["ptm_mask"][-1].item() == 0.0


def test_multiple_ptm_sites(tokenizer):
    """测试多个PTM位点会被正确编码并对齐到token位置。"""
    from src.data.datasets import ESMTokenizedDataset

    df = pd.DataFrame(
        [
            {
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps(
                    [
                        {"position": 2, "type": "phosphorylation", "amino_acid": "C"},
                        {"position": 7, "type": "acetylation", "amino_acid": "H"},
                    ]
                ),
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer)
    sample = dataset[0]

    assert sample["ptm_mask"][2].item() == 1.0
    assert sample["ptm_mask"][7].item() == 1.0
    assert sample["ptm_mask"].sum().item() == 2.0
    assert sample["ptm_types"][2].item() == 1
    assert sample["ptm_types"][7].item() == 2


def test_ptm_at_sequence_end(tokenizer):
    """测试序列末尾氨基酸上的PTM可正确映射，且不落在<eos>。"""
    from src.data.datasets import ESMTokenizedDataset

    sequence = "ACDEFGHIKL"
    df = pd.DataFrame(
        [
            {
                "sequence": sequence,
                "ptm_sites": json.dumps([{"position": len(sequence), "type": "phosphorylation"}]),
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer)
    sample = dataset[0]

    assert sample["ptm_mask"][len(sequence)].item() == 1.0
    assert sample["ptm_mask"][-1].item() == 0.0


def test_invalid_ptm_json(tokenizer):
    """测试无效PTM JSON格式时降级为空列表处理。"""
    from src.data.datasets import ESMTokenizedDataset

    df = pd.DataFrame(
        [
            {
                "sequence": "ACDEFGHIKL",
                "ptm_sites": "{invalid_json]",
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer)
    sample = dataset[0]

    assert sample["ptm_mask"].sum().item() == 0.0
    assert sample["ptm_types"].sum().item() == 0


def test_missing_ptm_sites_column(tokenizer):
    """测试缺少ptm_sites列时返回全零PTM掩码。"""
    from src.data.datasets import ESMTokenizedDataset

    df = pd.DataFrame(
        [
            {
                "sequence": "ACDEFGHIKL",
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer)
    sample = dataset[0]

    assert "ptm_mask" in sample
    assert "ptm_types" in sample
    assert sample["ptm_mask"].sum().item() == 0.0
    assert sample["ptm_types"].sum().item() == 0


def test_long_sequence_truncation(tokenizer):
    """测试长序列截断后，超出范围的PTM位点不会被标注。"""
    from src.data.datasets import ESMTokenizedDataset

    sequence = "A" * 1200
    df = pd.DataFrame(
        [
            {
                "sequence": sequence,
                "ptm_sites": json.dumps(
                    [
                        {"position": 1, "type": "phosphorylation"},
                        {"position": 1022, "type": "acetylation"},
                        {"position": 1023, "type": "methylation"},
                        {"position": 1200, "type": "ubiquitination"},
                    ]
                ),
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer, max_length=1024)
    sample = dataset[0]

    assert sample["input_ids"].shape[0] == 1024
    assert sample["ptm_mask"][1].item() == 1.0
    assert sample["ptm_mask"][1022].item() == 1.0
    assert sample["ptm_mask"][1023].item() == 0.0
    assert sample["ptm_mask"].sum().item() == 2.0


def test_ptm_position_out_of_range(tokenizer):
    """测试越界或非法位置的PTM位点会被忽略。"""
    from src.data.datasets import ESMTokenizedDataset

    df = pd.DataFrame(
        [
            {
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps(
                    [
                        {"position": 0, "type": "phosphorylation"},
                        {"position": -3, "type": "acetylation"},
                        {"position": 999, "type": "methylation"},
                    ]
                ),
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer)
    sample = dataset[0]

    assert sample["ptm_mask"].sum().item() == 0.0
    assert sample["ptm_types"].sum().item() == 0


def test_dataset_with_config(tokenizer):
    """测试传入config中的ptm_types映射会生效。"""
    from src.data.datasets import ESMTokenizedDataset

    config = {"data": {"ptm_types": ["alpha", "beta", "gamma"]}}
    df = pd.DataFrame(
        [
            {
                "sequence": "ACDEFGHIKL",
                "ptm_sites": json.dumps([{"position": 4, "type": "beta"}]),
                "cell_state": "Activated",
            }
        ]
    )
    dataset = ESMTokenizedDataset(df=df, tokenizer=tokenizer, config=config)
    sample = dataset[0]

    assert sample["ptm_mask"][4].item() == 1.0
    assert sample["ptm_types"][4].item() == 2
