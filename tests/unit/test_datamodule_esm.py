"""
PTMDataModule tokenizer参数集成测试
验证传入tokenizer时切换为ESMTokenizedDataset，不传时保持PTMDataset（向后兼容）
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
def sample_dfs():
    rows = [
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
        {
            "sequence": "ACDEFGHIKL",
            "ptm_sites": json.dumps([]),
            "cell_state": "Activated",
        },
    ]
    df = pd.DataFrame(rows)
    train_df = df.iloc[:2].reset_index(drop=True)
    val_df = df.iloc[1:2].reset_index(drop=True)
    test_df = df.iloc[2:].reset_index(drop=True)
    return train_df, val_df, test_df


def test_without_tokenizer_uses_ptm_dataset(sample_dfs):
    from src.data.lightning_datamodule import PTMDataModule
    from src.data.datasets import PTMDataset
    train_df, val_df, test_df = sample_dfs
    dm = PTMDataModule(train_df=train_df, val_df=val_df, test_df=test_df, config={})
    dm.setup("fit")
    assert isinstance(dm.train_dataset, PTMDataset)


def test_with_tokenizer_uses_esm_dataset(sample_dfs, tokenizer):
    from src.data.lightning_datamodule import PTMDataModule
    from src.data.datasets import ESMTokenizedDataset
    train_df, val_df, test_df = sample_dfs
    dm = PTMDataModule(train_df=train_df, val_df=val_df, test_df=test_df, config={}, tokenizer=tokenizer)
    dm.setup("fit")
    assert isinstance(dm.train_dataset, ESMTokenizedDataset)
    assert isinstance(dm.val_dataset, ESMTokenizedDataset)


def test_train_dataloader_batch_has_input_ids(sample_dfs, tokenizer):
    from src.data.lightning_datamodule import PTMDataModule
    train_df, val_df, test_df = sample_dfs
    dm = PTMDataModule(
        train_df=train_df, val_df=val_df, test_df=test_df,
        config={"training": {"batch_size": 2}, "data": {"num_workers": 0, "pin_memory": False}},
        tokenizer=tokenizer,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert batch["input_ids"].dtype == torch.long


def test_batch_shapes_consistent(sample_dfs, tokenizer):
    from src.data.lightning_datamodule import PTMDataModule
    train_df, val_df, test_df = sample_dfs
    dm = PTMDataModule(
        train_df=train_df, val_df=val_df, test_df=test_df,
        config={"training": {"batch_size": 2}, "data": {"num_workers": 0, "pin_memory": False}},
        tokenizer=tokenizer,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    B, L = batch["input_ids"].shape
    assert batch["attention_mask"].shape == (B, L)
    assert batch["ptm_mask"].shape == (B, L)
    assert batch["ptm_types"].shape == (B, L)
