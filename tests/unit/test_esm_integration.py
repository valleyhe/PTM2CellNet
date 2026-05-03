"""
ESM端到端集成测试
验证 tokenize -> dataset -> datamodule -> model.forward 的完整数据流
"""
import os
import json

import pytest
import torch
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@pytest.fixture(scope="module")
def esm2_encoder():
    """提供ESM2Encoder实例供集成测试复用。"""
    from src.models.pretrained_encoders import ESM2Encoder

    return ESM2Encoder(model_size="8M", freeze=True)


@pytest.fixture(scope="module")
def integration_dfs():
    """构建train/val/test三份DataFrame。"""
    rows = [
        {
            "sequence": "ACDEFGHIKL",
            "ptm_sites": json.dumps([{"position": 3, "type": "phosphorylation"}]),
            "cell_state": "Activated",
        },
        {
            "sequence": "MNPQRSTVWY",
            "ptm_sites": json.dumps([{"position": 5, "type": "acetylation"}]),
            "cell_state": "Quiescent",
        },
        {
            "sequence": "ACDMNPQRST",
            "ptm_sites": json.dumps([]),
            "cell_state": "Activated",
        },
    ]
    df = pd.DataFrame(rows)
    train_df = df.iloc[:2].reset_index(drop=True)
    val_df = df.iloc[1:2].reset_index(drop=True)
    test_df = df.iloc[2:].reset_index(drop=True)
    return train_df, val_df, test_df


def test_esm_full_pipeline_flow(esm2_encoder, integration_dfs):
    """测试ESM完整数据流可连通并产出logits。"""
    from src.data.datasets import ESMTokenizedDataset
    from src.data.lightning_datamodule import PTMDataModule
    from src.models.architectures import PTM2CellNet

    train_df, val_df, test_df = integration_dfs

    tokenized = esm2_encoder.tokenize([train_df.iloc[0]["sequence"]])
    assert tokenized["input_ids"].shape[0] == 1
    assert tokenized["attention_mask"].shape == tokenized["input_ids"].shape

    dataset = ESMTokenizedDataset(df=train_df, tokenizer=esm2_encoder.tokenizer)
    sample = dataset[0]
    assert {"input_ids", "attention_mask", "ptm_mask", "ptm_types", "label"}.issubset(sample.keys())

    dm = PTMDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config={"training": {"batch_size": 2}, "data": {"num_workers": 0, "pin_memory": False}},
        tokenizer=esm2_encoder.tokenizer,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    model = PTM2CellNet(encoder_type="esm2_8M", num_classes=2, freeze_encoder=True)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    assert "logits" in output
    assert "predictions" in output
    assert output["logits"].shape == (2, 2)
    assert output["predictions"].shape == (2,)


def test_esm_pipeline_with_multiple_ptm_type_combinations(esm2_encoder):
    """测试多种PTM类型组合在数据集与模型前向中的一致性。"""
    from src.data.datasets import ESMTokenizedDataset
    from src.models.architectures import PTM2CellNet

    config = {"data": {"ptm_types": ["phosphorylation", "acetylation", "methylation"]}}
    df = pd.DataFrame([
        {
            "sequence": "ACDEFGHIKLMN",
            "ptm_sites": json.dumps([
                {"position": 2, "type": "phosphorylation"},
                {"position": 5, "type": "acetylation"},
                {"position": 9, "type": "methylation"},
            ]),
            "cell_state": "Activated",
        }
    ])

    dataset = ESMTokenizedDataset(
        df=df,
        tokenizer=esm2_encoder.tokenizer,
        config=config,
    )
    sample = dataset[0]

    assert sample["ptm_mask"][2].item() == 1.0
    assert sample["ptm_mask"][5].item() == 1.0
    assert sample["ptm_mask"][9].item() == 1.0
    assert sample["ptm_types"][2].item() == 1
    assert sample["ptm_types"][5].item() == 2
    assert sample["ptm_types"][9].item() == 3

    model = PTM2CellNet(
        encoder_type="esm2_8M",
        num_classes=2,
        num_ptm_types=3,
        freeze_encoder=True,
    )
    model.eval()

    batch = {
        "input_ids": sample["input_ids"].unsqueeze(0),
        "attention_mask": sample["attention_mask"].unsqueeze(0),
        "ptm_mask": sample["ptm_mask"].unsqueeze(0),
        "ptm_types": sample["ptm_types"].unsqueeze(0),
    }
    with torch.no_grad():
        output = model(batch)

    assert output["logits"].shape == (1, 2)
    assert output["predictions"].shape == (1,)
