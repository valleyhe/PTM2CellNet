"""第一阶段稳定性修复回归测试。"""

import torch

from src.data.datasets import PTMDataset
from src.models.architectures import PTM2CellNet
from src.training.optimizers import configure_optimizer


def test_sequence_encoding_reserves_zero_for_padding():
    dataset = PTMDataset(
        df=__import__("pandas").DataFrame(
            [
                {"sequence": "AC", "ptm_sites": "[]", "cell_state": "x"},
                {"sequence": "LM", "ptm_sites": "[]", "cell_state": "y"},
            ]
        ),
        config={"data": {"max_sequence_length": 8, "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"}},
    )

    encoded = dataset._encode_sequence("AC")
    # 0仅用于padding，真实氨基酸编码应从1开始
    assert int(encoded[0].item()) == 1
    assert int(encoded[1].item()) == 2
    assert int(encoded[2].item()) == 0


def test_pretrained_encoder_type_case_insensitive_and_dimension_aligned():
    config = {
        "model": {
            "encoder_type": "ESM2_8m",
            "hidden_dim": 64,
            "num_layers": 1,
            "num_heads": 2,
            "num_classes": 3,
            "freeze_encoder": True,
        },
        "data": {
            "max_sequence_length": 16,
            "ptm_types": ["phosphorylation", "acetylation"],
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
        },
    }

    model = PTM2CellNet.from_config(config)

    # 8M应解析为facebook/esm2_t6_8M_UR50D，而不是回退默认150M
    assert getattr(model.encoder, "model_size", "") in {"8m", "8M"}

    # PTM模块与预测头输入维度应与最终encoder输出维度一致
    assert model.ptm_module.embedding.type_embedding.embedding_dim == model.embed_dim
    assert model.predictor.classifier.in_features == model.embed_dim


def test_configure_optimizer_returns_scheduler_without_import_error():
    model = torch.nn.Linear(8, 2)
    cfg = {
        "training": {
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "scheduler": "cosine",
            "max_epochs": 5,
        }
    }
    optimizer, scheduler = configure_optimizer(model, cfg)
    assert optimizer is not None
    assert scheduler is not None
