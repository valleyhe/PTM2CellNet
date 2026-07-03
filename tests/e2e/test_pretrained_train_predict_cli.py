"""端到端测试：预训练模型训练脚本 -> 推理 CLI（mocked HF，P0-2）。

验证 ``scripts/train_pretrained.py`` 现在导出与 ``train.py`` /
``train_lightning.py`` 对齐的推理 artifact，可被 ``scripts/predict.py`` 直接
消费。

为避免 CI 下载真实 ESM-2 权重，整个训练过程在进程内执行（HF AutoModel/
AutoTokenizer/AutoConfig 用轻量 dummy 替换，与
``tests/unit/test_pretrained_encoders.py`` 同款 mock）。这样进程内的 monkeypatch
能直接作用于脚本，无需将补丁传递给子进程。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Tiny HF stand-ins (no network).
# ---------------------------------------------------------------------------

class _DummyTokenizer:
    """Minimal tokenizer stand-in returning fixed-size id/mask tensors."""

    def __call__(
        self, sequences, return_tensors="pt", padding=True, truncation=True,
        max_length=1024,
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
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyHFModel(nn.Module):
    """Tiny HF AutoModel stand-in with a configurable hidden size."""

    def __init__(self, hidden_size=16, num_layers=2):
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
            b, s = input_ids.shape
            attentions = tuple(torch.ones(b, 1, s, s) for _ in self.layers)
        return SimpleNamespace(last_hidden_state=hidden, attentions=attentions)


@pytest.fixture
def mocked_hf(monkeypatch):
    """Patch HF AutoConfig/AutoModel/AutoTokenizer to avoid network downloads.

    In-process so the patch reaches the script under test directly. Also force
    CPU so the test runs in CI without a GPU.
    """
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setattr(
        "src.models.pretrained_encoders.AutoConfig.from_pretrained",
        lambda model_name, cache_dir=None: SimpleNamespace(hidden_size=16),
    )
    monkeypatch.setattr(
        "src.models.pretrained_encoders.AutoModel.from_pretrained",
        lambda model_name, cache_dir=None: _DummyHFModel(hidden_size=16, num_layers=2),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_name, cache_dir=None: _DummyTokenizer(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _esm_config(tmp_path: Path, num_classes: int = 2) -> tuple[Path, list[str]]:
    """Write a tiny ESM2 training config and return (path, cell_states)."""
    cell_states = [f"c{i}" for i in range(num_classes)]
    cfg = {
        "model": {
            "encoder_type": "esm2_8m",
            "model_size": "8m",
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "freeze_encoder": True,
            "num_classes": num_classes,
        },
        "data": {
            "max_sequence_length": 64,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "ptm_types": ["phosphorylation", "acetylation", "methylation"],
            "cell_states": cell_states,
            "num_workers": 0,
            "pin_memory": False,
            "preprocessing": {"remove_duplicates": False},
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32",
            "drop_last": False,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 0.0,
            "checkpoint": {"enabled": True, "monitor": "step", "save_top_k": 1},
            "early_stopping": {"enabled": False},
            "logger": {"type": "tensorboard", "save_dir": str(tmp_path / "logs")},
        },
        "paths": {"outputs_models": str(tmp_path / "models")},
    }
    path = tmp_path / "esm_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path, cell_states


def _make_data(tmp_path: Path, num_classes: int) -> Path:
    csv = tmp_path / "train.csv"
    rows = ["sequence,cell_state,ptm_sites"]
    # Equal-length but distinct sequences so the ESM dataset collates them
    # without variable-length padding, and dedup keeps all rows.
    seqs = [
        "ACDEFGHIKLMNPQRSTVWY",
        "AYDEFGHIKLMNPQRSTVWC",
        "ACDefGHIKLMNPQRSTVWE",  # distinct
        "ACDEFGHIKLMNPQRSTVWD",
        "ACDEFGHIKLMNPQRSTVWG",
        "ACDEFGHIKLMNPQRSTVWH",
    ]
    for i in range(6):
        rows.append(f'{seqs[i]},c{i % num_classes},[]')
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv


class TestPretrainedArtifactContract:
    """P0-2: train_pretrained.py exports a predict-consumable artifact."""

    def test_exports_best_model_and_manifest(self, mocked_hf, tmp_path):
        num_classes = 2
        cfg_path, cell_states = _esm_config(tmp_path, num_classes)
        data_csv = _make_data(tmp_path, num_classes)
        models_dir = tmp_path / "models"

        # Run train_pretrained.py in-process by invoking main() with argv.
        import scripts.train_pretrained as tp

        sys.argv = [
            "train_pretrained.py",
            "--model", "esm2_8M",
            "--config", str(cfg_path),
            "--data", str(data_csv),
            "--max-epochs", "1",
            "--batch-size", "2",
            "--freeze",
        ]
        tp.main()

        # P0-2: all three contract files must exist.
        ckpt = models_dir / "best_model.pt"
        cfg = models_dir / "best_model.config.yaml"
        manifest = models_dir / "artifact_manifest.json"
        assert ckpt.exists(), "best_model.pt 未生成"
        assert cfg.exists(), "best_model.config.yaml 未生成"
        assert manifest.exists(), "artifact_manifest.json 未生成"

        saved = yaml.safe_load(cfg.read_text())
        # Labels synced from data, not the config's num_classes default.
        assert saved["model"]["num_classes"] == num_classes
        assert saved["data"]["cell_states"] == cell_states

        man = json.loads(manifest.read_text())
        assert man["cell_states"] == cell_states
        assert man["num_classes"] == num_classes
        assert man["training_entrypoint"] == "scripts/train_pretrained.py"
        # Encoder metadata so inference can identify the backbone/tokenizer.
        assert "encoder" in man
        assert man["encoder"]["pretrained_backbone"] == "esm2_8M"
        # Provenance + release gate (P1-1/P1-3).
        assert "model_kind" in man
        assert "data_provenance" in man
        assert "deployable" in man
        assert "dataset_hash" in man

    def test_predict_consumes_pretrained_artifact(self, mocked_hf, tmp_path):
        """The exported artifact loads in predict.py with the right labels."""
        num_classes = 2
        cfg_path, cell_states = _esm_config(tmp_path, num_classes)
        data_csv = _make_data(tmp_path, num_classes)
        models_dir = tmp_path / "models"

        import scripts.train_pretrained as tp

        sys.argv = [
            "train_pretrained.py",
            "--model", "esm2_8M",
            "--config", str(cfg_path),
            "--data", str(data_csv),
            "--max-epochs", "1",
            "--batch-size", "2",
            "--freeze",
        ]
        tp.main()

        ckpt = models_dir / "best_model.pt"
        out = tmp_path / "pred.csv"

        # Run predict.py in-process too, so the HF mock applies to its
        # ESM2Encoder build as well.
        import scripts.predict as pred

        sys.argv = [
            "predict.py",
            "--model", str(ckpt),
            "--sequence", "ACDEFGHIKLMNPQRSTVWY",
            "--output", str(out),
            "--device", "cpu",
        ]
        pred.main()

        df = pd.read_csv(out)
        for state in cell_states:
            assert f"prob_{state}" in df.columns, f"Missing prob_{state}"
