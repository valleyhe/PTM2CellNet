"""API startup auto-init label-semantics tests (P0-1).

Validates that ``src.api.app._try_auto_initialize`` resolves cell-state labels
from the training artifact (config/manifest) and refuses to silently fall back
to a hard-coded order. Covers the production Docker/K8s deployment path where
the model is loaded from environment variables at lifespan startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.autoinit import (
    _infer_logits_dim,
    _resolve_autoinit_cell_states,
)
from src.api.routes.state import STATE, reset_state
from src.models.architectures import PTM2CellNet


@pytest.fixture(autouse=True)
def _reset_state_around():
    reset_state()
    yield
    reset_state()


# ---------------------------------------------------------------------------
# Pure label-resolution logic
# ---------------------------------------------------------------------------

class TestResolveCellStatesPriority:
    def _normalize(self, labels, source):
        return list(labels), source

    def test_config_cell_states_wins(self):
        labels, source = _resolve_autoinit_cell_states(
            {"data": {"cell_states": ["a", "b"]}}, {"cell_states": ["x"]}, "c,d"
        )
        assert labels == ["a", "b"]
        assert "config" in source

    def test_manifest_cell_states_wins_when_config_missing(self):
        labels, source = _resolve_autoinit_cell_states(
            {}, {"cell_states": ["x", "y", "z"]}, "c,d"
        )
        assert labels == ["x", "y", "z"]
        assert "manifest" in source

    def test_env_used_with_warning_when_no_artifact(self, caplog):
        with caplog.at_level("WARNING"):
            labels, source = _resolve_autoinit_cell_states({}, {}, "p,d")
        assert labels == ["p", "d"]
        assert "env" in source
        assert any("PTM2CELLNET_CELL_STATES" in rec.message for rec in caplog.records)

    def test_no_source_returns_none(self):
        """P0-1 core invariant: a hard-coded default is never used implicitly."""
        labels, source = _resolve_autoinit_cell_states({}, {}, None)
        assert labels is None
        assert source == "none"


# ---------------------------------------------------------------------------
# Logits-dim inference helper
# ---------------------------------------------------------------------------

class TestInferLogitsDim:
    def test_returns_none_for_empty(self):
        assert _infer_logits_dim({}) is None

    def test_detects_head_weight(self):
        sd = {"predictor.head.weight": torch.zeros(3, 8)}
        assert _infer_logits_dim(sd) == 3

    def test_returns_none_for_huge_dim(self):
        # A 4096-wide tensor is a hidden layer, not a class head.
        sd = {"layer.weight": torch.zeros(4096, 8)}
        assert _infer_logits_dim(sd) is None

    def test_trusts_small_final_bias(self):
        sd = {"some.bias": torch.zeros(5)}
        assert _infer_logits_dim(sd) == 5


# ---------------------------------------------------------------------------
# Full auto-init via FastAPI lifespan
# ---------------------------------------------------------------------------

def _train_tiny_artifact(output_dir: Path, cell_states: list[str]) -> Path:
    """Train a minimal CNN and write best_model.pt + config + manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(cell_states)
    cfg = {
        "model": {
            "encoder_type": "cnn",
            "vocab_size": 21,
            "embed_dim": 16,
            "num_filters": 16,
            "kernel_sizes": [3],
            "max_seq_len": 64,
            "num_ptm_types": 5,
            "num_classes": num_classes,
            "pool_type": "mean",
            "dropout": 0.0,
        },
        "data": {
            "max_sequence_length": 64,
            "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            "cell_states": cell_states,
            "ptm_types": ["phosphorylation", "acetylation", "methylation"],
        },
    }
    model = PTM2CellNet.from_config(cfg)
    ckpt = output_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt)
    cfg_path = output_dir / "best_model.config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    manifest = {
        "checkpoint_path": str(ckpt),
        "config_path": str(cfg_path),
        "cell_states": cell_states,
        "num_classes": num_classes,
        "model_kind": "real",
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return ckpt


class TestAutoInitLifespan:
    def test_auto_init_uses_config_cell_states_without_env(self, tmp_path, monkeypatch):
        """3-class artifact auto-loads using config labels; PTM2CELLNET_CELL_STATES unset."""
        cell_states = ["growth", "arrest", "death"]
        ckpt = _train_tiny_artifact(tmp_path, cell_states)
        monkeypatch.setenv("PTM2CELLNET_CHECKPOINT", str(ckpt))
        monkeypatch.setenv("PTM2CELLNET_CONFIG", str(ckpt.with_name("best_model.config.yaml")))
        monkeypatch.delenv("PTM2CELLNET_CELL_STATES", raising=False)

        # Use the context-manager form so the FastAPI lifespan (which runs
        # _try_auto_initialize) actually executes.
        with TestClient(create_app()) as client:
            assert STATE.model is not None
            assert STATE.cell_states == cell_states

            # /predict must only expose the 3 artifact labels.
            r = client.post("/api/v1/predict", json={"sequence": "ACDEFGHIK", "ptm_sites": []})
            assert r.status_code == 200, r.text
            probs = r.json()["probabilities"]
            assert set(probs.keys()) == set(cell_states)

    def test_auto_init_refuses_when_no_label_source(self, tmp_path, monkeypatch):
        """No config/manifest/env labels -> model must NOT load (no hard-coded fallback)."""
        ckpt = _train_tiny_artifact(tmp_path, ["a", "b"])
        # Strip labels from both config and manifest.
        bad_cfg = {"model": {"encoder_type": "cnn", "vocab_size": 21, "embed_dim": 16,
                             "num_filters": 16, "kernel_sizes": [3], "max_seq_len": 64,
                             "num_ptm_types": 5, "num_classes": 2, "pool_type": "mean"}}
        bad_cfg_path = tmp_path / "no_labels.config.yaml"
        bad_cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
        monkeypatch.setenv("PTM2CELLNET_CHECKPOINT", str(ckpt))
        monkeypatch.setenv("PTM2CELLNET_CONFIG", str(bad_cfg_path))
        monkeypatch.delenv("PTM2CELLNET_CELL_STATES", raising=False)

        with TestClient(create_app()):
            assert STATE.model is None, "model loaded despite having no label source"

    def test_auto_init_fails_on_label_count_mismatch(self, tmp_path, monkeypatch):
        """config cell_states count != num_classes must abort auto-init."""
        cell_states = ["a", "b", "c"]
        ckpt = _train_tiny_artifact(tmp_path, cell_states)
        # Corrupt config: 3 labels but num_classes=4.
        bad_cfg = {
            "model": {"encoder_type": "cnn", "vocab_size": 21, "embed_dim": 16,
                      "num_filters": 16, "kernel_sizes": [3], "max_seq_len": 64,
                      "num_ptm_types": 5, "num_classes": 4, "pool_type": "mean"},
            "data": {"cell_states": cell_states},
        }
        bad_cfg_path = tmp_path / "mismatch.config.yaml"
        bad_cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
        monkeypatch.setenv("PTM2CELLNET_CHECKPOINT", str(ckpt))
        monkeypatch.setenv("PTM2CELLNET_CONFIG", str(bad_cfg_path))
        monkeypatch.delenv("PTM2CELLNET_CELL_STATES", raising=False)

        with TestClient(create_app()):
            assert STATE.model is None, "model loaded despite label/count mismatch"
