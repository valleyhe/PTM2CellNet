"""Lightweight unit tests for CLI scripts.

These tests avoid heavy model loading and focus on:
- scripts being syntactically valid (py_compile)
- argument parsers accepting expected flags
- small helper functions behaving correctly
"""

import importlib.util
import py_compile
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _import_script(module_name: str, script_path: Path) -> ModuleType:
    """Import a script from the scripts/ directory without requiring __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def predict_module():
    return _import_script("scripts.predict", SCRIPTS_DIR / "predict.py")


@pytest.fixture
def batch_predict_module():
    return _import_script("scripts.batch_predict", SCRIPTS_DIR / "batch_predict.py")


@pytest.fixture
def mamba_test_module():
    return _import_script("scripts.test_mamba_standalone", SCRIPTS_DIR / "test_mamba_standalone.py")


class TestPredictScript:
    """Tests for scripts/predict.py."""

    def test_py_compile(self):
        py_compile.compile(SCRIPTS_DIR / "predict.py", doraise=True)

    def test_parse_args_defaults(self, predict_module):
        args = predict_module.parse_args([])
        assert args.model == "outputs/models/best_model.pt"
        assert args.config == "configs/default.yaml"
        assert args.batch_size == 32
        assert args.pathway_analysis is False

    def test_parse_args_batch_size(self, predict_module):
        args = predict_module.parse_args(["--batch-size", "64"])
        assert args.batch_size == 64

    def test_predict_in_batches(self, predict_module):
        """_predict_in_batches should stack rows and call model.forward once per batch."""
        num_samples = 5
        batch_size = 2
        num_classes = 4
        preprocessed_rows = [
            {
                "sequence": torch.zeros(100, dtype=torch.long),
                "ptm_mask": torch.zeros(100, dtype=torch.float32),
                "ptm_types": torch.zeros(100, dtype=torch.long),
            }
            for _ in range(num_samples)
        ]

        forward_calls = []

        class FakeModel:
            def __call__(self, batch):
                forward_calls.append({k: v.shape for k, v in batch.items()})
                batch_len = batch["sequence"].shape[0]
                logits = torch.randn(batch_len, num_classes)
                return {"probabilities": torch.softmax(logits, dim=-1)}

        predictions, confidences, prob_rows = predict_module._predict_in_batches(
            FakeModel(), preprocessed_rows, device="cpu", batch_size=batch_size
        )

        # 5 samples / batch_size 2 -> 3 batches (2, 2, 1)
        assert len(forward_calls) == 3
        assert forward_calls[0]["sequence"][0] == 2
        assert forward_calls[1]["sequence"][0] == 2
        assert forward_calls[2]["sequence"][0] == 1
        assert len(predictions) == num_samples
        assert len(confidences) == num_samples
        assert all(0 <= p < num_classes for p in predictions)
        assert all(0.0 <= c <= 1.0 for c in confidences)
        # P0-1: _predict_in_batches now also returns per-class probability rows.
        # When cell_states is not provided, names default to class_<i>.
        assert len(prob_rows) == num_samples
        assert all(set(row.keys()) == {f"prob_class_{i}" for i in range(num_classes)} for row in prob_rows)

    def test_predict_in_batches_accepts_cell_states(self, predict_module):
        """Explicit cell_states name the per-class probability columns."""
        num_samples = 3
        num_classes = 4
        cell_states = ["apoptosis", "differentiation", "proliferation", "quiescence"]
        rows = [
            {
                "sequence": torch.zeros(100, dtype=torch.long),
                "ptm_mask": torch.zeros(100, dtype=torch.float32),
                "ptm_types": torch.zeros(100, dtype=torch.long),
            }
            for _ in range(num_samples)
        ]

        class FakeModel:
            def __call__(self, batch):
                n = batch["sequence"].shape[0]
                return {"probabilities": torch.softmax(torch.randn(n, num_classes), dim=-1)}

        _, _, prob_rows = predict_module._predict_in_batches(
            FakeModel(),
            rows,
            device="cpu",
            batch_size=2,
            cell_states=cell_states,
        )
        assert len(prob_rows) == num_samples
        for row in prob_rows:
            assert set(row.keys()) == {f"prob_{s}" for s in cell_states}

    def test_is_demo_model_detects_demo_via_config(self, predict_module):
        """_is_demo_model flags model.model_kind == 'demo' (P1-3)."""
        from src.utils.config import Config

        assert predict_module._is_demo_model(Config({"model": {"model_kind": "demo"}})) is True
        assert predict_module._is_demo_model(Config({"model": {"model_kind": "real"}})) is False
        # data_provenance.training_data containing 'synthetic' also flags demo
        assert predict_module._is_demo_model(Config({"data_provenance": {"training_data": "synthetic_random"}})) is True

    def test_is_demo_model_detects_demo_via_manifest(self, predict_module):
        """_is_demo_model reads sibling manifest model_card/data_provenance."""
        manifest = {"model_card": {"model_kind": "demo", "training_data": "synthetic"}}
        assert predict_module._is_demo_model(None, manifest) is True
        assert predict_module._is_demo_model(None, {"model_card": {"model_kind": "real"}}) is False


class TestBatchPredictScript:
    """Tests for scripts/batch_predict.py."""

    def test_py_compile(self):
        py_compile.compile(SCRIPTS_DIR / "batch_predict.py", doraise=True)

    def test_canonical_ptm_site_script_exists(self):
        assert (SCRIPTS_DIR / "predict_ptm_sites.py").exists()

    def test_parse_args_fasta_batch_size(self, batch_predict_module):
        args = batch_predict_module.parse_args(
            ["fasta", "--input", "in.fasta", "--output", "out.csv", "--model", "m.pt"]
        )
        assert args.command == "fasta"
        assert args.batch_size == 256

    def test_parse_args_single_batch_size(self, batch_predict_module):
        args = batch_predict_module.parse_args(
            ["single", "--sequence", "ACDEFGH", "--model", "m.pt", "--batch-size", "128"]
        )
        assert args.command == "single"
        assert args.batch_size == 128

    def test_parse_args_variant_batch_size(self, batch_predict_module):
        args = batch_predict_module.parse_args(
            [
                "variant",
                "--variants",
                "vars.csv",
                "--sequences",
                "seqs.fasta",
                "--output",
                "out.csv",
                "--model",
                "m.pt",
                "--batch-size",
                "512",
            ]
        )
        assert args.command == "variant"
        assert args.batch_size == 512

    def test_help_text_clarifies_ptm_site_prediction(self, batch_predict_module):
        parser = batch_predict_module._build_parser()
        help_text = parser.format_help()
        assert "PTM" in help_text
        assert "位点" in help_text
        assert "细胞状态" not in help_text

    def test_encode_sequence_window_padding(self, batch_predict_module, monkeypatch):
        # Avoid loading a real checkpoint by mocking the model loader.
        monkeypatch.setattr(
            batch_predict_module.BatchPTMPredictor,
            "_load_model",
            lambda self: batch_predict_module.MultiTaskPTMPredictor(),
        )
        predictor = batch_predict_module.BatchPTMPredictor(
            model_path="dummy.pt",
            model_type="cnn_multitask",
            device="cpu",
            batch_size=256,
        )
        # 短序列应被填充到 window_size=31
        tensor = predictor._encode_sequence_window("ACDEFGHIKLMNPQRSTVWY", position=10)
        assert tensor.shape == (predictor.window_size,)
        assert tensor.dtype == torch.long


class TestMambaStandaloneScript:
    """Tests for scripts/test_mamba_standalone.py."""

    def test_py_compile(self):
        py_compile.compile(SCRIPTS_DIR / "test_mamba_standalone.py", doraise=True)

    def test_parse_args_defaults(self, mamba_test_module):
        args = mamba_test_module.parse_args([])
        assert args.batch_size == 2
        assert args.seq_len == 64

    def test_parse_args_custom(self, mamba_test_module):
        args = mamba_test_module.parse_args(["--batch-size", "4", "--seq-len", "32"])
        assert args.batch_size == 4
        assert args.seq_len == 32

    def test_end_to_end_forward(self, mamba_test_module):
        """Run the script's main logic directly (requires torch + src.models.mamba_encoder)."""
        # Patch sys.argv so argparse inside main() sees no extra arguments from pytest.
        original_argv = sys.argv
        sys.argv = [str(SCRIPTS_DIR / "test_mamba_standalone.py")]
        try:
            mamba_test_module.main()
        finally:
            sys.argv = original_argv
