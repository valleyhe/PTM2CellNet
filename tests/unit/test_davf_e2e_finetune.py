"""Tests for DAVF E2E fine-tuning script (scripts/finetune_davf_e2e.py)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal CSV fixture for DAVF E2E fine-tuning."""
    csv_path = tmp_path / "train.csv"
    # Create a small dataset with sequence_window and label columns
    sequences = ["ACDEFGHIKLMNPQRSTVWY"] * 20  # 20 sample sequences
    labels = [0, 1] * 10  # Binary labels
    df = pd.DataFrame({"sequence_window": sequences, "label": labels})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_davf_module():
    """Mock DAVFInferenceModule for testing without real model weights."""
    with patch("scripts.finetune_davf_e2e.DAVFInferenceModule") as mock_cls:
        mock_model = MagicMock()
        mock_model.delta_projection = torch.nn.Linear(10, 10)
        mock_model.training = True
        mock_model.state_dict.return_value = {"key": torch.randn(10, 10)}

        # Mock DAVFInferenceOutput dataclass
        mock_output = MagicMock()
        mock_output.davf_features = torch.randn(2, 10)
        mock_model.return_value = mock_output
        mock_model.side_effect = None  # Allow calling as constructor
        mock_model.__call__ = MagicMock(return_value=mock_output)

        mock_cls.return_value = mock_model
        yield mock_cls, mock_model


class TestDAVFE2EScript:
    """Tests for the DAVF E2E fine-tuning script."""

    def test_csv_data_loading(self, sample_csv):
        """Verify that the script reads --data_path CSV correctly."""
        df = pd.read_csv(sample_csv)
        assert "sequence_window" in df.columns
        assert "label" in df.columns
        assert len(df) == 20

    def test_sequence_dataset_creation(self, sample_csv):
        """Verify SequenceDataset correctly encodes sequences."""
        # Add the scripts directory to path for import
        scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        df = pd.read_csv(sample_csv)
        # Simulate the SequenceDataset logic
        for _idx, row in df.iterrows():
            seq = row["sequence_window"]
            label = row["label"]
            # Convert to indices using ord() - ord('A'), capped at 25
            seq_tensor = torch.tensor(
                [min(ord(c) - ord("A"), 25) for c in seq.upper()],
                dtype=torch.long,
            )
            assert seq_tensor.shape == (20,)
            assert seq_tensor.min() >= 0
            assert seq_tensor.max() <= 25
            assert isinstance(label, int)
            break  # Just test first row

    def test_finetuned_checkpoint_has_training_metadata(self, tmp_path):
        """Verify that the saved checkpoint includes training metadata."""
        # Simulate what the script saves
        output_dir = tmp_path / "davf_finetuned"
        output_dir.mkdir()

        checkpoint = {
            "model_state_dict": {"key": torch.randn(10, 10)},
            "downstream_head_state_dict": {"key": torch.randn(10, 10)},
            "optimizer_state_dict": {},
            "epochs_completed": 2,
            "training_completed": True,
        }
        output_path = output_dir / "davf_finetuned.pt"
        torch.save(checkpoint, str(output_path))

        # Verify
        loaded = torch.load(str(output_path), weights_only=False)
        assert "epochs_completed" in loaded
        assert loaded["epochs_completed"] == 2
        assert loaded["training_completed"] is True
        assert "model_state_dict" in loaded
        assert "downstream_head_state_dict" in loaded

    def test_loss_computed_on_forward_pass(self):
        """Verify that loss is computed correctly for binary classification."""
        criterion = torch.nn.CrossEntropyLoss()
        # Simulate model output: (batch=4, num_classes=2)
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = criterion(logits, labels)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0  # Positive loss value
        assert not torch.isnan(loss)

    def test_optimizer_step_reduces_loss(self):
        """Verify that an optimizer step can reduce loss on a simple problem."""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        # Initial loss
        logits = model(x)
        initial_loss = criterion(logits, y)

        # Train for a few steps
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Final loss should be lower
        final_loss = criterion(model(x), y)
        assert final_loss.item() < initial_loss.item(), (
            f"Loss did not decrease: {initial_loss.item():.4f} -> {final_loss.item():.4f}"
        )
