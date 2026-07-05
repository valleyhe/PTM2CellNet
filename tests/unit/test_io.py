"""
IO工具模块单元测试
"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils import io as io_module


class TestHDF5IO:
    """HDF5读写测试"""

    def test_save_and_load_hdf5_round_trip(self, tmp_path):
        file_path = tmp_path / "artifacts.h5"
        payload = {
            "array": np.arange(6, dtype=np.float32).reshape(2, 3),
            "tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "frame": pd.DataFrame(
                {
                    "gene": ["A", "B"],
                    "score": [0.1, 0.2],
                }
            ),
            "label": "ptm2cellnet",
        }

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert isinstance(loaded["array"], np.ndarray)
        assert np.array_equal(loaded["array"], payload["array"])
        assert isinstance(loaded["tensor"], torch.Tensor)
        assert torch.equal(loaded["tensor"], payload["tensor"])
        assert isinstance(loaded["frame"], pd.DataFrame)
        pd.testing.assert_frame_equal(loaded["frame"], payload["frame"])
        assert loaded["label"] == payload["label"]

    def test_hdf5_pickle_fallback_without_h5py(self, tmp_path, monkeypatch, caplog):
        file_path = tmp_path / "fallback.h5"
        payload = {
            "array": np.array([1, 2, 3]),
            "tensor": torch.tensor([4, 5, 6]),
            "frame": pd.DataFrame({"gene": ["X"], "score": [1.0]}),
            "label": "fallback",
        }

        monkeypatch.setattr(io_module, "h5py", None)

        with caplog.at_level(logging.WARNING, logger="src.utils.io"):
            io_module.save_hdf5(payload, str(file_path))
            loaded = io_module.load_hdf5(str(file_path))

        assert "h5py is not available" in caplog.text
        assert np.array_equal(loaded["array"], payload["array"])
        assert torch.equal(loaded["tensor"], payload["tensor"])
        pd.testing.assert_frame_equal(loaded["frame"], payload["frame"])
        assert loaded["label"] == payload["label"]


class TestSafeTorchLoad:
    """safe_torch_load helper behavior."""

    def test_safe_torch_load_falls_back_with_warning(self, monkeypatch, tmp_path, caplog):
        checkpoint_path = tmp_path / "legacy.pt"
        checkpoint_path.write_bytes(b"placeholder")
        expected = {"model_state_dict": {}}
        calls = []

        def fake_torch_load(path, map_location=None, **kwargs):
            calls.append((path, map_location, dict(kwargs)))
            if kwargs.get("weights_only") is True:
                raise ValueError("weights_only unsupported")
            return expected

        monkeypatch.setattr(torch, "load", fake_torch_load)

        import logging
        with caplog.at_level(logging.WARNING, logger="src.utils.io"):
            loaded = io_module.safe_torch_load(
                str(checkpoint_path),
                map_location="cpu",
                enforce_safe_only=False,  # opt in to the fallback for this test
            )

        assert loaded == expected
        assert len(calls) == 2
        assert calls[0][2]["weights_only"] is True
        assert calls[1][2]["weights_only"] is False
        assert any("rejected types" in record.message for record in caplog.records)

    def test_safe_torch_load_enforce_safe_only_blocks_fallback(self, monkeypatch, tmp_path):
        """With enforce_safe_only=True (default), a weights_only=True failure
        is re-raised rather than falling back to unsafe deserialization."""
        checkpoint_path = tmp_path / "legacy.pt"
        checkpoint_path.write_bytes(b"placeholder")

        def failing_torch_load(path, map_location=None, **kwargs):
            if kwargs.get("weights_only") is True:
                raise ValueError("weights_only unsupported")
            return {}

        monkeypatch.setattr(torch, "load", failing_torch_load)

        with pytest.raises(ValueError, match="weights_only unsupported"):
            io_module.safe_torch_load(str(checkpoint_path), map_location="cpu")
