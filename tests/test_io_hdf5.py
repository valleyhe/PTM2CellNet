"""
HDF5 读写工具的单元测试

覆盖 save_hdf5 / load_hdf5 的 round-trip 一致性以及缺失文件的异常行为。
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils import io as io_module


class TestHDF5RoundTrip:
    """save_hdf5 -> load_hdf5 写读一致性测试。"""

    def test_round_trip_ndarray(self, tmp_path):
        file_path = tmp_path / "ndarray.h5"
        payload = {"array": np.arange(12, dtype=np.float64).reshape(3, 4)}

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert isinstance(loaded["array"], np.ndarray)
        assert loaded["array"].dtype == payload["array"].dtype
        np.testing.assert_array_equal(loaded["array"], payload["array"])

    def test_round_trip_string_array(self, tmp_path):
        file_path = tmp_path / "strings.h5"
        payload = {"genes": np.array(["TP53", "BRCA1", "EGFR"], dtype=object)}

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert loaded["genes"].tolist() == payload["genes"].tolist()

    def test_round_trip_scalar_string(self, tmp_path):
        file_path = tmp_path / "scalar_str.h5"
        payload = {"name": "PTM2CellNet"}

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert loaded["name"] == "PTM2CellNet"

    def test_round_trip_torch_tensor(self, tmp_path):
        file_path = tmp_path / "tensor.h5"
        payload = {"tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert isinstance(loaded["tensor"], torch.Tensor)
        assert torch.equal(loaded["tensor"], payload["tensor"])

    def test_round_trip_scalars(self, tmp_path):
        file_path = tmp_path / "scalars.h5"
        payload = {"int_val": 7, "float_val": 3.14, "bool_val": True}

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert loaded["int_val"] == 7
        assert loaded["float_val"] == pytest.approx(3.14)
        assert loaded["bool_val"] is True

    def test_round_trip_mixed_payload(self, tmp_path):
        file_path = tmp_path / "mixed.h5"
        payload = {
            "array": np.linspace(0, 1, 5, dtype=np.float32),
            "tensor": torch.tensor([10, 20, 30]),
            "frame": pd.DataFrame({"gene": ["A", "B"], "score": [0.1, 0.9]}),
            "label": "ptm2cellnet",
            "count": 42,
        }

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        np.testing.assert_array_equal(loaded["array"], payload["array"])
        assert torch.equal(loaded["tensor"], payload["tensor"])
        pd.testing.assert_frame_equal(loaded["frame"], payload["frame"])
        assert loaded["label"] == payload["label"]
        assert loaded["count"] == 42

    def test_round_trip_creates_parent_dirs(self, tmp_path):
        file_path = tmp_path / "nested" / "subdir" / "out.h5"
        payload = {"array": np.array([1, 2, 3])}

        io_module.save_hdf5(payload, str(file_path))
        loaded = io_module.load_hdf5(str(file_path))

        assert file_path.exists()
        np.testing.assert_array_equal(loaded["array"], payload["array"])


class TestHDF5MissingFile:
    """load_hdf5 对不存在文件的行为。"""

    def test_load_missing_file_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "does_not_exist.h5"

        with pytest.raises(FileNotFoundError):
            io_module.load_hdf5(str(missing))

    def test_load_missing_file_message_contains_path(self, tmp_path):
        missing = tmp_path / "ghost.h5"

        with pytest.raises(FileNotFoundError, match="ghost.h5"):
            io_module.load_hdf5(str(missing))
