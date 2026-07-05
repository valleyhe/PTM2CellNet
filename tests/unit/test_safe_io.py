"""Tests for safe I/O utilities (SafeUnpickler, safe_pickle_load, safe_torch_load)."""
import io
import os
import pickle
import tempfile
import pytest
import torch
import numpy as np

from src.utils.safe_io import SafeUnpickler, safe_pickle_load, safe_torch_load


class TestSafeUnpickler:
    """Tests for SafeUnpickler restricted unpickler."""

    def test_rejects_unsafe_module(self):
        """SafeUnpickler's find_class should reject 'os' module."""
        unpickler = SafeUnpickler(io.BytesIO(b""))
        with pytest.raises(pickle.UnpicklingError, match="Unsafe module"):
            unpickler.find_class("os", "system")

    def test_rejects_subprocess_module(self):
        """'subprocess' module should also be rejected."""
        unpickler = SafeUnpickler(io.BytesIO(b""))
        with pytest.raises(pickle.UnpicklingError, match="Unsafe module"):
            unpickler.find_class("subprocess", "Popen")

    def test_accepts_numpy(self):
        """SafeUnpickler should accept numpy.ndarray."""
        unpickler = SafeUnpickler(io.BytesIO(b""))
        result = unpickler.find_class("numpy", "ndarray")
        assert result is not None

    def test_accepts_collections(self):
        """SafeUnpickler should accept collections.OrderedDict."""
        unpickler = SafeUnpickler(io.BytesIO(b""))
        result = unpickler.find_class("collections", "OrderedDict")
        assert result is not None

    def test_rejects_unsafe_class_in_safe_module(self):
        """Even for a safe module, an unsafe class name should be rejected."""
        unpickler = SafeUnpickler(io.BytesIO(b""))
        with pytest.raises(pickle.UnpicklingError, match="Unsafe class"):
            unpickler.find_class("builtins", "exec")


class TestSafePickleLoad:
    """Tests for safe_pickle_load helper."""

    def test_loads_safe_data(self):
        """safe_pickle_load should load dicts with only safe types."""
        data = {"key": [1, 2, 3], "value": 4.5, "label": "test"}
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_pickle_load(buf)
        assert result["key"] == [1, 2, 3]
        assert result["value"] == 4.5
        assert result["label"] == "test"

    def test_rejects_malicious_data(self):
        """safe_pickle_load should reject pickles with unsafe content (callable)."""
        buf = io.BytesIO()
        # A callable like eval cannot be pickled through the safe allowlist
        # because 'builtins.eval' is not in _SAFE_CLASSES.
        pickle.dump(eval, buf)
        buf.seek(0)
        with pytest.raises((ValueError, pickle.UnpicklingError)):
            safe_pickle_load(buf)

    def test_rejects_os_module_reference(self):
        """Pickle built from an unsafe module reference should be rejected."""
        # We can't directly pickle os.getcwd (it resolves differently), but
        # we can verify via find_class that 'os' module is rejected.
        unpickler = SafeUnpickler(io.BytesIO(b""))
        with pytest.raises(pickle.UnpicklingError, match="Unsafe module"):
            unpickler.find_class("os", "system")

    def test_empty_bytesio_raises(self):
        """Loading from empty BytesIO should raise."""
        buf = io.BytesIO(b"")
        with pytest.raises(Exception):
            safe_pickle_load(buf)


class TestSafeTorchLoad:
    """Tests for safe_torch_load wrapper."""

    def test_loads_tensor_checkpoint(self):
        """safe_torch_load should load a tensor saved via torch.save."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save(tensor, path)

        try:
            result = safe_torch_load(path)
            assert torch.equal(result, tensor)
        finally:
            os.unlink(path)

    def test_raises_on_nonexistent_file(self):
        """safe_torch_load should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            safe_torch_load("/nonexistent/path/model.pt")

    def test_enforce_safe_only_defaults_true(self):
        """safe_torch_load should default enforce_safe_only=True."""
        import inspect
        sig = inspect.signature(safe_torch_load)
        assert sig.parameters["enforce_safe_only"].default is True

    def test_forbids_explicit_weights_only_false_when_enforced(self):
        """Passing weights_only=False with enforce_safe_only=True should raise."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save(torch.tensor(1.0), path)
        try:
            with pytest.raises(ValueError, match="weights_only=False is forbidden"):
                safe_torch_load(path, weights_only=False, enforce_safe_only=True)
        finally:
            os.unlink(path)
