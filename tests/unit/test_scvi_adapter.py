"""Unit tests for src/models/scvi_adapter.py (V22-02)."""

import logging
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.models.scvi_adapter import (
    ScVIAdapter,
    ScVIAdapterConfig,
    SCVI_AVAILABLE,
    _check_scvi_available,
)


class TestScVIAdapterConfig:
    def test_defaults(self):
        cfg = ScVIAdapterConfig()
        assert cfg.n_latent == 10
        assert cfg.model_path is None
        assert cfg.gene_layer is None
        assert cfg.batch_key is None


class TestScVIAdapterAvailability:
    def test_scvi_available_flag_is_bool(self):
        assert isinstance(SCVI_AVAILABLE, bool)

    def test_module_imports_without_scvi(self):
        """The module must import cleanly regardless of scvi availability."""
        # Importing here proves the lazy-import guard works.
        assert ScVIAdapter is not None

    def test_check_scvi_available_logs_install_command(self, monkeypatch, caplog):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "scvi":
                raise ModuleNotFoundError("No module named 'pkg_resources'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        with caplog.at_level(logging.WARNING, logger="src.models.scvi_adapter"):
            assert _check_scvi_available() is False

        assert "pkg_resources" in caplog.text
        assert "scvi-tools>=1.2.0" in caplog.text


class TestScVIAdapterWrappedMock:
    """Use a mock scVI model so tests run even when scvi-tools is absent."""

    def _make_mock_model(self, n_genes=50):
        model = MagicMock()
        model.summary = {"n_vars": n_genes}
        return model

    def test_repr(self):
        adapter = ScVIAdapter(self._make_mock_model())
        assert "ScVIAdapter" in repr(adapter)
        assert "has_model=True" in repr(adapter)

    def test_n_genes_from_summary(self):
        adapter = ScVIAdapter(self._make_mock_model(n_genes=42))
        assert adapter.n_genes == 42

    def test_n_genes_logs_warning_when_summary_lookup_fails(self, caplog):
        model = MagicMock()
        model.summary = MagicMock()
        model.summary.get.side_effect = RuntimeError("broken summary")
        adapter = ScVIAdapter(model)

        with caplog.at_level("WARNING"):
            assert adapter.n_genes == 0

        assert "Could not read n_vars from model summary" in caplog.text

    def test_encode_calls_get_latent_representation(self):
        model = self._make_mock_model()
        expected = np.random.randn(4, 10).astype(np.float32)
        model.get_latent_representation.return_value = expected
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig())

        out = adapter.encode(adata=MagicMock())
        assert out.shape == (4, 10)
        model.get_latent_representation.assert_called_once()

    def test_encode_without_model_raises(self):
        adapter = ScVIAdapter(None)
        with pytest.raises(RuntimeError, match="no model loaded"):
            adapter.encode(adata=MagicMock())

    def test_decode_without_model_raises(self):
        adapter = ScVIAdapter(None)
        with pytest.raises(RuntimeError, match="no model loaded"):
            adapter.decode(np.zeros((2, 10)))

    def test_decode_rejects_1d_input(self):
        model = self._make_mock_model()
        adapter = ScVIAdapter(model)
        with pytest.raises(ValueError, match="must be 2D"):
            adapter.decode(np.zeros(10))


class TestScVIAdapterConstructorsUnavailable:
    """When scvi-tools is unavailable, constructors raise ImportError."""

    @pytest.mark.skipif(SCVI_AVAILABLE, reason="scvi-tools is installed")
    def test_from_trained_model_raises_without_scvi(self, tmp_path):
        with pytest.raises(ImportError, match=r"scvi-tools>=1\.2\.0"):
            ScVIAdapter.from_trained_model(str(tmp_path))

    @pytest.mark.skipif(SCVI_AVAILABLE, reason="scvi-tools is installed")
    def test_from_anndata_raises_without_scvi(self):
        with pytest.raises(ImportError, match=r"scvi-tools>=1\.2\.0"):
            ScVIAdapter.from_anndata(adata=MagicMock())


class TestScVIAdapterPredictRequiresModel:
    def test_predict_without_model_raises(self):
        adapter = ScVIAdapter(None)
        with pytest.raises(RuntimeError, match="no model loaded"):
            adapter.predict(
                adata=MagicMock(),
                latent_davf=MagicMock(),
                gene_ids=None,
                directions=None,
            )
