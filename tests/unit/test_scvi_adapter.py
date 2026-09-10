"""Unit tests for src/models/scvi_adapter.py (V22-02)."""

import logging
import numpy as np
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.models.scvi_adapter import (
    ScVIAdapter,
    ScVIAdapterConfig,
    SCVI_AVAILABLE,
    _check_scvi_available,
)


class TestScVIAdapterConfig:
    def test_defaults(self):
        cfg = ScVIAdapterConfig()
        assert cfg.n_latent is None
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
        model.get_var_names.return_value = [f"gene_{index}" for index in range(n_genes)]
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
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig(n_latent=10))

        adata = MagicMock()
        adata.n_vars = 50
        adata.var_names = [f"gene_{index}" for index in range(50)]
        out = adapter.encode(adata=adata)
        assert out.shape == (4, 10)
        model.get_latent_representation.assert_called_once()

    def test_encode_rejects_gene_order_mismatch(self):
        model = self._make_mock_model(n_genes=3)
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig(n_latent=10))
        adata = MagicMock()
        adata.n_vars = 3
        adata.var_names = ["gene_1", "gene_0", "gene_2"]

        with pytest.raises(ValueError, match="gene order"):
            adapter.encode(adata)

    def test_encode_rejects_gene_count_mismatch(self):
        model = self._make_mock_model(n_genes=3)
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig(n_latent=10))
        adata = MagicMock()
        adata.n_vars = 2
        adata.var_names = ["gene_0", "gene_1"]

        with pytest.raises(ValueError, match="gene count"):
            adapter.encode(adata)
        model.get_latent_representation.assert_not_called()

    def test_validate_compatibility_rejects_latent_mismatch(self):
        model = self._make_mock_model(n_genes=3)
        model.module.n_latent = 64
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig(n_latent=64))

        with pytest.raises(ValueError, match="latent dimension mismatch"):
            adapter.validate_compatibility(expected_latent_dim=10, expected_num_genes=3)

    def test_validate_target_gene_indices_uses_checkpoint_order(self):
        model = self._make_mock_model(n_genes=3)
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig(n_latent=10))

        adapter.validate_target_gene_indices([1], ["gene_1"])
        with pytest.raises(ValueError, match="does not match scVI gene order"):
            adapter.validate_target_gene_indices([1], ["gene_0"])

    def test_resolve_target_gene_indices_uses_live_gene_names(self):
        model = self._make_mock_model(n_genes=3)
        adapter = ScVIAdapter(model, config=ScVIAdapterConfig(n_latent=10))

        assert adapter.resolve_target_gene_indices(["gene_2", "gene_0"]) == [2, 0]
        with pytest.raises(ValueError, match="absent from the scVI decoder vocabulary"):
            adapter.resolve_target_gene_indices(["not_in_scvi"])

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


class _StrictDecoder(torch.nn.Module):
    def __init__(self, n_genes=3, n_batch=1):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.n_latent = 2
        self.n_batch = n_batch
        self.n_genes = n_genes

    def generative(self, z, library, batch_index, cont_covs=None, cat_covs=None):
        assert z.shape[1] == self.n_latent
        assert library.shape == (z.shape[0], 1)
        assert batch_index.shape == (z.shape[0], 1)
        return {"px": SimpleNamespace(mu=torch.ones((z.shape[0], self.n_genes), device=z.device))}


class _StrictDecoderModel:
    def __init__(self, n_genes=3, n_batch=1):
        self.module = _StrictDecoder(n_genes=n_genes, n_batch=n_batch)
        self.summary_stats = SimpleNamespace(n_vars=n_genes)
        self._gene_names = [f"gene_{index}" for index in range(n_genes)]

    def get_var_names(self):
        return self._gene_names


class TestScVIAdapterDecoderContract:
    def test_decode_uses_scvi_14_generative_signature(self):
        adapter = ScVIAdapter(
            _StrictDecoderModel(),
            config=ScVIAdapterConfig(n_latent=2),
        )

        output = adapter.decode(np.zeros((2, 2), dtype=np.float32))

        assert output.shape == (2, 3)
        assert np.isfinite(output).all()

    def test_decode_requires_context_for_covariate_conditioned_model(self):
        adapter = ScVIAdapter(
            _StrictDecoderModel(n_batch=2),
            config=ScVIAdapterConfig(n_latent=2),
        )

        with pytest.raises(ValueError, match="requires AnnData context"):
            adapter.decode(np.zeros((2, 2), dtype=np.float32))

    def test_decode_rejects_latent_width_mismatch(self):
        adapter = ScVIAdapter(
            _StrictDecoderModel(),
            config=ScVIAdapterConfig(n_latent=2),
        )

        with pytest.raises(ValueError, match="latent dimension"):
            adapter.decode(np.zeros((2, 3), dtype=np.float32))

    def test_decode_rejects_decoder_gene_width_mismatch(self):
        adapter = ScVIAdapter(
            _StrictDecoderModel(n_genes=2),
            config=ScVIAdapterConfig(n_latent=2),
        )
        adapter.model.summary_stats = SimpleNamespace(n_vars=3)

        with pytest.raises(ValueError, match="gene dimension"):
            adapter.decode(np.zeros((2, 2), dtype=np.float32))


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


class TestScVIAdapterCheckpointContract:
    def test_prepare_load_context_preserves_all_zero_sparse_values(self):
        anndata = pytest.importorskip("anndata")
        from scipy import sparse

        adata = anndata.AnnData(
            X=sparse.csr_matrix((2, 3), dtype=np.float32),
            obs={"batch": ["a", "a"]},
        )

        prepared = ScVIAdapter._prepare_adata_for_scvi_load(adata)

        assert prepared is not adata
        assert adata.X.nnz == 0
        assert prepared.X.nnz == 2
        assert prepared.X.sum() == 0
        np.testing.assert_array_equal(prepared.X.toarray(), adata.X.toarray())

    @staticmethod
    def _make_loaded_model(n_genes=3, n_latent=64):
        model = MagicMock()
        model.module = SimpleNamespace(n_latent=n_latent)
        model.summary_stats = SimpleNamespace(n_vars=n_genes)
        model.get_var_names.return_value = [f"gene_{index}" for index in range(n_genes)]
        return model

    @pytest.mark.skipif(
        not Path("checkpoints/scvi/ibd_norman_model/model.pt").is_file(),
        reason="canonical local scVI checkpoint is not present",
    )
    def test_find_default_model_includes_canonical_local_checkpoint(self):
        assert ScVIAdapter.find_default_model() == str(Path("checkpoints/scvi/ibd_norman_model").resolve())

    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
    def test_from_trained_model_uses_current_load_contract(self, tmp_path):
        model_path = tmp_path / "scvi_model"
        model_path.mkdir()
        loaded_model = self._make_loaded_model()

        with patch("scvi.model.SCVI.load", return_value=loaded_model) as load:
            adapter = ScVIAdapter.from_trained_model(
                model_path,
                config=ScVIAdapterConfig(n_latent=64, device="cpu"),
            )

        load.assert_called_once_with(
            str(model_path),
            adata=False,
            accelerator="cpu",
            device="auto",
        )
        assert adapter.n_latent == 64
        assert adapter.n_genes == 3

    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
    def test_from_trained_model_derives_schema_when_config_is_omitted(self, tmp_path):
        model_path = tmp_path / "scvi_model"
        model_path.mkdir()
        loaded_model = self._make_loaded_model(n_latent=64)

        with patch("scvi.model.SCVI.load", return_value=loaded_model):
            adapter = ScVIAdapter.from_trained_model(model_path)

        assert adapter.config.n_latent == 64
        assert adapter.gene_names == ("gene_0", "gene_1", "gene_2")

    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
    def test_from_trained_model_rejects_explicit_latent_mismatch(self, tmp_path):
        model_path = tmp_path / "scvi_model"
        model_path.mkdir()
        loaded_model = self._make_loaded_model(n_latent=64)

        with patch("scvi.model.SCVI.load", return_value=loaded_model):
            with pytest.raises(ValueError, match="latent dimension mismatch"):
                ScVIAdapter.from_trained_model(
                    model_path,
                    config=ScVIAdapterConfig(n_latent=10),
                )

    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="scvi-tools is not installed")
    def test_from_trained_model_rejects_input_gene_order_mismatch(self, tmp_path):
        model_path = tmp_path / "scvi_model"
        model_path.mkdir()
        loaded_model = self._make_loaded_model(n_genes=3, n_latent=64)
        adata = SimpleNamespace(n_vars=3, var_names=["gene_1", "gene_0", "gene_2"])

        with patch("scvi.model.SCVI.load", return_value=loaded_model):
            with pytest.raises(ValueError, match="gene order"):
                ScVIAdapter.from_trained_model(model_path, adata=adata)


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
