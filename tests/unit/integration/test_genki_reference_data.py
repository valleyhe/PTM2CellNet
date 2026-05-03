import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.integration.genki.reference_data import ReferenceDataLoader


class TestBackendDetection:
    def test_runtime_ready_with_explicit_files(self) -> None:
        loader = ReferenceDataLoader(
            ref_root="/tmp",
            gene_list_file="g.txt",
            network_file="n.npy",
            counts_file="c.npy",
        )
        info = loader.get_backend_info()
        assert info["backend"] == "array_files"
        assert info["runtime_ready"] is True

    def test_runtime_ready_false_raises(self) -> None:
        loader = ReferenceDataLoader(ref_root="/nonexistent")
        with pytest.raises(RuntimeError) as exc_info:
            loader.validate_runtime_ready()
        assert "unknown" in str(exc_info.value).lower()

    def test_genki_source_missing_dependencies(self, tmp_path: Path, monkeypatch) -> None:
        genki_dir = tmp_path / "GenKI"
        genki_dir.mkdir()
        (genki_dir / "dataLoader.py").write_text("# dummy")
        loader = ReferenceDataLoader(ref_root=str(tmp_path))
        monkeypatch.setattr(loader, "_probe_dependencies", lambda deps: ["torch_geometric"])
        info = loader.get_backend_info()
        assert info["backend"] == "genki_source"
        assert info["runtime_ready"] is False
        with pytest.raises(RuntimeError) as exc_info:
            loader.validate_runtime_ready()
        assert "torch_geometric" in str(exc_info.value)


class TestReferenceDataErrors:
    def test_load_reference_data_missing_files_raises(self) -> None:
        loader = ReferenceDataLoader(ref_root="/no/files/here")
        with pytest.raises(FileNotFoundError):
            loader.load_reference_data()

    def test_load_reference_data_caching(self) -> None:
        loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        ref1 = loader.load_reference_data()
        ref2 = loader.load_reference_data()
        assert ref1 is ref2

    def test_probe_dependencies_import_exception(self, monkeypatch) -> None:
        loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        original_import_module = importlib.import_module

        def raise_for_scipy(name, package=None):
            if name == "scipy":
                raise Exception("import failed")
            return original_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", raise_for_scipy)
        assert "scipy" in loader._probe_dependencies(["scipy"])

    def test_probe_dependencies_find_spec_none(self, monkeypatch) -> None:
        loader = ReferenceDataLoader(ref_root="tests/fixtures/genki")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        deps = loader._probe_dependencies(["nonexistent_module_xyz"])
        assert "nonexistent_module_xyz" in deps

    def test_load_reference_data_genki_source(self, tmp_path: Path, monkeypatch) -> None:
        import numpy as np
        import scipy.sparse as sp

        genki_dir = tmp_path / "GenKI"
        genki_dir.mkdir()
        (genki_dir / "dataLoader.py").write_text("# dummy")
        grn_dir = tmp_path / "GRN"
        grn_dir.mkdir()
        # Create a minimal sparse matrix saved as npz
        sparse_net = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        sp.save_npz(grn_dir / "pcNet.npz", sparse_net)

        # Mock anndata.read_h5ad to return a minimal object
        mock_adata = MagicMock()
        mock_adata.var_names.tolist.return_value = ["G1", "G2"]
        mock_adata.X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        with patch("src.integration.genki.reference_data.ad.read_h5ad", return_value=mock_adata):
            loader = ReferenceDataLoader(
                ref_root=str(tmp_path),
                adata_file=str(tmp_path / "data.h5ad"),
                grn_file_dir=str(grn_dir),
            )
            ref = loader.load_reference_data()
            assert ref["backend"] == "genki_source"
            assert ref["gene_names"] == ["G1", "G2"]
            assert "adata_file" in ref
            assert "grn_file_dir" in ref
            assert ref["counts"].shape == (2, 2)
            assert ref["network"].shape == (2, 2)


class TestInitFallbacks:
    def test_integration_init_import_fallback(self, monkeypatch) -> None:
        import src.integration as integration_module

        monkeypatch.setattr(
            sys.modules["src.integration.genki"],
            "ReferenceDataLoader",
            None,
        )
        monkeypatch.setattr(
            sys.modules["src.integration.genki"],
            "PerturbationExecutor",
            None,
        )
        monkeypatch.setattr(
            sys.modules["src.integration.genki"],
            "SignificanceAnalyzer",
            None,
        )
        monkeypatch.setattr(
            sys.modules["src.integration.genki"],
            "GraphUtilities",
            None,
        )
        reloaded = importlib.reload(integration_module)
        assert reloaded.ReferenceDataLoader is None
        assert reloaded.PerturbationExecutor is None
        assert reloaded.SignificanceAnalyzer is None
        assert reloaded.GraphUtilities is None

    def test_integration_init_adapter_fallback(self, monkeypatch) -> None:
        import src.integration as integration_module

        monkeypatch.setattr(
            sys.modules["src.integration.genki_adapter"],
            "GenKIAdapter",
            None,
        )
        reloaded = importlib.reload(integration_module)
        assert reloaded.GenKIAdapter is None
