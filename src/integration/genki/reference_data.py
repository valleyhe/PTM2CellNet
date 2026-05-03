"""Reference data loading and backend discovery for GenKI integration."""

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata as ad
import numpy as np
import scipy.sparse as sp


class ReferenceDataLoader:
    """Load cached reference assets and detect available GenKI backend."""

    def __init__(
        self,
        ref_root: str,
        gene_list_file: Optional[str] = None,
        network_file: Optional[str] = None,
        counts_file: Optional[str] = None,
        adata_file: Optional[str] = None,
        grn_file_dir: Optional[str] = None,
        pcnet_name: str = "pcNet",
        cutoff: int = 85,
        target_cell: Optional[str] = None,
        obs_label: str = "ident",
        scoring_method: str = "shift",
        trainer_epochs: int = 10,
        trainer_lr: float = 7e-4,
        trainer_beta: float = 1e-4,
        trainer_seed: Optional[int] = None,
        trainer_out_channels: int = 2,
        null_permutations: int = 32,
        null_seed: Optional[int] = None,
        significance_alpha: float = 0.05,
        bagging_threshold: float = 0.05,
        bagging_cutoff: float = 0.95,
    ) -> None:
        self.ref_root = ref_root
        self.gene_list_file = gene_list_file
        self.network_file = network_file
        self.counts_file = counts_file
        self.adata_file = adata_file
        self.grn_file_dir = grn_file_dir
        self.pcnet_name = pcnet_name
        self.cutoff = cutoff
        self.target_cell = target_cell
        self.obs_label = obs_label
        self.scoring_method = scoring_method
        self.trainer_epochs = trainer_epochs
        self.trainer_lr = trainer_lr
        self.trainer_beta = trainer_beta
        self.trainer_seed = trainer_seed
        self.trainer_out_channels = trainer_out_channels
        self.null_permutations = max(0, int(null_permutations))
        self.null_seed = null_seed
        self.significance_alpha = float(significance_alpha)
        self.bagging_threshold = float(bagging_threshold)
        self.bagging_cutoff = float(bagging_cutoff)
        self._reference_cache: Dict[str, Any] | None = None

    def get_backend_info(self) -> Dict[str, Any]:
        explicit_files = all([self.gene_list_file, self.network_file, self.counts_file])
        if explicit_files:
            return {
                "backend": "array_files",
                "runtime_ready": True,
                "ref_root": self.ref_root,
                "missing_dependencies": [],
                "uses_explicit_files": True,
                "scoring_method": self.scoring_method,
                "null_permutations": self.null_permutations,
                "bagging_threshold": self.bagging_threshold,
                "bagging_cutoff": self.bagging_cutoff,
            }

        root = Path(self.ref_root)
        fixture_files_exist = all(
            [
                (root / "mock_gene_list.txt").exists(),
                (root / "mock_network.npy").exists(),
                (root / "mock_counts.npy").exists(),
            ]
        )
        if fixture_files_exist:
            return {
                "backend": "array_files",
                "runtime_ready": True,
                "ref_root": self.ref_root,
                "missing_dependencies": [],
                "uses_explicit_files": False,
                "scoring_method": self.scoring_method,
                "null_permutations": self.null_permutations,
                "bagging_threshold": self.bagging_threshold,
                "bagging_cutoff": self.bagging_cutoff,
            }

        genki_loader = root / "GenKI" / "dataLoader.py"
        if genki_loader.exists():
            missing_dependencies = self._probe_dependencies(["torch_geometric", "anndata", "scipy", "scanpy"])
            return {
                "backend": "genki_source",
                "runtime_ready": len(missing_dependencies) == 0,
                "ref_root": self.ref_root,
                "missing_dependencies": missing_dependencies,
                "uses_explicit_files": False,
                "has_adata_file": self.adata_file is not None,
                "has_grn_dir": self.grn_file_dir is not None,
                "scoring_method": self.scoring_method,
                "null_permutations": self.null_permutations,
                "bagging_threshold": self.bagging_threshold,
                "bagging_cutoff": self.bagging_cutoff,
            }

        return {
            "backend": "unknown",
            "runtime_ready": False,
            "ref_root": self.ref_root,
            "missing_dependencies": [],
            "uses_explicit_files": explicit_files,
            "scoring_method": self.scoring_method,
            "null_permutations": self.null_permutations,
            "bagging_threshold": self.bagging_threshold,
            "bagging_cutoff": self.bagging_cutoff,
        }

    def validate_runtime_ready(self) -> None:
        backend = self.get_backend_info()
        if backend["runtime_ready"]:
            return
        missing = ", ".join(backend["missing_dependencies"]) or "unknown"
        raise RuntimeError(
            f"Backend {backend['backend']} is not ready; missing or broken dependencies: {missing}"
        )

    def load_reference_data(self) -> Dict[str, Any]:
        if self._reference_cache is not None:
            return self._reference_cache

        backend_info = self.get_backend_info()
        if backend_info["backend"] == "genki_source":
            return self._load_reference_data_from_genki_source()

        root = Path(self.ref_root)
        gene_path = Path(self.gene_list_file) if self.gene_list_file else root / "mock_gene_list.txt"
        network_path = Path(self.network_file) if self.network_file else root / "mock_network.npy"
        counts_path = Path(self.counts_file) if self.counts_file else root / "mock_counts.npy"
        if not gene_path.exists() or not network_path.exists() or not counts_path.exists():
            raise FileNotFoundError(f"Unsupported or incomplete reference root: {self.ref_root}")

        gene_names = [line.strip() for line in gene_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        network = np.load(network_path)
        counts = np.load(counts_path)
        self._reference_cache = {
            "gene_names": gene_names,
            "network": network,
            "counts": counts,
            "backend": backend_info["backend"],
        }
        return self._reference_cache

    def _load_reference_data_from_genki_source(self) -> Dict[str, Any]:
        self.validate_runtime_ready()
        if self.adata_file is None or self.grn_file_dir is None:
            raise ValueError("genki_source backend requires adata_file and grn_file_dir")

        adata = ad.read_h5ad(self.adata_file)
        gene_names = [str(gene_name) for gene_name in adata.var_names.tolist()]
        net_path = Path(self.grn_file_dir) / f"{self.pcnet_name}.npz"
        if not net_path.exists():
            raise FileNotFoundError(f"GRN file not found: {net_path}")
        network = sp.load_npz(net_path)
        counts = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=float)

        self._reference_cache = {
            "gene_names": gene_names,
            "network": network.toarray(),
            "counts": counts,
            "backend": "genki_source",
            "adata_file": self.adata_file,
            "grn_file_dir": self.grn_file_dir,
        }
        return self._reference_cache

    def _probe_dependencies(self, module_names: List[str]) -> List[str]:
        missing_dependencies: List[str] = []
        for module_name in module_names:
            if importlib.util.find_spec(module_name) is None:
                missing_dependencies.append(module_name)
                continue
            try:
                importlib.import_module(module_name)
            except Exception:
                missing_dependencies.append(module_name)
        return missing_dependencies
