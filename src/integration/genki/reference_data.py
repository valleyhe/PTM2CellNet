"""Reference data loading and backend discovery for GenKI integration."""

import importlib
import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import anndata as ad
import numpy as np
import scipy.sparse as sp
from typing_extensions import TypedDict


class _BackendInfo(TypedDict, total=False):
    backend: str
    runtime_ready: bool
    ref_root: str
    missing_dependencies: List[str]
    missing_files: List[str]
    uses_explicit_files: bool
    has_adata_file: bool
    has_grn_dir: bool
    scoring_method: str
    null_permutations: int
    bagging_threshold: float
    bagging_cutoff: float


class _ReferenceData(TypedDict, total=False):
    gene_names: List[str]
    network: Any
    counts: Any
    backend: str
    loaded_at: float
    adata_file: Optional[str]
    grn_file_dir: Optional[str]

logger = logging.getLogger(__name__)


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
        self._reference_cache: _ReferenceData | None = None
        # Timestamp + backend fingerprint used to invalidate the cache when the
        # underlying reference files change or exceed the TTL.
        self._cache_ttl_seconds: float = 86400.0  # 24 hours
        self._cache_signature: str | None = None

    def get_backend_info(self) -> _BackendInfo:
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
            # Validate that the adata file and GRN directory actually exist
            # and are readable; import-ability alone is not sufficient because
            # the reference data cannot be loaded without these files.
            missing_files = self._missing_reference_files()
            runtime_ready = len(missing_dependencies) == 0 and len(missing_files) == 0
            return {
                "backend": "genki_source",
                "runtime_ready": runtime_ready,
                "ref_root": self.ref_root,
                "missing_dependencies": missing_dependencies,
                "missing_files": missing_files,
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
        missing_deps = ", ".join(backend.get("missing_dependencies", []))
        missing_files = ", ".join(backend.get("missing_files", []))
        parts: List[str] = []
        if missing_deps:
            parts.append(f"missing or broken dependencies: {missing_deps}")
        if missing_files:
            parts.append(f"missing or unreadable reference files: {missing_files}")
        if not parts:
            parts.append("unknown")
        raise RuntimeError(
            f"Backend {backend['backend']} is not ready; " + "; ".join(parts)
        )

    def _reference_files_signature(self) -> str:
        """Build a fingerprint of the reference input files + backend.

        Combines file mtimes/sizes so the cache is invalidated when the
        underlying data changes, plus a TTL based on load time.
        """
        import hashlib

        parts: List[str] = [self.ref_root]
        for path_attr in ("gene_list_file", "network_file", "counts_file", "adata_file", "grn_file_dir"):
            value = getattr(self, path_attr, None)
            if value:
                p = Path(value)
                try:
                    stat = p.stat()
                    parts.append(f"{path_attr}:{stat.st_mtime}:{stat.st_size}")
                except OSError:
                    parts.append(f"{path_attr}:missing")
        backend_info = self.get_backend_info()
        parts.append(f"backend:{backend_info.get('backend')}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _cache_is_valid(self) -> bool:
        """Return True if the in-memory cache is fresh enough to reuse."""
        if self._reference_cache is None:
            return False
        loaded_at = self._reference_cache.get("loaded_at")
        if loaded_at is None:
            return False
        if (time.time() - float(loaded_at)) > self._cache_ttl_seconds:
            return False
        if self._cache_signature is None or self._cache_signature != self._reference_files_signature():
            return False
        return True

    def load_reference_data(self) -> _ReferenceData:
        if self._cache_is_valid():
            return cast(_ReferenceData, self._reference_cache)

        backend_info = self.get_backend_info()
        if backend_info["backend"] == "genki_source":
            return self._load_reference_data_from_genki_source()

        root = Path(self.ref_root)
        gene_path = Path(self.gene_list_file) if self.gene_list_file else root / "mock_gene_list.txt"
        network_path = Path(self.network_file) if self.network_file else root / "mock_network.npy"
        counts_path = Path(self.counts_file) if self.counts_file else root / "mock_counts.npy"
        for path in (gene_path, network_path, counts_path):
            if not path.exists():
                raise FileNotFoundError(f"Unsupported or incomplete reference root: {self.ref_root}")
            if not os.access(str(path), os.R_OK):
                raise PermissionError(f"Reference file is not readable: {path}")

        gene_names = [line.strip() for line in gene_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        network = np.load(network_path)
        counts = np.load(counts_path)
        self._cache_signature = self._reference_files_signature()
        self._reference_cache = {
            "gene_names": gene_names,
            "network": network,
            "counts": counts,
            "backend": backend_info["backend"],
            "loaded_at": time.time(),
        }
        return self._reference_cache

    def _load_reference_data_from_genki_source(self) -> _ReferenceData:
        self.validate_runtime_ready()
        if self.adata_file is None or self.grn_file_dir is None:
            raise ValueError("genki_source backend requires adata_file and grn_file_dir")

        try:
            adata = ad.read_h5ad(self.adata_file)
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - depends on file content
            raise ValueError(
                f"adata_file {self.adata_file} is not a valid .h5ad file: {exc}"
            ) from exc
        # Sanity-check the AnnData before consuming it.
        if not hasattr(adata, "var_names") or adata.var_names is None:
            raise ValueError(f"adata_file {self.adata_file} has no var_names")
        if adata.n_vars == 0:
            raise ValueError(f"adata_file {self.adata_file} has zero genes (n_vars=0)")
        gene_names = [str(gene_name) for gene_name in adata.var_names.tolist()]
        net_path = Path(self.grn_file_dir) / f"{self.pcnet_name}.npz"
        # File existence/readability is already gated by validate_runtime_ready;
        # these defensive checks remain for direct callers.
        if not net_path.exists():
            raise FileNotFoundError(f"GRN file not found: {net_path}")
        if not os.access(str(net_path), os.R_OK):
            raise PermissionError(f"GRN file is not readable: {net_path}")
        # Keep the GRN as a sparse matrix to avoid OOM on large (>20k gene)
        # networks; downstream scoring densifies on demand.
        try:
            network = sp.load_npz(net_path)
        except (OSError, ValueError) as exc:  # pragma: no cover - depends on file content
            raise ValueError(
                f"GRN file {net_path} is not a valid scipy sparse npz: {exc}"
            ) from exc
        if not sp.issparse(network):
            network = sp.csr_matrix(network)
        if network.ndim != 2:
            raise ValueError(
                f"GRN file {net_path} is not a 2D matrix (ndim={network.ndim})"
            )
        counts = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=float)

        self._cache_signature = self._reference_files_signature()
        self._reference_cache = {
            "gene_names": gene_names,
            "network": network,
            "counts": counts,
            "backend": "genki_source",
            "adata_file": self.adata_file,
            "grn_file_dir": self.grn_file_dir,
            "loaded_at": time.time(),
        }
        return self._reference_cache

    def _missing_reference_files(self) -> List[str]:
        """Return descriptors for adata/GRN reference files that are missing
        or unreadable.

        Only relevant for the ``genki_source`` backend, which requires both
        ``adata_file`` (an .h5ad file) and ``grn_file_dir`` (a directory
        containing the ``pcNet.npz`` GRN file).
        """
        missing: List[str] = []
        if self.adata_file is None:
            missing.append("adata_file (not configured)")
        else:
            apath = Path(self.adata_file)
            if not apath.exists():
                missing.append(f"adata_file (not found: {self.adata_file})")
            elif not os.access(str(apath), os.R_OK):
                missing.append(f"adata_file (not readable: {self.adata_file})")
        if self.grn_file_dir is None:
            missing.append("grn_file_dir (not configured)")
        else:
            gpath = Path(self.grn_file_dir)
            if not gpath.exists() or not gpath.is_dir():
                missing.append(f"grn_file_dir (not a directory: {self.grn_file_dir})")
            else:
                net_path = gpath / f"{self.pcnet_name}.npz"
                if not net_path.exists():
                    missing.append(f"GRN file (not found: {net_path})")
                elif not os.access(str(net_path), os.R_OK):
                    missing.append(f"GRN file (not readable: {net_path})")
        return missing

    def _probe_dependencies(self, module_names: List[str]) -> List[str]:
        missing_dependencies: List[str] = []
        for module_name in module_names:
            if importlib.util.find_spec(module_name) is None:
                missing_dependencies.append(module_name)
                continue
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                logger.warning("Failed to import optional dependency %s: %s", module_name, e)
                missing_dependencies.append(module_name)
        return missing_dependencies
