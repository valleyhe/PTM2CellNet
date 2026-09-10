"""
scVI Adapter (V22-02): bridge between gene-space data and the scVI latent space.

Bridges the gap described in ``latent_davf.py``:

    gene -> scVI.encode -> latent flow (LatentDAVF) -> scVI.decode -> gene

Until this module the ``encode``/``decode`` steps were only documented in
comments; this adapter provides the actual implementation using ``scvi-tools``.

Design:
    * ``scvi`` is imported lazily so the module imports cleanly even when
      ``scvi-tools`` is not installed (matching the optional-dependency pattern
      used by ``src/analysis/pathway_integration.py`` for ``sspa``).
    * ``SCVIAvailable`` reports availability at runtime.
    * ``ScVIAdapter`` wraps a trained ``scvi.model.SCVI`` model and exposes:
          - ``encode(adata) -> latent [B, latent_dim]``
          - ``decode(latent, library_size) -> gene expression [B, num_genes]``
          - ``predict(z_0, perturbation, latent_davf) -> gene-space prediction``
            i.e. the full latent→gene mapping pipeline.

Integration with LatentDAVF
---------------------------
``LatentDAVF.predict`` / ``predict_delta`` operate purely in latent space. This
adapter supplies the surrounding encode/decode so an end-to-end
``gene -> gene`` prediction becomes::

    adapter = ScVIAdapter.from_trained_model("checkpoints/scvi_model")
    z_0 = adapter.encode(adata)                       # gene -> latent
    z_1 = latent_davf.predict(z_0, gene_ids, directions)   # latent -> latent'
    gene_pred = adapter.decode(z_1)                   # latent' -> gene
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SCVI_INSTALL_COMMAND = "pip install 'setuptools>=68.0,<81' 'anndata>=0.10,<0.12' 'scvi-tools>=1.2.0'"
_SCVI_IMPORT_ERROR: Optional[BaseException] = None


def _format_scvi_installation_message(exc: BaseException, *, action: str) -> str:
    """Build a consistent error message for missing or broken scvi-tools deps."""
    return (
        f"scvi-tools is required to {action}, but the import failed: {exc}. "
        f"Install or repair the dependency stack with: {_SCVI_INSTALL_COMMAND}"
    )


def _check_scvi_available() -> bool:
    """Return True if ``scvi-tools`` can be imported in the current env."""
    global _SCVI_IMPORT_ERROR
    try:
        import scvi  # noqa: F401

        _SCVI_IMPORT_ERROR = None
        return True
    except ImportError as e:  # pragma: no cover - environment dependent
        _SCVI_IMPORT_ERROR = e
        logger.warning(
            "scvi-tools import failed; scVI-dependent features are disabled: %s. Install with: %s",
            e,
            _SCVI_INSTALL_COMMAND,
        )
        return False


#: Runtime flag mirroring the ``SSPA_AVAILABLE`` pattern.
SCVI_AVAILABLE: bool = _check_scvi_available()

# Default checkpoint search paths (checked in order when no explicit path is given)
_DEFAULT_SCVI_PATHS: List[Union[str, Path]] = [
    Path(__file__).resolve().parents[2] / "checkpoints" / "scvi" / "ibd_norman_model",
    "checkpoints/scvi_model",
    "models/scvi_model",
    Path(__file__).resolve().parents[2] / "checkpoints" / "scvi_model",
]


def _as_positive_int(value: Any) -> Optional[int]:
    """Return a positive integer value without coercing arbitrary mocks."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    value = int(value)
    return value if value > 0 else None


def _read_field(container: Any, name: str) -> Any:
    """Read a field from scVI's mapping-like or attribute-like metadata."""

    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def _as_name_tuple(value: Any) -> tuple[str, ...]:
    """Normalize an AnnData/scVI gene-name collection to an immutable tuple."""

    if value is None or isinstance(value, (str, bytes)):
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


@dataclass
class ScVIAdapterConfig:
    """Configuration for :class:`ScVIAdapter`.

    Attributes:
        model_path: Path to a saved ``scvi.model.SCVI`` model directory.
        n_latent: Expected latent dimensionality. ``None`` derives the value
            from the saved scVI module; an explicit value is checked strictly
            against the checkpoint.
        gene_layer: Which AnnData layer to read for gene counts. ``None`` uses
            ``adata.X``.
        batch_key: Optional expected training batch key. scVI uses the key
            stored in its saved AnnData registry; it is not passed as a
            ``get_latent_representation`` data-loader argument.
        device: Device override (``None`` = auto-detect).
    """

    model_path: Optional[str] = None
    n_latent: Optional[int] = None
    gene_layer: Optional[str] = None
    batch_key: Optional[str] = None
    device: Optional[str] = None


class ScVIAdapter:
    """Adapter wrapping a trained scVI model for encode/decode (V22-02).

    The adapter is the single integration point between the project and
    ``scvi-tools``. All scVI-specific imports are performed lazily so that
    importing this module never fails when scVI is unavailable.

    Usage::

        adapter = ScVIAdapter.from_trained_model("checkpoints/scvi_model")
        z = adapter.encode(adata)            # [B, n_latent]
        gene = adapter.decode(z)             # [B, n_genes]
    """

    def __init__(
        self,
        model: Any,
        config: Optional[ScVIAdapterConfig] = None,
        adata_registry: Optional[Any] = None,
    ):
        """Wrap an already-loaded scVI model.

        Prefer :meth:`from_trained_model` / :meth:`from_anndata` for
        construction; this constructor is for cases where a model is already
        in memory (e.g. tests with a mock).

        Args:
            model: A ``scvi.model.SCVI`` instance (or compatible mock).
            config: Adapter configuration.
            adata_registry: Optional AnnData the model was set up with; needed
                for ``decode`` to know the gene vocabulary.
        """
        self.model = model
        self.config = config or ScVIAdapterConfig()
        self._adata_registry = adata_registry

    @staticmethod
    def _build_load_kwargs(
        config: ScVIAdapterConfig,
        *,
        adata: Optional[Any],
    ) -> dict[str, Any]:
        """Build the scvi-tools 1.4.x checkpoint loading arguments."""

        load_kwargs: dict[str, Any] = {
            # ``False`` is the scvi-tools sentinel for loading a checkpoint
            # without an embedded AnnData. Passing ``None`` is rejected by
            # scvi-tools 1.4.3 when the save directory has no anndata file.
            "adata": adata if adata is not None else False,
            "accelerator": "auto",
            "device": "auto",
        }
        if config.device is not None:
            if str(config.device).lower() == "auto":
                return load_kwargs
            requested_device = torch.device(config.device)
            if requested_device.type == "cpu":
                load_kwargs.update(accelerator="cpu", device="auto")
            elif requested_device.type == "cuda":
                load_kwargs.update(
                    accelerator="gpu",
                    device=requested_device.index if requested_device.index is not None else "auto",
                )
            else:
                load_kwargs.update(accelerator="auto", device=str(requested_device))
        return load_kwargs

    @staticmethod
    def _prepare_adata_for_scvi_load(adata: Optional[Any]) -> Optional[Any]:
        """Prepare a covariate context for scvi-tools checkpoint loading.

        scvi-tools 1.4.x identifies a minified ``AnnData`` by checking
        ``sparse_X.nnz == 0`` before it validates the saved setup.  A decoder
        context is allowed to contain no expression counts because the
        decoder only needs its batch/categorical covariates, so an ordinary
        all-zero sparse context is valid for this adapter but is rejected by
        that heuristic.  Preserve the logical all-zero matrix while retaining
        explicit zero entries in a private copy used only during ``SCVI.load``.

        Real minified AnnData carrying latent parameters is left untouched so
        scvi-tools can report the actual model/data incompatibility.
        """

        if adata is None:
            return None
        from scipy import sparse

        matrix = getattr(adata, "X", None)
        if not sparse.issparse(matrix) or matrix.nnz != 0:
            return adata
        obsm = getattr(adata, "obsm", {})
        has_latent_params = "_scvi_latent_qzm" in obsm and "_scvi_latent_qzv" in obsm
        if has_latent_params:
            return adata
        n_obs, n_vars = matrix.shape
        if n_obs <= 0 or n_vars <= 0:
            return adata

        prepared = adata.copy()
        explicit_zero_values = np.zeros(n_obs, dtype=matrix.dtype)
        row_pointers = np.arange(n_obs + 1, dtype=np.int64)
        zero_columns = np.zeros(n_obs, dtype=np.int64)
        prepared.X = sparse.csr_matrix(
            (explicit_zero_values, zero_columns, row_pointers),
            shape=(n_obs, n_vars),
        )
        return prepared

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_trained_model(
        cls,
        model_path: Union[str, Path],
        config: Optional[ScVIAdapterConfig] = None,
        adata: Optional[Any] = None,
    ) -> "ScVIAdapter":
        """Load a saved ``scvi.model.SCVI`` model from ``model_path``.

        Args:
            model_path: Directory previously written via ``model.save()``.
            config: Optional adapter configuration (``n_latent`` etc.).
            adata: Optional AnnData to bind while loading. If omitted, scVI
                binds the AnnData supplied later to :meth:`encode` using the
                saved registry.

        Returns:
            A configured :class:`ScVIAdapter`.

        Raises:
            ImportError: If ``scvi-tools`` is not available.
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not SCVI_AVAILABLE:
            raise ImportError(
                _format_scvi_installation_message(
                    _SCVI_IMPORT_ERROR or ModuleNotFoundError("scvi import check failed"),
                    action="load a trained scVI model",
                )
            )
        import scvi  # noqa: F401  (lazy import)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"scVI model not found at {model_path}")

        from scvi.model import SCVI

        # An omitted config means “derive the schema from this checkpoint”.
        cfg = config or ScVIAdapterConfig(model_path=str(model_path), n_latent=None)
        if cfg.model_path != str(model_path):
            cfg = replace(cfg, model_path=str(model_path))

        # ``use_gpu`` was removed from scvi-tools 1.4.x.  Passing the current
        # accelerator/device contract also makes the selected runtime device
        # explicit instead of relying on a version-specific legacy flag.
        model = SCVI.load(
            str(model_path),
            **cls._build_load_kwargs(
                cfg,
                adata=cls._prepare_adata_for_scvi_load(adata),
            ),
        )
        adapter = cls(model, config=cfg)
        adapter._validate_saved_setup()
        if adata is not None:
            adapter._validate_adata_schema(adata)
            adapter._adata_registry = adata
        actual_n_latent = adapter.n_latent
        if actual_n_latent <= 0:
            raise RuntimeError(f"Could not determine n_latent from scVI checkpoint {model_path}")
        if cfg.n_latent is not None and cfg.n_latent != actual_n_latent:
            raise ValueError(
                "scVI checkpoint latent dimension mismatch: "
                f"config expects {cfg.n_latent}, checkpoint provides {actual_n_latent}"
            )
        if cfg.n_latent is None:
            adapter.config = replace(cfg, n_latent=actual_n_latent)
        return adapter

    @classmethod
    def find_default_model(cls) -> Optional[str]:
        """Search for a default scVI model checkpoint.

        Checks common checkpoint directories and returns the first one found.
        Returns None if no checkpoint is found.
        """
        for path_str in _DEFAULT_SCVI_PATHS:
            path = Path(path_str)
            if path.exists() and (path / "model.pt").exists():
                logger.info("Found default scVI model at %s", path)
                return str(path)
        return None

    @classmethod
    def from_default_model(cls, config: Optional[ScVIAdapterConfig] = None) -> Optional["ScVIAdapter"]:
        """Try to load a scVI model from default checkpoint paths.

        Returns None if no default model is found (rather than raising).
        """
        model_path = cls.find_default_model()
        if model_path is None:
            logger.info("No default scVI model found in standard paths")
            return None
        try:
            return cls.from_trained_model(model_path, config=config)
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            logger.warning("Failed to load default scVI model from %s: %s", model_path, e)
            return None

    @classmethod
    def from_anndata(
        cls,
        adata: Any,
        n_latent: int = 10,
        n_layers: int = 1,
        gene_likelihood: str = "nb",
        config: Optional[ScVIAdapterConfig] = None,
    ) -> "ScVIAdapter":
        """Build and train a *new* scVI model on ``adata``.

        This is the from-scratch path: it instantiates an untrained scVI model,
        runs ``setup_anndata`` + a short training loop, and wraps the result.

        Args:
            adata: ``anndata.AnnData`` with raw gene counts in ``adata.X``.
            n_latent: Latent dimensionality.
            n_layers: Number of decoder/encoder layers.
            gene_likelihood: One of ``"nb"``, ``"zinb"``, ``"poisson"``, ``"normal"``.
            config: Optional adapter configuration.

        Returns:
            A configured :class:`ScVIAdapter` wrapping the trained model.
        """
        if not SCVI_AVAILABLE:
            raise ImportError(
                _format_scvi_installation_message(
                    _SCVI_IMPORT_ERROR or ModuleNotFoundError("scvi import check failed"),
                    action="build a scVI model",
                )
            )
        from scvi.model import SCVI

        SCVI.setup_anndata(adata)
        model = SCVI(
            adata,
            n_latent=n_latent,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
        )
        model.train(max_epochs=10, check_val_every_n_epoch=1)
        cfg = config or ScVIAdapterConfig(n_latent=n_latent)
        if cfg.n_latent is not None and cfg.n_latent != n_latent:
            raise ValueError(
                f"new scVI model latent dimension mismatch: config expects {cfg.n_latent}, requested {n_latent}"
            )
        if cfg.n_latent is None:
            cfg = replace(cfg, n_latent=n_latent)
        return cls(model, config=cfg, adata_registry=adata)

    # ------------------------------------------------------------------
    # Encode / Decode (V22-02 core)
    # ------------------------------------------------------------------
    def encode(self, adata, batch_size: Optional[int] = None):
        """Encode AnnData into the checkpoint's latent coordinates.

        ``batch_size`` is an optional scVI data-loader size. It is exposed for
        the large real scPerturb cohorts used to build DAVF pairs; omitting it
        preserves the adapter's historical scVI default.
        """
        if self.model is None:
            raise RuntimeError("ScVIAdapter has no model loaded; cannot encode.")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer when supplied")
        self._validate_adata_schema(adata)
        # Checkpoints saved without an h5ad keep the registry metadata but do
        # not necessarily restore an AnnDataManager (scvi-tools 1.4.3). Bind
        # the caller's AnnData explicitly before asking scVI for latents.
        if getattr(self.model, "adata_manager", None) is None:
            # Some scvi-tools 1.4.x builds can transfer a saved registry to a
            # new manager here; others leave the manager unset when the
            # checkpoint has no embedded h5ad. Reloading with the caller's
            # AnnData is the one supported path that works in both cases.
            model_path = self.config.model_path
            if model_path is not None:
                if not SCVI_AVAILABLE:
                    raise ImportError(
                        _format_scvi_installation_message(
                            _SCVI_IMPORT_ERROR or ModuleNotFoundError("scvi import check failed"),
                            action="bind AnnData to a trained scVI model",
                        )
                    )
                from scvi.model import SCVI

                self.model = SCVI.load(
                    str(model_path),
                    **self._build_load_kwargs(
                        self.config,
                        adata=self._prepare_adata_for_scvi_load(adata),
                    ),
                )
                self._validate_saved_setup()
            else:
                validate_anndata = getattr(self.model, "_validate_anndata", None)
                if not callable(validate_anndata):
                    raise RuntimeError("loaded scVI model has no AnnDataManager and cannot bind the input AnnData")
                validate_anndata(adata)
        # ``batch_key`` belongs to scVI's saved AnnData setup, not to the
        # ``get_latent_representation`` data-loader kwargs. Passing it here
        # reaches PyTorch DataLoader in scvi-tools 1.4.x and raises an
        # unexpected-keyword error. The schema check above ensures the input
        # AnnData matches that saved setup before inference starts.
        loader_kwargs = {} if batch_size is None else {"batch_size": int(batch_size)}
        representation = self.model.get_latent_representation(adata, **loader_kwargs)
        representation = np.asarray(representation)
        if representation.ndim != 2:
            raise ValueError(f"scVI encoder must return a 2D matrix, got shape {representation.shape}")
        expected_latent = self.n_latent
        if expected_latent > 0 and representation.shape[1] != expected_latent:
            raise ValueError(
                "scVI encoder returned an unexpected latent dimension: "
                f"expected {expected_latent}, got {representation.shape[1]}"
            )
        if not np.isfinite(representation).all():
            raise ValueError("scVI encoder returned non-finite latent values")
        # Keep the exact cell-level covariate context for a subsequent
        # latent->gene decode in the same round trip.
        self._adata_registry = adata
        return representation

    def decode(
        self,
        latent: np.ndarray,
        library_size: Optional[float] = None,
        n_samples: int = 1,
        adata: Optional[Any] = None,
    ) -> np.ndarray:
        """Map latent vectors back to gene-space expression (latent -> gene).

        Args:
            latent: ``[B, n_latent]`` latent vectors.
            library_size: Optional total counts per cell. ``None`` uses the
                model's default (``1.0``).
            n_samples: Number of posterior samples to draw and average.
            adata: Optional AnnData providing the cell-level batch and
                categorical covariates used by the trained scVI model. If
                omitted, the AnnData most recently passed to :meth:`encode`
                is used.

        Returns:
            ``np.ndarray`` of shape ``[B, n_genes]`` decoded gene expression.
        """
        if self.model is None:
            raise RuntimeError("ScVIAdapter has no model loaded; cannot decode.")
        latent = np.asarray(latent)
        if latent.ndim != 2:
            raise ValueError(f"latent must be 2D [B, n_latent], got shape {latent.shape}")
        expected_latent = self.n_latent
        if expected_latent > 0 and latent.shape[1] != expected_latent:
            raise ValueError(
                "latent dimension does not match the scVI checkpoint: "
                f"expected {expected_latent}, got {latent.shape[1]}"
            )
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        # scVI's decoder is accessed via the module's generative distribution.
        # We use the public posterior API for robustness across scvi-tools
        # versions: get the model's underlying AnnData registry to know the
        # gene count, then call the decoder module.
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError(f"torch is required for scVI decode but unavailable: {e}") from e

        module = self.model.module
        module.eval()
        try:
            module_device = next(module.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            module_device = torch.device("cpu")
        z = torch.as_tensor(latent, dtype=torch.float32, device=module_device)
        batch_index, cont_covs, cat_covs = self._decoder_covariates(
            adata if adata is not None else self._adata_registry,
            batch_size=z.shape[0],
            device=module_device,
        )

        with torch.no_grad():
            if library_size is None:
                # Default library size of 1.0; scale output later if needed.
                library = torch.ones((z.shape[0], 1), dtype=torch.float32, device=module_device)
            else:
                library = torch.full(
                    (z.shape[0], 1),
                    float(library_size),
                    dtype=torch.float32,
                    device=module_device,
                )

            outputs = []
            for _ in range(n_samples):
                # scVI-tools 1.4.x requires the decoder covariates explicitly.
                generative = module.generative(
                    z,
                    library,
                    batch_index,
                    cont_covs=cont_covs,
                    cat_covs=cat_covs,
                )

                # The decoded distribution is typically under "px".
                px = generative.get("px", None)
                if px is None:
                    raise RuntimeError("scVI generative output did not contain 'px' distribution.")
                rate = getattr(px, "mu", None)
                if rate is None:
                    rate = getattr(px, "mean", None)
                if rate is None:
                    raise RuntimeError("Could not extract rate/mu from decoded distribution.")
                outputs.append(rate.cpu().numpy())

        decoded = np.asarray(np.mean(np.stack(outputs, axis=0), axis=0))
        if decoded.ndim != 2 or decoded.shape[0] != latent.shape[0]:
            raise ValueError(
                "scVI decoder must return a 2D matrix with the input batch size; "
                f"got shape {decoded.shape} for batch size {latent.shape[0]}"
            )
        expected_genes = self.n_genes
        if expected_genes > 0 and decoded.shape[1] != expected_genes:
            raise ValueError(
                f"scVI decoder returned an unexpected gene dimension: expected {expected_genes}, got {decoded.shape[1]}"
            )
        if not np.isfinite(decoded).all():
            raise ValueError("scVI decoder returned non-finite expression values")
        return decoded

    # ------------------------------------------------------------------
    # End-to-end latent->gene prediction (V22-02)
    # ------------------------------------------------------------------
    def predict(
        self,
        adata: Any,
        latent_davf: Any,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Full gene -> latent -> perturbed latent -> gene pipeline.

        Implements the round-trip described in ``latent_davf.py``::

            gene -> scVI.encode -> LatentDAVF.predict -> scVI.decode -> gene

        Args:
            adata: Gene-space input data (``anndata.AnnData``).
            latent_davf: A trained ``LatentDAVF`` instance.
            gene_ids: ``[B, K]`` perturbation gene indices for LatentDAVF.
            directions: ``[B, K]`` perturbation direction codes.
            magnitudes: Optional ``[B, K]`` perturbation magnitudes.
            num_steps: ODE integration steps for LatentDAVF.
            attention_mask: ``[B, K]`` validity mask.

        Returns:
            ``np.ndarray`` of shape ``[B, n_genes]`` predicted gene expression
            after the perturbation.
        """
        if self.model is None:
            raise RuntimeError("ScVIAdapter has no model loaded; cannot predict.")

        import torch

        # 1. gene -> latent
        z_0 = self.encode(adata)
        z_0_t = torch.as_tensor(z_0, dtype=torch.float32)

        # 2. latent -> perturbed latent (via LatentDAVF ODE integration)
        z_1 = latent_davf.predict(
            z_0_t,
            gene_ids=gene_ids,
            directions=directions,
            magnitudes=magnitudes,
            num_steps=num_steps,
            attention_mask=attention_mask,
        )
        z_1_np = z_1.detach().cpu().numpy() if hasattr(z_1, "detach") else np.asarray(z_1)

        # 3. latent -> gene
        gene_pred = self.decode(z_1_np, adata=adata)
        return gene_pred

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def n_latent(self) -> int:
        """Latent dimension recorded by the loaded scVI module."""

        module = getattr(self.model, "module", None)
        value = _as_positive_int(_read_field(module, "n_latent"))
        if value is not None:
            return value
        return _as_positive_int(self.config.n_latent) or 0

    @property
    def n_genes(self) -> int:
        """Number of genes in the underlying scVI model's vocabulary."""
        for summary_name in ("summary_stats", "summary"):
            summary = getattr(self.model, summary_name, None)
            value = _as_positive_int(_read_field(summary, "n_vars"))
            if value is not None:
                return value
        gene_names = self.gene_names
        if gene_names:
            return len(gene_names)
        if self._adata_registry is not None:
            value = _as_positive_int(getattr(self._adata_registry, "n_vars", None))
            if value is not None:
                return value
        if any(getattr(self.model, name, None) is not None for name in ("summary_stats", "summary")):
            logger.warning("Could not read n_vars from model summary; defaulting to 0")
        return 0

    @property
    def gene_names(self) -> tuple[str, ...]:
        """Ordered gene vocabulary used by the scVI checkpoint decoder."""
        get_var_names = getattr(self.model, "get_var_names", None)
        if callable(get_var_names):
            try:
                names = _as_name_tuple(get_var_names())
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Could not read scVI gene vocabulary: %s", exc)
                return ()
            if names:
                return names
        registry = getattr(self.model, "registry_", {})
        field_registries = _read_field(registry, "field_registries") or {}
        x_registry = _read_field(field_registries, "X") or {}
        x_state = _read_field(x_registry, "state_registry") or {}
        names = _as_name_tuple(_read_field(x_state, "column_names"))
        if names:
            return names
        if self._adata_registry is not None:
            names = _as_name_tuple(getattr(self._adata_registry, "var_names", None))
            if names:
                return names
        return ()

    def _validate_saved_setup(self) -> None:
        """Validate optional adapter settings against scVI's saved setup."""

        registry = getattr(self.model, "registry_", {})
        setup_args = _read_field(registry, "setup_args") or {}
        if self.config.batch_key is not None:
            saved_batch_key = _read_field(setup_args, "batch_key")
            if saved_batch_key != self.config.batch_key:
                raise ValueError(
                    "scVI batch_key mismatch: "
                    f"config expects {self.config.batch_key!r}, "
                    f"checkpoint uses {saved_batch_key!r}"
                )
        if self.config.gene_layer is not None:
            saved_layer = _read_field(setup_args, "layer")
            if saved_layer != self.config.gene_layer:
                raise ValueError(
                    "scVI gene layer mismatch: "
                    f"config expects {self.config.gene_layer!r}, "
                    f"checkpoint uses {saved_layer!r}"
                )

    def _validate_adata_schema(self, adata: Any) -> None:
        """Reject AnnData whose dimensions or order differ from the checkpoint."""

        if adata is None:
            raise ValueError("adata must not be None for encoding")
        n_vars = _as_positive_int(getattr(adata, "n_vars", None))
        if n_vars is None:
            raise TypeError("adata must expose a positive integer n_vars")
        expected_genes = self.n_genes
        if expected_genes <= 0:
            raise RuntimeError("Could not determine the scVI checkpoint gene vocabulary")
        if n_vars != expected_genes:
            raise ValueError(
                f"AnnData gene count does not match the scVI checkpoint: expected {expected_genes}, got {n_vars}"
            )

        expected_names = self.gene_names
        actual_names = _as_name_tuple(getattr(adata, "var_names", None))
        if not expected_names:
            raise RuntimeError("Could not determine the scVI checkpoint gene order")
        if actual_names != expected_names:
            first_mismatch = next(
                (
                    index
                    for index, (expected, actual) in enumerate(zip(expected_names, actual_names, strict=False))
                    if expected != actual
                ),
                min(len(expected_names), len(actual_names)),
            )
            expected = expected_names[first_mismatch] if first_mismatch < len(expected_names) else "<missing>"
            actual = actual_names[first_mismatch] if first_mismatch < len(actual_names) else "<missing>"
            raise ValueError(
                "AnnData gene order does not match the scVI checkpoint at index "
                f"{first_mismatch}: expected {expected!r}, got {actual!r}"
            )

    def validate_compatibility(
        self,
        *,
        expected_latent_dim: int,
        expected_num_genes: int,
        expected_gene_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Validate the strict DAVF ↔ scVI dimensional/schema contract."""

        if self.model is None:
            raise RuntimeError("ScVIAdapter has no model loaded")
        if expected_latent_dim <= 0 or expected_num_genes <= 0:
            raise ValueError("expected_latent_dim and expected_num_genes must be positive")
        if self.n_latent <= 0:
            raise RuntimeError("Could not determine n_latent from the scVI checkpoint")
        if self.n_genes <= 0:
            raise RuntimeError("Could not determine n_vars from the scVI checkpoint")
        if self.n_latent != expected_latent_dim:
            raise ValueError(
                "DAVF/scVI latent dimension mismatch: "
                f"DAVF expects {expected_latent_dim}, scVI provides {self.n_latent}"
            )
        if self.n_genes != expected_num_genes:
            raise ValueError(
                f"DAVF/scVI gene dimension mismatch: DAVF expects {expected_num_genes}, scVI provides {self.n_genes}"
            )
        if expected_gene_names is not None:
            expected_names = tuple(str(name) for name in expected_gene_names)
            if self.gene_names != expected_names:
                raise ValueError("DAVF/scVI gene vocabulary order does not match")

    def validate_target_gene_indices(
        self,
        gene_indices: Sequence[int],
        gene_symbols: Sequence[str],
    ) -> None:
        """Ensure decoder indices refer to the supplied scVI gene symbols."""

        names = self.gene_names
        if not names:
            raise RuntimeError("scVI gene vocabulary is required for target index validation")
        if len(gene_indices) != len(gene_symbols):
            raise ValueError("gene_indices and gene_symbols must have the same length")
        for row, (raw_index, symbol) in enumerate(zip(gene_indices, gene_symbols, strict=True)):
            if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
                raise ValueError(f"gene index at row {row} must be an integer")
            index = int(raw_index)
            if index < 0 or index >= len(names):
                raise ValueError(f"gene index {index} at row {row} is outside scVI vocabulary [0, {len(names)})")
            if names[index] != str(symbol):
                raise ValueError(
                    "target gene index does not match scVI gene order at row "
                    f"{row}: index {index} is {names[index]!r}, not {str(symbol)!r}"
                )

    def resolve_target_gene_indices(self, gene_symbols: Sequence[str]) -> list[int]:
        """Resolve target symbols against the live scVI decoder vocabulary.

        This is the only supported source of decoder column indices for the
        DAVF direction path.  PerturbGen token IDs belong to a separate input
        vocabulary and must never be reused as scVI output indices.
        """

        names = self.gene_names
        if not names:
            raise RuntimeError("scVI gene vocabulary is required to resolve target gene indices")
        index_by_name = {name: index for index, name in enumerate(names)}
        indices: list[int] = []
        for symbol in gene_symbols:
            normalized = str(symbol)
            if normalized not in index_by_name:
                raise ValueError(f"target gene symbol {normalized!r} is absent from the scVI decoder vocabulary")
            indices.append(index_by_name[normalized])
        return indices

    def _decoder_covariates(
        self,
        adata: Optional[Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return scVI decoder covariates in the same row order as ``adata``."""

        if adata is not None and getattr(self.model, "adata_manager", None) is None:
            model_path = self.config.model_path
            if model_path is None:
                raise RuntimeError(
                    "scVI model_path is required to bind AnnData before decoding with covariates"
                )
            if not SCVI_AVAILABLE:
                raise ImportError(
                    _format_scvi_installation_message(
                        _SCVI_IMPORT_ERROR or ModuleNotFoundError("scvi import check failed"),
                        action="bind AnnData before scVI decoding",
                    )
                )
            from scvi.model import SCVI

            self._validate_adata_schema(adata)
            self.model = SCVI.load(
                str(model_path),
                **self._build_load_kwargs(
                    self.config,
                    adata=self._prepare_adata_for_scvi_load(adata),
                ),
            )
            self._validate_saved_setup()
            self._adata_registry = adata

        module = self.model.module
        if adata is None:
            n_batch = _as_positive_int(_read_field(module, "n_batch")) or 1
            registry = getattr(self.model, "registry_", {})
            setup_args = _read_field(registry, "setup_args") or {}
            field_registries = _read_field(registry, "field_registries") or {}
            extra_cat_registry = _read_field(field_registries, "extra_categorical_covs") or {}
            extra_cont_registry = _read_field(field_registries, "extra_continuous_covs") or {}
            extra_cat_state = _read_field(extra_cat_registry, "state_registry") or {}
            extra_cont_state = _read_field(extra_cont_registry, "state_registry") or {}
            has_extra_covariates = bool(
                _read_field(setup_args, "batch_key")
                or _read_field(setup_args, "categorical_covariate_keys")
                or _read_field(setup_args, "continuous_covariate_keys")
                or _read_field(extra_cat_state, "field_keys")
                or _read_field(extra_cont_state, "n_keys")
            )
            if n_batch > 1 or has_extra_covariates:
                raise ValueError(
                    "scVI decoder requires AnnData context because the checkpoint uses batch or other covariates"
                )
            return (
                torch.zeros((batch_size, 1), dtype=torch.long, device=device),
                None,
                None,
            )

        self._validate_adata_schema(adata)
        n_obs = _as_positive_int(getattr(adata, "n_obs", None))
        if n_obs is None or n_obs != batch_size:
            raise ValueError(
                f"AnnData context row count must match latent batch size: expected {batch_size}, got {n_obs or 0}"
            )

        validate_anndata = getattr(self.model, "_validate_anndata", None)
        make_data_loader = getattr(self.model, "_make_data_loader", None)
        if not callable(validate_anndata) or not callable(make_data_loader):
            raise RuntimeError(
                "loaded scVI model does not expose the data-manager API needed to recover decoder covariates"
            )
        validate_anndata(adata)
        loader = make_data_loader(
            adata,
            indices=list(range(n_obs)),
            batch_size=n_obs,
            shuffle=False,
        )
        try:
            batch = next(iter(loader))
        except StopIteration as exc:
            raise RuntimeError("scVI data loader returned no decoder context") from exc

        batch_index = batch.get("batch")
        if batch_index is None:
            raise RuntimeError("scVI data loader did not return batch covariates")
        cont_covs = batch.get("extra_continuous_covs")
        cat_covs = batch.get("extra_categorical_covs")
        return (
            batch_index.to(device=device),
            None if cont_covs is None else cont_covs.to(device=device),
            None if cat_covs is None else cat_covs.to(device=device),
        )

    def __repr__(self) -> str:
        return f"ScVIAdapter(available={SCVI_AVAILABLE}, has_model={self.model is not None}, n_latent={self.n_latent})"


__all__ = [
    "ScVIAdapter",
    "ScVIAdapterConfig",
    "SCVI_AVAILABLE",
]
