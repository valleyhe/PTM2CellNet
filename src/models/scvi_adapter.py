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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SCVI_INSTALL_COMMAND = (
    "pip install 'setuptools>=68.0,<81' 'anndata>=0.10,<0.12' 'scvi-tools>=1.2.0'"
)
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
            "scvi-tools import failed; scVI-dependent features are disabled: %s. "
            "Install with: %s",
            e,
            _SCVI_INSTALL_COMMAND,
        )
        return False


#: Runtime flag mirroring the ``SSPA_AVAILABLE`` pattern.
SCVI_AVAILABLE: bool = _check_scvi_available()

# Default checkpoint search paths (checked in order when no explicit path is given)
_DEFAULT_SCVI_PATHS = [
    "checkpoints/scvi_model",
    "models/scvi_model",
    Path(__file__).resolve().parents[2] / "checkpoints" / "scvi_model",
]


@dataclass
class ScVIAdapterConfig:
    """Configuration for :class:`ScVIAdapter`.

    Attributes:
        model_path: Path to a saved ``scvi.model.SCVI`` model directory.
        n_latent: Latent dimensionality of the scVI model (must match the
            trained model; default 10 to match ``LatentDAVFConfig.latent_dim``).
        gene_layer: Which AnnData layer to read for gene counts. ``None`` uses
            ``adata.X``.
        batch_key: Optional batch key passed to ``get_latent_representation``.
        device: Device override (``None`` = auto-detect).
    """
    model_path: Optional[str] = None
    n_latent: int = 10
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

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_trained_model(
        cls,
        model_path: Union[str, Path],
        config: Optional[ScVIAdapterConfig] = None,
    ) -> "ScVIAdapter":
        """Load a saved ``scvi.model.SCVI`` model from ``model_path``.

        Args:
            model_path: Directory previously written via ``model.save()``.
            config: Optional adapter configuration (``n_latent`` etc.).

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
            raise FileNotFoundError(
                f"scVI model not found at {model_path}"
            )

        from scvi.model import SCVI
        model = SCVI.load(str(model_path), adata=None, use_gpu=False)
        cfg = config or ScVIAdapterConfig(model_path=str(model_path))
        return cls(model, config=cfg)

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
        return cls(model, config=cfg, adata_registry=adata)

    # ------------------------------------------------------------------
    # Encode / Decode (V22-02 core)
    # ------------------------------------------------------------------
    def encode(self, adata):
        if self.model is None:
            raise RuntimeError("ScVIAdapter has no model loaded; cannot encode.")
        # Validate adata format
        if adata is None:
            raise ValueError("adata must not be None for encoding")
        if hasattr(adata, 'X') and hasattr(adata, 'n_vars'):
            n_vars = adata.n_vars
            expected = self.n_genes
            if expected > 0 and n_vars != expected:
                logger.warning(
                    "adata has %d variables but model expects %d. "
                    "Results may be incorrect.",
                    n_vars, expected,
                )
        representation = self.model.get_latent_representation(adata, batch_key=self.config.batch_key)
        return np.asarray(representation)

    def decode(
        self,
        latent: np.ndarray,
        library_size: Optional[float] = None,
        n_samples: int = 1,
    ) -> np.ndarray:
        """Map latent vectors back to gene-space expression (latent -> gene).

        Args:
            latent: ``[B, n_latent]`` latent vectors.
            library_size: Optional total counts per cell. ``None`` uses the
                model's default (``1.0``).
            n_samples: Number of posterior samples to draw and average.

        Returns:
            ``np.ndarray`` of shape ``[B, n_genes]`` decoded gene expression.
        """
        if self.model is None:
            raise RuntimeError("ScVIAdapter has no model loaded; cannot decode.")
        latent = np.asarray(latent)
        if latent.ndim != 2:
            raise ValueError(
                f"latent must be 2D [B, n_latent], got shape {latent.shape}"
            )

        # scVI's decoder is accessed via the module's generative distribution.
        # We use the public posterior API for robustness across scvi-tools
        # versions: get the model's underlying AnnData registry to know the
        # gene count, then call the decoder module.
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                f"torch is required for scVI decode but unavailable: {e}"
            ) from e

        module = self.model.module
        module.eval()
        z = torch.as_tensor(latent, dtype=torch.float32)

        with torch.no_grad():
            if library_size is None:
                # Default library size of 1.0; scale output later if needed.
                library = torch.ones((z.shape[0], 1), dtype=torch.float32)
            else:
                library = torch.full(
                    (z.shape[0], 1), float(library_size), dtype=torch.float32
                )

            outputs = []
            for _ in range(max(1, n_samples)):
                # scVI generative model expects (z, library); newer versions
                # accept kwargs. Use getattr to be version-robust.
                try:
                    generative = module.generative(z, library)
                except TypeError:
                    generative = module.generative({"z": z, "library": library})

                # The decoded distribution is typically under "px".
                px = generative.get("px", None)
                if px is None:
                    raise RuntimeError(
                        "scVI generative output did not contain 'px' distribution."
                    )
                rate = getattr(px, "mu", None)
                if rate is None:
                    rate = getattr(px, "mean", None)
                if rate is None:
                    raise RuntimeError(
                        "Could not extract rate/mu from decoded distribution."
                    )
                outputs.append(rate.cpu().numpy())

        decoded = np.asarray(np.mean(np.stack(outputs, axis=0), axis=0))
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
        gene_pred = self.decode(z_1_np)
        return gene_pred

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def n_genes(self) -> int:
        """Number of genes in the underlying scVI model's vocabulary."""
        if self._adata_registry is not None:
            return int(self._adata_registry.n_vars)
        summary = getattr(self.model, "summary", None)
        if summary is not None:
            try:
                return int(summary.get("n_vars", 0))
            except (TypeError, AttributeError, KeyError, RuntimeError) as exc:
                logger.warning("Could not read n_vars from model summary; defaulting to 0: %s", exc)
        return 0

    def __repr__(self) -> str:
        return (
            f"ScVIAdapter(available={SCVI_AVAILABLE}, "
            f"has_model={self.model is not None}, "
            f"n_latent={self.config.n_latent})"
        )


__all__ = [
    "ScVIAdapter",
    "ScVIAdapterConfig",
    "SCVI_AVAILABLE",
]
