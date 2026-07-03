"""
DeltaPredictor (V22-01-DELTA-PREDICTOR): direct latent-delta predictor.

A drop-in alternative to :class:`~src.models.latent_davf.LatentDAVF` that
predicts the latent-space delta ``Δz`` *directly* via an MLP conditioned on
the perturbation, instead of integrating an ODE velocity field.

Why this exists
---------------
``LatentDAVF`` integrates a velocity field over ``num_steps`` ODE steps to
produce ``z_1`` from ``z_0``. For some workflows a single-step (amortized)
predictor is preferable: it is faster, exposes a ``delta_mlp`` parameter block
(distinguished at checkpoint-detection time, see plan line 546), and serves as
a lightweight baseline for the latent-space perturbation model.

The class is constructed with the same signature as ``LatentDAVF`` so that
callers (e.g. ``DAVFInferenceModule`` checkpoint loading) can instantiate
either interchangeably based on the checkpoint format::

    from src.models.delta_predictor import DeltaPredictor
    if uses_delta_mlp:
        model = DeltaPredictor(config, gene_names=gene_names,
                               pretrained_gene_embeddings=emb)
    else:
        model = LatentDAVF(config, gene_names=gene_names,
                           pretrained_gene_embeddings=emb)

Interface parity with ``LatentDAVF``:
    - ``predict(z_0, gene_ids, directions, attention_mask, ...) -> z_1``
    - ``predict_delta(z_0, ...) -> Δz``

Checkpoint compatibility
------------------------
State-dict keys include a ``delta_mlp.*`` prefix so that checkpoint-format
detection (``any("delta_mlp" in k for k in state_keys)``) correctly identifies
DeltaPredictor checkpoints vs. LatentDAVF checkpoints.
"""

from __future__ import annotations

import logging
from typing import List, Optional, cast

import torch
import torch.nn as nn

from src.models.latent_davf import LatentDAVF, LatentDAVFConfig

logger = logging.getLogger(__name__)


class DeltaPredictor(LatentDAVF):
    """Amortized single-step predictor of latent-space perturbation deltas.

    Subclass of :class:`LatentDAVF` that overrides the ODE integration path
    with a direct MLP mapping::

        condition(gene_ids, directions) -> Δz   (via delta_mlp)

    The BiPerturbEncoder condition embedding (inherited from LatentDAVF) is
    projected to a ``latent_dim`` delta, so::

        z_1 = z_0 + Δz

    Args:
        config: ``LatentDAVFConfig`` describing latent/gene dimensions.
        embedding_loader: Optional Geneformer embedding loader.
        gene_names: Optional gene-name list for embedding lookup.
        pretrained_gene_embeddings: Optional ``[num_genes, embed_dim]`` tensor
            used to initialize the gene embedding table.
        delta_hidden_mult: Multiplier for the delta MLP hidden width, relative
            to ``config.hidden_dim`` (default 2 → hidden = 2 * hidden_dim).
        dropout: Dropout probability inside the delta MLP.
    """

    def __init__(
        self,
        config: LatentDAVFConfig,
        embedding_loader=None,
        gene_names: Optional[List[str]] = None,
        pretrained_gene_embeddings: Optional[torch.Tensor] = None,
        delta_hidden_mult: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__(
            config,
            embedding_loader=embedding_loader,
            gene_names=gene_names,
            pretrained_gene_embeddings=pretrained_gene_embeddings,
        )

        hidden = config.hidden_dim * max(1, delta_hidden_mult)
        drop = config.dropout if dropout is None else dropout

        # The delta_mlp maps a condition embedding [B, hidden_dim] directly to
        # a latent delta [B, latent_dim]. Named ``delta_mlp`` deliberately so
        # checkpoint-format detection keys on it.
        self.delta_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, config.latent_dim),
        )

        logger.info(
            "DeltaPredictor initialized: latent_dim=%d hidden_dim=%d delta_hidden=%d",
            config.latent_dim, config.hidden_dim, hidden,
        )

    def _compute_delta(
        self,
        gene_ids: Optional[torch.Tensor],
        directions: Optional[torch.Tensor],
        magnitudes: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        external_condition: Optional[torch.Tensor],
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``Δz`` from the perturbation condition.

        Reuses the inherited condition-resolution logic from LatentDAVF so the
        same ``condition_source`` semantics apply (internal targets, external
        condition, attention masking, etc.).
        """
        condition = self._resolve_condition(
            gene_ids=gene_ids, directions=directions, magnitudes=magnitudes,
            condition_source="internal_targets" if external_condition is None else "external",
            external_condition=external_condition,
            attention_mask=attention_mask,
        )
        condition = self._align_condition(
            condition,
            batch_size=batch_size,
            condition_source="internal_targets" if external_condition is None else "external",
            reference=reference,
        )
        return cast(torch.Tensor, self.delta_mlp(condition))

    def predict(
        self,
        z_0: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        use_ema: bool = False,
        ema_alpha: float = 0.5,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict ``z_1`` directly from ``z_0`` and the perturbation condition.

        Overrides the ODE-integration ``predict`` inherited from ``LatentDAVF``
        with a single amortized MLP step. ``num_steps`` / ``use_ema`` /
        ``ema_alpha`` are accepted for signature parity with ``LatentDAVF``
        but are ignored (a warning is logged when ``num_steps != 1``).

        Args:
            z_0: ``[B, latent_dim]`` initial latent state.
            gene_ids: ``[B, K]`` perturbation gene indices.
            directions: ``[B, K]`` direction codes (0=KO, 1=KD, 2=OE).
            magnitudes: Optional ``[B, K]`` magnitudes.
            num_steps: Ignored (single-step predictor); kept for API parity.
            use_ema: Ignored.
            ema_alpha: Ignored.
            condition_source: Kept for API parity (see LatentDAVF).
            external_condition: Optional ``[B, hidden_dim]`` external condition.
            attention_mask: ``[B, K]`` validity mask.

        Returns:
            ``[B, latent_dim]`` predicted ``z_1``.
        """
        was_training = self.training
        self.eval()
        try:
            if z_0.ndim != 2:
                raise ValueError(f"z_0 must be 2D [B, latent_dim], got {tuple(z_0.shape)}")
            if z_0.shape[1] != self.config.latent_dim:
                raise ValueError(
                    f"z_0 latent_dim mismatch: expected {self.config.latent_dim}, "
                    f"got {z_0.shape[1]}"
                )
            if num_steps != 1:
                logger.debug(
                    "DeltaPredictor is single-step; num_steps=%d ignored.", num_steps
                )

            B = z_0.shape[0]
            with torch.no_grad():
                delta_z = self._compute_delta(
                    gene_ids=gene_ids,
                    directions=directions,
                    magnitudes=magnitudes,
                    attention_mask=attention_mask,
                    external_condition=external_condition,
                    batch_size=B,
                    reference=z_0,
                )
            return z_0 + delta_z
        finally:
            if was_training:
                self.train()

    def predict_delta(
        self,
        z_0: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        use_ema: bool = False,
        ema_alpha: float = 0.5,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the latent delta ``Δz = z_1 - z_0`` directly.

        Args mirror :meth:`predict`. Returns ``[B, latent_dim]`` delta.
        """
        was_training = self.training
        self.eval()
        try:
            if z_0.ndim != 2:
                raise ValueError(f"z_0 must be 2D [B, latent_dim], got {tuple(z_0.shape)}")
            if z_0.shape[1] != self.config.latent_dim:
                raise ValueError(
                    f"z_0 latent_dim mismatch: expected {self.config.latent_dim}, "
                    f"got {z_0.shape[1]}"
                )
            B = z_0.shape[0]
            with torch.no_grad():
                delta_z = self._compute_delta(
                    gene_ids=gene_ids,
                    directions=directions,
                    magnitudes=magnitudes,
                    attention_mask=attention_mask,
                    external_condition=external_condition,
                    batch_size=B,
                    reference=z_0,
                )
            return delta_z
        finally:
            if was_training:
                self.train()

    def forward(
        self,
        z_0: torch.Tensor,
        z_1: Optional[torch.Tensor] = None,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training-time forward.

        When ``z_1`` (target) is provided, returns a supervised MSE loss
        against the amortized delta prediction. Otherwise returns the predicted
        delta (inference mode).
        """
        if z_0.ndim != 2:
            raise ValueError(f"z_0 must be 2D [B, latent_dim], got {tuple(z_0.shape)}")
        if z_0.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"z_0 latent_dim mismatch: expected {self.config.latent_dim}, "
                f"got {z_0.shape[1]}"
            )
        B = z_0.shape[0]

        delta_pred = self._compute_delta(
            gene_ids=gene_ids,
            directions=directions,
            magnitudes=magnitudes,
            attention_mask=attention_mask,
            external_condition=external_condition,
            batch_size=B,
            reference=z_0,
        )

        if z_1 is not None:
            if z_1.shape != z_0.shape:
                raise ValueError(
                    f"z_1 shape {tuple(z_1.shape)} must match z_0 {tuple(z_0.shape)}"
                )
            delta_target = z_1 - z_0
            loss = nn.functional.mse_loss(delta_pred, delta_target)
            return {"delta_pred": delta_pred, "delta_target": delta_target, "loss": loss}

        return {"delta_pred": delta_pred, "z_1_pred": z_0 + delta_pred}


__all__ = ["DeltaPredictor"]
