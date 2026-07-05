"""
DAVF: Direction-Aware Velocity Field

Combines BiPerturb direction encoding with CellFlow's Flow Matching framework
for direction-aware gene perturbation prediction.

Core innovation:
1. Direction encoding (KO=-0.5, KD=-0.25, OE=+0.5) for perturbation awareness
2. Flow Matching for continuous trajectory learning
3. Conditional velocity field v(x_t, t, condition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, cast

from src.models.biperturb import (
    BiPerturbEncoder
)
from src.models.geneformer_embedding import GeneformerEmbeddingLoader

# Re-exported symbols from sub-modules for backward compatibility.
# Existing imports like ``from src.models.davf import DAVFConfig`` continue to work.
from .davf_encoder import DAVFConfig, TimeEncoder, GeneSpecificModulation  # noqa: F401
from .davf_losses import DAVFLoss, DirectionConsistencyLoss  # noqa: F401
from .davf_velocity import ConditionalVelocityField  # noqa: F401


class DAVF(nn.Module):
    """
    Direction-Aware Velocity Field model.

    Combines BiPerturb direction encoding with CellFlow's Flow Matching
    for direction-aware perturbation prediction.

    Forward pass (training):
    1. Encode perturbation conditions (direction + magnitude)
    2. Interpolate x_t = (1-t)*x_0 + t*x_1
    3. Predict velocity v(x_t, t, condition)
    4. Compute loss against target velocity u_t = x_1 - x_0

    Inference:
    1. ODE integration from x_0 to x_1 using predicted velocity
    """

    # Default architecture constants — single source of truth (DAVFConfig)
    DEFAULT_HIDDEN_DIM = DAVFConfig._DEFAULT_HIDDEN_DIM
    DEFAULT_LATENT_DIM = DAVFConfig._DEFAULT_X_ENCODER_HIDDEN
    DEFAULT_MODULATION_DIM = DAVFConfig._DEFAULT_MODULATION_DIM
    DEFAULT_NUM_GENES = DAVFConfig._DEFAULT_NUM_GENES
    DEFAULT_NUM_STEPS = DAVFConfig._DEFAULT_NUM_STEPS

    def __init__(
        self,
        config: DAVFConfig,
        embedding_loader: Optional[GeneformerEmbeddingLoader] = None,
        gene_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.config = config

        # Unified embedding loader (supports Geneformer and scGPT via EmbeddingLoaderBase)
        self.embedding_loader = embedding_loader
        self.gene_names = gene_names if gene_names is not None else []

        # Projection layer for pretrained embeddings (e.g., 1152-dim Geneformer -> 192-dim DAVF)
        self.gene_embed_proj = None
        if embedding_loader is not None and gene_names is not None and len(gene_names) > 0:
            pretrained_dim = embedding_loader.get_embedding_dim()
            if pretrained_dim != config.gene_embed_dim:
                self.gene_embed_proj = nn.Linear(pretrained_dim, config.gene_embed_dim)
            self.gene_embed_table = None
        else:
            # Learnable embedding table when no pretrained loader available
            self.gene_embed_table = nn.Embedding(config.num_genes, config.gene_embed_dim)

        # BiPerturbEncoder for condition encoding
        biperturb_config = config.to_biperturb_config()
        self.biperturb_encoder = BiPerturbEncoder(biperturb_config)

        # ConditionalVelocityField for velocity prediction
        self.velocity_field = ConditionalVelocityField(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier uniform, skipping modules with special initialization."""
        # Skip modules that have special initialization (e.g., FiLM identity mapping)
        if hasattr(module, 'no_init_weights') and module.no_init_weights:
            return

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _resolve_condition(
        self,
        gene_ids: Optional[torch.Tensor],
        directions: Optional[torch.Tensor],
        magnitudes: Optional[torch.Tensor],
        condition_source: str,
        external_condition: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if condition_source not in {"internal_targets", "external_embedding"}:
            raise ValueError(
                "condition_source must be 'internal_targets' or 'external_embedding'"
            )

        if condition_source == "external_embedding":
            if external_condition is None:
                raise ValueError("external_condition is required when condition_source='external_embedding'")
            return external_condition

        if gene_ids is None or directions is None:
            raise ValueError("gene_ids and directions are required when condition_source='internal_targets'")

        gene_ids = self._sanitize_gene_ids(gene_ids, attention_mask=attention_mask)
        gene_embeddings = self._get_gene_embeddings(gene_ids)
        condition, _ = self.biperturb_encoder(
            gene_embeddings,
            directions,
            magnitudes,
            return_attention=False,
            attention_mask=attention_mask,
        )
        return cast(torch.Tensor, condition)

    def _align_condition(
        self,
        condition: torch.Tensor,
        *,
        batch_size: int,
        condition_source: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        condition = condition.to(device=reference.device, dtype=reference.dtype)
        expected = (batch_size, self.config.hidden_dim)
        if condition.ndim != 2 or tuple(condition.shape) != expected:
            if condition_source == "external_embedding":
                raise ValueError(
                    f"external_condition must be [B, {self.config.hidden_dim}], got {tuple(condition.shape)}"
                )
            raise ValueError(
                f"condition embedding must be [B, {self.config.hidden_dim}], got {tuple(condition.shape)}"
            )
        return condition

    def forward_flow_matching(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,  # Kept for API compatibility, but ignored
        t: Optional[torch.Tensor] = None,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching forward pass.

        Args:
            x_0: [batch_size, num_genes] - control expression
            x_1: [batch_size, num_genes] - perturbed expression
            gene_ids: [batch_size, num_targets] - target gene indices
            directions: [batch_size, num_targets] - 0=KO, 1=KD, 2=OE
            magnitudes: [batch_size, num_targets] - DEPRECATED, ignored (magnitude encoding removed)
            t: [batch_size] - optional time steps (sample if None)
            condition_source: source of condition, either "internal_targets" or "external_embedding"
            external_condition: [batch_size, hidden_dim] - external condition embedding

        Returns:
            Dictionary with:
            - x_t: interpolated expression at time t
            - v_t: predicted velocity
            - u_t: target velocity (x_1 - x_0)
            - condition: perturbation embedding
            - loss: MSE(v_t, u_t)
        """
        if x_0.ndim != 2 or x_1.ndim != 2:
            raise ValueError("x_0 and x_1 must be 2D tensors [B, num_genes]")
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0 and x_1 must have identical shape, got {x_0.shape} vs {x_1.shape}")

        B = x_0.shape[0]
        device = x_0.device
        if t is not None and tuple(t.shape) != (B,):
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")

        # Sample time if not provided
        if t is None:
            t = torch.rand(B, device=device)

        # Compute target velocity (ground truth for flow matching)
        u_t = x_1 - x_0  # [batch_size, num_genes]

        # Interpolate to get x_t
        x_t = (1 - t).view(B, 1) * x_0 + t.view(B, 1) * x_1

        # Resolve perturbation condition
        condition = self._resolve_condition(
            gene_ids=gene_ids,
            directions=directions,
            magnitudes=magnitudes,
            condition_source=condition_source,
            external_condition=external_condition,
            attention_mask=attention_mask,
        )
        condition = self._align_condition(
            condition,
            batch_size=B,
            condition_source=condition_source,
            reference=x_0,
        )

        # Predict velocity
        v_t = self.velocity_field(x_t, t, condition)

        # Compute loss
        loss = F.mse_loss(v_t, u_t)

        return {
            'x_t': x_t,
            'v_t': v_t,
            'u_t': u_t,
            't': t,
            'condition': condition,
            'loss': loss
        }

    def predict(
        self,
        x_0: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,  # Kept for API compatibility, but ignored
        num_steps: int = DAVFConfig._DEFAULT_NUM_STEPS,
        use_ema: bool = True,
        ema_alpha: float = 0.85,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict perturbed expression using ODE integration.

        Args:
            x_0: [batch_size, num_genes] - control expression
            gene_ids: [batch_size, num_targets] - target gene indices
            directions: [batch_size, num_targets] - 0=KO, 1=KD, 2=OE
            magnitudes: [batch_size, num_targets] - DEPRECATED, ignored (magnitude encoding removed)
            num_steps: number of Euler integration steps
            use_ema: whether to apply EMA trajectory smoothing (default True)
            ema_alpha: EMA smoothing coefficient; ema_alpha=0.0 approximates no smoothing, ema_alpha->1.0 stronger smoothing (default 0.85)
            condition_source: source of condition, either "internal_targets" or "external_embedding"
            external_condition: [batch_size, hidden_dim] - external condition embedding

        Returns:
            x_1_pred: [batch_size, num_genes] - predicted perturbed expression
        """
        was_training = self.training
        self.eval()
        try:
            B = x_0.shape[0]
            device = x_0.device

            # Resolve perturbation condition
            with torch.no_grad():
                condition = self._resolve_condition(
                    gene_ids=gene_ids,
                    directions=directions,
                    magnitudes=magnitudes,
                    condition_source=condition_source,
                    external_condition=external_condition,
                    attention_mask=attention_mask,
                )
                condition = self._align_condition(
                    condition,
                    batch_size=B,
                    condition_source=condition_source,
                    reference=x_0,
                )

            if not (0.0 <= ema_alpha <= 1.0):
                raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")

            if num_steps <= 0:
                raise ValueError(f"num_steps must be > 0, got {num_steps}")

            # Euler integration
            dt = 1.0 / num_steps
            x_t = x_0.clone()
            x_ema = x_0.clone() if use_ema else None  # EMA state only needed if smoothing enabled

            with torch.no_grad():
                for step in range(num_steps):
                    t = torch.full((B,), step * dt, device=device)
                    v_t = self.velocity_field(x_t, t, condition)

                    # Euler update
                    x_t = x_t + dt * v_t

                    # Optional EMA trajectory smoothing
                    if use_ema and x_ema is not None:
                        x_ema = ema_alpha * x_ema + (1 - ema_alpha) * x_t
                        x_t = x_ema

            return x_t
        finally:
            if was_training:
                self.train()

    def _get_gene_embeddings(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """
        Get gene embeddings for given gene indices.

        Args:
            gene_ids: [batch_size, num_targets] - target gene indices

        Returns:
            gene_embeddings: [batch_size, num_targets, gene_embed_dim]
        """
        if (gene_ids < 0).any():
            raise ValueError("gene_ids must be non-negative")

        B, K = gene_ids.shape

        # Use pretrained embeddings (Geneformer or scGPT) if loader is available
        if self.embedding_loader is not None and len(self.gene_names) > 0:
            if gene_ids.max().item() >= len(self.gene_names):
                raise ValueError(
                    f"gene_ids must be in [0, {len(self.gene_names)}), got max {gene_ids.max().item()}"
                )
            # Vectorized batch lookup (replaces the previous double Python
            # ``for i in range(B): for j in range(K):`` loop that called
            # ``.item()`` B*K times and hit the embedding loader once per row).
            #
            # Strategy:
            #   1. Move gene_ids to CPU once and convert to a flat numpy array
            #      so we can index self.gene_names (a Python list) in one shot
            #      via list comprehension over the flat index array.
            #   2. Build a single flat list of gene names (length B*K) and
            #      issue ONE call to ``get_gene_embedding`` instead of B calls.
            #   3. Reshape the [B*K, D] result back to [B, K, D].
            #
            # This removes one full axis of Python overhead and collapses
            # B embedding-loader calls into one, which is the actual hotspot
            # the audit flagged (davf.py:739-771).

            flat_cpu = gene_ids.detach().cpu().numpy().reshape(-1)
            gene_names_arr = list(self.gene_names)
            flat_gene_names = [gene_names_arr[int(idx)] for idx in flat_cpu]

            # Single batched call into the embedding loader.
            flat_embeddings = self.embedding_loader.get_gene_embedding(flat_gene_names)
            # Reshape back to [B, K, D] and move to the projection device.
            embed_dim = flat_embeddings.shape[-1]
            embeddings = flat_embeddings.reshape(B, K, embed_dim)
            projection_device = (
                self.gene_embed_proj.weight.device
                if self.gene_embed_proj is not None
                else gene_ids.device
            )
            embeddings = embeddings.to(projection_device)

            # Project to target dimension if needed
            if self.gene_embed_proj is not None:
                embeddings = self.gene_embed_proj(embeddings)

            return embeddings.to(gene_ids.device)
        else:
            # Learnable embedding table lookup
            if self.gene_embed_table is None:
                raise RuntimeError("gene_embed_table is not initialized")
            return cast(torch.Tensor, self.gene_embed_table(gene_ids))

    def _sanitize_gene_ids(
        self,
        gene_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            if attention_mask.shape != gene_ids.shape:
                raise ValueError(
                    f"attention_mask must match gene_ids shape {tuple(gene_ids.shape)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            valid_mask = attention_mask.to(dtype=torch.bool)
            if ((gene_ids < 0) & valid_mask).any():
                raise ValueError("masked-in gene_ids must be non-negative")
            gene_ids = torch.where(valid_mask, gene_ids, torch.zeros_like(gene_ids))
        elif (gene_ids < 0).any():
            raise ValueError("gene_ids must be non-negative")
        return gene_ids

    def verify_direction_accuracy(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        gene_ids: torch.Tensor,
        directions: torch.Tensor,
        gene_names: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """验证方向预测的准确性。

        Args:
            attention_mask: [B, K] - 1 for real targets, 0 for padding

        Returns:
            dict包含:
            - oe_direction_acc: OE方向准确率（预测x_1 > x_0的基因比例）
            - ko_direction_acc: KO方向准确率（预测x_1 < x_0的基因比例）
            - overall_acc: 整体方向准确率
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                # 预测
                x_1_pred = self.predict(
                    x_0, gene_ids, directions, num_steps=DAVFConfig._DEFAULT_NUM_STEPS, attention_mask=attention_mask
                )

                # 计算delta并提取目标位点
                delta_pred = x_1_pred - x_0
                delta_true = x_1 - x_0
                # Sanitize gene_ids: replace -1 padding with 0 before gather
                safe_gene_ids = torch.where(
                    gene_ids >= 0, gene_ids, torch.zeros_like(gene_ids)
                )
                target_delta_pred = torch.gather(delta_pred, 1, safe_gene_ids)
                target_delta_true = torch.gather(delta_true, 1, safe_gene_ids)

                valid_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(directions, dtype=torch.bool)

                # 按方向分组计算准确率
                results = {}
                available_metrics = []
                for dir_name, dir_idx in [('KO', 0), ('KD', 1), ('OE', 2)]:
                    mask = (directions == dir_idx) & valid_mask
                    if mask.sum() == 0:
                        results[f'{dir_name.lower()}_direction_acc'] = 0.0
                        continue

                    # 对于OE，delta_pred应该>0；对于KO，delta_pred应该<0
                    if dir_idx == 2:  # OE
                        correct = (target_delta_pred[mask] > 0) == (target_delta_true[mask] > 0)
                    else:  # KO/KD
                        correct = (target_delta_pred[mask] < 0) == (target_delta_true[mask] < 0)

                    acc = correct.float().mean().item()
                    results[f'{dir_name.lower()}_direction_acc'] = acc
                    available_metrics.append(acc)

                results['overall_acc'] = float(sum(available_metrics) / len(available_metrics)) if available_metrics else 0.0

                return results
        finally:
            if was_training:
                self.train()

