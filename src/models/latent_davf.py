"""
LatentDAVF: Direction-Aware Velocity Field in scVI latent space.

Core flow:
    gene -> scVI.encode -> latent flow -> scVI.decode -> gene

LatentDAVF operates on the latent space portion only:
    z_0 + condition -> predict z_1 (or delta_z)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, cast
from dataclasses import dataclass

from src.models.davf_encoder import TimeEncoder
from src.models.biperturb import BiPerturbConfig, BiPerturbEncoder
from src.models.geneformer_embedding import GeneformerEmbeddingLoader


@dataclass
class LatentDAVFConfig:
    """Configuration for LatentDAVF model."""
    # Default architecture constants — aligned with DAVFConfig
    _DEFAULT_HIDDEN_DIM: int = 256
    _DEFAULT_X_ENCODER_HIDDEN: int = 512
    _DEFAULT_VELOCITY_HIDDEN: int = 512
    _DEFAULT_NUM_GENES: int = 5000
    _DEFAULT_NUM_STEPS: int = 50

    latent_dim: int = 10
    num_genes: int = _DEFAULT_NUM_GENES
    gene_embed_dim: int = 192
    gene_embed_frozen: bool = True
    num_directions: int = 3
    direction_embed_dim: int = 64
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    num_heads: int = 4
    attention_dropout: float = 0.1
    time_embed_dim: int = 64
    x_encoder_hidden: int = _DEFAULT_X_ENCODER_HIDDEN
    velocity_hidden: int = _DEFAULT_VELOCITY_HIDDEN
    num_velocity_layers: int = 3
    use_residual: bool = True
    residual_gate_init: float = 0.1
    condition_projection_depth: int = 3
    dropout: float = 0.1

    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {self.num_genes}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.num_heads > self.hidden_dim:
            raise ValueError(
                f"num_heads ({self.num_heads}) cannot exceed hidden_dim ({self.hidden_dim})"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.gene_embed_dim <= 0:
            raise ValueError(f"gene_embed_dim must be positive, got {self.gene_embed_dim}")
        if self.gene_embed_dim % self.num_heads != 0:
            raise ValueError(
                f"gene_embed_dim ({self.gene_embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.direction_embed_dim <= 0:
            raise ValueError(f"direction_embed_dim must be positive, got {self.direction_embed_dim}")
        if self.num_directions <= 0:
            raise ValueError(f"num_directions must be positive, got {self.num_directions}")
        if not (0.0 < self.residual_gate_init < 1.0):
            raise ValueError(
                f"residual_gate_init must be in (0, 1), got {self.residual_gate_init}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")

    def to_biperturb_config(self) -> BiPerturbConfig:
        return BiPerturbConfig(
            gene_embed_dim=self.gene_embed_dim,
            gene_embed_frozen=self.gene_embed_frozen,
            num_directions=self.num_directions,
            direction_embed_dim=self.direction_embed_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            num_genes=self.num_genes,
        )


class LatentConditionalVelocityField(nn.Module):
    """Velocity field operating in scVI latent space."""

    def __init__(self, config: LatentDAVFConfig):
        super().__init__()
        self.config = config

        # x_encoder: latent_dim -> x_encoder_hidden (2-layer MLP)
        self.x_encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.x_encoder_hidden),
            nn.LayerNorm(config.x_encoder_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.x_encoder_hidden, config.x_encoder_hidden),
            nn.LayerNorm(config.x_encoder_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.time_encoder = TimeEncoder(embed_dim=config.time_embed_dim)

        # Condition projection: hidden_dim -> x_encoder_hidden
        condition_dim = config.hidden_dim
        if config.condition_projection_depth == 3:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, config.x_encoder_hidden),
                nn.LayerNorm(config.x_encoder_hidden),
                nn.GELU(),
                nn.Linear(config.x_encoder_hidden, config.x_encoder_hidden),
                nn.LayerNorm(config.x_encoder_hidden),
                nn.GELU(),
                nn.Linear(config.x_encoder_hidden, config.x_encoder_hidden),
                nn.GELU(),
            )
        elif config.condition_projection_depth == 2:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, config.x_encoder_hidden),
                nn.LayerNorm(config.x_encoder_hidden),
                nn.GELU(),
                nn.Linear(config.x_encoder_hidden, config.x_encoder_hidden),
                nn.GELU(),
            )
        else:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, config.x_encoder_hidden),
                nn.LayerNorm(config.x_encoder_hidden),
                nn.GELU(),
            )

        # Residual gate
        self.use_residual = config.use_residual
        if config.use_residual:
            init_logit = np.log(config.residual_gate_init / (1 - config.residual_gate_init))
            self.residual_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # Velocity network
        total_input_dim = config.time_embed_dim + config.x_encoder_hidden + config.x_encoder_hidden
        velocity_layers = []
        in_dim = total_input_dim
        for i in range(config.num_velocity_layers):
            hidden_dim = config.velocity_hidden if i < config.num_velocity_layers - 1 else config.x_encoder_hidden
            velocity_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if i < config.num_velocity_layers - 1 else nn.Identity(),
                nn.GELU() if i < config.num_velocity_layers - 1 else nn.Identity(),
                nn.Dropout(config.dropout) if i < config.num_velocity_layers - 1 else nn.Identity(),
            ])
            in_dim = hidden_dim
        velocity_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.velocity_net = nn.Sequential(*velocity_layers)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        z_emb = self.x_encoder(z_t)
        t_emb = self.time_encoder(t)
        cond_emb = self.condition_projection(condition)

        combined = torch.cat([t_emb, z_emb, cond_emb], dim=-1)
        velocity = cast(torch.Tensor, self.velocity_net(combined))

        if self.use_residual:
            gate = torch.sigmoid(self.residual_logit).clamp(min=0.01, max=0.99)
            velocity = velocity + gate * z_t

        return cast(torch.Tensor, velocity)


class LatentDAVF(nn.Module):
    """Direction-Aware Velocity Field in scVI latent space."""

    def __init__(
        self,
        config: LatentDAVFConfig,
        embedding_loader: Optional[GeneformerEmbeddingLoader] = None,
        gene_names: Optional[List[str]] = None,
        pretrained_gene_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.embedding_loader = embedding_loader
        self.gene_names = gene_names if gene_names is not None else []

        # Projection for pretrained embeddings
        self.gene_embed_proj: Optional[nn.Linear] = None
        self.gene_embed_table: Optional[nn.Embedding] = None
        pretrained_dim = None
        if pretrained_gene_embeddings is not None:
            # Use pretrained gene embeddings (e.g., from Geneformer)
            pretrained_dim = pretrained_gene_embeddings.shape[1]
            num_genes = pretrained_gene_embeddings.shape[0]
            self.gene_embed_table = nn.Embedding(num_genes, pretrained_dim)
            self.gene_embed_table.weight.data.copy_(pretrained_gene_embeddings)
            if pretrained_dim != config.gene_embed_dim:
                self.gene_embed_proj = nn.Linear(pretrained_dim, config.gene_embed_dim)
        elif embedding_loader is not None and gene_names is not None and len(gene_names) > 0:
            pretrained_dim = embedding_loader.get_embedding_dim()
            if pretrained_dim != config.gene_embed_dim:
                self.gene_embed_proj = nn.Linear(pretrained_dim, config.gene_embed_dim)
            self.gene_embed_table = None
        else:
            self.gene_embed_table = nn.Embedding(config.num_genes, config.gene_embed_dim)

        # BiPerturbEncoder for condition encoding
        biperturb_config = config.to_biperturb_config()
        self.biperturb_encoder = BiPerturbEncoder(biperturb_config)

        # Velocity field in latent space
        self.velocity_field = LatentConditionalVelocityField(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module is not self.gene_embed_table:
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
                raise ValueError(
                    "external_condition is required when condition_source='external_embedding'"
                )
            return external_condition
        if gene_ids is None or directions is None:
            raise ValueError(
                "gene_ids and directions are required when condition_source='internal_targets'"
            )
        gene_ids = self._sanitize_gene_ids(gene_ids, attention_mask=attention_mask)
        gene_embeddings = self._get_gene_embeddings(gene_ids)
        condition, _ = self.biperturb_encoder(
            gene_embeddings, directions, magnitudes,
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
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_0.ndim != 2 or z_1.ndim != 2:
            raise ValueError("z_0 and z_1 must be 2D tensors [B, latent_dim]")
        if z_0.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"z_0 latent_dim mismatch: expected {self.config.latent_dim}, got {z_0.shape[1]}"
            )
        if z_1.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"z_1 latent_dim mismatch: expected {self.config.latent_dim}, got {z_1.shape[1]}"
            )

        B = z_0.shape[0]
        device = z_0.device
        if t is not None:
            if tuple(t.shape) != (B,):
                raise ValueError(f"t must be [B], got {tuple(t.shape)}")
            if (t < 0).any() or (t > 1).any():
                raise ValueError("t must be in [0, 1]")

        if t is None:
            t = torch.rand(B, device=device, dtype=z_0.dtype)

        u_t = z_1 - z_0
        z_t = (1 - t).view(B, 1) * z_0 + t.view(B, 1) * z_1

        condition = self._resolve_condition(
            gene_ids=gene_ids, directions=directions, magnitudes=magnitudes,
            condition_source=condition_source,
            external_condition=external_condition,
            attention_mask=attention_mask,
        )
        condition = self._align_condition(
            condition, batch_size=B, condition_source=condition_source, reference=z_0,
        )

        v_t = self.velocity_field(z_t, t, condition)
        loss = F.mse_loss(v_t, u_t)

        return {"z_t": z_t, "v_t": v_t, "u_t": u_t, "loss": loss, "condition": condition}

    def predict(
        self,
        z_0: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        num_steps: int = LatentDAVFConfig._DEFAULT_NUM_STEPS,
        use_ema: bool = False,
        ema_alpha: float = 0.5,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict z_1 from z_0 using ODE integration.
        
        NOTE: use_ema=False by default. EMA was found to cause collapse
        to identity mapping (z_1_pred ≈ z_0) because it pulls z_t back
        towards z_0 at every step.
        """
        was_training = self.training
        self.eval()
        try:
            if z_0.ndim != 2:
                raise ValueError(f"z_0 must be 2D [B, latent_dim], got {tuple(z_0.shape)}")
            if z_0.shape[1] != self.config.latent_dim:
                raise ValueError(
                    f"z_0 latent_dim mismatch: expected {self.config.latent_dim}, got {z_0.shape[1]}"
                )
            if num_steps <= 0:
                raise ValueError(f"num_steps must be > 0, got {num_steps}")
            if not (0.0 <= ema_alpha <= 1.0):
                raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")
            B = z_0.shape[0]
            device = z_0.device

            with torch.no_grad():
                condition = self._resolve_condition(
                    gene_ids=gene_ids, directions=directions, magnitudes=magnitudes,
                    condition_source=condition_source,
                    external_condition=external_condition,
                    attention_mask=attention_mask,
                )
                condition = self._align_condition(
                    condition, batch_size=B, condition_source=condition_source, reference=z_0,
                )

            dt = 1.0 / num_steps
            z_t = z_0.clone()

            with torch.no_grad():
                for step in range(num_steps):
                    t = torch.full((B,), step * dt, device=device, dtype=z_0.dtype)
                    v_t = self.velocity_field(z_t, t, condition)
                    z_t = z_t + dt * v_t
                    if use_ema:
                        z_t = ema_alpha * z_0 + (1 - ema_alpha) * z_t

            return z_t
        finally:
            if was_training:
                self.train()

    def predict_delta(
        self,
        z_0: torch.Tensor,
        gene_ids: Optional[torch.Tensor] = None,
        directions: Optional[torch.Tensor] = None,
        magnitudes: Optional[torch.Tensor] = None,
        num_steps: int = LatentDAVFConfig._DEFAULT_NUM_STEPS,
        use_ema: bool = False,
        ema_alpha: float = 0.5,
        *,
        condition_source: str = "internal_targets",
        external_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_1 = self.predict(
            z_0, gene_ids=gene_ids, directions=directions, magnitudes=magnitudes,
            num_steps=num_steps, use_ema=use_ema, ema_alpha=ema_alpha,
            condition_source=condition_source,
            external_condition=external_condition,
            attention_mask=attention_mask,
        )
        return z_1 - z_0

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

    def _get_gene_embeddings(self, gene_ids: torch.Tensor) -> torch.Tensor:
        if (gene_ids < 0).any():
            raise ValueError("gene_ids must be non-negative")
        B, K = gene_ids.shape

        if self.embedding_loader is not None and len(self.gene_names) > 0:
            if gene_ids.max().item() >= len(self.gene_names):
                raise ValueError(
                    f"gene_ids must be in [0, {len(self.gene_names)}), got max {gene_ids.max().item()}"
                )
            gene_names_list = []
            for i in range(B):
                batch_gene_names = []
                for j in range(K):
                    idx = int(gene_ids[i, j].item())
                    batch_gene_names.append(self.gene_names[idx])
                gene_names_list.append(batch_gene_names)

            embeddings_list = []
            for batch_gene_names in gene_names_list:
                embs = self.embedding_loader.get_gene_embedding(batch_gene_names)
                embeddings_list.append(embs)

            embeddings = torch.stack(embeddings_list, dim=0)
        else:
            if self.gene_embed_table is None:
                raise RuntimeError("gene_embed_table is not initialized")
            if gene_ids.max().item() >= self.gene_embed_table.num_embeddings:
                raise ValueError(
                    f"gene_ids must be in [0, {self.gene_embed_table.num_embeddings}), "
                    f"got max {gene_ids.max().item()}"
                )
            embeddings = cast(torch.Tensor, self.gene_embed_table(gene_ids))

        # Apply projection if needed (e.g., Geneformer 1152 -> gene_embed_dim 192)
        if self.gene_embed_proj is not None:
            projection_device = self.gene_embed_proj.weight.device
            embeddings = embeddings.to(projection_device)
            embeddings = self.gene_embed_proj(embeddings)
            embeddings = embeddings.to(gene_ids.device)

        return embeddings
