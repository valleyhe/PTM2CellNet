"""
ConditionalVelocityField: the core velocity prediction network for DAVF.

Extracted from davf.py to keep that file under 500 lines.
"""

import torch
import torch.nn as nn
from typing import cast

from .davf_encoder import DAVFConfig, TimeEncoder, GeneSpecificModulation


class ConditionalVelocityField(nn.Module):
    """
    Conditional velocity field for Flow Matching.

    Predicts v(x_t, t, condition) where:
    - x_t: expression at time t (interpolated between x_0 and x_1)
    - t: time step in [0, 1]
    - condition: direction embedding from BiPerturbEncoder

    Architecture (per D-03 Concatenation):
    - x_encoder: MLP encoding x_t
    - time_encoder: sinusoidal encoding of t
    - conditioning: concat [time_emb, x_emb, condition_emb]
    - velocity_head: MLP predicting velocity vector
    """

    def __init__(self, config: DAVFConfig):
        super().__init__()
        self.config = config

        # x_encoder: encodes expression x_t
        self.x_encoder = nn.Sequential(
            nn.Linear(config.num_genes, config.x_encoder_hidden),
            nn.LayerNorm(config.x_encoder_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.x_encoder_hidden, config.x_encoder_hidden),
            nn.LayerNorm(config.x_encoder_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # time_encoder: sinusoidal encoding of t
        self.time_encoder = TimeEncoder(embed_dim=config.time_embed_dim)

        # condition_projection: projects BiPerturbEncoder output
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
                nn.GELU()
            )
        elif config.condition_projection_depth == 2:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, config.x_encoder_hidden),
                nn.LayerNorm(config.x_encoder_hidden),
                nn.GELU(),
                nn.Linear(config.x_encoder_hidden, config.x_encoder_hidden),
                nn.GELU()
            )
        else:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, config.x_encoder_hidden),
                nn.LayerNorm(config.x_encoder_hidden),
                nn.GELU()
            )

        # 残差连接 (v2.0)
        self.use_residual = config.use_residual
        if config.use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(config.residual_gate_init))

        # velocity_net: predicts v(x_t, t, condition)
        # Input: concat [time_emb, x_emb, condition_emb]
        total_input_dim = config.time_embed_dim + config.x_encoder_hidden + config.x_encoder_hidden

        velocity_layers = []
        in_dim = total_input_dim
        for i in range(config.num_velocity_layers):
            hidden_dim = config.velocity_hidden if i < config.num_velocity_layers - 1 else config.x_encoder_hidden
            velocity_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if i < config.num_velocity_layers - 1 else nn.Identity(),
                nn.GELU() if i < config.num_velocity_layers - 1 else nn.Identity(),
                nn.Dropout(config.dropout) if i < config.num_velocity_layers - 1 else nn.Identity()
            ])
            in_dim = hidden_dim

        # Final layer to gene dimension
        velocity_layers.append(nn.Linear(in_dim, config.num_genes))

        self.velocity_net = nn.Sequential(*velocity_layers)

        # Hybrid injection path (D19-01~05)
        if config.condition_injection not in ("hybrid", "concat"):
            raise ValueError(
                "condition_injection must be 'hybrid' or 'concat', "
                f"got {config.condition_injection}"
            )
        self.condition_injection = config.condition_injection

        if config.condition_injection == "hybrid":
            num_mod_heads = max(1, config.num_kv_heads // 2)
            if config.modulation_dim % num_mod_heads != 0:
                raise ValueError(
                    f"modulation_dim ({config.modulation_dim}) must be divisible by "
                    f"derived modulation heads ({num_mod_heads} = max(1, num_kv_heads//2))"
                )
            self.gene_modulation = GeneSpecificModulation(
                num_genes=config.num_genes,
                hidden_dim=config.hidden_dim,
                modulation_dim=config.modulation_dim,
                num_kv=config.num_kv_heads,
                num_heads=num_mod_heads,
                dropout=config.dropout
            )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity at time t given current state and condition.

        Args:
            x_t: [batch_size, num_genes] - expression at time t
            t: [batch_size] - time step in [0, 1]
            condition: [batch_size, hidden_dim] - from BiPerturbEncoder

        Returns:
            velocity: [batch_size, num_genes] - predicted velocity
        """
        # Encode x_t
        x_emb = self.x_encoder(x_t)

        # Encode time
        t_emb = self.time_encoder(t)

        # Project condition
        cond_emb = self.condition_projection(condition)

        # Concatenate all embeddings
        combined = torch.cat([t_emb, x_emb, cond_emb], dim=-1)

        # Predict velocity
        velocity = self.velocity_net(combined)

        # Hybrid path: gene-specific modulation (D19-01)
        if self.condition_injection == "hybrid":
            velocity = self.gene_modulation(condition, velocity)

        # 残差连接
        if self.use_residual:
            velocity = velocity + self.residual_gate * x_t

        return cast(torch.Tensor, velocity)
