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
from typing import Optional, Tuple, Dict, List, cast
from dataclasses import dataclass, field

from src.models.biperturb import (
    BiPerturbConfig,
    DirectionEncoder,
    BiPerturbEncoder
)
from src.models.geneformer_embedding import GeneformerEmbeddingLoader


@dataclass
class DAVFConfig:
    """Configuration for DAVF (Direction-Aware Velocity Field) model.

    Extends BiPerturbConfig with Flow Matching specific fields.
    """
    # Gene embedding (from BiPerturb)
    gene_embed_dim: int = 192
    gene_embed_frozen: bool = True

    # Direction encoding (from BiPerturb)
    num_directions: int = 3  # KO=0, KD=1, OE=2
    direction_embed_dim: int = 64

    # Magnitude encoding (optional, for backward compatibility with old checkpoints)
    magnitude_embed_dim: Optional[int] = None
    magnitude_hidden_dim: Optional[int] = None

    # Multi-target attention (from BiPerturb)
    hidden_dim: int = 256
    num_heads: int = 4
    attention_dropout: float = 0.1

    # Flow Matching specific
    time_embed_dim: int = 64
    x_encoder_hidden: int = 512  # 256 → 512 (增大容量)
    velocity_hidden: int = 512    # 256 → 512 (增大容量)
    num_velocity_layers: int = 3

    # 残差连接 (v2.0改进)
    use_residual: bool = True
    residual_gate_init: float = 0.1

    # 增强condition projection (v2.0/v5.0改进)
    condition_projection_depth: int = 3  # 2 → 3 (增强非线性表达能力)

    # Output
    num_genes: int = 5000

    # Regularization
    dropout: float = 0.1

    # Hybrid injection fields (D19-01~05)
    condition_injection: str = "hybrid"  # "hybrid" | "concat"
    modulation_dim: int = 128            # Cross-Attention + FiLM internal dim
    num_kv_heads: int = 8                # Number of condition sub-vectors for K/V

    def __post_init__(self):
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
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.gene_embed_dim <= 0:
            raise ValueError(f"gene_embed_dim must be positive, got {self.gene_embed_dim}")
        if self.gene_embed_dim % self.num_heads != 0:
            raise ValueError(
                f"gene_embed_dim ({self.gene_embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {self.num_genes}")
        if self.num_directions <= 0:
            raise ValueError(f"num_directions must be positive, got {self.num_directions}")
        if self.direction_embed_dim <= 0:
            raise ValueError(f"direction_embed_dim must be positive, got {self.direction_embed_dim}")
        if self.condition_injection not in ("hybrid", "concat"):
            raise ValueError(
                f"condition_injection must be 'hybrid' or 'concat', "
                f"got {self.condition_injection!r}"
            )
        if not (0.0 < self.residual_gate_init < 1.0):
            raise ValueError(
                f"residual_gate_init must be in (0, 1), got {self.residual_gate_init}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")

    def to_biperturb_config(self) -> BiPerturbConfig:
        """Convert to BiPerturbConfig for encoder initialization."""
        return BiPerturbConfig(
            gene_embed_dim=self.gene_embed_dim,
            gene_embed_frozen=self.gene_embed_frozen,
            num_directions=self.num_directions,
            direction_embed_dim=self.direction_embed_dim,
            magnitude_embed_dim=self.magnitude_embed_dim,
            magnitude_hidden_dim=self.magnitude_hidden_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            num_genes=self.num_genes
        )


class TimeEncoder(nn.Module):
    """Sinusoidal positional encoding for time step t in Flow Matching.

    Encodes time t in [0, 1] into a dense embedding using sinusoidal
    positional encoding (similar to transformer positional encodings).
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time step t using sinusoidal positional encoding.

        Args:
            t: [batch_size] - time values in [0, 1]

        Returns:
            time_embedding: [batch_size, embed_dim]
        """
        B = t.shape[0]
        device = t.device

        # Create sinusoidal encoding
        # Use log-space for better interpolation
        half_dim = self.embed_dim // 2

        # Transform t to log-space for better handling of [0, 1] range
        # t_log = log(t + 1) to spread out early times
        t_log = torch.log1p(t)

        # Create frequency bands
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device)) *
            torch.arange(0, half_dim, dtype=torch.float, device=device) / half_dim
        )

        # Compute angles
        angles = t_log.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half_dim]

        # Concatenate sin and cos
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)  # [B, embed_dim]

        # Project to target dimension
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros(B, 1, device=device)], dim=-1)

        return cast(torch.Tensor, self.linear(embedding))


class GeneSpecificModulation(nn.Module):
    """Hybrid Cross-Attention + FiLM gene-specific modulation.

    Per D19-01~03: Each gene receives differentiated velocity modulation via:
    1. Cross-Attention: gene embeddings (Q) attend to condition sub-vectors (K/V)
    2. FiLM: cross-attn output generates per-gene gamma (scale) and beta (shift)

    This solves the aggregation collapse problem where all genes received the same
    condition influence in the original concat-based injection.
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int,
        modulation_dim: int = 128,
        num_kv: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_genes = num_genes
        self.modulation_dim = modulation_dim
        self.num_kv = num_kv

        # Gene embedding table (learnable, per-gene representation as Query)
        self.gene_embedding = nn.Parameter(torch.randn(num_genes, modulation_dim) * 0.02)

        # Condition splitter: [B, hidden_dim] -> [B, num_kv * modulation_dim]
        self.condition_splitter = nn.Sequential(
            nn.Linear(hidden_dim, num_kv * modulation_dim),
            nn.LayerNorm(num_kv * modulation_dim),
            nn.GELU()
        )

        # Cross-attention: gene_emb (Q) attend to condition sub-vectors (K/V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=modulation_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(modulation_dim)

        # FiLM generator: cross-attn output -> per-gene (gamma, beta)
        self.film_generator = nn.Sequential(
            nn.Linear(modulation_dim, modulation_dim),
            nn.GELU(),
            nn.Linear(modulation_dim, 2)  # gamma and beta per gene
        )

        # FiLM initialization critical: gamma=1, beta=0 for identity mapping
        # This prevents disruption of pretrained checkpoints when new module is added
        with torch.no_grad():
            self.film_generator[-1].weight.zero_()
            self.film_generator[-1].bias.copy_(torch.tensor([1.0, 0.0]))

        # Mark FiLM layer to skip standard initialization (prevents overwrite by DAVF._init_weights)
        self.film_generator[-1].no_init_weights = True

    def forward(
        self,
        condition: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Apply gene-specific modulation to velocity.

        Args:
            condition: [B, hidden_dim] - from BiPerturbEncoder
            velocity: [B, num_genes] - base velocity from velocity_net

        Returns:
            modulated_velocity: [B, num_genes] - gamma * velocity + beta
        """
        B = condition.shape[0]

        # 1. Split condition into sub-vectors as K/V
        cond_split = self.condition_splitter(condition)  # [B, num_kv * modulation_dim]
        cond_kv = cond_split.view(B, self.num_kv, self.modulation_dim)  # [B, num_kv, mod_dim]

        # 2. Gene embedding as Query (expand to batch size)
        gene_q = self.gene_embedding.unsqueeze(0).expand(B, -1, -1)  # [B, num_genes, mod_dim]

        # 3. Cross-attention: genes attend to condition aspects
        attn_out, _ = self.cross_attention(gene_q, cond_kv, cond_kv)  # [B, num_genes, mod_dim]
        attn_out = self.cross_attn_norm(gene_q + attn_out)  # residual + norm

        # 4. FiLM: generate per-gene gamma and beta
        film_params = self.film_generator(attn_out)  # [B, num_genes, 2]
        gamma = film_params[:, :, 0]  # [B, num_genes]
        beta = film_params[:, :, 1]   # [B, num_genes]

        # 5. Modulate velocity
        return cast(torch.Tensor, gamma * velocity + beta)


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
        num_steps: int = 50,
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
            import numpy as _np

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
                    x_0, gene_ids, directions, num_steps=50, attention_mask=attention_mask
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


class DAVFLoss(nn.Module):
    """
    Loss function for DAVF Flow Matching training (v2.0改进).

    Computes MSE + Magnitude Loss between predicted velocity and target velocity.
    - MSE: 确保方向正确
    - Magnitude Loss: 确保预测幅度与真实幅度匹配
    """

    def __init__(self, mse_weight: float = 1.0, mag_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.mag_weight = mag_weight

    def forward(
        self,
        v_t: torch.Tensor,
        u_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Flow Matching loss with magnitude supervision.

        Args:
            v_t: [batch_size, num_genes] - predicted velocity
            u_t: [batch_size, num_genes] - target velocity (x_1 - x_0)

        Returns:
            Dictionary with loss components
        """
        # 1. MSE Loss
        mse = F.mse_loss(v_t, u_t)

        # 2. Magnitude Loss (v2.0改进)
        # 鼓励预测幅度的均值与真实幅度的均值匹配
        pred_mag = torch.abs(v_t).mean()
        true_mag = torch.abs(u_t).mean()
        # 使用相对误差，避免除零
        mag_loss = torch.abs(pred_mag - true_mag) / (true_mag.detach() + 1e-8)

        # 复合损失
        total = self.mse_weight * mse + self.mag_weight * mag_loss

        return {
            'mse': mse,
            'mag': mag_loss,
            'total': total
        }


class DirectionConsistencyLoss(nn.Module):
    """
    Direction consistency loss for target genes.

    Penalizes predicted velocity at target gene positions when its sign
    disagrees with the expected perturbation direction:
    - OE (direction=2): target gene velocity should be positive (up-regulated)
    - KO (direction=0): target gene velocity should be negative (down-regulated)

    This provides explicit directional supervision complementary to the
    global MSE flow matching loss.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        v_t: torch.Tensor,
        u_t: torch.Tensor,
        gene_ids: torch.Tensor,
        directions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            v_t: [B, num_genes] - predicted velocity
            u_t: [B, num_genes] - target velocity
            gene_ids: [B, max_targets] - target gene indices
            directions: [B, max_targets] - 0=KO, 1=KD, 2=OE
            attention_mask: [B, max_targets] - 1 for real targets, 0 for padding

        Returns:
            direction_loss: scalar tensor
        """
        # Sanitize gene_ids: replace -1 padding with 0 before gather
        safe_gene_ids = torch.where(
            gene_ids >= 0, gene_ids, torch.zeros_like(gene_ids)
        )
        # Extract velocity at target gene positions
        target_v = torch.gather(v_t, 1, safe_gene_ids)  # [B, K]
        target_u = torch.gather(u_t, 1, safe_gene_ids)  # [B, K]

        # Expected direction: OE -> +1, KO/KD -> -1
        dir_signs = torch.where(directions == 2, 1.0, -1.0)  # [B, K]

        # Sign agreement: positive means correct direction
        sign_agreement = target_v * dir_signs  # [B, K]

        # BCE with logits: encourage sign_agreement > 0 (correct direction)
        per_target_loss = F.binary_cross_entropy_with_logits(
            sign_agreement,
            torch.ones_like(sign_agreement),
            reduction='none'
        )  # [B, K]

        # Mask padding if attention_mask provided
        if attention_mask is not None:
            per_target_loss = per_target_loss * attention_mask
            return per_target_loss.sum() / (attention_mask.sum() + 1e-8)
        else:
            return per_target_loss.mean()
