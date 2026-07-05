"""
DAVF shared encoder modules and configuration.

Extracted from davf.py to avoid circular imports and improve modularity.

Provides:
- DAVFConfig: configuration dataclass for DAVF models
- TimeEncoder: sinusoidal time encoding for Flow Matching
- GeneSpecificModulation: hybrid cross-attention + FiLM modulation
"""

import torch
import torch.nn as nn
from typing import Optional, cast
from dataclasses import dataclass

from src.models.biperturb import BiPerturbConfig


@dataclass
class DAVFConfig:
    """Configuration for DAVF (Direction-Aware Velocity Field) model.

    Extends BiPerturbConfig with Flow Matching specific fields.
    """
    # Default architecture constants (also mirrored as DAVF class-level attrs)
    _DEFAULT_HIDDEN_DIM: int = 256
    _DEFAULT_X_ENCODER_HIDDEN: int = 512
    _DEFAULT_VELOCITY_HIDDEN: int = 512
    _DEFAULT_MODULATION_DIM: int = 128
    _DEFAULT_NUM_GENES: int = 5000
    _DEFAULT_NUM_STEPS: int = 50

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
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    num_heads: int = 4
    attention_dropout: float = 0.1

    # Flow Matching specific
    time_embed_dim: int = 64
    x_encoder_hidden: int = _DEFAULT_X_ENCODER_HIDDEN  # 256 -> 512
    velocity_hidden: int = _DEFAULT_VELOCITY_HIDDEN     # 256 -> 512
    num_velocity_layers: int = 3

    # Residual connection (v2.0 improvement)
    use_residual: bool = True
    residual_gate_init: float = 0.1

    # Enhanced condition projection (v2.0/v5.0 improvement)
    condition_projection_depth: int = 3  # 2 -> 3 (increased nonlinearity)

    # Output
    num_genes: int = _DEFAULT_NUM_GENES

    # Regularization
    dropout: float = 0.1

    # Hybrid injection fields (D19-01~05)
    condition_injection: str = "hybrid"  # "hybrid" | "concat"
    modulation_dim: int = _DEFAULT_MODULATION_DIM   # Cross-Attention + FiLM internal dim
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
