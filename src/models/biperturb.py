"""
BiPerturb: Bidirectional Gene Perturbation Prediction Model

Core innovation:
1. Direction encoding for KO/KD/OE perturbation types
2. Magnitude encoding for continuous expression change degree
3. Direction-aware multi-target attention for mixed perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, cast
from dataclasses import dataclass

from src.models.davf_attention import DirectionAwareAttention


@dataclass
class BiPerturbConfig:
    """Configuration for BiPerturb model."""

    # Default architecture constants
    _DEFAULT_HIDDEN_DIM: int = 256
    _DEFAULT_NUM_GENES: int = 5000

    # Gene embedding
    gene_embed_dim: int = 192  # Geneformer output dimension
    gene_embed_frozen: bool = True

    # Direction encoding
    num_directions: int = 3  # KO=0, KD=1, OE=2
    direction_embed_dim: int = 64

    # Magnitude encoding (optional, for backward compatibility)
    magnitude_embed_dim: Optional[int] = None
    magnitude_hidden_dim: Optional[int] = None

    # Multi-target attention
    hidden_dim: int = _DEFAULT_HIDDEN_DIM
    num_heads: int = 4
    attention_dropout: float = 0.1

    # GNN (GEARS backbone)
    gnn_hidden_dim: int = _DEFAULT_HIDDEN_DIM
    gnn_num_layers: int = 2

    # Output
    num_genes: int = _DEFAULT_NUM_GENES  # Number of genes to predict

    # Regularization
    dropout: float = 0.1


class DirectionEncoder(nn.Module):
    """
    Encodes perturbation direction (KO/KD/OE) into embeddings.

    KO (Knockout): Gene completely removed, effect = -1
    KD (Knockdown): Gene partially inhibited, effect = -0.5 to -1
    OE (Overexpression): Gene overexpressed, effect = +1

    Enhanced with MLP projection for learnable expressiveness.
    """

    def __init__(self, num_directions: int = 3, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_directions, embed_dim)

        # MLP projection for learnable expressiveness
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Initialize base embedding with directional prior
        with torch.no_grad():
            if num_directions >= 1:
                self.embedding.weight[0] = -torch.ones(embed_dim) * 0.5  # KO
            if num_directions >= 2:
                self.embedding.weight[1] = -torch.ones(embed_dim) * 0.25  # KD
            if num_directions >= 3:
                self.embedding.weight[2] = torch.ones(embed_dim) * 0.5  # OE

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            directions: [batch_size, num_targets] - integer indices (0=KO, 1=KD, 2=OE)

        Returns:
            direction_embeddings: [batch_size, num_targets, embed_dim]
        """
        emb = self.embedding(directions)
        return cast(torch.Tensor, self.projection(emb))


class MagnitudeEncoder(nn.Module):
    """
    Encodes expression change magnitude into embeddings.

    Magnitude represents the degree of expression change:
    - KO: -1 (complete knockout)
    - KD: -0.5 to -1 (partial knockdown)
    - OE: +1 (overexpression)

    Note: Ablation experiments showed magnitude encoding contributes zero to performance.
    This class exists only for backward compatibility with old checkpoints.
    """

    def __init__(self, output_dim: int = 64, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitudes: [batch_size, num_targets] - continuous values in [-1, 1]

        Returns:
            magnitude_embeddings: [batch_size, num_targets, output_dim]
        """
        # Ensure magnitudes have correct shape for linear layer
        if magnitudes.dim() == 2:
            magnitudes = magnitudes.unsqueeze(-1)  # [batch, num_targets, 1]
        return cast(torch.Tensor, self.encoder(magnitudes))


class BiPerturbEncoder(nn.Module):
    """
    Bidirectional perturbation encoder with direction and magnitude awareness.

    This is the core innovation of BiPerturb:
    1. Gene embeddings from pretrained model (Geneformer)
    2. Direction embeddings for perturbation type
    3. Magnitude embeddings for expression change degree (optional)
    4. Direction-aware attention for target interactions
    """

    def __init__(self, config: BiPerturbConfig):
        super().__init__()
        self.config = config

        # Direction encoder
        self.direction_encoder = DirectionEncoder(
            num_directions=config.num_directions, embed_dim=config.direction_embed_dim, dropout=config.dropout
        )

        # Magnitude encoder (optional, for backward compatibility with old checkpoints)
        mag_dim = config.magnitude_embed_dim
        self.use_magnitude = mag_dim is not None and mag_dim > 0
        if self.use_magnitude and mag_dim is not None:
            self.magnitude_encoder = MagnitudeEncoder(
                output_dim=mag_dim, hidden_dim=config.magnitude_hidden_dim or 32, dropout=config.dropout
            )

        # Cross-attention: gene embeddings attend to direction embeddings
        self.direction_cross_attention = nn.MultiheadAttention(
            config.gene_embed_dim, config.num_heads, dropout=config.attention_dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(config.gene_embed_dim)
        # Project direction embeddings to gene embedding dimension for cross-attention
        self.dir_to_gene_proj = nn.Linear(config.direction_embed_dim, config.gene_embed_dim)

        # Project combined embeddings to hidden dimension
        total_input_dim: int = config.gene_embed_dim + config.direction_embed_dim
        if self.use_magnitude and mag_dim is not None:
            total_input_dim += mag_dim
        self.input_projection = nn.Linear(total_input_dim, config.hidden_dim)

        # Direction-aware multi-target attention using DirectionAwareAttention
        self.direction_aware_attention = DirectionAwareAttention(
            embed_dim=config.hidden_dim,
            direction_embed_dim=config.direction_embed_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
        )

        # Direction modulation layer (deeper for more expressiveness)
        self.direction_modulation = nn.Sequential(
            nn.Linear(config.direction_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
        )

        # Feed-forward network
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

        # Output projection for combined perturbation embedding
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        gene_embeddings: torch.Tensor,
        directions: torch.Tensor,
        magnitudes: Optional[torch.Tensor] = None,  # Kept for API compatibility, but ignored
        return_attention: bool = True,
        disable_direction: bool = False,  # For proper ablation
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode multi-target perturbations with direction awareness.

        Args:
            gene_embeddings: [batch_size, num_targets, gene_embed_dim]
            directions: [batch_size, num_targets] - 0=KO, 1=KD, 2=OE
            magnitudes: [batch_size, num_targets] - DEPRECATED, ignored (magnitude encoding removed)
            return_attention: whether to return attention weights
            disable_direction: if True, skip direction encoding (for ablation)

        Returns:
            perturbation_embedding: [batch_size, hidden_dim]
            attention_weights: [batch_size, num_targets, num_targets] or None
        """
        B, K, _ = gene_embeddings.shape
        device = gene_embeddings.device

        if attention_mask is not None:
            if attention_mask.shape != (B, K):
                raise ValueError(f"attention_mask must be [B, K], got {tuple(attention_mask.shape)}")

        # Encode direction (or use zeros for ablation)
        if disable_direction:
            dir_emb = torch.zeros(B, K, self.config.direction_embed_dim, device=device)
        else:
            dir_emb = self.direction_encoder(directions)  # [B, K, dir_dim]

        # Cross-attention: gene embeddings refined by direction (skip if disabled)
        if not disable_direction:
            dir_emb_proj = self.dir_to_gene_proj(dir_emb)  # [B, K, gene_dim]
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.to(dtype=torch.bool)

            cross_attn_out, _ = self.direction_cross_attention(
                gene_embeddings,
                dir_emb_proj,
                dir_emb_proj,
                need_weights=False,
                key_padding_mask=key_padding_mask,
            )
            gene_embeddings = self.cross_attn_norm(gene_embeddings + cross_attn_out)

        # Concatenate embeddings (include magnitude if enabled)
        if self.use_magnitude:
            if magnitudes is None:
                # Use default magnitudes based on direction (for backward compatibility)
                # KO=-1, KD=-0.5, OE=+1
                default_magnitudes = self._get_direction_signs(directions).unsqueeze(-1)  # [B, K, 1]
                mag_emb = self.magnitude_encoder(default_magnitudes)  # [B, K, mag_dim]
            else:
                mag_emb = self.magnitude_encoder(magnitudes)  # [B, K, mag_dim]
            combined = torch.cat([gene_embeddings, dir_emb, mag_emb], dim=-1)
        else:
            combined = torch.cat([gene_embeddings, dir_emb], dim=-1)
        combined = self.input_projection(combined)  # [B, K, hidden_dim]

        # Apply direction modulation (skip if disabled)
        if disable_direction:
            modulated = combined  # No modulation
        else:
            dir_modulation = self.direction_modulation(dir_emb)  # [B, K, hidden_dim]
            modulated = combined * (0.5 + dir_modulation)  # OE enhanced, KO suppressed

        # Direction-aware attention over targets
        # DirectionAwareAttention handles batch masking internally
        attn_out, attn_weights = self.direction_aware_attention(
            modulated, dir_emb, directions, attention_mask=attention_mask
        )

        # Residual + Norm + FFN
        x = self.norm1(modulated + attn_out)
        x = self.norm2(x + self.ffn(x))

        # Aggregate to single perturbation embedding
        # Weight by direction sign for proper combination (skip if disabled)
        if disable_direction:
            if attention_mask is not None:
                mask_float = attention_mask.unsqueeze(-1)  # [B, K, 1]
                valid_count = mask_float.sum(dim=1).clamp(min=1.0)  # Ensure at least 1 valid
                aggregated = (x * mask_float).sum(dim=1) / valid_count
            else:
                # Simple mean aggregation without direction weighting
                aggregated = x.mean(dim=1)  # [B, hidden_dim]
        else:
            direction_signs = self._get_direction_signs(directions)  # [B, K]
            if attention_mask is not None:
                mask_float = attention_mask.unsqueeze(-1)  # [B, K, 1]
                valid_count = mask_float.sum(dim=1).clamp(min=1.0)  # Ensure at least 1 valid
                aggregated = (x * direction_signs.unsqueeze(-1) * mask_float).sum(dim=1) / valid_count
            else:
                aggregated = (x * direction_signs.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]

        # Final projection
        perturbation_embedding = self.output_projection(aggregated)

        if return_attention:
            return perturbation_embedding, attn_weights
        return perturbation_embedding, None

    def _get_direction_signs(self, directions: torch.Tensor) -> torch.Tensor:
        """Get soft signs for aggregation based on direction type."""
        signs = torch.zeros_like(directions, dtype=torch.float)
        signs[directions == 0] = -1.0  # KO
        signs[directions == 1] = -0.5  # KD
        signs[directions == 2] = 1.0  # OE
        return signs


class PerturbationGNN(nn.Module):
    """
    Graph Neural Network for modeling gene regulatory relationships.

    Simplified version of GEARS GNN for perturbation effect prediction.
    """

    def __init__(self, hidden_dim: int, num_genes: int, num_layers: int = 2):
        super().__init__()
        self.num_genes = num_genes

        # Gene-specific embeddings
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)

        # GNN layers (simplified message passing)
        self.gnn_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # Output MLP
        self.output_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, baseline_expression: torch.Tensor, perturbation_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict expression changes based on perturbation.

        Args:
            baseline_expression: [batch_size, num_genes]
            perturbation_embedding: [batch_size, hidden_dim]

        Returns:
            delta_expression: [batch_size, num_genes]
        """
        B, G = baseline_expression.shape

        # Gene embeddings
        gene_ids = torch.arange(G, device=baseline_expression.device)
        gene_emb = self.gene_embedding(gene_ids)  # [G, hidden_dim]
        gene_emb = gene_emb.unsqueeze(0).expand(B, -1, -1)  # [B, G, hidden_dim]

        # Message passing (simplified)
        for gnn_layer in self.gnn_layers:
            gene_emb = F.relu(gnn_layer(gene_emb))

        # Combine with perturbation embedding
        perturb_expanded = perturbation_embedding.unsqueeze(1).expand(-1, G, -1)
        combined = torch.cat([gene_emb, perturb_expanded], dim=-1)

        # Predict delta
        delta = self.output_mlp(combined).squeeze(-1)  # [B, G]

        return cast(torch.Tensor, delta)


class BiPerturb(nn.Module):
    """
    BiPerturb: Bidirectional Gene Perturbation Prediction Model

    A unified framework for predicting gene expression changes under
    knockout, knockdown, and overexpression perturbations.

    Architecture:
        Gene Embedding → Direction Encoding → Magnitude Encoding →
        Direction-Aware Attention → GNN → Expression Delta Prediction
    """

    def __init__(self, config: BiPerturbConfig):
        super().__init__()
        self.config = config

        if config.hidden_dim != config.gnn_hidden_dim:
            raise ValueError(
                f"hidden_dim ({config.hidden_dim}) must match "
                f"gnn_hidden_dim ({config.gnn_hidden_dim}) for "
                f"encoder-to-GNN dimension compatibility"
            )

        # Perturbation encoder (main contribution)
        self.encoder = BiPerturbEncoder(config)

        # GNN for gene regulatory modeling
        self.gnn = PerturbationGNN(
            hidden_dim=config.gnn_hidden_dim, num_genes=config.num_genes, num_layers=config.gnn_num_layers
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        gene_embeddings: torch.Tensor,
        directions: torch.Tensor,
        baseline_expression: torch.Tensor,
        magnitudes: Optional[torch.Tensor] = None,  # Kept for API compatibility, but ignored
        return_attention: bool = True,
        disable_direction: bool = False,  # For proper ablation
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict expression changes under bidirectional perturbation.

        Args:
            gene_embeddings: [batch_size, num_targets, gene_embed_dim]
            directions: [batch_size, num_targets] - 0=KO, 1=KD, 2=OE
            baseline_expression: [batch_size, num_genes]
            magnitudes: [batch_size, num_targets] - DEPRECATED, ignored (magnitude encoding removed)
            return_attention: whether to return attention weights
            disable_direction: if True, skip direction encoding (for ablation)
            attention_mask: [batch_size, num_targets] - 1 for valid targets, 0 for padding

        Returns:
            Dictionary containing:
                - predicted_expression: [batch_size, num_genes]
                - delta_expression: [batch_size, num_genes]
                - perturbation_embedding: [batch_size, hidden_dim]
                - attention_weights: [batch_size, num_targets, num_targets] or None
        """
        # Encode perturbation
        perturb_emb, attn_weights = self.encoder(
            gene_embeddings,
            directions,
            magnitudes,
            return_attention,
            disable_direction=disable_direction,
            attention_mask=attention_mask,
        )

        # Predict expression changes
        delta_expression = self.gnn(baseline_expression, perturb_emb)

        # Compute final expression
        predicted_expression = baseline_expression + delta_expression

        return {
            "predicted_expression": predicted_expression,
            "delta_expression": delta_expression,
            "perturbation_embedding": perturb_emb,
            "attention_weights": attn_weights,
        }

    def predict_knockout(
        self,
        gene_embeddings: torch.Tensor,
        baseline_expression: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method for knockout-only prediction."""
        B, K, _ = gene_embeddings.shape
        directions = torch.zeros(B, K, dtype=torch.long, device=gene_embeddings.device)
        result = self.forward(
            gene_embeddings,
            directions,
            baseline_expression,
            attention_mask=attention_mask,
        )
        return result["predicted_expression"]

    def predict_overexpression(
        self,
        gene_embeddings: torch.Tensor,
        baseline_expression: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method for overexpression-only prediction."""
        B, K, _ = gene_embeddings.shape
        directions = torch.full((B, K), 2, dtype=torch.long, device=gene_embeddings.device)
        result = self.forward(
            gene_embeddings,
            directions,
            baseline_expression,
            attention_mask=attention_mask,
        )
        return result["predicted_expression"]

    def predict_mixed(
        self,
        gene_embeddings: torch.Tensor,
        directions: torch.Tensor,
        baseline_expression: torch.Tensor,
        magnitudes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict expression under mixed perturbations (some KO, some OE)."""
        result = self.forward(
            gene_embeddings,
            directions,
            baseline_expression,
            magnitudes,
            attention_mask=attention_mask,
        )
        return result["predicted_expression"]


class BiPerturbLoss(nn.Module):
    """
    Loss function for BiPerturb training.

    Combines:
    1. Reconstruction loss (MSE)
    2. Pearson correlation loss
    3. Direction consistency loss (optional)
    """

    def __init__(self, mse_weight: float = 1.0, pearson_weight: float = 0.5, direction_weight: float = 0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.direction_weight = direction_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        directions: Optional[torch.Tensor] = None,
        delta_pred: Optional[torch.Tensor] = None,
        target_gene_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predicted: [batch_size, num_genes] - predicted expression
            target: [batch_size, num_genes] - true expression
            directions: [batch_size, num_targets] - perturbation directions
            delta_pred: [batch_size, num_genes] - predicted delta
            target_gene_ids: [batch_size, num_targets] - indices of perturbed genes
            attention_mask: [batch_size, num_targets] - 1 for real targets, 0 for padding

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # MSE loss
        mse = F.mse_loss(predicted, target)
        losses["mse"] = mse

        # Pearson correlation loss (maximize correlation)
        pearson = 1 - self._pearson_correlation(predicted, target).mean()
        losses["pearson"] = pearson

        # Direction consistency loss: penalize when delta sign disagrees
        # with expected perturbation direction
        if directions is not None and delta_pred is not None and target_gene_ids is not None:
            direction_signs = torch.where(
                directions == 2,
                torch.ones_like(directions, dtype=torch.float),
                -torch.ones_like(directions, dtype=torch.float),
            )
            # Sanitize target_gene_ids: replace -1 padding with 0 before gather
            safe_target_gene_ids = torch.where(target_gene_ids >= 0, target_gene_ids, torch.zeros_like(target_gene_ids))
            delta_at_targets = delta_pred.gather(1, safe_target_gene_ids)  # [B, K]
            direction_match = direction_signs * delta_at_targets
            # Mask out padding positions (attention_mask available in batch)
            if attention_mask is not None:
                direction_match = direction_match * attention_mask
                n_valid = attention_mask.sum().clamp(min=1)
                direction_loss = F.relu(-direction_match).sum() / n_valid
            else:
                direction_loss = F.relu(-direction_match).mean()
            losses["direction"] = direction_loss

        # Total loss
        total = self.mse_weight * mse + self.pearson_weight * pearson
        if self.direction_weight > 0 and "direction" in losses:
            total = total + self.direction_weight * losses["direction"]
        losses["total"] = total

        return losses

    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient."""
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)

        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = torch.sqrt((x_centered**2).sum(dim=-1) * (y_centered**2).sum(dim=-1) + 1e-8)

        return numerator / denominator
