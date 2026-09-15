"""
DAVF loss functions.

Extracted from davf.py to improve modularity.

Provides:
- DAVFLoss: Flow Matching combined MSE + magnitude loss
- DirectionConsistencyLoss: directional supervision for target genes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DAVFLoss(nn.Module):
    """
    Loss function for DAVF Flow Matching training (v2.0改进).

    Computes MSE + Magnitude Loss between predicted velocity and target velocity.
    - MSE: 确保方向正确
    - Magnitude Loss: 确保预测幅度与真实幅度匹配
    """

    def __init__(self, mse_weight: float = 1.0, mag_weight: float = 0.3, adaptive_weights: bool = False):
        super().__init__()
        self.mse_weight = mse_weight
        self.mag_weight = mag_weight
        self.adaptive_weights = adaptive_weights
        self._step_count: int = 0

    def forward(self, v_t: torch.Tensor, u_t: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        pred_mag = torch.abs(v_t).mean()
        true_mag = torch.abs(u_t).mean()
        mag_loss = torch.abs(pred_mag - true_mag) / (true_mag.detach() + 1e-8)

        # 3. Direction consistency (sign agreement)
        sign_agreement = (v_t * u_t).sum(dim=-1) / (torch.norm(v_t, dim=-1) * torch.norm(u_t, dim=-1) + 1e-8)
        dir_loss = 1.0 - sign_agreement.mean()

        # Adaptive weights: reduce mse_weight as training progresses
        # (mag_loss and dir_loss become more important)
        if self.adaptive_weights:
            self._step_count += 1
            decay = min(1.0, self._step_count / 10000.0)
            effective_mse_weight = self.mse_weight * (1.0 - 0.3 * decay)
            effective_mag_weight = self.mag_weight * (1.0 + 0.5 * decay)
        else:
            effective_mse_weight = self.mse_weight
            effective_mag_weight = self.mag_weight

        total = effective_mse_weight * mse + effective_mag_weight * mag_loss + 0.1 * dir_loss

        return {"mse": mse, "mag": mag_loss, "dir": dir_loss, "total": total}


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
        attention_mask: Optional[torch.Tensor] = None,
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
        safe_gene_ids = torch.where(gene_ids >= 0, gene_ids, torch.zeros_like(gene_ids))
        # Extract velocity at target gene positions
        target_v = torch.gather(v_t, 1, safe_gene_ids)  # [B, K]
        target_u = torch.gather(u_t, 1, safe_gene_ids)  # [B, K]

        # Expected direction: OE -> +1, KO/KD -> -1
        dir_signs = torch.where(directions == 2, 1.0, -1.0)  # [B, K]

        # Sign agreement: positive means correct direction
        sign_agreement = target_v * dir_signs  # [B, K]

        # BCE with logits: encourage sign_agreement > 0 (correct direction)
        per_target_loss = F.binary_cross_entropy_with_logits(
            sign_agreement, torch.ones_like(sign_agreement), reduction="none"
        )  # [B, K]

        # Mask padding if attention_mask provided
        if attention_mask is not None:
            per_target_loss = per_target_loss * attention_mask
            return per_target_loss.sum() / (attention_mask.sum() + 1e-8)
        else:
            return per_target_loss.mean()
