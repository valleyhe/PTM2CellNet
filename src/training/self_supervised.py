"""Self-supervised PTM pretraining utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


class MaskedPTMPrediction(nn.Module):
    """Mask PTM tokens and predict the original PTM type at masked sites."""

    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int = 128,
        max_position: int = 1000,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        mask_probability: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_ptm_types = num_ptm_types
        self.mask_token_id = num_ptm_types
        self.mask_probability = mask_probability
        hidden_dim = hidden_dim or embed_dim

        self.type_embedding = nn.Embedding(num_ptm_types + 1, embed_dim)
        self.position_embedding = nn.Embedding(max_position, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_ptm_types)
        self.max_position = max_position

    def mask_inputs(
        self,
        ptm_types: torch.Tensor,
        ptm_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample masked positions and replace them with the mask token."""
        if ptm_mask is None:
            valid_positions = ptm_types > 0
        else:
            valid_positions = ptm_mask.to(dtype=torch.bool)

        sampled = valid_positions & (
            torch.rand(ptm_types.shape, device=ptm_types.device) < self.mask_probability
        )
        if valid_positions.any() and not sampled.any():
            sampled = sampled.clone()
            first_valid = valid_positions.nonzero(as_tuple=False)[0]
            sampled[first_valid[0], first_valid[1]] = True

        masked_ptm_types = ptm_types.clone()
        masked_ptm_types[sampled] = self.mask_token_id
        return masked_ptm_types, sampled

    def forward(
        self,
        masked_ptm_types: torch.Tensor,
        ptm_positions: torch.Tensor,
        ptm_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        positions = ptm_positions.clamp(0, self.max_position - 1)
        embeddings = self.type_embedding(masked_ptm_types) + self.position_embedding(positions)
        hidden = self.encoder(embeddings)
        logits = self.classifier(hidden)
        predictions = torch.argmax(logits, dim=-1)

        if ptm_mask is not None:
            predictions = predictions.masked_fill(~ptm_mask.to(dtype=torch.bool), 0)

        return {
            "logits": logits,
            "predictions": predictions,
        }


def pretrain_masked_ptm(
    model: MaskedPTMPrediction,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epochs: int = 10,
) -> List[float]:
    """Run masked-PTM pretraining and return mean loss per epoch."""
    model.to(device)
    history: List[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            ptm_types = batch["ptm_types"].to(device)
            ptm_positions = batch.get("ptm_positions")
            if ptm_positions is None:
                ptm_positions = torch.arange(ptm_types.size(1), device=device).unsqueeze(0).expand_as(ptm_types)
            else:
                ptm_positions = ptm_positions.to(device)

            ptm_mask = batch.get("ptm_mask")
            if ptm_mask is None:
                ptm_mask = (ptm_types > 0).float()
            else:
                ptm_mask = ptm_mask.to(device)

            masked_ptm_types, masked_positions = model.mask_inputs(ptm_types, ptm_mask)
            if not masked_positions.any():
                continue

            outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
            logits = outputs["logits"][masked_positions]
            targets = ptm_types[masked_positions]
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        history.append(total_loss / max(steps, 1))

    return history
