"""Self-supervised PTM pretraining utilities."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


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


def _validate(
    model: MaskedPTMPrediction,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: str | torch.device,
    use_amp: bool,
) -> float:
    """Run one validation epoch and return mean loss."""
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
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

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
                    logits = outputs["logits"][masked_positions]
                    targets = ptm_types[masked_positions]
                    loss = F.cross_entropy(logits, targets)
            else:
                outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
                logits = outputs["logits"][masked_positions]
                targets = ptm_types[masked_positions]
                loss = F.cross_entropy(logits, targets)

            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(steps, 1)


def pretrain_masked_ptm(
    model: MaskedPTMPrediction,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epochs: int = 10,
    validation_split: float = 0.0,
    validate_every: int = 1,
    checkpoint_dir: str | None = None,
    use_amp: bool = False,
    lr_scheduler: str | None = None,
    warmup_steps: int = 0,
    grad_clip_norm: float | None = None,
    log_interval: int = 10,
    patience: int | None = None,
    min_delta: float = 0.0,
) -> List[float]:
    """Run masked-PTM pretraining and return mean loss per epoch.

    Parameters
    ----------
    validation_split : float
        Fraction of training data to hold out for validation (0.0 = no validation).
    validate_every : int
        Run validation every N epochs.
    checkpoint_dir : str or None
        If set, save best model checkpoint to this directory.
    use_amp : bool
        Enable automatic mixed precision (only effective when CUDA is available).
    lr_scheduler : str or None
        ``'cosine'`` or ``'plateau'`` for built-in LR schedulers.
    warmup_steps : int
        Linearly ramp LR from 0 to the initial optimizer LR over this many batches.
    grad_clip_norm : float or None
        Max gradient norm for ``clip_grad_norm_`` (no clipping when None).
    log_interval : int
        Log loss every N batches.
    patience : int or None
        Early stopping patience: stop when validation loss has not improved for
        ``patience`` consecutive validation epochs. Requires a validation loader
        (``validation_split > 0``). ``None`` disables early stopping.
    min_delta : float
        Minimum decrease in validation loss to count as an improvement.
    """
    model.to(device)

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    train_loader: Iterable[Dict[str, torch.Tensor]] = dataloader
    val_loader: Iterable[Dict[str, torch.Tensor]] | None = None

    if validation_split > 0.0:
        if hasattr(dataloader, "dataset") and hasattr(dataloader, "batch_size"):
            dataset = dataloader.dataset
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            bs = dataloader.batch_size
            cf = getattr(dataloader, "collate_fn", None)
            ss = getattr(dataloader, "sampler", None)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=bs, shuffle=(ss is None),
                collate_fn=cf,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=bs, shuffle=False,
                collate_fn=cf,
            )
            logger.info(
                "Split dataset: %d train, %d validation (%.1f%%)",
                train_size, val_size, validation_split * 100,
            )
        else:
            logger.warning(
                "validation_split > 0 but dataloader is not a DataLoader; skipping split"
            )

    use_amp = use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    scheduler = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    initial_lr: float | None = None
    if warmup_steps > 0 and optimizer.param_groups:
        initial_lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    best_val_loss = float("inf")
    epochs_since_improvement = 0
    early_stop = False
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    history: List[float] = []

    for epoch in range(epochs):
        if early_stop:
            break
        model.train()
        total_loss = 0.0
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
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

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
                    logits = outputs["logits"][masked_positions]
                    targets = ptm_types[masked_positions]
                    loss = F.cross_entropy(logits, targets)
            else:
                outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
                logits = outputs["logits"][masked_positions]
                targets = ptm_types[masked_positions]
                loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            if warmup_steps > 0 and initial_lr is not None and epoch == 0:
                if batch_idx + 1 < warmup_steps:
                    warmup_factor = (batch_idx + 1) / warmup_steps
                    for group in optimizer.param_groups:
                        group["lr"] = initial_lr * warmup_factor
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = initial_lr

            total_loss += float(loss.item())
            steps += 1

            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d/%d, Batch %d, Loss: %.4f",
                    epoch + 1, epochs, batch_idx + 1, loss.item(),
                )

        epoch_loss = total_loss / max(steps, 1)
        history.append(epoch_loss)
        logger.info("Epoch %d/%d training loss: %.4f", epoch + 1, epochs, epoch_loss)

        if scheduler is not None and lr_scheduler == "cosine":
            scheduler.step()

        val_loss = None
        if val_loader is not None and (epoch + 1) % validate_every == 0:
            val_loss = _validate(model, val_loader, device, use_amp)
            logger.info("Epoch %d/%d validation loss: %.4f", epoch + 1, epochs, val_loss)

            if scheduler is not None and lr_scheduler == "plateau":
                scheduler.step(val_loss)

            # Checkpoint and early stopping share the same improvement test
            # (b < a - min_delta), consistent with callbacks.EarlyStopping.is_improved.
            improved = val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_loss
                if checkpoint_dir is not None:
                    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info("Saved best model to %s", ckpt_path)

            if patience is not None:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    logger.info(
                        "早停等待中: %d/%d", epochs_since_improvement, patience
                    )
                    if epochs_since_improvement >= patience:
                        logger.info("早停触发于 epoch %d", epoch + 1)
                        early_stop = True

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d LR: %.6f", epoch + 1, epochs, current_lr)

    return history
