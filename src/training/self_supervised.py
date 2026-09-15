"""Self-supervised PTM pretraining utilities."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)

from .amp_compat import make_grad_scaler, amp_autocast
from ..utils.safe_io import safe_torch_load


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

        sampled = valid_positions & (torch.rand(ptm_types.shape, device=ptm_types.device) < self.mask_probability)
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


class PTMContrastiveLearning(nn.Module):
    """Contrastive learning module for PTM representation pretraining.

    Learns to distinguish PTM-augmented sequences from their unmodified
    counterparts using a simple NT-Xent (normalized temperature-scaled
    cross-entropy) loss. Positive pairs are (original, PTM-augmented)
    views of the same sequence; negatives are all other samples in the batch.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.07,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        # Projection head: embed_dim -> projection_dim
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute contrastive loss between anchor and positive pairs.

        Args:
            anchor: [B, embed_dim] embeddings from original sequences.
            positive: [B, embed_dim] embeddings from PTM-augmented sequences.

        Returns:
            Dictionary with 'contrastive_loss' and 'similarity' tensors.
        """
        # Project to lower-dimensional space
        z_anchor = self.dropout(self.projector(anchor))
        z_positive = self.dropout(self.projector(positive))

        # L2 normalize
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_positive = F.normalize(z_positive, dim=-1)

        # Compute similarity matrix [2B, 2B]
        z = torch.cat([z_anchor, z_positive], dim=0)  # [2B, proj_dim]
        sim = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        batch_size = anchor.shape[0]
        mask = torch.eye(2 * batch_size, device=sim.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Labels: for each anchor, the positive is at offset +batch_size
        labels = torch.arange(batch_size, 2 * batch_size, device=sim.device)
        labels = torch.cat([labels, torch.arange(0, batch_size, device=sim.device)])

        loss = F.cross_entropy(sim, labels)
        similarity = torch.sum(z_anchor * z_positive, dim=-1).mean()

        return {
            "contrastive_loss": loss,
            "similarity": similarity,
        }


class PTMDenoisingAutoEncoder(nn.Module):
    """Denoising autoencoder for PTM representations.

    Corrupts input PTM features with noise and trains the model
    to reconstruct the original (clean) features.
    """

    def __init__(
        self,
        num_ptm_types: int,
        embed_dim: int = 128,
        hidden_dim: int | None = None,
        noise_factor: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_ptm_types = num_ptm_types
        self.noise_factor = noise_factor
        hidden_dim = hidden_dim or embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(num_ptm_types, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ptm_types),
        )

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input features."""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def forward(
        self,
        ptm_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Denoise PTM features.

        Args:
            ptm_features: [B, num_ptm_types] one-hot or multi-hot PTM features.

        Returns:
            Dictionary with 'reconstructed', 'encoded', 'denoising_loss'.
        """
        noisy = self.add_noise(ptm_features)
        encoded = self.encoder(noisy)
        reconstructed = self.decoder(encoded)

        loss = F.mse_loss(reconstructed, ptm_features)

        return {
            "reconstructed": reconstructed,
            "encoded": encoded,
            "denoising_loss": loss,
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
            loss, _ = _compute_masked_loss(model, batch, device, use_amp)
            if loss is None:
                continue
            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(steps, 1)


def _split_train_val(
    dataloader: Iterable[Dict[str, torch.Tensor]],
    validation_split: float,
) -> tuple[Iterable[Dict[str, torch.Tensor]], Iterable[Dict[str, torch.Tensor]] | None]:
    """Hold out ``validation_split`` of a DataLoader as a validation loader.

    Returns ``(dataloader, None)`` unchanged when no split is requested or the
    input is not a DataLoader (warning logged in the latter case).
    """
    if validation_split <= 0.0:
        return dataloader, None

    if not (hasattr(dataloader, "dataset") and hasattr(dataloader, "batch_size")):
        logger.warning("validation_split > 0 but dataloader is not a DataLoader; skipping split")
        return dataloader, None

    dataset = dataloader.dataset
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    bs = dataloader.batch_size
    cf = getattr(dataloader, "collate_fn", None)
    ss = getattr(dataloader, "sampler", None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=(ss is None),
        collate_fn=cf,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=cf,
    )
    logger.info(
        "Split dataset: %d train, %d validation (%.1f%%)",
        train_size,
        val_size,
        validation_split * 100,
    )
    return train_loader, val_loader


def _build_lr_schedulers(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str | None,
    epochs: int,
) -> tuple[
    torch.optim.lr_scheduler.CosineAnnealingLR | None,
    torch.optim.lr_scheduler.ReduceLROnPlateau | None,
]:
    """Build the requested LR scheduler pair (at most one is non-None)."""
    cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
    plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if lr_scheduler == "cosine":
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return cosine_scheduler, plateau_scheduler


def _restore_checkpoint(
    model: MaskedPTMPrediction,
    optimizer: torch.optim.Optimizer,
    cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None,
    resume_from: str,
    device: str | torch.device,
) -> tuple[int, float, int, List[float]]:
    """Restore model/optimizer/scheduler state from a resume checkpoint.

    N20: 与 TD-M01 callbacks 恢复语义对齐（检查点缺字段时仅告警继续）。
    Returns ``(start_epoch, best_val_loss, epochs_since_improvement, history)``.
    """
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    history: List[float] = []

    state = safe_torch_load(resume_from, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and cosine_scheduler is not None:
            cosine_scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        epochs_since_improvement = int(state.get("epochs_since_improvement", 0))
        restored_history = state.get("history")
        if isinstance(restored_history, list):
            history = [float(v) for v in restored_history]
        logger.info(
            "Resumed %s from epoch %d (best_val_loss=%.6f)",
            resume_from,
            start_epoch,
            best_val_loss,
        )
    else:
        # 旧格式：裸 state_dict（仅权重）
        if not isinstance(state, dict):
            raise ValueError(f"Unrecognized checkpoint format for masked pretraining: {resume_from}")
        model.load_state_dict(state)
        logger.warning(
            "Legacy checkpoint %s contains only model weights; optimizer/scheduler/epoch state not restored",
            resume_from,
        )
    return start_epoch, best_val_loss, epochs_since_improvement, history


def _compute_masked_loss(
    model: MaskedPTMPrediction,
    batch: Dict[str, torch.Tensor],
    device: str | torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Mask one batch, run the model and return ``(loss, masked_positions)``.

    ``loss`` is ``None`` when no position was sampled for masking (the batch
    is skipped by callers without running a forward pass).
    """
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
        return None, masked_positions

    if use_amp:
        with amp_autocast():
            outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
            logits = outputs["logits"][masked_positions]
            targets = ptm_types[masked_positions]
            loss = F.cross_entropy(logits, targets)
    else:
        outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
        logits = outputs["logits"][masked_positions]
        targets = ptm_types[masked_positions]
        loss = F.cross_entropy(logits, targets)
    return loss, masked_positions


def _train_one_epoch(
    model: MaskedPTMPrediction,
    train_loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    epoch: int,
    epochs: int,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float | None,
    warmup_steps: int,
    initial_lr: float | None,
    log_interval: int,
) -> float:
    """Run one training epoch (AMP/grad-clip/warmup aware); return mean loss."""
    model.train()
    total_loss = 0.0
    steps = 0

    for batch_idx, batch in enumerate(train_loader):
        loss, masked_positions = _compute_masked_loss(model, batch, device, use_amp)
        if loss is None:
            continue

        optimizer.zero_grad()

        if use_amp:
            assert scaler is not None
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
                epoch + 1,
                epochs,
                batch_idx + 1,
                loss.item(),
            )

    return total_loss / max(steps, 1)


def _save_best_checkpoint(
    model: MaskedPTMPrediction,
    optimizer: torch.optim.Optimizer,
    cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None,
    checkpoint_dir: str,
    *,
    epoch: int,
    best_val_loss: float,
    epochs_since_improvement: int,
    history: List[float],
    training_config: Dict[str, object],
) -> None:
    """Save the best-model checkpoint with full resume state."""
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (cosine_scheduler.state_dict() if cosine_scheduler is not None else None),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_since_improvement": epochs_since_improvement,
            "config": training_config,
            "history": history,
        },
        ckpt_path,
    )
    logger.info("Saved best model to %s", ckpt_path)


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
    resume_from: str | None = None,
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
    resume_from : str or None
        Path to a checkpoint written by a previous ``pretrain_masked_ptm``
        run (or a legacy bare ``state_dict``). Restores model/optimizer/
        scheduler state, ``best_val_loss`` and epoch counter so training
        continues exactly where it stopped. Old-format checkpoints (bare
        ``state_dict``) are supported with a warning.
    """
    model.to(device)

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    train_loader, val_loader = _split_train_val(dataloader, validation_split)

    use_amp = use_amp and torch.cuda.is_available()
    scaler = make_grad_scaler() if use_amp else None

    cosine_scheduler, plateau_scheduler = _build_lr_schedulers(optimizer, lr_scheduler, epochs)

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

    if resume_from is not None:
        start_epoch, best_val_loss, epochs_since_improvement, history = _restore_checkpoint(
            model, optimizer, cosine_scheduler, resume_from, device
        )
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        if early_stop:
            break
        epoch_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            epochs=epochs,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=warmup_steps,
            initial_lr=initial_lr,
            log_interval=log_interval,
        )
        history.append(epoch_loss)
        logger.info("Epoch %d/%d training loss: %.4f", epoch + 1, epochs, epoch_loss)

        if cosine_scheduler is not None:
            cosine_scheduler.step()

        if val_loader is not None and (epoch + 1) % validate_every == 0:
            val_loss = _validate(model, val_loader, device, use_amp)
            logger.info("Epoch %d/%d validation loss: %.4f", epoch + 1, epochs, val_loss)

            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)

            # Checkpoint and early stopping share the same improvement test
            # (b < a - min_delta), consistent with callbacks.EarlyStopping.is_improved.
            improved = val_loss < best_val_loss - min_delta
            if improved:
                best_val_loss = val_loss
                if checkpoint_dir is not None:
                    _save_best_checkpoint(
                        model,
                        optimizer,
                        cosine_scheduler,
                        checkpoint_dir,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        epochs_since_improvement=epochs_since_improvement,
                        history=history,
                        training_config={
                            "epochs": epochs,
                            "use_amp": use_amp,
                            "validation_split": validation_split,
                            "validate_every": validate_every,
                            "lr_scheduler": lr_scheduler,
                            "warmup_steps": warmup_steps,
                            "grad_clip_norm": grad_clip_norm,
                            "log_interval": log_interval,
                            "patience": patience,
                            "min_delta": min_delta,
                        },
                    )

            if patience is not None:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    logger.info("早停等待中: %d/%d", epochs_since_improvement, patience)
                    if epochs_since_improvement >= patience:
                        logger.info("早停触发于 epoch %d", epoch + 1)
                        early_stop = True

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d LR: %.6f", epoch + 1, epochs, current_lr)

    return history


def pretrain_combined(
    model: MaskedPTMPrediction,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epochs: int = 10,
    contrastive_weight: float = 0.3,
    denoising_weight: float = 0.2,
    masked_weight: float = 1.0,
    checkpoint_dir: str | None = None,
    use_amp: bool = False,
    log_interval: int = 10,
    resume_from: str | None = None,
) -> List[Dict[str, float]]:
    """Combined pretraining with masked prediction, contrastive learning, and denoising.

    Runs all three pretraining objectives simultaneously with configurable weights.
    This provides richer pretraining than masked prediction alone.

    Parameters
    ----------
    contrastive_weight : float
        Weight for the contrastive learning loss component.
    denoising_weight : float
        Weight for the denoising autoencoder loss component.
    masked_weight : float
        Weight for the masked PTM prediction loss component (primary objective).
    resume_from : str or None
        Path to a checkpoint written by a previous ``pretrain_combined`` run.
        Restores all three module weights, optimizer state, ``best_loss`` and
        the epoch counter so training continues exactly where it stopped.
        Legacy checkpoints (weights only, written before this field existed)
        are supported with a warning.

    Returns
    -------
    List[Dict[str, float]]
        Per-epoch loss breakdown for each component.
    """
    model.to(device)

    num_ptm_types = model.num_ptm_types
    embed_dim = model.classifier.out_features if hasattr(model, "classifier") else 128

    contrastive_module = PTMContrastiveLearning(
        embed_dim=embed_dim,
    ).to(device)

    denoising_module = PTMDenoisingAutoEncoder(
        num_ptm_types=num_ptm_types,
        embed_dim=embed_dim,
    ).to(device)

    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    best_loss = float("inf")
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    history: List[Dict[str, float]] = []

    start_epoch = 0
    if resume_from is not None:
        # N20: 恢复三模块权重 + optimizer + best_loss + epoch 计数，
        # 使断点续训后 best 语义与 TD-M01 checkpoint 恢复一致。
        state = safe_torch_load(resume_from, map_location=device)
        if isinstance(state, dict) and "masked_model" in state:
            model.load_state_dict(state["masked_model"])
            contrastive_module.load_state_dict(state["contrastive_module"])
            denoising_module.load_state_dict(state["denoising_module"])
            if "optimizer" in state and state["optimizer"] is not None:
                # 新格式（2026-08-16 起）：完整训练状态
                optimizer.load_state_dict(state["optimizer"])
                start_epoch = int(state.get("epoch", -1)) + 1
                best_loss = float(state.get("best_loss", float("inf")))
                restored_history = state.get("history")
                if isinstance(restored_history, list):
                    history = [
                        {k: float(v) for k, v in entry.items()} for entry in restored_history if isinstance(entry, dict)
                    ]
                logger.info(
                    "Resumed %s from epoch %d (best_loss=%.6f)",
                    resume_from,
                    start_epoch,
                    best_loss,
                )
            else:
                # 旧格式：三模块裸权重（2026-08 前产物）
                logger.warning(
                    "Legacy checkpoint %s contains only module weights; optimizer/epoch state not restored",
                    resume_from,
                )
        else:
            raise ValueError(f"Unrecognized checkpoint format for combined pretraining: {resume_from}")

    for epoch in range(start_epoch, epochs):
        model.train()
        contrastive_module.train()
        denoising_module.train()

        epoch_losses = {"masked": 0.0, "contrastive": 0.0, "denoising": 0.0, "total": 0.0}
        steps = 0

        for batch_idx, batch in enumerate(dataloader):
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

            optimizer.zero_grad()

            # 1. Masked PTM prediction
            masked_ptm_types, masked_positions = model.mask_inputs(ptm_types, ptm_mask)
            if not masked_positions.any():
                continue

            outputs = model(masked_ptm_types, ptm_positions, ptm_mask)
            logits = outputs["logits"][masked_positions]
            targets = ptm_types[masked_positions]
            masked_loss = F.cross_entropy(logits, targets)

            # 2. Contrastive learning (if batch size > 1)
            contrastive_loss = torch.tensor(0.0, device=device)
            if ptm_types.shape[0] > 1:
                # Create positive pairs: original vs masked
                original_emb = outputs["logits"].detach()
                masked_emb = outputs["logits"]
                # Normalize to embeddings for contrastive
                # R-02: logits 为 [B, L, D]，按 L 维平均得到 [B, D] 样本向量
                # （原实现 unsqueeze 后 adaptive_avg_pool1d 收到 4 维张量必崩）
                anchor = original_emb.mean(dim=1)
                positive = masked_emb.mean(dim=1)
                if anchor.shape[-1] != contrastive_module.projector[0].in_features:
                    # Project to matching dim
                    min_dim = min(anchor.shape[-1], contrastive_module.projector[0].in_features)
                    anchor_proj = torch.zeros(
                        anchor.shape[0], contrastive_module.projector[0].in_features, device=device
                    )
                    positive_proj = torch.zeros_like(anchor_proj)
                    anchor_proj[:, :min_dim] = anchor[:, :min_dim]
                    positive_proj[:, :min_dim] = positive[:, :min_dim]
                    c_out = contrastive_module(anchor_proj, positive_proj)
                else:
                    c_out = contrastive_module(anchor, positive)
                contrastive_loss = c_out["contrastive_loss"]

            # 3. Denoising autoencoder
            # Convert PTM types to one-hot for denoising input
            ptm_onehot = F.one_hot(ptm_types.clamp(0, num_ptm_types - 1), num_classes=num_ptm_types).float()
            ptm_feature = ptm_onehot.mean(dim=1)  # [B, num_ptm_types]
            d_out = denoising_module(ptm_feature)
            denoising_loss = d_out["denoising_loss"]

            # Combined loss
            total_loss = (
                masked_weight * masked_loss + contrastive_weight * contrastive_loss + denoising_weight * denoising_loss
            )

            total_loss.backward()
            optimizer.step()

            epoch_losses["masked"] += masked_loss.item()
            epoch_losses["contrastive"] += contrastive_loss.item()
            epoch_losses["denoising"] += denoising_loss.item()
            epoch_losses["total"] += total_loss.item()
            steps += 1

            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d/%d, Batch %d, Loss: %.4f (masked=%.4f, cl=%.4f, denoise=%.4f)",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    total_loss.item(),
                    masked_loss.item(),
                    contrastive_loss.item(),
                    denoising_loss.item(),
                )

        for key in epoch_losses:
            epoch_losses[key] /= max(steps, 1)
        history.append(epoch_losses)

        logger.info(
            "Epoch %d/%d — total: %.4f (masked=%.4f, contrastive=%.4f, denoising=%.4f)",
            epoch + 1,
            epochs,
            epoch_losses["total"],
            epoch_losses["masked"],
            epoch_losses["contrastive"],
            epoch_losses["denoising"],
        )

        # Save best model
        if checkpoint_dir is not None and epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            ckpt_path = os.path.join(checkpoint_dir, "best_combined_model.pt")
            torch.save(
                {
                    "masked_model": model.state_dict(),
                    "contrastive_module": contrastive_module.state_dict(),
                    "denoising_module": denoising_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "config": {
                        "epochs": epochs,
                        "contrastive_weight": contrastive_weight,
                        "denoising_weight": denoising_weight,
                        "masked_weight": masked_weight,
                        "use_amp": use_amp,
                        "log_interval": log_interval,
                    },
                    "history": history,
                },
                ckpt_path,
            )
            logger.info("Saved best combined model to %s", ckpt_path)

    return history
