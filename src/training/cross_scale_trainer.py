"""Dedicated cross-scale trainer with self-describing exact-resume checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class CrossScaleTrainingError(RuntimeError):
    """Raised when training cannot produce a valid finite result."""


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _rng_state() -> Dict[str, Any]:
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": {
            "kind": numpy_state[0],
            # torch serialization does not support uint32 storages on every
            # supported version, so persist MT19937 words losslessly as int64.
            "state": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping) and isinstance(numpy_state.get("state"), Tensor):
        np.random.set_state(
            (
                str(numpy_state["kind"]),
                numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    cpu_state = state.get("torch_cpu")
    if isinstance(cpu_state, Tensor):
        torch.set_rng_state(cpu_state.cpu())
    cuda_states = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


class CrossScaleTrainer:
    """Train/evaluate ``CrossScalePTM2CellNet`` without standard-head assumptions."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Optional[Any] = None,
        device: str | torch.device = "cpu",
        gradient_clip_norm: float = 1.0,
        accumulation_steps: int = 1,
        mixed_precision: bool = False,
        artifact_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if gradient_clip_norm < 0 or accumulation_steps <= 0:
            raise ValueError("gradient_clip_norm must be non-negative and accumulation_steps positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CrossScaleTrainingError("请求了 CUDA，但当前环境不可用")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.accumulation_steps = int(accumulation_steps)
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.artifact_context = dict(artifact_context or {})
        self.start_epoch = 0
        self.best_val_loss = math.inf
        self.history: list[Dict[str, float]] = []

    def _run_epoch(self, loader: DataLoader[Any], *, training: bool) -> Dict[str, float]:
        self.model.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "delta_expression_rmse": 0.0, "cell_state_accuracy": 0.0}
        sample_count = 0
        steps = len(loader)
        if steps == 0:
            raise CrossScaleTrainingError("数据加载器为空")
        for step, raw_batch in enumerate(loader):
            inputs = _move(raw_batch["inputs"], self.device)
            targets = _move(raw_batch["targets"], self.device)
            batch_size = int(targets["cell_state"].shape[0])
            with torch.set_grad_enabled(training), torch.autocast(
                device_type=self.device.type,
                enabled=self.mixed_precision,
            ):
                outputs = self.model(inputs)
                loss = self.model.compute_loss(outputs, targets)
                scaled_loss = loss / self.accumulation_steps
            if not torch.isfinite(loss):
                raise CrossScaleTrainingError(f"检测到非有限 loss: {float(loss.detach().cpu())}")
            if training:
                self.scaler.scale(scaled_loss).backward()
                should_step = (step + 1) % self.accumulation_steps == 0 or step + 1 == steps
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            delta = outputs["delta_expression"].detach()
            state_logits = outputs["cell_state_logits"].detach()
            rmse = torch.sqrt(torch.mean((delta - targets["delta_expression"]) ** 2))
            accuracy = (state_logits.argmax(dim=-1) == targets["cell_state"].long()).float().mean()
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["delta_expression_rmse"] += float(rmse.cpu()) * batch_size
            totals["cell_state_accuracy"] += float(accuracy.cpu()) * batch_size
            sample_count += batch_size
        return {key: value / sample_count for key, value in totals.items()}

    def evaluate(self, loader: DataLoader[Any]) -> Dict[str, float]:
        return self._run_epoch(loader, training=False)

    def _checkpoint_payload(self, epoch: int) -> Dict[str, Any]:
        return {
            "checkpoint_schema_version": "ptm2cellnet.cross-scale.checkpoint.v1",
            "epoch": int(epoch),
            "best_val_loss": float(self.best_val_loss),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "rng_state": _rng_state(),
            "history": list(self.history),
            "artifact_context": dict(self.artifact_context),
            "model_info": self.model.get_model_info(),
        }

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
        torch.save(self._checkpoint_payload(epoch), temporary)
        temporary.replace(destination)

    def load_checkpoint(self, path: str | Path) -> Dict[str, Any]:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"跨尺度 checkpoint 不存在: {checkpoint_path}")
        try:
            payload = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CrossScaleTrainingError(f"无法安全加载 checkpoint {checkpoint_path}: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("checkpoint_schema_version") != "ptm2cellnet.cross-scale.checkpoint.v1":
            raise CrossScaleTrainingError("checkpoint schema 不受支持")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler_state = payload.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        scaler_state = payload.get("scaler_state_dict")
        if isinstance(scaler_state, Mapping):
            self.scaler.load_state_dict(dict(scaler_state))
        self.start_epoch = int(payload["epoch"]) + 1
        self.best_val_loss = float(payload.get("best_val_loss", math.inf))
        self.history = [dict(item) for item in payload.get("history", [])]
        rng_state = payload.get("rng_state")
        if isinstance(rng_state, Mapping):
            _restore_rng_state(rng_state)
        return dict(payload)

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        max_epochs: int,
        output_dir: str | Path,
        patience: int = 10,
    ) -> Dict[str, Any]:
        if max_epochs <= 0 or patience < 0:
            raise ValueError("max_epochs must be positive and patience non-negative")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, max_epochs):
            sampler = getattr(train_loader, "sampler", None)
            set_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self.evaluate(val_loader)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            improved = val_metrics["loss"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            record = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.append(record)
            self.save_checkpoint(output / "last.pt", epoch)
            if improved:
                self.save_checkpoint(output / "best.pt", epoch)
            if patience and epochs_without_improvement >= patience:
                break
        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "last_epoch": int(self.history[-1]["epoch"]) if self.history else self.start_epoch - 1,
        }

    def write_artifact_manifest(
        self,
        output_dir: str | Path,
        *,
        test_metrics: Mapping[str, float],
    ) -> Path:
        output = Path(output_dir)
        payload = {
            "artifact_schema_version": "ptm2cellnet.cross-scale.artifact.v1",
            "model_type": "cross_scale",
            "best_checkpoint": "best.pt",
            "last_checkpoint": "last.pt",
            "best_val_loss": self.best_val_loss,
            "test_metrics": dict(test_metrics),
            "history": self.history,
            **self.artifact_context,
        }
        destination = output / "artifact_manifest.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return destination


__all__ = ["CrossScaleTrainingError", "CrossScaleTrainer"]
