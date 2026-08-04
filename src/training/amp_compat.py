"""AMP compatibility helpers for PyTorch 2.x / 1.x.

Provides version-agnostic ``GradScaler`` and ``autocast`` factories that
prefer the modern ``torch.amp`` API (PyTorch >= 2.0) and fall back to the
deprecated ``torch.cuda.amp`` API on older versions.

These were originally duplicated in ``trainers.py`` and ``self_supervised.py``
(see TD-M7).  Keeping them in a single module avoids code drift and makes the
version-compat logic easier to update when the minimum-supported PyTorch
version eventually rises.
"""

from typing import Any


def make_grad_scaler() -> Any:
    """Construct a CUDA AMP ``GradScaler`` using the modern API when available.

    PyTorch deprecated ``torch.cuda.amp.GradScaler`` in favour of
    ``torch.amp.GradScaler("cuda")`` (the new API takes a ``device`` string).
    The new form landed in PyTorch 2.0+; we fall back to the legacy form on
    older versions so the project keeps working on its full declared range.

    We resolve the legacy class via ``getattr`` so that static type checkers
    do not flag the deprecated attribute on installs where it still exists
    but is hidden behind a deprecation shim.
    """
    import torch

    if hasattr(torch.amp, "GradScaler"):
        # Modern API (PyTorch >= 2.0). The string arg selects the device.
        return torch.amp.GradScaler("cuda")
    legacy = getattr(torch.cuda.amp, "GradScaler", None)
    if legacy is None:  # pragma: no cover - very old torch
        raise RuntimeError("torch.amp.GradScaler unavailable on this PyTorch")
    return legacy()


def amp_autocast() -> Any:
    """Return an autocast context manager for CUDA AMP (version-compat).

    Mirrors :func:`make_grad_scaler`: prefer the non-deprecated
    ``torch.amp.autocast(device_type="cuda")`` (PyTorch >= 2.0) and fall back
    to the legacy ``torch.cuda.amp.autocast()`` on older versions.
    """
    import torch

    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    legacy = getattr(torch.cuda.amp, "autocast", None)
    if legacy is None:  # pragma: no cover - very old torch
        raise RuntimeError("torch.amp.autocast unavailable on this PyTorch")
    return legacy()
