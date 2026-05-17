"""Device selection, deterministic seeding and stream synchronisation helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device() -> torch.device:
    """Return the best available torch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch (CPU/CUDA/MPS) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def device_sync(device: torch.device) -> None:
    """Block until queued device work is finished so timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def pin_memory_flag(device: torch.device) -> bool:
    """Return whether DataLoader should use pinned memory on this device."""
    return device.type == "cuda"
