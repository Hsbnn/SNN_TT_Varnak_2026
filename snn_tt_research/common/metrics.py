"""Parameter and MAC counters used in the efficiency tables."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dense_layer_mac(n_out: int, n_in: int, batch: int = 1) -> int:
    """MAC count for a single dense linear layer (multiply-add counted once)."""
    return n_out * n_in * batch


def tt_layer_mac_cached(n_out: int, n_in: int, r1: int, batch: int = 1) -> int:
    """MAC count for the cached TT layer (``z → h → y``)."""
    return batch * (n_in * r1 + r1 * n_out)


def lowrank_layer_mac(n_out: int, n_in: int, r: int, batch: int = 1) -> int:
    """MAC count for an SVD low-rank layer ``z → z_r → y``."""
    return batch * (n_in * r + r * n_out)


def efficiency_index(accuracy_pct: float, n_params: int) -> float:
    """Heuristic efficiency: accuracy in percent per kilo-parameter."""
    return accuracy_pct * 1000.0 / max(n_params, 1)


def relative_error(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    """Relative Frobenius error ``||A - B|| / ||A||`` as a Python float."""
    return float(((reference - approximation).norm() / (reference.norm() + 1e-8)).item())
