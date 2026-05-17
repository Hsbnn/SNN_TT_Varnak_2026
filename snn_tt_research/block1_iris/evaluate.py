"""Evaluation utilities for the Iris block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..common.device import pin_memory_flag
from ..common.spike import poisson_from_rates
from .config import IrisConfig
from .models import SNN_rf_stdp_hybrid


@torch.no_grad()
def eval_ann(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy of an ANN on the supplied loader."""
    pin = pin_memory_flag(device)
    model.eval()
    cor, n = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
        cor += (model(xb).argmax(1) == yb).sum().item()
        n += yb.size(0)
    return 100.0 * cor / n


@torch.no_grad()
def eval_snn(
    model: nn.Module,
    loader: DataLoader,
    cfg: IrisConfig,
    device: torch.device,
    runs: int = 8,
) -> float:
    """Mean SNN accuracy averaged over ``runs`` independent Poisson samples.

    Averaging stabilises the stochastic Poisson encoder; the prediction is
    taken from the argmax of the accumulated output counts.
    """
    pin = pin_memory_flag(device)
    model.eval()
    cor, n = 0, 0
    for rb, yb in loader:
        rb, yb = rb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
        acc = torch.zeros(yb.size(0), cfg.num_classes, device=device)
        for _ in range(runs):
            seq = poisson_from_rates(rb, cfg.t_steps, gain=cfg.rf_gain, max_prob=cfg.poisson_max_p)
            acc += model(seq)
        cor += (acc.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return 100.0 * cor / n


@torch.no_grad()
def eval_stdp_hybrid(
    model: SNN_rf_stdp_hybrid,
    loader: DataLoader,
    cfg: IrisConfig,
    device: torch.device,
    runs: int = 16,
) -> float:
    """Accuracy of the STDP hybrid model averaged over Poisson samples.

    The class is selected from the readout logits applied to the pooled
    hidden spike counts.
    """
    pin = pin_memory_flag(device)
    model.eval()
    cor, n = 0, 0
    for rb, yb in loader:
        rb, yb = rb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
        acc = torch.zeros(yb.size(0), cfg.num_classes, device=device)
        for _ in range(runs):
            seq = poisson_from_rates(rb, cfg.t_steps, gain=cfg.rf_gain, max_prob=cfg.poisson_max_p)
            _, hid_sum = model.forward_with_hidden_sum(seq)
            acc += F.linear(hid_sum, model.w2, model.b2)
        cor += (acc.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return 100.0 * cor / n
