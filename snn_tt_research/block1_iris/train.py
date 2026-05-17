"""Training loops for the Iris block (ANN, SNN, STDP hybrid)."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..common.device import pin_memory_flag, set_seed
from ..common.spike import poisson_from_rates
from .config import IrisConfig
from .models import SNN_rf_stdp_hybrid


def train_ann(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """One supervised epoch for an ANN: loss is cross-entropy on raw outputs.

    Returns the mean training loss and the training accuracy in percent.
    """
    pin = pin_memory_flag(device)
    model.train()
    tot, cor, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        tot += loss.item() * yb.size(0)
        cor += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return tot / n, 100.0 * cor / n


def train_snn(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: IrisConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """One supervised epoch for an SNN with Poisson-encoded RF inputs.

    Each mini-batch is converted into a fresh ``[T, B, D]`` spike train using
    the configured rate gain.  The cross-entropy loss is back-propagated
    through the surrogate-gradient spike function.
    """
    pin = pin_memory_flag(device)
    model.train()
    tot, cor, n = 0.0, 0, 0
    for rb, yb in loader:
        rb, yb = rb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
        seq = poisson_from_rates(rb, cfg.t_steps, gain=cfg.rf_gain, max_prob=cfg.poisson_max_p)
        opt.zero_grad()
        logits = model(seq)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        tot += loss.item() * yb.size(0)
        cor += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return tot / n, 100.0 * cor / n


def train_stdp_hybrid_epoch(
    model: SNN_rf_stdp_hybrid,
    loader: DataLoader,
    opt_w2: torch.optim.Optimizer,
    cfg: IrisConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """One epoch combining local STDP on ``w1`` with supervised updates on ``w2``.

    Steps per mini-batch:
        1. Sample a Poisson spike train from the RF intensities.
        2. Apply the STDP rule to the input weights (no gradients).
        3. Replay the network to collect pooled hidden spike counts.
        4. Train ``w2``/``b2`` with cross-entropy on those counts.
    """
    pin = pin_memory_flag(device)
    model.train()
    tot, cor, n = 0.0, 0, 0
    for rb, yb in loader:
        rb = rb.to(device, non_blocking=pin)
        yb = yb.to(device, non_blocking=pin)
        seq = poisson_from_rates(rb, cfg.t_steps, gain=cfg.rf_gain, max_prob=cfg.poisson_max_p)
        with torch.no_grad():
            model.stdp_update_w1(seq)
            _, hid_sum = model.forward_with_hidden_sum(seq)
        opt_w2.zero_grad()
        logits = F.linear(hid_sum, model.w2, model.b2)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_([model.w2, model.b2], 2.0)
        opt_w2.step()
        tot += loss.item() * yb.size(0)
        cor += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return tot / n, 100.0 * cor / n


def run_supervised_training(
    name: str,
    build_model: Callable[[], nn.Module],
    train_step: Callable[[nn.Module, torch.optim.Optimizer], Tuple[float, float]],
    eval_step: Callable[[nn.Module], float],
    cfg: IrisConfig,
    device: torch.device,
    log_every: int = 5,
):
    """Build, optimise and evaluate one network for ``cfg.epochs`` epochs.

    ``train_step`` and ``eval_step`` are closures that already capture the
    loader, the data shape and any auxiliary state, so a single loop drives
    both ANN and SNN experiments without per-model conditionals.
    """
    set_seed(42)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_step(model, opt)
        te_acc = eval_step(model)
        history.append((tr_loss, tr_acc, te_acc))
        if ep == 1 or ep == cfg.epochs or ep % max(1, cfg.epochs // log_every) == 0:
            print(f"  [{name}] ep {ep}/{cfg.epochs}  train_acc={tr_acc:.1f}%  test={te_acc:.1f}%")
    return model, history
