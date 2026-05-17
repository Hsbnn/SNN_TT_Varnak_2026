"""Training loops and joint best-by-validation routine for all four models."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..common.device import pin_memory_flag
from .config import FashionConfig
from .data import FashionLoaders
from .models import (
    ANN_MLP,
    SNN_MLP,
    SNN_MLP_LowRank,
    SNN_MLP_TT,
    poisson_encode_images,
)


def train_epoch_ann(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device
) -> Tuple[float, float]:
    """One supervised cross-entropy epoch for an ANN."""
    pin = pin_memory_flag(device)
    model.train()
    tot, cor, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item() * y.size(0)
        cor += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return tot / n, 100.0 * cor / n


def train_epoch_snn(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: FashionConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """One epoch of surrogate-gradient SNN training with Poisson-encoded inputs."""
    pin = pin_memory_flag(device)
    model.train()
    tot, cor, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)
        seq = poisson_encode_images(x, cfg)
        opt.zero_grad()
        logits = model(seq)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item() * y.size(0)
        cor += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return tot / n, 100.0 * cor / n


@torch.no_grad()
def eval_ann(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy of an ANN on the given loader."""
    pin = pin_memory_flag(device)
    model.eval()
    cor, n = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)
        cor += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return 100.0 * cor / n


@torch.no_grad()
def eval_snn(
    model: nn.Module,
    loader: DataLoader,
    cfg: FashionConfig,
    device: torch.device,
    n_runs: int | None = None,
) -> float:
    """SNN accuracy averaged over ``n_runs`` independent Poisson samples."""
    pin = pin_memory_flag(device)
    n_runs = n_runs if n_runs is not None else cfg.eval_runs_snn
    model.eval()
    cor, n = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        y = y.to(device, non_blocking=pin)
        acc = torch.zeros(x.size(0), cfg.num_classes, device=device)
        for _ in range(n_runs):
            acc += model(poisson_encode_images(x, cfg))
        cor += (acc.argmax(1) == y).sum().item()
        n += y.size(0)
    return 100.0 * cor / n


def fit_all_models(
    loaders: FashionLoaders,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    cfg: FashionConfig,
    device: torch.device,
):
    """Train ANN, SNN dense, SNN+TT and SNN+LR sharing the same initialisation.

    A cosine learning-rate schedule is used for each model, validation is
    measured every epoch, and the best-by-validation checkpoint is restored
    before returning.

    Returns:
        ``(models, val_history, test_accuracies)`` where ``test_accuracies``
        is a dict containing one number per variant.
    """
    ann = ANN_MLP(cfg).to(device)
    snn_d = SNN_MLP.from_shared_init(cfg, w1, b1, w2, b2).to(device)
    snn_t = SNN_MLP_TT.from_shared_init(cfg, w1, b1, w2, b2).to(device)
    snn_lr = SNN_MLP_LowRank.from_shared_init(cfg, w1, b1, w2, b2).to(device)

    opt_a = torch.optim.AdamW(ann.parameters(), lr=cfg.lr_ann, weight_decay=cfg.wd_ann)
    opt_d = torch.optim.AdamW(snn_d.parameters(), lr=cfg.lr_snn, weight_decay=cfg.wd_snn)
    opt_t = torch.optim.AdamW(snn_t.parameters(), lr=cfg.lr_snn, weight_decay=cfg.wd_snn)
    opt_lr = torch.optim.AdamW(snn_lr.parameters(), lr=cfg.lr_snn, weight_decay=cfg.wd_snn)

    schedulers = {
        "ann": torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=cfg.epochs),
        "snn_d": torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=cfg.epochs),
        "snn_t": torch.optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=cfg.epochs),
        "snn_lr": torch.optim.lr_scheduler.CosineAnnealingLR(opt_lr, T_max=cfg.epochs),
    }

    models = {
        "ann": (ann, opt_a, schedulers["ann"]),
        "snn_d": (snn_d, opt_d, schedulers["snn_d"]),
        "snn_t": (snn_t, opt_t, schedulers["snn_t"]),
        "snn_lr": (snn_lr, opt_lr, schedulers["snn_lr"]),
    }

    best_states: Dict[str, dict] = {}
    best_vals: Dict[str, float] = {k: -1.0 for k in models}
    history: Dict[str, list] = {k: [] for k in models}

    for ep in range(1, cfg.epochs + 1):
        train_loader_ep = loaders.make_train_loader(ep)
        for name, (model, opt, sched) in models.items():
            if name == "ann":
                tr_loss, tr_acc = train_epoch_ann(model, train_loader_ep, opt, device)
                val_acc = eval_ann(model, loaders.val_loader, device)
            else:
                tr_loss, tr_acc = train_epoch_snn(model, train_loader_ep, opt, cfg, device)
                val_acc = eval_snn(model, loaders.val_loader, cfg, device)
            sched.step()
            history[name].append((tr_loss, tr_acc, val_acc))
            if val_acc > best_vals[name]:
                best_vals[name] = val_acc
                best_states[name] = copy.deepcopy(model.state_dict())

        print(
            f"ep {ep:02d}/{cfg.epochs} | "
            f"ANN val={history['ann'][-1][2]:.2f}% | "
            f"SNN dense val={history['snn_d'][-1][2]:.2f}% | "
            f"SNN+TT val={history['snn_t'][-1][2]:.2f}% | "
            f"SNN+LR val={history['snn_lr'][-1][2]:.2f}%"
        )

    for name, (model, _, _) in models.items():
        model.load_state_dict(best_states[name])

    test_accs = {
        "ann": eval_ann(ann, loaders.test_loader, device),
        "snn_d": eval_snn(snn_d, loaders.test_loader, cfg, device),
        "snn_t": eval_snn(snn_t, loaders.test_loader, cfg, device),
        "snn_lr": eval_snn(snn_lr, loaders.test_loader, cfg, device),
    }
    trained = {name: model for name, (model, _, _) in models.items()}
    return trained, history, test_accs
