"""STDP unsupervised training and supervised readout training loops."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..common.device import pin_memory_flag
from .config import MNISTStdpConfig
from .data import MNISTRFEncoder
from .models import STDP, SNNEncoder
from .readout import build_readout


@torch.no_grad()
def stdp_train_epoch(
    encoder: SNNEncoder,
    stdp: STDP,
    loader: DataLoader,
    rf_encoder: MNISTRFEncoder,
    cfg: MNISTStdpConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run one unsupervised STDP epoch over the train loader.

    For every mini-batch the LIF layer is rolled out for ``cfg.sim_time``
    steps; the STDP rule is invoked at each step and the per-neuron threshold
    is updated homeostatically toward :attr:`MNISTStdpConfig.target_rate`.

    Returns:
        Diagnostics including the mean hidden firing rate and the mean/std
        of the adaptive threshold.
    """
    pin = pin_memory_flag(device)
    encoder.train()
    spike_frac = 0.0
    per_neuron_rate = torch.zeros(encoder.hidden.n_out, device=device)
    n_batches = 0

    for x, _ in tqdm(loader, desc="STDP"):
        x = x.to(device, non_blocking=pin)
        B = x.size(0)
        rates = rf_encoder.image_to_rates(x)
        x_seq = rf_encoder.poisson(rates, cfg.sim_time)

        mem, _ = encoder.hidden.reset(B, x_seq.device)
        stdp.init_traces(B, x_seq.device)

        batch_spikes = torch.zeros(B, encoder.hidden.n_out, device=x_seq.device)
        for t in range(cfg.sim_time):
            pre = x_seq[t]
            post, mem = encoder.hidden(pre, mem)
            stdp.step(pre, post)
            batch_spikes += post

        r = batch_spikes.mean(dim=0) / cfg.sim_time
        encoder.hidden.theta.data += cfg.theta_lr * (r - cfg.target_rate)
        encoder.hidden.theta.data.clamp_(cfg.theta_min, cfg.theta_max)

        spike_frac += batch_spikes.sum().item() / (B * cfg.sim_time * encoder.hidden.n_out)
        per_neuron_rate += r
        n_batches += 1

    return {
        "mean_hidden_spike_rate_pct": 100.0 * spike_frac / max(n_batches, 1),
        "theta_mean": float(encoder.hidden.theta.mean().item()),
        "theta_std": float(encoder.hidden.theta.std().item()),
        "per_neuron_rate": (per_neuron_rate / max(n_batches, 1)).detach().cpu(),
    }


def train_readout_head(
    mode: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_hidden: int,
    n_rf: int,
    cfg: MNISTStdpConfig,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    """Supervised cross-entropy training of the readout head.

    Best-by-validation-accuracy weights are restored before returning.

    Returns:
        ``(readout, best_val_acc)``.
    """
    readout = build_readout(mode, n_hidden, n_rf, device)
    opt = torch.optim.AdamW(readout.parameters(), lr=cfg.readout_lr, weight_decay=cfg.readout_wd)

    ds = TensorDataset(X_train.float().to(device), y_train.long().to(device))
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    X_val = X_val.float().to(device)
    y_val = y_val.long().to(device)

    best_state = None
    best_val = -1.0

    for ep in range(1, cfg.readout_epochs + 1):
        readout.train()
        tot, cor, n = 0.0, 0, 0
        for xb, yb in dl:
            opt.zero_grad()
            logits = readout(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), cfg.readout_clip)
            opt.step()
            pred = logits.argmax(dim=1)
            cor += (pred == yb).sum().item()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)

        readout.eval()
        with torch.no_grad():
            val_pred = readout(X_val).argmax(dim=1)
            val_acc = 100.0 * (val_pred == y_val).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(readout.state_dict())

        if ep == 1 or ep == cfg.readout_epochs or ep % 5 == 0:
            print(
                f"  readout {ep}/{cfg.readout_epochs}  "
                f"loss={tot/n:.4f}  train_acc={100*cor/n:.2f}%  val_acc={val_acc:.2f}%"
            )

    readout.load_state_dict(best_state)
    return readout, best_val


@torch.no_grad()
def evaluate_readout(readout: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, tag: str = "Test") -> float:
    """Top-1 accuracy of a readout head on a feature tensor and its labels."""
    readout.eval()
    X = X.float().to(device)
    y = y.long().to(device)
    pred = readout(X).argmax(dim=1)
    acc = 100.0 * (pred == y).float().mean().item()
    print(f"{tag} accuracy: {acc:.2f}%")
    return acc
