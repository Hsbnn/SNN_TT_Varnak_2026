"""Spike-count and RF feature extraction pipelines."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from tqdm.auto import tqdm

from ..common.device import pin_memory_flag
from .data import MNISTRFEncoder


@torch.no_grad()
def spike_features(
    encoder,
    rates: torch.Tensor,
    rf_encoder: MNISTRFEncoder,
    T: int,
    n_avg: int,
) -> torch.Tensor:
    """Average hidden spike counts across ``n_avg`` Poisson trials.

    Returned values are total spike counts per neuron (not normalised by
    ``T``) because the larger dynamic range is more informative for the
    readout head.
    """
    B = rates.size(0)
    acc = torch.zeros(B, encoder.n_hidden, device=rates.device, dtype=rates.dtype)
    for _ in range(n_avg):
        seq = rf_encoder.poisson(rates, T)
        acc = acc + encoder.forward_spikes(seq)
    return acc / float(n_avg)


@torch.no_grad()
def collect_feature_blocks(
    encoder,
    loader: Iterable,
    rf_encoder: MNISTRFEncoder,
    T: int,
    n_avg: int,
    device: torch.device,
    desc: str = "Features",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iterate the loader, collecting spike features, RF rates and labels.

    Returns three CPU tensors so the same arrays can be reused across
    several readout-head trainings without holding GPU memory.
    """
    pin = pin_memory_flag(device)
    encoder.eval()
    X_sp, X_rf, ys = [], [], []
    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=pin)
        rates = rf_encoder.image_to_rates(x)
        sp = spike_features(encoder, rates, rf_encoder, T, n_avg)
        X_sp.append(sp.cpu())
        X_rf.append(rates.cpu())
        ys.append(y.long().clone())
    return torch.cat(X_sp, dim=0), torch.cat(X_rf, dim=0), torch.cat(ys, dim=0)
