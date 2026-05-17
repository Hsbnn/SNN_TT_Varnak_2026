"""Latency benchmarks for full models and for the first layer in isolation."""

from __future__ import annotations

import statistics
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..common.benchmarks import median_bench
from ..common.device import device_sync
from .config import FashionConfig
from .models import (
    SNN_MLP,
    SNN_MLP_LowRank_Infer,
    SNN_MLP_TT_Infer,
    poisson_encode_images,
)


@torch.no_grad()
def build_real_eval_batches(
    loader, cfg: FashionConfig, device: torch.device, n_batches: int, seed: int = 999
):
    """Materialise ANN-shaped batches and matching Poisson spike batches.

    Both lists are aligned: the ``k``-th spike batch is sampled from the
    ``k``-th ANN batch, so each model variant sees the same underlying images.
    """
    pin = device.type == "cuda"
    ann_batches: List[torch.Tensor] = []
    snn_batches: List[torch.Tensor] = []
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for x, _ in loader:
        x = x.to(device, non_blocking=pin)
        ann_batches.append(x)
        snn_batches.append(poisson_encode_images(x, cfg))
        if len(ann_batches) >= n_batches:
            break
    return ann_batches, snn_batches


def benchmark_inference(
    ann,
    snn_d,
    snn_t_infer: SNN_MLP_TT_Infer,
    snn_lr_infer: SNN_MLP_LowRank_Infer,
    ann_batches,
    snn_batches,
    cfg: FashionConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Measure median per-batch inference latency for the four model variants.

    Returns a dict keyed by model name with values in seconds.
    """
    @torch.no_grad()
    def run_ann():
        """Run the ANN over every prepared ANN batch (one forward each)."""
        for x in ann_batches:
            ann(x)

    @torch.no_grad()
    def run(model):
        """Run an SNN over every prepared spike batch (full T-step rollout)."""
        for seq in snn_batches:
            model(seq)

    for _ in range(cfg.bench_warmup):
        run_ann()
        run(snn_d)
        run(snn_t_infer)
        run(snn_lr_infer)
        device_sync(device)

    def measure(fn):
        """Time ``fn`` once and return the per-batch median across trials."""
        samples = []
        for _ in range(cfg.bench_trials):
            device_sync(device)
            t0 = time.perf_counter()
            fn()
            device_sync(device)
            samples.append((time.perf_counter() - t0) / max(1, len(ann_batches)))
        return statistics.median(samples)

    return {
        "ann": measure(run_ann),
        "snn_d": measure(lambda: run(snn_d)),
        "snn_t": measure(lambda: run(snn_t_infer)),
        "snn_lr": measure(lambda: run(snn_lr_infer)),
    }


@torch.no_grad()
def _dense_layer1_step(z0: torch.Tensor, model: SNN_MLP) -> torch.Tensor:
    """Apply the dense input layer to a single spike step."""
    return F.linear(z0, model.w1, model.b1)


@torch.no_grad()
def _tt_layer1_step(z0: torch.Tensor, model: SNN_MLP_TT_Infer) -> torch.Tensor:
    """Apply the cached TT input layer to a single spike step."""
    h = F.linear(z0, model.W_mid)
    return F.linear(h, model.G1, model.b1)


@torch.no_grad()
def _lr_layer1_step(z0: torch.Tensor, model: SNN_MLP_LowRank_Infer) -> torch.Tensor:
    """Apply the low-rank input layer to a single spike step."""
    z1 = F.linear(z0, model.w_down)
    return F.linear(z1, model.w_up, model.b1)


def benchmark_layer1(
    snn_d: SNN_MLP,
    snn_t_infer: SNN_MLP_TT_Infer,
    snn_lr_infer: SNN_MLP_LowRank_Infer,
    seq_fix: torch.Tensor,
    cfg: FashionConfig,
    device: torch.device,
    repeats_step: int = 200,
    repeats_episode: int = 80,
) -> Dict[str, Dict[str, float]]:
    """Compare dense / TT / LR input layers per step and over a full episode.

    The single-step benchmark isolates the matrix multiplication cost; the
    episode benchmark amortises constant overheads over ``T`` rollouts.
    """
    z0 = seq_fix[0]
    Tn = seq_fix.shape[0]

    one_step = {
        "dense": median_bench(lambda: _dense_layer1_step(z0, snn_d), device, warmup=50, trials=repeats_step),
        "tt": median_bench(lambda: _tt_layer1_step(z0, snn_t_infer), device, warmup=50, trials=repeats_step),
        "lr": median_bench(lambda: _lr_layer1_step(z0, snn_lr_infer), device, warmup=50, trials=repeats_step),
    }

    def episode_dense():
        """Sum the dense layer over the whole episode."""
        out = 0.0
        for t in range(Tn):
            out = out + _dense_layer1_step(seq_fix[t], snn_d)
        return out

    def episode_tt():
        """Sum the TT layer over the whole episode."""
        out = 0.0
        for t in range(Tn):
            out = out + _tt_layer1_step(seq_fix[t], snn_t_infer)
        return out

    def episode_lr():
        """Sum the low-rank layer over the whole episode."""
        out = 0.0
        for t in range(Tn):
            out = out + _lr_layer1_step(seq_fix[t], snn_lr_infer)
        return out

    episode = {
        "dense": median_bench(episode_dense, device, warmup=20, trials=repeats_episode),
        "tt": median_bench(episode_tt, device, warmup=20, trials=repeats_episode),
        "lr": median_bench(episode_lr, device, warmup=20, trials=repeats_episode),
    }
    return {"one_step": one_step, "episode": episode}
