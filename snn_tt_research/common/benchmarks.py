"""Latency benchmark primitives used in all three research blocks."""

from __future__ import annotations

import statistics
import time
from typing import Callable, List

import torch

from .device import device_sync
from .spike import poisson_from_rates_with_generator


def median_bench(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int = 10,
    trials: int = 7,
) -> float:
    """Run a callable repeatedly and return the median wall-clock time in seconds.

    The device is synchronised before and after each measurement so that
    asynchronous CUDA/MPS work is fully accounted for.
    """
    for _ in range(warmup):
        fn()
        device_sync(device)
    samples: List[float] = []
    for _ in range(trials):
        device_sync(device)
        t0 = time.perf_counter()
        fn()
        device_sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


@torch.no_grad()
def build_real_spike_batches(
    loader,
    n_batches: int,
    T: int,
    device: torch.device,
    rate_fn: Callable[[torch.Tensor], torch.Tensor],
    gain: float,
    max_prob: float,
    seed: int = 2026,
) -> List[torch.Tensor]:
    """Materialise a fixed list of real Poisson spike sequences from a loader.

    Both the dense baseline and the compressed model are evaluated on these
    identical batches, which is required for a fair side-by-side benchmark.

    Args:
        loader: torch ``DataLoader`` over the underlying dataset.
        n_batches: number of spike batches to collect.
        T: simulation length in time steps.
        device: target device for the returned tensors.
        rate_fn: callable mapping a batch of inputs to intensities.
        gain: Poisson rate scaling factor.
        max_prob: clamp on per-step Bernoulli probability.
        seed: RNG seed for reproducibility.
    """
    pin = device.type == "cuda"
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    batches: List[torch.Tensor] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=pin)
        rates = rate_fn(x)
        seq = poisson_from_rates_with_generator(rates, T, g, gain=gain, max_prob=max_prob)
        batches.append(seq)
        if len(batches) >= n_batches:
            break
    return batches
