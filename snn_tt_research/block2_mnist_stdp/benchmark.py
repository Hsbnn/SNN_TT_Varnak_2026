"""Latency and approximation-error benchmarks for the dense vs TT hidden layer."""

from __future__ import annotations

import statistics
import time
from typing import Dict, List

import numpy as np
import torch

from ..common.device import device_sync
from .models import LIFHidden, LIFHiddenTensorTrain, rollout_hidden_layer


def benchmark_dense_vs_tt(
    dense_h: LIFHidden,
    tt_h: LIFHiddenTensorTrain,
    batches: List[torch.Tensor],
    device: torch.device,
    warmup: int,
    n_trials: int,
) -> Dict[str, float]:
    """Compare dense and TT hidden layers on the same real Poisson batches.

    Measured quantities:
        * median latency per batch for each variant,
        * speed-up of dense over TT,
        * relative and absolute error of the pooled hidden spike counts.
    """
    @torch.no_grad()
    def one_run(h):
        """Time and return ``(per-batch seconds, list of outputs)`` for ``h``."""
        device_sync(device)
        t0 = time.perf_counter()
        outs = []
        for x_seq in batches:
            outs.append(rollout_hidden_layer(h, x_seq))
        device_sync(device)
        return (time.perf_counter() - t0) / len(batches), outs

    for _ in range(warmup):
        one_run(dense_h)
        one_run(tt_h)

    dense_times: List[float] = []
    tt_times: List[float] = []
    dense_ref = None
    tt_ref = None
    for _ in range(n_trials):
        td, out_d = one_run(dense_h)
        tt, out_t = one_run(tt_h)
        dense_times.append(td)
        tt_times.append(tt)
        dense_ref = out_d
        tt_ref = out_t

    med_d = statistics.median(dense_times)
    med_t = statistics.median(tt_times)

    rel_errs = []
    abs_diff = []
    for a, b in zip(dense_ref, tt_ref):
        rel = (a - b).norm() / (a.norm() + 1e-8)
        rel_errs.append(float(rel.item()))
        abs_diff.append(float((a - b).abs().mean().item()))

    return {
        "dense_ms_per_batch": med_d * 1000.0,
        "tt_ms_per_batch": med_t * 1000.0,
        "speedup_dense_over_tt": med_d / med_t,
        "hidden_rel_err_mean": float(np.mean(rel_errs)),
        "hidden_abs_diff_mean": float(np.mean(abs_diff)),
    }
