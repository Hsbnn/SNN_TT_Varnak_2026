"""TT-SVD rank sweep and final TT-layer construction for the STDP encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ..common.metrics import dense_layer_mac, relative_error, tt_layer_mac_cached
from ..common.tt_decomp import (
    reconstruct_weight_tt,
    tt_core_num_params,
    tt_runtime_param_count_cached,
)
from .benchmark import benchmark_dense_vs_tt
from .config import MNISTStdpConfig
from .models import LIFHidden, LIFHiddenTensorTrain


@dataclass
class TTCandidate:
    """Single row of the TT rank sweep table."""

    r1: int
    r2: int
    rel_weight_err: float
    tt_core_params: int
    compression_ratio: float
    tt_ms_per_batch: float
    speedup_dense_over_tt: float
    hidden_rel_err_mean: float
    hidden_abs_diff_mean: float
    layer: LIFHiddenTensorTrain


def evaluate_tt_candidate(
    r1: int,
    r2: int,
    dense_weight: torch.Tensor,
    theta_vec: torch.Tensor,
    bench_batches: List[torch.Tensor],
    dense_layer: LIFHidden,
    cfg: MNISTStdpConfig,
    device: torch.device,
) -> TTCandidate:
    """Build one TT candidate, benchmark it and return the full diagnostic row.

    The function is intentionally side-effect free: every candidate is
    constructed from scratch so the sweep can be reproduced deterministically.
    """
    n_rf = cfg.n_rf
    layer_tt = LIFHiddenTensorTrain.from_dense_weight(
        n_in=n_rf,
        n_out=cfg.hidden,
        side=cfg.side,
        weight_row_major=dense_weight,
        theta_vec=theta_vec,
        r1=r1,
        r2=r2,
        cfg=cfg,
    ).to(device)

    W_hat = reconstruct_weight_tt(layer_tt.tt_G1, layer_tt.tt_G2, layer_tt.tt_G3)
    rel_w = relative_error(dense_weight.view(cfg.hidden, cfg.side, cfg.side), W_hat)

    bench = benchmark_dense_vs_tt(
        dense_layer, layer_tt, bench_batches, device, warmup=2, n_trials=2
    )
    core_params = tt_core_num_params(layer_tt.tt_G1, layer_tt.tt_G2, layer_tt.tt_G3)
    dense_params = cfg.hidden * n_rf

    return TTCandidate(
        r1=r1,
        r2=r2,
        rel_weight_err=rel_w,
        tt_core_params=core_params,
        compression_ratio=dense_params / max(core_params, 1),
        tt_ms_per_batch=bench["tt_ms_per_batch"],
        speedup_dense_over_tt=bench["speedup_dense_over_tt"],
        hidden_rel_err_mean=bench["hidden_rel_err_mean"],
        hidden_abs_diff_mean=bench["hidden_abs_diff_mean"],
        layer=layer_tt,
    )


def sweep_tt_ranks(
    dense_weight: torch.Tensor,
    theta_vec: torch.Tensor,
    bench_batches: List[torch.Tensor],
    dense_layer: LIFHidden,
    cfg: MNISTStdpConfig,
    device: torch.device,
) -> Tuple[List[TTCandidate], Tuple[int, int]]:
    """Evaluate every TT rank pair in ``cfg.tt_candidates``.

    The best rank pair is selected by the smallest mean relative error on
    pooled hidden spike counts, restricted to candidates whose stored size
    is at least 1.2× smaller than dense (i.e. compression is meaningful).
    """
    candidates: List[TTCandidate] = []
    for r1, r2 in cfg.tt_candidates:
        print(f"[TT sweep] ranks=({r1},{r2})")
        cand = evaluate_tt_candidate(
            r1, r2, dense_weight, theta_vec, bench_batches, dense_layer, cfg, device
        )
        candidates.append(cand)

    compressing = [c for c in candidates if c.compression_ratio > 1.2] or candidates
    best = min(compressing, key=lambda c: (c.hidden_rel_err_mean, c.rel_weight_err))
    return candidates, (best.r1, best.r2)


def build_final_tt_layer(
    dense_weight: torch.Tensor,
    theta_vec: torch.Tensor,
    r1: int,
    r2: int,
    cfg: MNISTStdpConfig,
    device: torch.device,
) -> Tuple[LIFHiddenTensorTrain, Dict[str, float]]:
    """Construct the TT layer at the selected ranks and report its statistics.

    Returns:
        ``(tt_layer, stats)`` where ``stats`` contains parameter counts, MAC
        estimates and the relative reconstruction error.
    """
    layer = LIFHiddenTensorTrain.from_dense_weight(
        n_in=cfg.n_rf,
        n_out=cfg.hidden,
        side=cfg.side,
        weight_row_major=dense_weight,
        theta_vec=theta_vec,
        r1=r1,
        r2=r2,
        cfg=cfg,
    ).to(device)

    W_hat = reconstruct_weight_tt(layer.tt_G1, layer.tt_G2, layer.tt_G3)
    rel_w = relative_error(dense_weight.view(cfg.hidden, cfg.side, cfg.side), W_hat)

    dense_params = cfg.hidden * cfg.n_rf
    tt_params = tt_core_num_params(layer.tt_G1, layer.tt_G2, layer.tt_G3)
    runtime_params = tt_runtime_param_count_cached(layer.tt_G1, layer.tt_G2, layer.tt_G3)

    stats = {
        "rel_weight_err": rel_w,
        "dense_params": dense_params,
        "tt_core_params": tt_params,
        "tt_runtime_params": runtime_params,
        "compression_ratio_cores": dense_params / max(tt_params, 1),
        "compression_ratio_runtime": dense_params / max(runtime_params, 1),
        "dense_mac": dense_layer_mac(cfg.hidden, cfg.n_rf),
        "tt_mac": tt_layer_mac_cached(cfg.hidden, cfg.n_rf, r1),
    }
    return layer, stats
