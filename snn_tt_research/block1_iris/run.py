"""End-to-end driver for the Iris efficiency block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from ..common.benchmarks import median_bench
from ..common.device import select_device, set_seed
from ..common.metrics import (
    count_params,
    dense_layer_mac,
    efficiency_index,
    tt_layer_mac_cached,
)
from ..common.spike import poisson_from_rates
from .config import IrisConfig
from .data import load_iris_dataloaders
from .evaluate import eval_ann, eval_snn, eval_stdp_hybrid
from .models import ANN_raw, ANN_rf, SNN_rf, SNN_rf_stdp_hybrid, SNN_rf_tt
from .train import (
    run_supervised_training,
    train_ann,
    train_snn,
    train_stdp_hybrid_epoch,
)


@dataclass
class Block1Report:
    """Numerical summary returned by :func:`run_block1`.

    Attributes:
        accuracies: test accuracy in percent for each model variant.
        param_counts: trainable parameter count for each model.
        latencies_ms: median single-batch inference latency in milliseconds.
        macs: estimated MAC counts of the first layer per training episode.
        efficiency: accuracy / 1000 parameters heuristic.
        stdp_accuracy: extra accuracy obtained by the STDP hybrid model.
    """

    accuracies: Dict[str, float]
    param_counts: Dict[str, int]
    latencies_ms: Dict[str, float]
    macs: Dict[str, int]
    efficiency: Dict[str, float]
    stdp_accuracy: float


def run_block1(cfg: IrisConfig | None = None, device: torch.device | None = None) -> Block1Report:
    """Train and benchmark all four Iris models, then run the STDP comparison.

    Steps:
        1. Build raw and RF DataLoaders.
        2. Train ANN_raw, ANN_rf, SNN_rf and SNN_rf_tt for ``cfg.epochs`` epochs.
        3. Train the STDP hybrid model with the same backbone.
        4. Measure inference latency on a fixed spike batch.
        5. Compute parameter counts, first-layer MAC estimates and an
           accuracy-per-kiloparameter efficiency index.
    """
    cfg = cfg or IrisConfig()
    device = device or select_device()
    set_seed(cfg.seed)
    print(f"[block1] device={device}")

    data = load_iris_dataloaders(cfg)

    def step_ann_raw(model, opt):
        """Single epoch closure for the raw-feature ANN."""
        return train_ann(model, data.raw_train, opt, device)

    def step_ann_rf(model, opt):
        """Single epoch closure for the RF-feature ANN."""
        return train_ann(model, data.rf_train, opt, device)

    def step_snn(model, opt):
        """Single epoch closure for SNN_rf and SNN_rf_tt."""
        return train_snn(model, data.rf_train, opt, cfg, device)

    def eval_raw(model):
        """Test-set evaluation closure for ANN_raw."""
        return eval_ann(model, data.raw_test, device)

    def eval_rf(model):
        """Test-set evaluation closure for ANN_rf."""
        return eval_ann(model, data.rf_test, device)

    def eval_snn_rf(model):
        """Test-set evaluation closure for any RF-input SNN."""
        return eval_snn(model, data.rf_test, cfg=cfg, device=device, runs=16)

    print("=== ANN_raw ===")
    m_raw, _ = run_supervised_training("ANN_raw", lambda: ANN_raw(cfg), step_ann_raw, eval_raw, cfg, device)
    print("=== ANN_rf ===")
    m_arf, _ = run_supervised_training("ANN_rf", lambda: ANN_rf(cfg), step_ann_rf, eval_rf, cfg, device)
    print("=== SNN_rf ===")
    m_snn, _ = run_supervised_training("SNN_rf", lambda: SNN_rf(cfg), step_snn, eval_snn_rf, cfg, device)
    print("=== SNN_rf_tt ===")
    m_stt, _ = run_supervised_training("SNN_rf_tt", lambda: SNN_rf_tt(cfg), step_snn, eval_snn_rf, cfg, device)

    acc_raw = eval_ann(m_raw, data.raw_test, device)
    acc_arf = eval_ann(m_arf, data.rf_test, device)
    acc_snn = eval_snn(m_snn, data.rf_test, cfg=cfg, device=device, runs=16)
    acc_stt = eval_snn(m_stt, data.rf_test, cfg=cfg, device=device, runs=16)

    set_seed(2026)
    print("=== SNN_rf STDP hybrid ===")
    m_stdp = SNN_rf_stdp_hybrid(cfg).to(device)
    opt_w2 = torch.optim.AdamW([m_stdp.w2, m_stdp.b2], lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_stdp_hybrid_epoch(m_stdp, data.rf_train, opt_w2, cfg, device)
        if ep == 1 or ep == cfg.epochs or ep % max(1, cfg.epochs // 5) == 0:
            te = eval_stdp_hybrid(m_stdp, data.rf_test, cfg=cfg, device=device, runs=12)
            print(f"  [STDP] ep {ep}/{cfg.epochs}  train_acc={tr_acc:.1f}%  test={te:.1f}%")
    acc_stdp = eval_stdp_hybrid(m_stdp, data.rf_test, cfg=cfg, device=device, runs=16)

    batch = min(cfg.batch_size, data.rf_test_tensor.shape[0])
    rates_fix = data.rf_test_tensor[:batch].to(device)
    raw_fix = data.raw_test_tensor[:batch].to(device)
    seq_fix = poisson_from_rates(rates_fix, cfg.t_steps, gain=cfg.rf_gain, max_prob=cfg.poisson_max_p)
    for m in (m_raw, m_arf, m_snn, m_stt):
        m.eval()

    t_raw = median_bench(lambda: m_raw(raw_fix), device, cfg.bench_warmup, cfg.bench_trials)
    t_arf = median_bench(lambda: m_arf(rates_fix), device, cfg.bench_warmup, cfg.bench_trials)
    t_sd = median_bench(lambda: m_snn(seq_fix), device, cfg.bench_warmup, cfg.bench_trials)
    t_st = median_bench(lambda: m_stt(seq_fix), device, cfg.bench_warmup, cfg.bench_trials)

    accuracies = {"ANN_raw": acc_raw, "ANN_rf": acc_arf, "SNN_rf": acc_snn, "SNN_rf_tt": acc_stt}
    param_counts = {
        "ANN_raw": count_params(m_raw),
        "ANN_rf": count_params(m_arf),
        "SNN_rf": count_params(m_snn),
        "SNN_rf_tt": count_params(m_stt),
    }
    latencies_ms = {
        "ANN_raw": t_raw * 1000.0,
        "ANN_rf": t_arf * 1000.0,
        "SNN_rf": t_sd * 1000.0,
        "SNN_rf_tt": t_st * 1000.0,
    }
    macs = {
        "SNN_rf_layer1_episode": cfg.t_steps * dense_layer_mac(cfg.hidden, cfg.n_rf, batch),
        "SNN_rf_tt_layer1_episode": cfg.t_steps * tt_layer_mac_cached(cfg.hidden, cfg.n_rf, cfg.r1, batch),
    }
    efficiency = {name: efficiency_index(accuracies[name], param_counts[name]) for name in accuracies}

    print("\n=== Block 1 summary (Iris) ===")
    for name in accuracies:
        print(
            f"  {name:10s}  acc={accuracies[name]:5.2f}%  "
            f"params={param_counts[name]:6d}  lat={latencies_ms[name]:7.3f} ms  "
            f"eff={efficiency[name]:6.3f}"
        )
    print(
        f"  STDP hybrid: acc={acc_stdp:.2f}%; "
        f"MAC ratio dense/TT (layer1, episode) = "
        f"{macs['SNN_rf_layer1_episode'] / max(1, macs['SNN_rf_tt_layer1_episode']):.2f}x; "
        f"bench median latency over {cfg.bench_trials} trials"
    )

    return Block1Report(
        accuracies=accuracies,
        param_counts=param_counts,
        latencies_ms=latencies_ms,
        macs=macs,
        efficiency=efficiency,
        stdp_accuracy=acc_stdp,
    )


if __name__ == "__main__":
    run_block1()
