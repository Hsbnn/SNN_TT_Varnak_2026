"""End-to-end driver for the MNIST STDP + TT-compression block."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..common.benchmarks import build_real_spike_batches
from ..common.device import select_device, set_seed
from ..common.evaluation import ClassificationReport
from .benchmark import benchmark_dense_vs_tt
from .config import MNISTStdpConfig
from .data import build_mnist_loaders
from .features import collect_feature_blocks
from .metrics_report import build_reports, export_reports
from .models import STDP, SNNEncoder, SNNEncoderTTWrapper
from .readout import BlockNormalizer
from .train import evaluate_readout, stdp_train_epoch, train_readout_head
from .tt_compression import build_final_tt_layer, sweep_tt_ranks


@dataclass
class Block2Report:
    """Numerical summary returned by :func:`run_block2`.

    Attributes:
        ablation_val: best validation accuracy for each ablation mode.
        ablation_test: test accuracy for each ablation mode.
        chosen_ranks: TT ranks selected by the sweep.
        tt_stats: reconstruction error and parameter / MAC summary.
        bench: dense vs TT inference benchmark.
        acc_dense_same: dense encoder accuracy on its own readout.
        acc_tt_same: TT encoder accuracy on the dense-trained readout.
        acc_tt_retrained: TT encoder accuracy after retraining the readout.
    """

    ablation_val: Dict[str, float]
    ablation_test: Dict[str, float]
    chosen_ranks: tuple
    tt_stats: Dict[str, float]
    bench: Dict[str, float]
    acc_dense_same: float
    acc_tt_same: float
    acc_tt_retrained: float
    reports: Dict[str, ClassificationReport]
    plot_paths: Dict[str, str]


def run_block2(
    cfg: MNISTStdpConfig | None = None,
    device: torch.device | None = None,
    data_root: str = "data",
    plots_dir: str = "plots/block2",
) -> Block2Report:
    """Train the STDP encoder, the readouts, then evaluate the TT compression.

    Pipeline:
        1. Load MNIST and build the Gaussian RF projection.
        2. Train the LIF + STDP encoder unsupervised.
        3. Collect spike + RF features and train three readout ablations.
        4. Sweep the TT ranks on real Poisson spike batches.
        5. Build the final TT layer at the chosen ranks.
        6. Benchmark dense vs TT inference latency and approximation error.
        7. Compare end-to-end accuracy under the strict and practical
           evaluation protocols.
    """
    cfg = cfg or MNISTStdpConfig()
    device = device or select_device()
    set_seed(cfg.seed)
    print(f"[block2] device={device}")

    train_loader, test_loader, rf_encoder = build_mnist_loaders(cfg, device, data_root=data_root)

    encoder = SNNEncoder(cfg.n_rf, cfg.hidden, cfg).to(device)
    stdp = STDP(encoder.hidden, cfg)

    print("=== A: STDP ===")
    t0 = time.perf_counter()
    for ep in range(1, cfg.stdp_epochs + 1):
        diag = stdp_train_epoch(encoder, stdp, train_loader, rf_encoder, cfg, device)
        print(
            f"  STDP {ep}/{cfg.stdp_epochs}  "
            f"rate={diag['mean_hidden_spike_rate_pct']:.2f}%  "
            f"theta={diag['theta_mean']:.3f}±{diag['theta_std']:.3f}"
        )
    print(f"  STDP wall-clock: {time.perf_counter() - t0:.1f} s")

    print("\n=== B: spike features + readout ablations ===")
    X_sp, X_rf, y_all = collect_feature_blocks(
        encoder, train_loader, rf_encoder, cfg.sim_time, cfg.feature_avg_train, device, desc="Features train"
    )
    idx = np.arange(len(y_all))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42, stratify=y_all.numpy())

    norm = BlockNormalizer()
    norm.fit(torch.log1p(X_sp[train_idx]), X_rf[train_idx])

    modes = ["rf_only", "spikes_only", "spikes_plus_rf"]
    readouts: Dict[str, torch.nn.Module] = {}
    val_results: Dict[str, float] = {}
    for mode in modes:
        print(f"\n--- readout: {mode} ---")
        X_tr = norm.transform(X_sp[train_idx], X_rf[train_idx], mode)
        X_val = norm.transform(X_sp[val_idx], X_rf[val_idx], mode)
        readout, best_val = train_readout_head(
            mode, X_tr, y_all[train_idx], X_val, y_all[val_idx], encoder.n_hidden, cfg.n_rf, cfg, device
        )
        readouts[mode] = readout
        val_results[mode] = best_val

    print("\n=== Eval on test ===")
    X_sp_te, X_rf_te, y_te = collect_feature_blocks(
        encoder, test_loader, rf_encoder, cfg.sim_time, cfg.feature_avg_test, device, desc="Features test"
    )
    test_results: Dict[str, float] = {}
    ablation_features: Dict[str, Dict[str, torch.Tensor]] = {}
    for mode in modes:
        X_te = norm.transform(X_sp_te, X_rf_te, mode)
        test_results[mode] = evaluate_readout(readouts[mode], X_te, y_te, device, tag=f"Test ({mode})")
        ablation_features[f"ablation/{mode}"] = {"readout": readouts[mode], "X": X_te, "y": y_te}

    print("\n=== C1: TT rank sweep ===")
    sweep_batches = build_real_spike_batches(
        test_loader,
        cfg.tt_sweep_batches,
        cfg.sim_time,
        device,
        rate_fn=rf_encoder.image_to_rates,
        gain=cfg.input_gain,
        max_prob=cfg.poisson_max_prob,
        seed=3030,
    )
    candidates, (best_r1, best_r2) = sweep_tt_ranks(
        encoder.hidden.weight.data.clone(),
        encoder.hidden.theta.data.clone(),
        sweep_batches,
        encoder.hidden,
        cfg,
        device,
    )
    print(f"Selected TT ranks: r1={best_r1}, r2={best_r2}")

    print("\n=== C2: final TT layer ===")
    tt_layer, tt_stats = build_final_tt_layer(
        encoder.hidden.weight.data,
        encoder.hidden.theta.data,
        best_r1,
        best_r2,
        cfg,
        device,
    )
    print(
        f"  dense_params={tt_stats['dense_params']:,}  "
        f"tt_core_params={tt_stats['tt_core_params']:,}  "
        f"compression_cores={tt_stats['compression_ratio_cores']:.2f}x  "
        f"rel_weight_err={tt_stats['rel_weight_err']:.4f}"
    )

    print("\n=== C3: dense vs TT benchmark ===")
    real_batches = build_real_spike_batches(
        test_loader,
        cfg.tt_final_bench_batches,
        cfg.sim_time,
        device,
        rate_fn=rf_encoder.image_to_rates,
        gain=cfg.input_gain,
        max_prob=cfg.poisson_max_prob,
        seed=2026,
    )
    bench = benchmark_dense_vs_tt(
        encoder.hidden, tt_layer, real_batches, device, cfg.bench_warmup, cfg.bench_trials
    )
    print(
        f"  dense={bench['dense_ms_per_batch']:.3f} ms  "
        f"tt={bench['tt_ms_per_batch']:.3f} ms  "
        f"speedup={bench['speedup_dense_over_tt']:.2f}x  "
        f"hidden_rel_err={bench['hidden_rel_err_mean']:.4f}"
    )

    print("\n=== C4: strict evaluation (same readout) ===")
    tt_encoder = SNNEncoderTTWrapper(tt_layer).to(device)
    set_seed(12345)
    X_sp_d, X_rf_d, y_d = collect_feature_blocks(
        encoder, test_loader, rf_encoder, cfg.sim_time, cfg.feature_avg_test, device, desc="Test dense"
    )
    X_te_dense_main = norm.transform(X_sp_d, X_rf_d, cfg.tt_main_mode)
    acc_dense_same = evaluate_readout(
        readouts[cfg.tt_main_mode], X_te_dense_main, y_d, device, tag=f"Test dense ({cfg.tt_main_mode})"
    )
    set_seed(12345)
    X_sp_t, X_rf_t, y_t = collect_feature_blocks(
        tt_encoder, test_loader, rf_encoder, cfg.sim_time, cfg.feature_avg_test, device, desc="Test TT"
    )
    X_te_tt_main = norm.transform(X_sp_t, X_rf_t, cfg.tt_main_mode)
    acc_tt_same = evaluate_readout(
        readouts[cfg.tt_main_mode], X_te_tt_main, y_t, device, tag=f"Test TT same readout ({cfg.tt_main_mode})"
    )

    metric_specs: Dict[str, Dict[str, torch.Tensor]] = dict(ablation_features)
    metric_specs["dense_encoder"] = {"readout": readouts[cfg.tt_main_mode], "X": X_te_dense_main, "y": y_d}
    metric_specs["tt_encoder_same_readout"] = {"readout": readouts[cfg.tt_main_mode], "X": X_te_tt_main, "y": y_t}

    acc_tt_retrained = acc_tt_same
    if cfg.tt_retrain_readout:
        print("\n=== C5: practical evaluation (retrained readout) ===")
        set_seed(54321)
        X_sp_tr, X_rf_tr, y_tr = collect_feature_blocks(
            tt_encoder, train_loader, rf_encoder, cfg.sim_time, cfg.tt_feature_avg_train, device, desc="Train TT"
        )
        idx_tt = np.arange(len(y_tr))
        tr_idx, va_idx = train_test_split(idx_tt, test_size=0.1, random_state=42, stratify=y_tr.numpy())
        tt_norm = BlockNormalizer()
        tt_norm.fit(torch.log1p(X_sp_tr[tr_idx]), X_rf_tr[tr_idx])
        tt_readout, _ = train_readout_head(
            cfg.tt_main_mode,
            tt_norm.transform(X_sp_tr[tr_idx], X_rf_tr[tr_idx], cfg.tt_main_mode),
            y_tr[tr_idx],
            tt_norm.transform(X_sp_tr[va_idx], X_rf_tr[va_idx], cfg.tt_main_mode),
            y_tr[va_idx],
            encoder.n_hidden,
            cfg.n_rf,
            cfg,
            device,
        )
        set_seed(54321)
        X_sp_te2, X_rf_te2, y_te2 = collect_feature_blocks(
            tt_encoder, test_loader, rf_encoder, cfg.sim_time, cfg.tt_feature_avg_test, device, desc="Test TT (retrained)"
        )
        X_te_tt_retr = tt_norm.transform(X_sp_te2, X_rf_te2, cfg.tt_main_mode)
        acc_tt_retrained = evaluate_readout(
            tt_readout, X_te_tt_retr, y_te2, device, tag=f"Test TT retrained ({cfg.tt_main_mode})"
        )
        metric_specs["tt_encoder_retrained"] = {"readout": tt_readout, "X": X_te_tt_retr, "y": y_te2}

    reports = build_reports(metric_specs, device)
    dense_params = tt_stats["dense_params"]
    tt_params = tt_stats["tt_core_params"]
    resource_points = {
        "dense_encoder": {
            "params": float(dense_params),
            "accuracy": reports["dense_encoder"].accuracy,
        },
        "tt_encoder_same_readout": {
            "params": float(tt_params),
            "accuracy": reports["tt_encoder_same_readout"].accuracy,
        },
    }
    if "tt_encoder_retrained" in reports:
        resource_points["tt_encoder_retrained"] = {
            "params": float(tt_params),
            "accuracy": reports["tt_encoder_retrained"].accuracy,
        }
    latencies_ms = {
        "dense": bench["dense_ms_per_batch"],
        "tt": bench["tt_ms_per_batch"],
    }
    speedups = {
        "dense": 1.0,
        "tt": bench["speedup_dense_over_tt"],
    }
    plot_paths = export_reports(
        reports,
        plots_dir=plots_dir,
        resource_points=resource_points,
        latencies_ms=latencies_ms,
        speedups=speedups,
        title_prefix="MNIST | ",
    )

    return Block2Report(
        ablation_val=val_results,
        ablation_test=test_results,
        chosen_ranks=(best_r1, best_r2),
        tt_stats=tt_stats,
        bench=bench,
        acc_dense_same=acc_dense_same,
        acc_tt_same=acc_tt_same,
        acc_tt_retrained=acc_tt_retrained,
        reports=reports,
        plot_paths=plot_paths,
    )


if __name__ == "__main__":
    run_block2()
