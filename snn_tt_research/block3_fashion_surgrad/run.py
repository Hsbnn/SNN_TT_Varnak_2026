"""End-to-end driver for the Fashion-MNIST surrogate-gradient block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from ..common.device import select_device, set_seed
from ..common.evaluation import ClassificationReport
from ..common.metrics import count_params, dense_layer_mac, lowrank_layer_mac, tt_layer_mac_cached
from .benchmark import benchmark_inference, benchmark_layer1, build_real_eval_batches
from .config import FashionConfig
from .data import build_fashion_loaders, make_shared_initial_tensors
from .metrics_report import build_fashion_reports, export_fashion_reports
from .models import SNN_MLP_LowRank_Infer, SNN_MLP_TT_Infer
from .train import fit_all_models


@dataclass
class Block3Report:
    """Numerical summary returned by :func:`run_block3`.

    Attributes:
        test_accuracies: best-by-val test accuracy per model variant.
        param_counts: trainable parameter count per model variant.
        layer1_params: parameter count of the first layer per variant.
        latencies_s: median per-batch inference latency in seconds.
        layer1_bench: results of the isolated first-layer benchmark.
        layer1_macs: estimated first-layer MACs per training episode.
    """

    test_accuracies: Dict[str, float]
    param_counts: Dict[str, int]
    layer1_params: Dict[str, int]
    latencies_s: Dict[str, float]
    layer1_bench: Dict[str, Dict[str, float]]
    layer1_macs: Dict[str, int]
    reports: Dict[str, ClassificationReport]
    plot_paths: Dict[str, str]


def run_block3(
    cfg: FashionConfig | None = None,
    device: torch.device | None = None,
    data_root: str = "data",
    plots_dir: str = "plots/block3",
) -> Block3Report:
    """Train ANN/SNN/SNN+TT/SNN+LR with shared init and benchmark them.

    Pipeline:
        1. Build Fashion-MNIST loaders with a stratified validation split.
        2. Sample the shared ``(w1, b1, w2, b2)`` and train all four models.
        3. Restore best-by-validation checkpoints and measure test accuracy.
        4. Build inference-only TT and LowRank wrappers.
        5. Run the full-model and the isolated first-layer benchmarks.
    """
    cfg = cfg or FashionConfig()
    device = device or select_device()
    set_seed(cfg.seed)
    print(f"[block3] device={device}")

    loaders = build_fashion_loaders(cfg, device, data_root=data_root)
    w1, b1, w2, b2 = make_shared_initial_tensors(cfg)

    trained, history, test_accs = fit_all_models(loaders, w1, b1, w2, b2, cfg, device)
    ann = trained["ann"]
    snn_d = trained["snn_d"]
    snn_t = trained["snn_t"]
    snn_lr = trained["snn_lr"]

    snn_t_infer = SNN_MLP_TT_Infer.from_trainable(snn_t).to(device)
    snn_lr_infer = SNN_MLP_LowRank_Infer.from_trainable(snn_lr).to(device)
    for model in (ann, snn_d, snn_t_infer, snn_lr_infer):
        model.eval()

    ann_batches, snn_batches = build_real_eval_batches(
        loaders.test_loader, cfg, device, cfg.bench_batches, seed=999
    )
    latencies = benchmark_inference(
        ann, snn_d, snn_t_infer, snn_lr_infer, ann_batches, snn_batches, cfg, device
    )
    layer1 = benchmark_layer1(snn_d, snn_t_infer, snn_lr_infer, snn_batches[0], cfg, device)

    param_counts = {
        "ann": count_params(ann),
        "snn_d": count_params(snn_d),
        "snn_t": count_params(snn_t),
        "snn_lr": count_params(snn_lr),
    }
    layer1_params = {
        "snn_d": cfg.hidden * cfg.flat + cfg.hidden,
        "snn_t": cfg.hidden * cfg.r1 + cfg.r1 * cfg.side * cfg.r2 + cfg.r2 * cfg.side + cfg.hidden,
        "snn_lr": min(cfg.rank_matrix, cfg.hidden, cfg.flat) * (cfg.flat + cfg.hidden) + cfg.hidden,
    }
    rank_eff = min(cfg.rank_matrix, cfg.hidden, cfg.flat)
    layer1_macs = {
        "snn_d_episode": cfg.t_steps * dense_layer_mac(cfg.hidden, cfg.flat, cfg.batch_size),
        "snn_t_episode": cfg.t_steps * tt_layer_mac_cached(cfg.hidden, cfg.flat, cfg.r1, cfg.batch_size),
        "snn_lr_episode": cfg.t_steps * lowrank_layer_mac(cfg.hidden, cfg.flat, rank_eff, cfg.batch_size),
    }

    print("\n=== Block 3 summary (Fashion-MNIST) ===")
    for name in ("ann", "snn_d", "snn_t", "snn_lr"):
        print(
            f"  {name:8s}  acc={test_accs[name]:5.2f}%  "
            f"params={param_counts[name]:7d}  lat={latencies[name]*1000:7.3f} ms"
        )
    print(
        f"  layer1 speedup dense/TT one_step="
        f"{layer1['one_step']['dense'] / max(1e-12, layer1['one_step']['tt']):.2f}x  "
        f"episode={layer1['episode']['dense'] / max(1e-12, layer1['episode']['tt']):.2f}x"
    )

    reports = build_fashion_reports(ann, snn_d, snn_t_infer, snn_lr_infer, loaders.test_loader, cfg, device)
    name_map = {"ann": "ann", "snn_d": "snn_dense", "snn_t": "snn_tt", "snn_lr": "snn_lowrank"}
    resource_points = {
        report_name: {
            "params": float(param_counts[short]),
            "accuracy": reports[report_name].accuracy,
        }
        for short, report_name in name_map.items()
    }
    latencies_ms = {report_name: latencies[short] * 1000.0 for short, report_name in name_map.items()}
    ann_dense_latency = latencies["ann"]
    snn_dense_latency = latencies["snn_d"]
    speedups = {
        "ann": ann_dense_latency / max(1e-12, latencies["ann"]),
        "snn_dense": 1.0,
        "snn_tt": snn_dense_latency / max(1e-12, latencies["snn_t"]),
        "snn_lowrank": snn_dense_latency / max(1e-12, latencies["snn_lr"]),
    }
    plot_paths = export_fashion_reports(
        reports,
        plots_dir=plots_dir,
        history=history,
        resource_points=resource_points,
        latencies_ms=latencies_ms,
        speedups=speedups,
    )

    return Block3Report(
        test_accuracies=test_accs,
        param_counts=param_counts,
        layer1_params=layer1_params,
        latencies_s=latencies,
        layer1_bench=layer1,
        layer1_macs=layer1_macs,
        reports=reports,
        plot_paths=plot_paths,
    )


if __name__ == "__main__":
    run_block3()
