"""Classification metrics and plots for the Fashion-MNIST surrogate-gradient block."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..common.device import pin_memory_flag
from ..common.evaluation import ClassificationReport, classification_report, report_table
from ..common.plots import write_all_plots
from .config import FashionConfig
from .models import poisson_encode_images


FASHION_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@torch.no_grad()
def predict_ann(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run an ANN over ``loader`` and return ``(y_true, y_pred, mean_logits)``.

    ``mean_logits`` is the raw logit matrix for top-k accuracy computation.
    """
    pin = pin_memory_flag(device)
    model.eval()
    all_y, all_pred, all_logits = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    return (
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_logits, axis=0),
    )


@torch.no_grad()
def predict_snn(
    model: nn.Module,
    loader: DataLoader,
    cfg: FashionConfig,
    device: torch.device,
    n_runs: Optional[int] = None,
):
    """Run an SNN over ``loader`` averaging logits across ``n_runs`` Poisson trials.

    Returns:
        ``(y_true, y_pred, mean_logits)`` as numpy arrays on the CPU.
    """
    pin = pin_memory_flag(device)
    runs = n_runs if n_runs is not None else cfg.eval_runs_snn
    model.eval()
    all_y, all_pred, all_logits = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=pin)
        acc = torch.zeros(x.size(0), cfg.num_classes, device=device)
        for _ in range(runs):
            acc += model(poisson_encode_images(x, cfg))
        acc = acc / max(runs, 1)
        all_logits.append(acc.cpu().numpy())
        all_pred.append(acc.argmax(dim=1).cpu().numpy())
        all_y.append(y.cpu().numpy())
    return (
        np.concatenate(all_y),
        np.concatenate(all_pred),
        np.concatenate(all_logits, axis=0),
    )


def build_fashion_reports(
    ann: nn.Module,
    snn_dense: nn.Module,
    snn_tt: nn.Module,
    snn_lr: nn.Module,
    test_loader: DataLoader,
    cfg: FashionConfig,
    device: torch.device,
) -> Dict[str, ClassificationReport]:
    """Compute a full classification report for every model variant."""
    reports: Dict[str, ClassificationReport] = {}

    y, p, logits = predict_ann(ann, test_loader, device)
    reports["ann"] = classification_report(y, p, logits=logits, class_names=FASHION_CLASS_NAMES)

    for name, model in (("snn_dense", snn_dense), ("snn_tt", snn_tt), ("snn_lowrank", snn_lr)):
        y, p, logits = predict_snn(model, test_loader, cfg, device)
        reports[name] = classification_report(y, p, logits=logits, class_names=FASHION_CLASS_NAMES)

    return reports


def export_fashion_reports(
    reports: Dict[str, ClassificationReport],
    plots_dir: str,
    history: Optional[Dict[str, list]] = None,
    resource_points: Optional[Dict[str, Dict[str, float]]] = None,
    latencies_ms: Optional[Dict[str, float]] = None,
    speedups: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Print a metrics table for the four variants and write every plot to disk."""
    print("\n=== Classification metrics (Fashion-MNIST) ===")
    print(report_table(reports))
    return write_all_plots(
        plots_dir=plots_dir,
        reports=reports,
        history=history,
        resource_points=resource_points,
        latencies_ms=latencies_ms,
        speedups=speedups,
        title_prefix="Fashion | ",
    )
