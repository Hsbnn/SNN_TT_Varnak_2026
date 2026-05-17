"""Build classification reports and plots for the MNIST STDP block."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..common.evaluation import ClassificationReport, classification_report, report_table
from ..common.plots import write_all_plots


MNIST_CLASS_NAMES = [str(i) for i in range(10)]


@torch.no_grad()
def readout_logits_and_labels(
    readout: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
):
    """Push a feature tensor through the readout head and return numpy arrays.

    Returns:
        ``(y_true, y_pred, logits)`` all as ``numpy`` arrays on CPU.
    """
    readout.eval()
    logits = readout(X.float().to(device))
    pred = logits.argmax(dim=1)
    return (
        y.long().cpu().numpy(),
        pred.cpu().numpy(),
        logits.cpu().numpy(),
    )


def build_reports(
    eval_specs: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, ClassificationReport]:
    """Produce a :class:`ClassificationReport` for each named evaluation.

    Args:
        eval_specs: mapping ``name -> {"readout": nn.Module, "X": Tensor, "y": Tensor}``.
        device: device on which the readout is executed.

    Returns:
        Mapping ``name -> ClassificationReport``.
    """
    out: Dict[str, ClassificationReport] = {}
    for name, spec in eval_specs.items():
        y_true, y_pred, logits = readout_logits_and_labels(
            spec["readout"], spec["X"], spec["y"], device
        )
        out[name] = classification_report(
            y_true,
            y_pred,
            logits=logits,
            class_names=MNIST_CLASS_NAMES,
        )
    return out


def export_reports(
    reports: Dict[str, ClassificationReport],
    plots_dir: str,
    history: Optional[Dict[str, list]] = None,
    resource_points: Optional[Dict[str, Dict[str, float]]] = None,
    latencies_ms: Optional[Dict[str, float]] = None,
    speedups: Optional[Dict[str, float]] = None,
    title_prefix: str = "MNIST | ",
) -> Dict[str, str]:
    """Print a text table of headline numbers and write every plot to ``plots_dir``."""
    print("\n=== Classification metrics (MNIST) ===")
    print(report_table(reports))
    return write_all_plots(
        plots_dir=plots_dir,
        reports=reports,
        history=history,
        resource_points=resource_points,
        latencies_ms=latencies_ms,
        speedups=speedups,
        title_prefix=title_prefix,
    )
