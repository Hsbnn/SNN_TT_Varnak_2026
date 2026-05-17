"""Matplotlib plotting helpers shared by the MNIST and Fashion-MNIST blocks.

All functions take a ``save_path`` argument — when supplied, the figure is
written to disk (the parent directory is created on the fly) instead of being
returned for interactive display.  ``show=True`` displays the figure.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - matplotlib is an optional runtime dep
    plt = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .evaluation import ClassificationReport


def _ensure_matplotlib() -> None:
    """Raise a helpful error if matplotlib is not available at runtime."""
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting helpers but failed to import: "
            f"{_IMPORT_ERROR}"
        )


def _finalize(fig, save_path: Optional[str], show: bool):
    """Save and/or display ``fig``; close it when only saving."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    report: ClassificationReport,
    title: str = "Confusion matrix",
    normalize: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Render a confusion matrix as a heatmap with cell annotations.

    Args:
        report: classification report produced by
            :func:`snn_tt_research.common.evaluation.classification_report`.
        title: figure title.
        normalize: if ``True``, divide each row by its support so cells show
            recall-style fractions in ``[0, 1]``.
        save_path: where to write the figure (``None`` to skip saving).
        show: if ``True``, call ``plt.show`` after rendering.
    """
    _ensure_matplotlib()
    cm = report.confusion.astype(np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sum, 1.0))
    n = cm.shape[0]

    fig, ax = plt.subplots(figsize=(0.6 + 0.45 * n, 0.6 + 0.45 * n))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=cm.max() if not normalize else 1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(report.class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(report.class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fmt = "{:.2f}" if normalize else "{:.0f}"
    threshold = 0.5 * (cm.max() if cm.max() > 0 else 1.0)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                fmt.format(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    _finalize(fig, save_path, show)


def plot_per_class_bars(
    report: ClassificationReport,
    title: str = "Per-class precision / recall / F1",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot per-class precision, recall and F1 side by side as grouped bars."""
    _ensure_matplotlib()
    n = len(report.class_names)
    x = np.arange(n)
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * n), 4.0))
    ax.bar(x - width, report.per_class_precision, width=width, label="precision", color="#4477aa")
    ax.bar(x, report.per_class_recall, width=width, label="recall", color="#228833")
    ax.bar(x + width, report.per_class_f1, width=width, label="F1", color="#ee7733")
    ax.set_xticks(x)
    ax.set_xticklabels(report.class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _finalize(fig, save_path, show)


def plot_model_comparison(
    metrics_by_model: Dict[str, Dict[str, float]],
    metric_keys: Sequence[str] = ("accuracy", "precision_macro", "recall_macro", "f1_macro"),
    title: str = "Models comparison",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Grouped bar chart comparing several scalar metrics across model variants.

    Args:
        metrics_by_model: mapping ``{model_name: {metric_name: value, ...}}``.
        metric_keys: which metrics to display, in plotting order.
        title: figure title.
        save_path: where to save the figure (``None`` to skip).
        show: if ``True``, call ``plt.show``.
    """
    _ensure_matplotlib()
    models = list(metrics_by_model.keys())
    n_models = len(models)
    n_metrics = len(metric_keys)

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * n_models, 4.2))
    width = 0.8 / n_metrics
    x = np.arange(n_models)
    palette = ["#4477aa", "#228833", "#ee7733", "#cc6677", "#882255", "#117733"]
    for i, key in enumerate(metric_keys):
        vals = [metrics_by_model[m].get(key, float("nan")) for m in models]
        offsets = x + (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(offsets, vals, width=width, label=key, color=palette[i % len(palette)])
        for off, v in zip(offsets, vals):
            if np.isfinite(v):
                ax.text(off, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _finalize(fig, save_path, show)


def plot_efficiency_scatter(
    points: Dict[str, Dict[str, float]],
    x_key: str = "params",
    y_key: str = "accuracy",
    title: str = "Accuracy vs parameter count",
    log_x: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Scatter plot of model accuracy versus a resource axis (params, MAC, latency)."""
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = ["#4477aa", "#228833", "#ee7733", "#cc6677", "#882255", "#117733"]
    for i, (name, p) in enumerate(points.items()):
        ax.scatter(p[x_key], p[y_key], color=palette[i % len(palette)], s=85, label=name)
        ax.annotate(name, (p[x_key], p[y_key]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _finalize(fig, save_path, show)


def plot_training_curves(
    history: Dict[str, List],
    metric_index: int = 2,
    metric_name: str = "val accuracy %",
    title: str = "Training curves",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot one curve per model from a training history dictionary.

    ``history[name]`` is expected to be a list of ``(train_loss, train_acc, val_acc)``
    tuples; the function selects column ``metric_index``.
    """
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    palette = ["#4477aa", "#228833", "#ee7733", "#cc6677", "#882255", "#117733"]
    for i, (name, rows) in enumerate(history.items()):
        if not rows:
            continue
        xs = np.arange(1, len(rows) + 1)
        ys = [row[metric_index] for row in rows]
        ax.plot(xs, ys, marker="o", markersize=3, color=palette[i % len(palette)], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _finalize(fig, save_path, show)


def plot_latency_vs_speedup(
    latencies_ms: Dict[str, float],
    speedups: Dict[str, float],
    title: str = "Inference latency and speedup over dense",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Twin-axis bar chart: latency in milliseconds and speedup over dense baseline."""
    _ensure_matplotlib()
    names = list(latencies_ms.keys())
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(names), 4.2))
    bars = ax.bar(x - 0.18, [latencies_ms[n] for n in names], width=0.36, color="#4477aa", label="latency (ms)")
    for bar, value in zip(bars, [latencies_ms[n] for n in names]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("ms / batch")
    ax.set_title(title)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.18, [speedups[n] for n in names], width=0.36, color="#ee7733", label="speedup ×")
    for bar, value in zip(bars2, [speedups[n] for n in names]):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}×", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("speedup over dense")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    _finalize(fig, save_path, show)


def write_all_plots(
    plots_dir: str,
    reports: Dict[str, ClassificationReport],
    history: Optional[Dict[str, List]] = None,
    resource_points: Optional[Dict[str, Dict[str, float]]] = None,
    latencies_ms: Optional[Dict[str, float]] = None,
    speedups: Optional[Dict[str, float]] = None,
    title_prefix: str = "",
) -> Dict[str, str]:
    """Convenience wrapper that writes the standard plot set for a research block.

    Returns a mapping ``{plot_name: save_path}`` for downstream logging.
    """
    _ensure_matplotlib()
    os.makedirs(plots_dir, exist_ok=True)
    written: Dict[str, str] = {}

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for name, r in reports.items():
        cm_path = os.path.join(plots_dir, f"confusion_{name}.png")
        plot_confusion_matrix(r, title=f"{title_prefix}{name}: confusion matrix", save_path=cm_path)
        written[f"confusion_{name}"] = cm_path

        per_path = os.path.join(plots_dir, f"per_class_{name}.png")
        plot_per_class_bars(r, title=f"{title_prefix}{name}: per-class metrics", save_path=per_path)
        written[f"per_class_{name}"] = per_path

        metrics_by_model[name] = {
            "accuracy": r.accuracy,
            "precision_macro": r.precision_macro,
            "recall_macro": r.recall_macro,
            "f1_macro": r.f1_macro,
            "balanced_accuracy": r.balanced_accuracy,
        }

    if metrics_by_model:
        cmp_path = os.path.join(plots_dir, "models_comparison.png")
        plot_model_comparison(
            metrics_by_model,
            title=f"{title_prefix}models comparison",
            save_path=cmp_path,
        )
        written["models_comparison"] = cmp_path

    if history is not None and history:
        curves_path = os.path.join(plots_dir, "training_curves.png")
        plot_training_curves(history, title=f"{title_prefix}validation accuracy per epoch", save_path=curves_path)
        written["training_curves"] = curves_path

    if resource_points is not None and resource_points:
        eff_path = os.path.join(plots_dir, "accuracy_vs_params.png")
        plot_efficiency_scatter(resource_points, title=f"{title_prefix}accuracy vs params", save_path=eff_path)
        written["accuracy_vs_params"] = eff_path

    if latencies_ms is not None and speedups is not None and latencies_ms:
        lat_path = os.path.join(plots_dir, "latency_speedup.png")
        plot_latency_vs_speedup(latencies_ms, speedups, title=f"{title_prefix}latency vs speedup", save_path=lat_path)
        written["latency_speedup"] = lat_path

    return written
