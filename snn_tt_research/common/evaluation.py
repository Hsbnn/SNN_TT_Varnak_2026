"""Classification metrics used across the MNIST and Fashion-MNIST blocks.

This module wraps :mod:`sklearn.metrics` to produce a compact, JSON-friendly
report (macro precision/recall/F1, Cohen's kappa, balanced accuracy, top-k
accuracy, per-class numbers and a confusion matrix) that the run scripts pass
both to the printers and to the plot helpers in :mod:`.plots`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


@dataclass
class ClassificationReport:
    """JSON-friendly classification summary for a single model.

    Attributes:
        accuracy: top-1 accuracy in percent.
        precision_macro: unweighted mean of per-class precisions, in percent.
        recall_macro: unweighted mean of per-class recalls, in percent.
        f1_macro: unweighted mean of per-class F1, in percent.
        precision_weighted: support-weighted precision, in percent.
        recall_weighted: support-weighted recall, in percent.
        f1_weighted: support-weighted F1, in percent.
        balanced_accuracy: mean of per-class recalls, in percent.
        cohen_kappa: Cohen's kappa coefficient (chance-corrected agreement).
        top_2_accuracy: top-2 accuracy in percent (None if no logits supplied).
        top_5_accuracy: top-5 accuracy in percent (None if no logits supplied).
        per_class_precision: per-class precision, in percent.
        per_class_recall: per-class recall, in percent.
        per_class_f1: per-class F1, in percent.
        per_class_support: per-class sample counts on the evaluation set.
        confusion: ``[C, C]`` integer confusion matrix.
        class_names: optional human-readable class names.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    balanced_accuracy: float
    cohen_kappa: float
    top_2_accuracy: Optional[float]
    top_5_accuracy: Optional[float]
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    per_class_support: List[int]
    confusion: np.ndarray
    class_names: List[str] = field(default_factory=list)

    def short_line(self, prefix: str = "") -> str:
        """Render the headline numbers as a single human-readable line."""
        return (
            f"{prefix}acc={self.accuracy:5.2f}%  "
            f"P={self.precision_macro:5.2f}%  "
            f"R={self.recall_macro:5.2f}%  "
            f"F1={self.f1_macro:5.2f}%  "
            f"kappa={self.cohen_kappa:.4f}  "
            f"balanced_acc={self.balanced_accuracy:5.2f}%"
        )


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> ClassificationReport:
    """Build a :class:`ClassificationReport` from arrays of labels and predictions.

    Args:
        y_true: ground-truth integer labels, shape ``[N]``.
        y_pred: predicted integer labels, shape ``[N]``.
        logits: optional ``[N, C]`` array of class scores used to compute
            top-2 and top-5 accuracy.
        class_names: optional list of class labels for downstream plots.

    Returns:
        Filled :class:`ClassificationReport` with all macro/weighted aggregates
        and per-class quantities expressed in percent (except Cohen's kappa).
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    p_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = np.bincount(y_true, minlength=int(max(y_true.max(), y_pred.max()) + 1))

    cm = confusion_matrix(y_true, y_pred)

    top_2 = top_5 = None
    if logits is not None:
        logits = np.asarray(logits)
        n_cls = logits.shape[1]
        labels = np.arange(n_cls)
        if n_cls >= 2:
            top_2 = 100.0 * top_k_accuracy_score(y_true, logits, k=min(2, n_cls), labels=labels)
        if n_cls >= 5:
            top_5 = 100.0 * top_k_accuracy_score(y_true, logits, k=5, labels=labels)

    return ClassificationReport(
        accuracy=100.0 * float((y_pred == y_true).mean()),
        precision_macro=100.0 * float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=100.0 * float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_weighted=100.0 * float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall_weighted=100.0 * float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_weighted=100.0 * float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        balanced_accuracy=100.0 * float(balanced_accuracy_score(y_true, y_pred)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        top_2_accuracy=top_2,
        top_5_accuracy=top_5,
        per_class_precision=[100.0 * v for v in p_class.tolist()],
        per_class_recall=[100.0 * v for v in r_class.tolist()],
        per_class_f1=[100.0 * v for v in f1_class.tolist()],
        per_class_support=[int(s) for s in support.tolist()],
        confusion=cm,
        class_names=list(class_names) if class_names is not None else [str(i) for i in range(cm.shape[0])],
    )


def predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax class indices for a ``[N, C]`` tensor of scores."""
    return logits.argmax(dim=1)


def report_table(reports: Dict[str, ClassificationReport]) -> str:
    """Pretty-print several reports side by side as a fixed-width text table."""
    headers = ["model", "acc", "P", "R", "F1", "bal_acc", "kappa", "top2", "top5"]
    rows: List[List[str]] = []
    for name, r in reports.items():
        rows.append(
            [
                name,
                f"{r.accuracy:.2f}",
                f"{r.precision_macro:.2f}",
                f"{r.recall_macro:.2f}",
                f"{r.f1_macro:.2f}",
                f"{r.balanced_accuracy:.2f}",
                f"{r.cohen_kappa:.4f}",
                "-" if r.top_2_accuracy is None else f"{r.top_2_accuracy:.2f}",
                "-" if r.top_5_accuracy is None else f"{r.top_5_accuracy:.2f}",
            ]
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    body = "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)
    return f"{header_line}\n{sep}\n{body}"
