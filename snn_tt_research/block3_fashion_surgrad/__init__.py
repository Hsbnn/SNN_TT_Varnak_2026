"""Block 3 — Fashion-MNIST MLPs trained with surrogate gradient, TT and LowRank."""

from .config import FashionConfig
from .data import build_fashion_loaders, make_shared_initial_tensors
from .models import (
    ANN_MLP,
    SNN_MLP,
    SNN_MLP_TT,
    SNN_MLP_LowRank,
    SNN_MLP_TT_Infer,
    SNN_MLP_LowRank_Infer,
)
from .train import train_epoch_ann, train_epoch_snn, eval_ann, eval_snn, fit_all_models
from .benchmark import benchmark_inference, benchmark_layer1
from .metrics_report import build_fashion_reports, export_fashion_reports
from .run import run_block3

__all__ = [
    "FashionConfig",
    "build_fashion_loaders",
    "make_shared_initial_tensors",
    "ANN_MLP",
    "SNN_MLP",
    "SNN_MLP_TT",
    "SNN_MLP_LowRank",
    "SNN_MLP_TT_Infer",
    "SNN_MLP_LowRank_Infer",
    "train_epoch_ann",
    "train_epoch_snn",
    "eval_ann",
    "eval_snn",
    "fit_all_models",
    "benchmark_inference",
    "benchmark_layer1",
    "build_fashion_reports",
    "export_fashion_reports",
    "run_block3",
]
