"""Block 1 — Iris with Gaussian receptive fields, ANN/SNN/SNN+TT, STDP comparison."""

from .config import IrisConfig
from .data import load_iris_dataloaders
from .models import ANN_raw, ANN_rf, SNN_rf, SNN_rf_tt, SNN_rf_stdp_hybrid
from .train import train_ann, train_snn, train_stdp_hybrid_epoch, run_supervised_training
from .evaluate import eval_ann, eval_snn, eval_stdp_hybrid
from .run import run_block1

__all__ = [
    "IrisConfig",
    "load_iris_dataloaders",
    "ANN_raw",
    "ANN_rf",
    "SNN_rf",
    "SNN_rf_tt",
    "SNN_rf_stdp_hybrid",
    "train_ann",
    "train_snn",
    "train_stdp_hybrid_epoch",
    "run_supervised_training",
    "eval_ann",
    "eval_snn",
    "eval_stdp_hybrid",
    "run_block1",
]
