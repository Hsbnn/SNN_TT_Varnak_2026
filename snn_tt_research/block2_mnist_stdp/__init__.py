"""Block 2 — STDP-trained SNN on MNIST with post-hoc Tensor-Train compression."""

from .config import MNISTStdpConfig
from .data import build_mnist_loaders, MNISTRFEncoder
from .models import LIFHidden, STDP, SNNEncoder, LIFHiddenTensorTrain, SNNEncoderTTWrapper
from .readout import RFOnlyReadout, SpikesOnlyReadout, FusionReadout, BlockNormalizer
from .features import collect_feature_blocks, spike_features
from .train import stdp_train_epoch, train_readout_head
from .tt_compression import sweep_tt_ranks, build_final_tt_layer
from .benchmark import benchmark_dense_vs_tt
from .metrics_report import build_reports, export_reports
from .run import run_block2

__all__ = [
    "MNISTStdpConfig",
    "build_mnist_loaders",
    "MNISTRFEncoder",
    "LIFHidden",
    "STDP",
    "SNNEncoder",
    "LIFHiddenTensorTrain",
    "SNNEncoderTTWrapper",
    "RFOnlyReadout",
    "SpikesOnlyReadout",
    "FusionReadout",
    "BlockNormalizer",
    "collect_feature_blocks",
    "spike_features",
    "stdp_train_epoch",
    "train_readout_head",
    "sweep_tt_ranks",
    "build_final_tt_layer",
    "benchmark_dense_vs_tt",
    "build_reports",
    "export_reports",
    "run_block2",
]
