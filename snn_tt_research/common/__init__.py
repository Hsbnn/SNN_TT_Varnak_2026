"""Shared utilities for all three research blocks."""

from .device import select_device, device_sync, set_seed
from .spike import SpikeFunction, spike_fn, poisson_from_rates
from .encoding import (
    build_rf_centers_sigma,
    gaussian_rf_encode_1d,
    build_gaussian_rf_weights_2d,
    image_to_rf_rates,
)
from .tt_decomp import (
    tt_svd_3way,
    reconstruct_weight_tt,
    tt_core_num_params,
    tt_runtime_param_count_cached,
    matrix_lowrank_from_dense,
)
from .metrics import (
    count_params,
    dense_layer_mac,
    tt_layer_mac_cached,
    lowrank_layer_mac,
    efficiency_index,
)
from .benchmarks import median_bench, build_real_spike_batches
from .evaluation import (
    ClassificationReport,
    classification_report,
    predictions_from_logits,
    report_table,
)
from .plots import (
    plot_confusion_matrix,
    plot_per_class_bars,
    plot_model_comparison,
    plot_efficiency_scatter,
    plot_training_curves,
    plot_latency_vs_speedup,
    write_all_plots,
)

__all__ = [
    "select_device",
    "device_sync",
    "set_seed",
    "SpikeFunction",
    "spike_fn",
    "poisson_from_rates",
    "build_rf_centers_sigma",
    "gaussian_rf_encode_1d",
    "build_gaussian_rf_weights_2d",
    "image_to_rf_rates",
    "tt_svd_3way",
    "reconstruct_weight_tt",
    "tt_core_num_params",
    "tt_runtime_param_count_cached",
    "matrix_lowrank_from_dense",
    "count_params",
    "dense_layer_mac",
    "tt_layer_mac_cached",
    "lowrank_layer_mac",
    "efficiency_index",
    "median_bench",
    "build_real_spike_batches",
    "ClassificationReport",
    "classification_report",
    "predictions_from_logits",
    "report_table",
    "plot_confusion_matrix",
    "plot_per_class_bars",
    "plot_model_comparison",
    "plot_efficiency_scatter",
    "plot_training_curves",
    "plot_latency_vs_speedup",
    "write_all_plots",
]
