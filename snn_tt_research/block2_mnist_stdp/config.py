"""Hyper-parameters for the MNIST STDP + TT-compression block."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MNISTStdpConfig:
    """Configuration for the STDP-trained SNN encoder and its TT compression.

    Attributes:
        img_size: side length of the MNIST images in pixels.
        rf_grid: side length of the Gaussian RF grid (``rf_grid ** 2`` units).
        hidden: number of hidden LIF neurons.
        sim_time: simulation length in time steps.
        batch_size: mini-batch size for both training and evaluation.
        stdp_epochs: number of unsupervised STDP epochs.
        readout_epochs: training epochs for the supervised readout head.
        readout_lr: learning rate for the readout AdamW optimiser.
        readout_wd: weight decay for the readout AdamW optimiser.
        readout_clip: gradient norm clip for the readout head.
        feature_avg_train: spike-encoding repetitions averaged at train time.
        feature_avg_test: spike-encoding repetitions averaged at test time.
        lif_beta: LIF membrane decay.
        synaptic_gain: scaling applied to the synaptic current.
        theta0: initial value of the adaptive threshold.
        target_rate: homeostatic firing-rate target.
        theta_lr: learning rate of the homeostatic threshold rule.
        theta_min: lower clamp for the adaptive threshold.
        theta_max: upper clamp for the adaptive threshold.
        tau_pre: pre-synaptic trace time constant in time steps.
        tau_post: post-synaptic trace time constant in time steps.
        a_plus: LTP scaling factor.
        a_minus: LTD scaling factor.
        w_min: lower clamp for the STDP weight.
        w_max: upper clamp for the STDP weight.
        row_sum_target: row-wise weight normalisation target.
        k_winners: number of winners-take-all hidden spikes per step.
        input_gain: Poisson rate gain for the RF encoder.
        poisson_max_prob: clamp on per-step Bernoulli spike probability.
        tt_candidates: TT rank pairs to sweep when selecting the compression.
        tt_main_mode: feature combination used when training the final readout.
        tt_sweep_batches: number of batches for the sweep benchmark.
        tt_final_bench_batches: number of batches for the final benchmark.
        tt_retrain_readout: whether to retrain the readout on TT features.
        tt_feature_avg_train: spike repetitions when collecting TT train features.
        tt_feature_avg_test: spike repetitions when collecting TT test features.
        bench_warmup: warmup runs before timing.
        bench_trials: number of timed runs whose median is reported.
        seed: master RNG seed.
    """

    img_size: int = 28
    rf_grid: int = 19
    hidden: int = 400
    sim_time: int = 50
    batch_size: int = 128

    stdp_epochs: int = 4
    readout_epochs: int = 25
    readout_lr: float = 1e-3
    readout_wd: float = 1e-4
    readout_clip: float = 1.0
    feature_avg_train: int = 6
    feature_avg_test: int = 10

    lif_beta: float = 0.90
    synaptic_gain: float = 2.8
    theta0: float = 0.90
    target_rate: float = 0.05
    theta_lr: float = 0.03
    theta_min: float = 0.35
    theta_max: float = 1.80

    tau_pre: float = 20.0
    tau_post: float = 20.0
    a_plus: float = 0.0012
    a_minus: float = 0.0014
    w_min: float = 0.0
    w_max: float = 1.0
    row_sum_target: float = 10.0

    k_winners: int = 20
    input_gain: float = 2.4
    poisson_max_prob: float = 0.22

    tt_candidates: List[Tuple[int, int]] = field(
        default_factory=lambda: [(64, 19), (96, 19), (128, 19), (160, 19), (192, 19)]
    )
    tt_main_mode: str = "spikes_plus_rf"
    tt_sweep_batches: int = 10
    tt_final_bench_batches: int = 24
    tt_retrain_readout: bool = True
    tt_feature_avg_train: int = 4
    tt_feature_avg_test: int = 6

    bench_warmup: int = 6
    bench_trials: int = 5
    seed: int = 0

    @property
    def n_rf(self) -> int:
        """Total dimensionality of the Gaussian RF code."""
        return self.rf_grid * self.rf_grid

    @property
    def side(self) -> int:
        """TT mode size: same as :attr:`rf_grid` since the code is laid out 2-D."""
        return self.rf_grid
