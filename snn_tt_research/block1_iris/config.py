"""Hyper-parameter container for the Iris efficiency study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IrisConfig:
    """Configuration of the Iris block.

    Attributes:
        n_per_feat: Gaussian centres per input feature.
        n_feat: number of raw input features (Iris has 4).
        hidden: hidden layer size shared between ANN and SNN.
        num_classes: number of target classes.
        t_steps: SNN simulation length in time steps.
        beta: LIF membrane decay.
        v_thresh: spike threshold.
        spike_surr_scale: steepness of the fast-sigmoid surrogate gradient.
        rf_gain: Poisson rate gain applied after RF encoding.
        poisson_max_p: clamp on per-step Bernoulli spike probability.
        r1: first TT rank of the compressed input layer.
        r2: second TT rank of the compressed input layer.
        batch_size: mini-batch size during training.
        epochs: total number of training epochs.
        lr: learning rate for AdamW.
        weight_decay: weight decay for AdamW.
        bench_warmup: warm-up iterations before timed runs.
        bench_trials: number of timed runs whose median is reported.
        stdp_trace_decay: pre/post trace decay used by hybrid STDP.
        stdp_a_plus: LTP scaling factor for STDP.
        stdp_a_minus: LTD scaling factor for STDP.
        stdp_w1_step: outer update step applied to the STDP-trained weight.
        stdp_w1_min: lower clamp on STDP weights.
        stdp_w1_max: upper clamp on STDP weights.
        seed: master seed for data split and model init.
    """

    n_per_feat: int = 10
    n_feat: int = 4
    hidden: int = 48
    num_classes: int = 3
    t_steps: int = 16
    beta: float = 0.9
    v_thresh: float = 1.0
    spike_surr_scale: float = 5.0
    rf_gain: float = 2.5
    poisson_max_p: float = 0.85
    r1: int = 10
    r2: int = 5
    batch_size: int = 16
    epochs: int = 40
    lr: float = 1e-2
    weight_decay: float = 1e-4
    bench_warmup: int = 30
    bench_trials: int = 7
    stdp_trace_decay: float = 0.93
    stdp_a_plus: float = 0.045
    stdp_a_minus: float = 0.035
    stdp_w1_step: float = 0.35
    stdp_w1_min: float = 0.02
    stdp_w1_max: float = 1.25
    seed: int = 0

    @property
    def n_rf(self) -> int:
        """Total dimensionality of the RF-encoded input."""
        return self.n_per_feat * self.n_feat

    @property
    def side_features(self) -> int:
        """Outer mode of the TT factorisation (feature axis)."""
        return self.n_feat

    @property
    def side_centers(self) -> int:
        """Inner mode of the TT factorisation (centre axis)."""
        return self.n_per_feat
