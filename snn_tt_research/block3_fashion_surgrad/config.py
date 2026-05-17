"""Hyper-parameters for the Fashion-MNIST surrogate-gradient block."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FashionConfig:
    """Configuration of the Fashion-MNIST block.

    Attributes:
        batch_size: training mini-batch size.
        epochs: number of training epochs per model variant.
        lr_ann: learning rate for the ANN baseline.
        lr_snn: learning rate for SNN variants.
        wd_ann: weight decay for the ANN baseline.
        wd_snn: weight decay for SNN variants.
        t_steps: SNN simulation length.
        beta: LIF membrane decay.
        v_thresh: spike threshold.
        spike_surr_scale: steepness of the fast-sigmoid surrogate gradient.
        input_gain: Poisson rate gain applied to the pixel values.
        poisson_max_p: clamp on per-step Bernoulli spike probability.
        hidden: hidden layer size.
        num_classes: number of target classes (10 for Fashion-MNIST).
        side: image side length.
        r1: first TT rank for the input layer.
        r2: second TT rank for the input layer.
        rank_matrix: rank for the matrix-level low-rank factorisation.
        val_split: fraction of training samples held out for validation.
        model_init_seed: seed used to build shared initial tensors.
        bench_warmup: warm-up iterations before timed runs.
        bench_trials: number of timed runs whose median is reported.
        bench_batches: number of batches drawn for the latency benchmark.
        eval_runs_snn: number of Poisson samples averaged at evaluation.
        seed: master RNG seed.
    """

    batch_size: int = 128
    epochs: int = 25
    lr_ann: float = 2e-3
    lr_snn: float = 5e-3
    wd_ann: float = 1e-4
    wd_snn: float = 1e-5

    t_steps: int = 32
    beta: float = 0.9
    v_thresh: float = 1.0
    spike_surr_scale: float = 5.0
    input_gain: float = 4.0
    poisson_max_p: float = 0.6

    hidden: int = 512
    num_classes: int = 10
    side: int = 28

    r1: int = 32
    r2: int = 16
    rank_matrix: int = 40

    val_split: float = 0.1
    model_init_seed: int = 42

    bench_warmup: int = 10
    bench_trials: int = 7
    bench_batches: int = 24
    eval_runs_snn: int = 4

    seed: int = 0

    @property
    def flat(self) -> int:
        """Flattened image dimensionality ``side * side``."""
        return self.side * self.side
