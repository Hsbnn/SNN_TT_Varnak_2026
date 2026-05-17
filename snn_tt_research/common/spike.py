"""Spike non-linearity with fast-sigmoid surrogate gradient and Poisson encoder."""

from __future__ import annotations

import torch


class SpikeFunction(torch.autograd.Function):
    """Heaviside step in the forward pass with a fast-sigmoid surrogate gradient.

    The forward pass emits a binary spike when the membrane minus threshold is
    non-negative.  In the backward pass the non-differentiable Heaviside is
    replaced by the derivative of the logistic with steepness ``scale``.
    """

    @staticmethod
    def forward(ctx, v_minus_thresh: torch.Tensor, scale: float) -> torch.Tensor:
        """Emit binary spikes and cache state needed for the surrogate gradient."""
        spikes = (v_minus_thresh >= 0).float()
        ctx.save_for_backward(v_minus_thresh)
        ctx.scale = scale
        return spikes

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor):
        """Compute the surrogate-gradient backward step using a fast-sigmoid."""
        (v,) = ctx.saved_tensors
        s = ctx.scale
        sig = torch.sigmoid(s * v)
        sg = s * sig * (1.0 - sig)
        return grad_spike * sg, None


def spike_fn(v_minus_thresh: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    """Apply :class:`SpikeFunction` with a default surrogate steepness."""
    return SpikeFunction.apply(v_minus_thresh, scale)


def poisson_from_rates(
    rates: torch.Tensor,
    T: int,
    gain: float = 1.0,
    max_prob: float = 0.85,
) -> torch.Tensor:
    """Sample a Bernoulli spike train of length ``T`` from per-neuron rates.

    Args:
        rates: tensor ``[B, D]`` with non-negative firing intensities.
        T: number of simulated time steps.
        gain: multiplier applied to ``rates`` before clamping.
        max_prob: upper clamp on the per-step Bernoulli probability.

    Returns:
        Binary tensor of shape ``[T, B, D]``.
    """
    p = (rates * gain).clamp(0.0, max_prob)
    rnd = torch.rand(T, *p.shape, device=p.device, dtype=p.dtype)
    return (rnd < p.unsqueeze(0)).float()


def poisson_from_rates_with_generator(
    rates: torch.Tensor,
    T: int,
    generator: torch.Generator,
    gain: float = 1.0,
    max_prob: float = 0.85,
) -> torch.Tensor:
    """Variant of :func:`poisson_from_rates` driven by an explicit RNG generator.

    Used to produce identical spike sequences for benchmarking the dense and the
    compressed model on the same input.
    """
    p = (rates * gain).clamp(0.0, max_prob)
    rnd = torch.rand(
        (T, *p.shape),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(rates.device)
    return (rnd < p.unsqueeze(0)).float()
