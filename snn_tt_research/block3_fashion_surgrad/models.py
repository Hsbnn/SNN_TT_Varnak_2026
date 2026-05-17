"""Dense / TT / LowRank SNNs and their inference-only counterparts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.spike import poisson_from_rates, spike_fn
from ..common.tt_decomp import fuse_tt_middle, matrix_lowrank_from_dense, tt_svd_3way
from .config import FashionConfig


def poisson_encode_images(images: torch.Tensor, cfg: FashionConfig) -> torch.Tensor:
    """Project ``[B, 1, 28, 28]`` images onto a Poisson spike train.

    Pixels are flattened, scaled by ``input_gain`` and clamped to
    ``poisson_max_p`` before sampling Bernoulli spikes for ``T`` steps.
    """
    B = images.size(0)
    flat = images.view(B, -1)
    return poisson_from_rates(flat, cfg.t_steps, gain=cfg.input_gain, max_prob=cfg.poisson_max_p)


class ANN_MLP(nn.Module):
    """Reference fully-connected ANN with one hidden ReLU layer."""

    def __init__(self, cfg: FashionConfig):
        """Construct the two-layer MLP sized from the config."""
        super().__init__()
        self.fc1 = nn.Linear(cfg.flat, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the input, then apply ``ReLU(fc1) → fc2``."""
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class SNN_MLP(nn.Module):
    """Two-layer LIF SNN with dense input weights trained via surrogate gradient."""

    def __init__(self, cfg: FashionConfig):
        """Initialise parameters with Kaiming-uniform and store SNN settings."""
        super().__init__()
        self.cfg = cfg
        self.w1 = nn.Parameter(torch.empty(cfg.hidden, cfg.flat))
        self.b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden))
        self.b2 = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    @classmethod
    def from_shared_init(
        cls,
        cfg: FashionConfig,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ) -> "SNN_MLP":
        """Build an :class:`SNN_MLP` and copy the supplied shared tensors."""
        m = cls(cfg)
        m.w1.data.copy_(w1)
        m.b1.data.copy_(b1)
        m.w2.data.copy_(w2)
        m.b2.data.copy_(b2)
        return m

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the dense LIF dynamics and return summed output spike counts."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)
        for t in range(Tn):
            z0 = spike_seq[t]
            v1 = cfg.beta * v1 + F.linear(z0, self.w1, self.b1)
            s1 = spike_fn(v1 - cfg.v_thresh, cfg.spike_surr_scale)
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = spike_fn(v2 - cfg.v_thresh, cfg.spike_surr_scale)
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out


class SNN_MLP_TT(nn.Module):
    """LIF SNN whose input layer is parameterised by three trainable TT cores."""

    def __init__(self, cfg: FashionConfig):
        """Initialise dense weights, then replace the input layer with TT cores."""
        super().__init__()
        self.cfg = cfg
        w1_full = torch.empty(cfg.hidden, cfg.flat)
        nn.init.kaiming_uniform_(w1_full, a=math.sqrt(5))
        G1, G2, G3 = tt_svd_3way(w1_full.view(cfg.hidden, cfg.side, cfg.side), cfg.r1, cfg.r2)

        self.tt_G1 = nn.Parameter(G1)
        self.tt_G2 = nn.Parameter(G2)
        self.tt_G3 = nn.Parameter(G3)
        self.b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden))
        self.b2 = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    @classmethod
    def from_shared_init(
        cls,
        cfg: FashionConfig,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ) -> "SNN_MLP_TT":
        """Build a TT model, factorise the shared ``w1`` and copy the other tensors."""
        m = cls(cfg)
        G1, G2, G3 = tt_svd_3way(w1.view(cfg.hidden, cfg.side, cfg.side), cfg.r1, cfg.r2)
        m.tt_G1.data.copy_(G1)
        m.tt_G2.data.copy_(G2)
        m.tt_G3.data.copy_(G3)
        m.b1.data.copy_(b1)
        m.w2.data.copy_(w2)
        m.b2.data.copy_(b2)
        return m

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the SNN using a per-pass fused middle contraction.

        The contraction ``G2 ⊗ G3`` is computed once and shared across all
        time steps, which is essential for the backward pass to remain
        cheap during training.
        """
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)

        W_mid = (self.tt_G2 @ self.tt_G3).reshape(self.tt_G2.shape[0], -1)
        for t in range(Tn):
            z0 = spike_seq[t]
            h = z0 @ W_mid.T
            v1 = cfg.beta * v1 + F.linear(h, self.tt_G1, self.b1)
            s1 = spike_fn(v1 - cfg.v_thresh, cfg.spike_surr_scale)
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = spike_fn(v2 - cfg.v_thresh, cfg.spike_surr_scale)
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out


class SNN_MLP_LowRank(nn.Module):
    """LIF SNN with a matrix-level low-rank input layer ``W1 ≈ W_up · W_down``."""

    def __init__(self, cfg: FashionConfig):
        """Initialise dense weights, then truncate them with SVD."""
        super().__init__()
        self.cfg = cfg
        w1_full = torch.empty(cfg.hidden, cfg.flat)
        nn.init.kaiming_uniform_(w1_full, a=math.sqrt(5))
        w_down, w_up = matrix_lowrank_from_dense(w1_full, cfg.rank_matrix)
        self.w_down = nn.Parameter(w_down)
        self.w_up = nn.Parameter(w_up)
        self.b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden))
        self.b2 = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    @classmethod
    def from_shared_init(
        cls,
        cfg: FashionConfig,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ) -> "SNN_MLP_LowRank":
        """Build the low-rank model and reproduce the shared dense input."""
        m = cls(cfg)
        w_down, w_up = matrix_lowrank_from_dense(w1, cfg.rank_matrix)
        m.w_down.data.copy_(w_down)
        m.w_up.data.copy_(w_up)
        m.b1.data.copy_(b1)
        m.w2.data.copy_(w2)
        m.b2.data.copy_(b2)
        return m

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the LIF dynamics with two linear maps in place of ``W1``."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)
        for t in range(Tn):
            z0 = spike_seq[t]
            z1 = F.linear(z0, self.w_down)
            v1 = cfg.beta * v1 + F.linear(z1, self.w_up, self.b1)
            s1 = spike_fn(v1 - cfg.v_thresh, cfg.spike_surr_scale)
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = spike_fn(v2 - cfg.v_thresh, cfg.spike_surr_scale)
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out


class SNN_MLP_TT_Infer(nn.Module):
    """Inference-only counterpart of :class:`SNN_MLP_TT`.

    The fused middle matrix is computed once at construction time and stored
    as a buffer; every forward pass then issues exactly two ``F.linear``
    calls per time step.  All tensors are registered as buffers, never as
    parameters, so no autograd state is allocated.
    """

    def __init__(
        self,
        cfg: FashionConfig,
        W_mid: torch.Tensor,
        G1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ):
        """Store the pre-fused tensors and the relevant SNN hyper-parameters."""
        super().__init__()
        self.cfg = cfg
        self.register_buffer("W_mid", W_mid.contiguous())
        self.register_buffer("G1", G1.contiguous())
        self.register_buffer("b1", b1.contiguous())
        self.register_buffer("w2", w2.contiguous())
        self.register_buffer("b2", b2.contiguous())

    @classmethod
    def from_trainable(cls, trainable: SNN_MLP_TT) -> "SNN_MLP_TT_Infer":
        """Detach all trainable tensors and pre-fuse the middle TT contraction."""
        with torch.no_grad():
            W_mid = fuse_tt_middle(trainable.tt_G2.detach(), trainable.tt_G3.detach())
            G1 = trainable.tt_G1.detach().contiguous()
            b1 = trainable.b1.detach().contiguous()
            w2 = trainable.w2.detach().contiguous()
            b2 = trainable.b2.detach().contiguous()
        return cls(trainable.cfg, W_mid, G1, b1, w2, b2)

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out using the cached fused tensors; mirrors :class:`SNN_MLP_TT`."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, self.G1.shape[0], device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, self.w2.shape[0], device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)
        for t in range(Tn):
            z0 = spike_seq[t]
            h = F.linear(z0, self.W_mid)
            v1 = cfg.beta * v1 + F.linear(h, self.G1, self.b1)
            s1 = spike_fn(v1 - cfg.v_thresh, cfg.spike_surr_scale)
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = spike_fn(v2 - cfg.v_thresh, cfg.spike_surr_scale)
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out


class SNN_MLP_LowRank_Infer(nn.Module):
    """Inference-only counterpart of :class:`SNN_MLP_LowRank`."""

    def __init__(
        self,
        cfg: FashionConfig,
        w_down: torch.Tensor,
        w_up: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ):
        """Register every tensor as a buffer so autograd is bypassed."""
        super().__init__()
        self.cfg = cfg
        self.register_buffer("w_down", w_down.contiguous())
        self.register_buffer("w_up", w_up.contiguous())
        self.register_buffer("b1", b1.contiguous())
        self.register_buffer("w2", w2.contiguous())
        self.register_buffer("b2", b2.contiguous())

    @classmethod
    def from_trainable(cls, trainable: SNN_MLP_LowRank) -> "SNN_MLP_LowRank_Infer":
        """Detach every tensor of the trainable model."""
        return cls(
            trainable.cfg,
            trainable.w_down.detach(),
            trainable.w_up.detach(),
            trainable.b1.detach(),
            trainable.w2.detach(),
            trainable.b2.detach(),
        )

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the LIF dynamics using cached low-rank tensors."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, self.w_up.shape[0], device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, self.w2.shape[0], device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)
        for t in range(Tn):
            z0 = spike_seq[t]
            z1 = F.linear(z0, self.w_down)
            v1 = cfg.beta * v1 + F.linear(z1, self.w_up, self.b1)
            s1 = spike_fn(v1 - cfg.v_thresh, cfg.spike_surr_scale)
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = spike_fn(v2 - cfg.v_thresh, cfg.spike_surr_scale)
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out
