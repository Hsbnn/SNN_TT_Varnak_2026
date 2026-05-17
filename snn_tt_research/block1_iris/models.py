"""Networks compared on Iris: dense ANNs, full SNN, TT-SNN and a STDP hybrid."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.spike import spike_fn
from ..common.tt_decomp import tt_svd_3way
from .config import IrisConfig


class ANN_raw(nn.Module):
    """Two-layer MLP operating on the raw 4-D Iris features."""

    def __init__(self, cfg: IrisConfig):
        """Initialise layers sized from the config."""
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_feat, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard ReLU MLP forward pass."""
        return self.fc2(F.relu(self.fc1(x)))


class ANN_rf(nn.Module):
    """Two-layer MLP operating on the Gaussian RF code of Iris."""

    def __init__(self, cfg: IrisConfig):
        """Initialise layers sized from the config."""
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_rf, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.num_classes)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Standard ReLU MLP forward pass on RF-encoded inputs."""
        return self.fc2(F.relu(self.fc1(r)))


class SNN_rf(nn.Module):
    """Two-layer LIF SNN with surrogate-gradient training on RF-encoded Iris."""

    def __init__(self, cfg: IrisConfig):
        """Initialise weights with Kaiming-uniform and store SNN hyper-parameters."""
        super().__init__()
        self.cfg = cfg
        self.w1 = nn.Parameter(torch.empty(cfg.hidden, cfg.n_rf))
        self.b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden))
        self.b2 = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the LIF dynamics and return aggregated output spike counts.

        Args:
            spike_seq: ``[T, B, N_RF]`` binary input spike train.
        """
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

    @torch.no_grad()
    def mean_spike_rates(self, spike_seq: torch.Tensor):
        """Return the per-step mean firing rate at the hidden and output layers."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        s1a, s2a = 0.0, 0.0
        for t in range(Tn):
            z0 = spike_seq[t]
            v1 = cfg.beta * v1 + F.linear(z0, self.w1, self.b1)
            s1 = (v1 >= cfg.v_thresh).float()
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = (v2 >= cfg.v_thresh).float()
            v2 = v2 - s2 * cfg.v_thresh
            s1a += float(s1.mean())
            s2a += float(s2.mean())
        return s1a / Tn, s2a / Tn


class SNN_rf_tt(nn.Module):
    """LIF SNN where the input layer ``W1`` is stored as a TT-3 tensor.

    The weight ``W1 ∈ R^{H × n_feat × n_per_feat}`` is replaced by three
    cores ``(G1, G2, G3)`` with ranks ``(r1, r2)``.  Both the cores and the
    output dense layer are trained end-to-end via surrogate gradients.
    """

    def __init__(self, cfg: IrisConfig):
        """Initialise dense weights then factorise the input layer with TT-SVD."""
        super().__init__()
        self.cfg = cfg
        w1_full = torch.empty(cfg.hidden, cfg.n_rf)
        nn.init.kaiming_uniform_(w1_full, a=math.sqrt(5))
        G1, G2, G3 = tt_svd_3way(
            w1_full.view(cfg.hidden, cfg.side_features, cfg.side_centers),
            cfg.r1,
            cfg.r2,
        )
        self.tt_G1 = nn.Parameter(G1)
        self.tt_G2 = nn.Parameter(G2)
        self.tt_G3 = nn.Parameter(G3)
        self.b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden))
        self.b2 = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def _fuse_middle(self) -> torch.Tensor:
        """Contract the inner cores into a ``[r1, n_rf]`` matrix once per pass."""
        return torch.matmul(self.tt_G2, self.tt_G3).reshape(self.tt_G2.shape[0], -1)

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the LIF dynamics using the TT-factorised input layer."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)
        W_mid = self._fuse_middle()
        for t in range(Tn):
            z0 = spike_seq[t]
            h = F.linear(z0, W_mid)
            v1 = cfg.beta * v1 + F.linear(h, self.tt_G1, self.b1)
            s1 = spike_fn(v1 - cfg.v_thresh, cfg.spike_surr_scale)
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = spike_fn(v2 - cfg.v_thresh, cfg.spike_surr_scale)
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out

    @torch.no_grad()
    def mean_spike_rates(self, spike_seq: torch.Tensor):
        """Return the per-step mean firing rate at hidden and output layers."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        W_mid = self._fuse_middle()
        s1a, s2a = 0.0, 0.0
        for t in range(Tn):
            z0 = spike_seq[t]
            h = F.linear(z0, W_mid)
            v1 = cfg.beta * v1 + F.linear(h, self.tt_G1, self.b1)
            s1 = (v1 >= cfg.v_thresh).float()
            v1 = v1 - s1 * cfg.v_thresh
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = (v2 >= cfg.v_thresh).float()
            v2 = v2 - s2 * cfg.v_thresh
            s1a += float(s1.mean())
            s2a += float(s2.mean())
        return s1a / Tn, s2a / Tn


class SNN_rf_stdp_hybrid(nn.Module):
    """Hybrid learning rule: STDP for ``W1`` plus surrogate gradient for ``W2``.

    The hidden layer uses a hard Heaviside spike, so its weight update follows
    a pair-based STDP rule with exponential pre/post traces.  The readout
    weights are updated with AdamW on the cross-entropy of pooled hidden spike
    counts.
    """

    def __init__(self, cfg: IrisConfig):
        """Initialise dense weights and freeze the hidden bias to zero."""
        super().__init__()
        self.cfg = cfg
        self.w1 = nn.Parameter(torch.empty(cfg.hidden, cfg.n_rf))
        self.b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.w2 = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden))
        self.b2 = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.b1.requires_grad_(False)

    @torch.no_grad()
    def forward_with_hidden_sum(self, spike_seq: torch.Tensor):
        """Roll out the network and also return the pooled hidden spike counts."""
        cfg = self.cfg
        Tn, B, _ = spike_seq.shape
        v1 = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        v2 = torch.zeros(B, cfg.num_classes, device=spike_seq.device, dtype=spike_seq.dtype)
        out = torch.zeros_like(v2)
        hid_sum = torch.zeros(B, cfg.hidden, device=spike_seq.device, dtype=spike_seq.dtype)
        for t in range(Tn):
            z0 = spike_seq[t]
            v1 = cfg.beta * v1 + F.linear(z0, self.w1, self.b1)
            s1 = (v1 >= cfg.v_thresh).float()
            v1 = v1 - s1 * cfg.v_thresh
            hid_sum = hid_sum + s1
            v2 = cfg.beta * v2 + F.linear(s1, self.w2, self.b2)
            s2 = (v2 >= cfg.v_thresh).float()
            v2 = v2 - s2 * cfg.v_thresh
            out = out + s2
        return out, hid_sum

    @torch.no_grad()
    def stdp_update_w1(self, spike_seq: torch.Tensor) -> None:
        """Apply a batched pair-based STDP update to the hidden weight.

        Long-term potentiation accumulates ``post_spike × pre_trace``;
        long-term depression accumulates ``post_trace × pre_spike``.  Traces
        decay at every step by :attr:`stdp_trace_decay`.
        """
        cfg = self.cfg
        Tn, B, D = spike_seq.shape
        device = spike_seq.device
        tr_pre = torch.zeros(B, D, device=device, dtype=spike_seq.dtype)
        tr_post = torch.zeros(B, cfg.hidden, device=device, dtype=spike_seq.dtype)
        v1 = torch.zeros(B, cfg.hidden, device=device, dtype=spike_seq.dtype)
        dw = torch.zeros_like(self.w1)
        dcy = cfg.stdp_trace_decay
        for t in range(Tn):
            z0 = spike_seq[t]
            v1 = cfg.beta * v1 + F.linear(z0, self.w1, self.b1)
            s1 = (v1 >= cfg.v_thresh).float()
            v1 = v1 - s1 * cfg.v_thresh
            dw = dw + cfg.stdp_a_plus * (s1.unsqueeze(2) * tr_pre.unsqueeze(1)).mean(0)
            dw = dw - cfg.stdp_a_minus * (tr_post.unsqueeze(2) * z0.unsqueeze(1)).mean(0)
            tr_pre = tr_pre * dcy + z0
            tr_post = tr_post * dcy + s1
        self.w1.add_(cfg.stdp_w1_step * dw)
        self.w1.clamp_(cfg.stdp_w1_min, cfg.stdp_w1_max)
