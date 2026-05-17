"""LIF hidden layer, STDP rule and its TT-factorised counterpart."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.tt_decomp import fuse_tt_middle, tt_svd_3way
from .config import MNISTStdpConfig


class LIFHidden(nn.Module):
    """Hard-threshold LIF layer with row-sum normalisation and k-WTA competition.

    The weight tensor is non-negative and stays inside ``[w_min, w_max]``.
    Lateral competition is implemented by keeping only the ``k_winners``
    neurons with the highest ``mem - theta`` margin at each time step.  The
    threshold ``theta`` is per-neuron and adapts homeostatically toward a
    target firing rate.
    """

    def __init__(self, n_in: int, n_out: int, cfg: MNISTStdpConfig):
        """Initialise weights uniformly in ``[0, 1/sqrt(n_in)]`` and normalise rows."""
        super().__init__()
        self.cfg = cfg
        self.n_in = n_in
        self.n_out = n_out

        scale = 1.0 / math.sqrt(n_in)
        w = torch.rand(n_out, n_in) * scale
        self.weight = nn.Parameter(w, requires_grad=False)
        self.theta = nn.Parameter(torch.full((n_out,), cfg.theta0), requires_grad=False)
        self.normalize_weights_()

    @torch.no_grad()
    def normalize_weights_(self) -> None:
        """Clamp the weight tensor and rescale every row to a fixed sum."""
        cfg = self.cfg
        self.weight.data.clamp_(cfg.w_min, cfg.w_max)
        row_sum = self.weight.data.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.weight.data = self.weight.data * (cfg.row_sum_target / row_sum)
        self.weight.data.clamp_(cfg.w_min, cfg.w_max)

    def reset(self, batch: int, device: torch.device):
        """Return zero-valued membrane and (unused) refractory state tensors."""
        zero = torch.zeros(batch, self.n_out, device=device)
        return zero, zero.clone()

    def forward(self, pre_spk: torch.Tensor, mem: torch.Tensor):
        """Advance the LIF dynamics one step and return ``(spikes, new_mem)``.

        After thresholding, ``k`` winners-take-all are selected by the size of
        their ``mem - theta`` margin.  The remainder of the spike vector is
        zeroed out, which sparsifies the resulting receptive fields.
        """
        cfg = self.cfg
        i = (pre_spk @ self.weight.T) * cfg.synaptic_gain
        mem = cfg.lif_beta * mem + i
        thr = self.theta.unsqueeze(0)

        raw_spk = (mem >= thr).float()

        if cfg.k_winners is not None and cfg.k_winners < self.n_out:
            if raw_spk.sum() > 0:
                score = mem - thr
                masked = torch.where(raw_spk > 0, score, torch.full_like(score, -1e9))
                topk_idx = torch.topk(masked, k=cfg.k_winners, dim=1).indices
                keep = torch.zeros_like(raw_spk)
                keep.scatter_(1, topk_idx, 1.0)
                spk = raw_spk * keep
            else:
                spk = raw_spk
        else:
            spk = raw_spk

        mem = mem - spk * thr
        return spk, mem


class STDP(nn.Module):
    """Pair-based exponential STDP rule operating on a :class:`LIFHidden` layer.

    The rule updates the underlying weight in-place using the standard
    pre/post traces with decays derived from ``tau_pre`` and ``tau_post``.
    """

    def __init__(self, layer: LIFHidden, cfg: MNISTStdpConfig):
        """Bind the rule to a layer and pre-compute the trace decay constants."""
        super().__init__()
        self.layer = layer
        self.cfg = cfg
        self.a_plus = cfg.a_plus
        self.a_minus = cfg.a_minus
        self.decay_pre = float(np.exp(-1.0 / cfg.tau_pre))
        self.decay_post = float(np.exp(-1.0 / cfg.tau_post))
        self.trace_pre: torch.Tensor | None = None
        self.trace_post: torch.Tensor | None = None

    def init_traces(self, batch: int, device: torch.device) -> None:
        """Allocate zero pre- and post-synaptic traces for a new mini-batch."""
        self.trace_pre = torch.zeros(batch, self.layer.n_in, device=device)
        self.trace_post = torch.zeros(batch, self.layer.n_out, device=device)

    @torch.no_grad()
    def step(self, pre_spk: torch.Tensor, post_spk: torch.Tensor) -> None:
        """Apply one STDP update step using the current pre/post spike pair."""
        self.trace_pre = self.trace_pre * self.decay_pre + pre_spk
        self.trace_post = self.trace_post * self.decay_post + post_spk

        dw_plus = torch.einsum("bo,bi->oi", post_spk, self.trace_pre)
        dw_minus = torch.einsum("bo,bi->oi", self.trace_post, pre_spk)
        dw = (self.a_plus * dw_plus - self.a_minus * dw_minus) / pre_spk.size(0)

        self.layer.weight.data += dw
        self.layer.weight.data.clamp_(self.cfg.w_min, self.cfg.w_max)
        self.layer.normalize_weights_()


class SNNEncoder(nn.Module):
    """Single hidden LIF layer used as an unsupervised feature extractor."""

    def __init__(self, n_rf: int, n_hidden: int, cfg: MNISTStdpConfig):
        """Compose a :class:`LIFHidden` of the requested width."""
        super().__init__()
        self.hidden = LIFHidden(n_rf, n_hidden, cfg)

    @property
    def n_hidden(self) -> int:
        """Convenience accessor for the hidden layer size."""
        return self.hidden.n_out

    @torch.no_grad()
    def forward_spikes(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the layer over ``T`` steps and return total spike counts."""
        T, B, _ = x_seq.shape
        mem, _ = self.hidden.reset(B, x_seq.device)
        total = torch.zeros(B, self.hidden.n_out, device=x_seq.device)
        for t in range(T):
            spk, mem = self.hidden(x_seq[t], mem)
            total = total + spk
        return total


class LIFHiddenTensorTrain(nn.Module):
    """TT-compressed counterpart of :class:`LIFHidden` for fast inference.

    The dense weight ``W ∈ R^{n_out × side × side}`` is replaced by three TT
    cores.  At inference time the middle and last cores are fused into a
    cached matrix so the forward pass uses two ``F.linear`` calls per step.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        side: int,
        G1: torch.Tensor,
        G2: torch.Tensor,
        G3: torch.Tensor,
        cfg: MNISTStdpConfig,
    ):
        """Store cores as buffers and cache the fused middle matrix."""
        super().__init__()
        assert n_in == side * side, "n_in must equal side*side for the 3-way factorisation"
        self.cfg = cfg
        self.n_in = n_in
        self.n_out = n_out
        self.side = side

        self.tt_G1 = nn.Parameter(G1, requires_grad=False)
        self.tt_G2 = nn.Parameter(G2, requires_grad=False)
        self.tt_G3 = nn.Parameter(G3, requires_grad=False)

        self.register_buffer("tt_W_mid", fuse_tt_middle(G2, G3))
        self.theta = nn.Parameter(torch.full((n_out,), cfg.theta0), requires_grad=False)

    @classmethod
    def from_dense_weight(
        cls,
        n_in: int,
        n_out: int,
        side: int,
        weight_row_major: torch.Tensor,
        theta_vec: torch.Tensor,
        r1: int,
        r2: int,
        cfg: MNISTStdpConfig,
    ) -> "LIFHiddenTensorTrain":
        """Factorise an existing dense weight matrix and copy the threshold vector."""
        W = weight_row_major.view(n_out, side, side).contiguous()
        G1, G2, G3 = tt_svd_3way(W, r1, r2)
        m = cls(n_in=n_in, n_out=n_out, side=side, G1=G1, G2=G2, G3=G3, cfg=cfg)
        m.theta.data.copy_(theta_vec.to(m.theta.device))
        return m

    def reset(self, batch: int, device: torch.device):
        """Return fresh zero membrane and (unused) refractory tensors."""
        zero = torch.zeros(batch, self.n_out, device=device)
        return zero, zero.clone()

    def forward(self, pre_spk: torch.Tensor, mem: torch.Tensor):
        """LIF step that uses the cached TT contractions in place of dense W."""
        cfg = self.cfg
        h = F.linear(pre_spk, self.tt_W_mid)
        lin = F.linear(h, self.tt_G1)
        i = lin * cfg.synaptic_gain
        mem = cfg.lif_beta * mem + i
        thr = self.theta.unsqueeze(0)
        raw_spk = (mem >= thr).float()

        if cfg.k_winners is not None and cfg.k_winners < self.n_out:
            if raw_spk.sum() > 0:
                score = mem - thr
                masked = torch.where(raw_spk > 0, score, torch.full_like(score, -1e9))
                topk_idx = torch.topk(masked, k=cfg.k_winners, dim=1).indices
                keep = torch.zeros_like(raw_spk)
                keep.scatter_(1, topk_idx, 1.0)
                spk = raw_spk * keep
            else:
                spk = raw_spk
        else:
            spk = raw_spk

        mem = mem - spk * thr
        return spk, mem


class SNNEncoderTTWrapper(nn.Module):
    """Replicates the :class:`SNNEncoder` API around a :class:`LIFHiddenTensorTrain`.

    Having a wrapper with the same surface keeps feature collection code
    agnostic of which weight representation is being benchmarked.
    """

    def __init__(self, tt_hidden: LIFHiddenTensorTrain):
        """Reference the supplied TT hidden layer."""
        super().__init__()
        self.hidden = tt_hidden

    @property
    def n_hidden(self) -> int:
        """Width of the wrapped TT hidden layer."""
        return self.hidden.n_out

    @torch.no_grad()
    def forward_spikes(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Roll out the TT layer over ``T`` steps and return total spike counts."""
        T, B, _ = x_seq.shape
        mem, _ = self.hidden.reset(B, x_seq.device)
        total = torch.zeros(B, self.hidden.n_out, device=x_seq.device)
        for t in range(T):
            spk, mem = self.hidden(x_seq[t], mem)
            total = total + spk
        return total


@torch.no_grad()
def rollout_hidden_layer(hidden: nn.Module, x_seq: torch.Tensor) -> torch.Tensor:
    """Generic ``LIFHidden``-compatible rollout that pools output spike counts."""
    T, B, _ = x_seq.shape
    mem, _ = hidden.reset(B, x_seq.device)
    total = torch.zeros(B, hidden.n_out, device=x_seq.device)
    for t in range(T):
        spk, mem = hidden(x_seq[t], mem)
        total = total + spk
    return total
