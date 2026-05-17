"""Supervised readout heads and the block-wise z-score normaliser."""

from __future__ import annotations

import torch
import torch.nn as nn


class BlockNormalizer:
    """Independent z-score normalisation for the spike block and the RF block.

    Statistics are computed on the train split only and re-used at test time;
    the spike branch is log-transformed first because spike counts have a
    heavy right tail.
    """

    def __init__(self) -> None:
        """Initialise the four moment tensors to ``None``."""
        self.sp_mean: torch.Tensor | None = None
        self.sp_std: torch.Tensor | None = None
        self.rf_mean: torch.Tensor | None = None
        self.rf_std: torch.Tensor | None = None

    def fit(self, X_sp_log1p: torch.Tensor, X_rf: torch.Tensor) -> None:
        """Estimate per-feature mean and standard deviation on the train split."""
        self.sp_mean = X_sp_log1p.mean(dim=0, keepdim=True)
        self.sp_std = X_sp_log1p.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.rf_mean = X_rf.mean(dim=0, keepdim=True)
        self.rf_std = X_rf.std(dim=0, keepdim=True).clamp_min(1e-6)

    def transform_spikes(self, X_sp_log1p: torch.Tensor) -> torch.Tensor:
        """Standardise the log-spike block to zero mean and unit variance."""
        return (X_sp_log1p - self.sp_mean) / self.sp_std

    def transform_rf(self, X_rf: torch.Tensor) -> torch.Tensor:
        """Standardise the RF block to zero mean and unit variance."""
        return (X_rf - self.rf_mean) / self.rf_std

    def transform(self, X_sp: torch.Tensor, X_rf: torch.Tensor, mode: str) -> torch.Tensor:
        """Return the feature tensor for the requested ablation mode.

        Supported modes:
            ``rf_only``         only the standardised RF block.
            ``spikes_only``     only the standardised log-spike block.
            ``spikes_plus_rf``  concatenation of both blocks.
        """
        if mode == "spikes_only":
            return self.transform_spikes(torch.log1p(X_sp))
        if mode == "rf_only":
            return self.transform_rf(X_rf)
        if mode == "spikes_plus_rf":
            sp = self.transform_spikes(torch.log1p(X_sp))
            rf = self.transform_rf(X_rf)
            return torch.cat([sp, rf], dim=1)
        raise ValueError(f"unknown mode: {mode!r}")


class RFOnlyReadout(nn.Module):
    """Two-layer MLP on the RF-block features (control baseline)."""

    def __init__(self, n_rf: int, n_cls: int = 10):
        """Construct a 256-unit ReLU MLP feeding ``n_cls`` logits."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rf, 256),
            nn.ReLU(),
            nn.Linear(256, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from the RF-only feature tensor."""
        return self.net(x)


class SpikesOnlyReadout(nn.Module):
    """Two-layer MLP on the spike-count block (measures STDP-only signal)."""

    def __init__(self, n_hidden: int, n_cls: int = 10):
        """Construct a 256-unit ReLU MLP feeding ``n_cls`` logits."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from the spike-only feature tensor."""
        return self.net(x)


class FusionReadout(nn.Module):
    """Two-branch readout with a learnable gate on the spike branch.

    The gate is a scalar sigmoid applied to the spike branch output; the RF
    branch is always passed through unscaled.  This lets the head learn how
    much of the STDP signal to keep on top of the RF baseline.
    """

    def __init__(self, n_hidden: int, n_rf: int, n_cls: int = 10):
        """Build both branches plus a small fusion head producing ``n_cls`` logits."""
        super().__init__()
        self.n_hidden = n_hidden
        self.n_rf = n_rf

        self.spike_branch = nn.Sequential(nn.Linear(n_hidden, 128), nn.ReLU())
        self.rf_branch = nn.Sequential(nn.Linear(n_rf, 128), nn.ReLU())
        self.spike_gate = nn.Parameter(torch.tensor(0.25))
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split the input into spike/RF parts, gate the spike branch and classify."""
        s = x[:, : self.n_hidden]
        r = x[:, self.n_hidden :]
        zs = self.spike_branch(s)
        zr = self.rf_branch(r)
        gate = torch.sigmoid(self.spike_gate)
        z = torch.cat([gate * zs, zr], dim=1)
        return self.head(z)


def build_readout(mode: str, n_hidden: int, n_rf: int, device: torch.device) -> nn.Module:
    """Instantiate the readout that matches the requested ablation mode."""
    if mode == "rf_only":
        return RFOnlyReadout(n_rf).to(device)
    if mode == "spikes_only":
        return SpikesOnlyReadout(n_hidden).to(device)
    if mode == "spikes_plus_rf":
        return FusionReadout(n_hidden, n_rf).to(device)
    raise ValueError(f"unknown mode: {mode!r}")
