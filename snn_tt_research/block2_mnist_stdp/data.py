"""MNIST loaders and the Gaussian RF projection used by the STDP encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..common.device import pin_memory_flag
from ..common.encoding import build_gaussian_rf_weights_2d, image_to_rf_rates
from ..common.spike import poisson_from_rates
from .config import MNISTStdpConfig


@dataclass
class MNISTRFEncoder:
    """Encapsulate the fixed Gaussian RF weights and the Poisson spike encoder.

    Holding both inside one object keeps the spike train pipeline reproducible
    across the dense and the TT-compressed model.
    """

    rf_weights: torch.Tensor
    cfg: MNISTStdpConfig

    def image_to_rates(self, x: torch.Tensor) -> torch.Tensor:
        """Project a batch of pixel-domain images onto the RF grid."""
        return image_to_rf_rates(x, self.rf_weights)

    def poisson(self, rates: torch.Tensor, T: int | None = None) -> torch.Tensor:
        """Sample a Poisson spike train using the encoder's clamp parameters."""
        T = T if T is not None else self.cfg.sim_time
        return poisson_from_rates(rates, T, gain=self.cfg.input_gain, max_prob=self.cfg.poisson_max_prob)


def build_mnist_loaders(cfg: MNISTStdpConfig, device: torch.device, data_root: str = "data"):
    """Return MNIST train/test loaders alongside the RF encoder.

    Images are kept as raw ``[0, 1]`` tensors; the Gaussian RF projection is
    applied lazily inside the encoder so the same loaders feed both the
    spiking pipeline and any baselines.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    pin = pin_memory_flag(device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    rf_weights = build_gaussian_rf_weights_2d(cfg.img_size, cfg.rf_grid, device)
    encoder_io = MNISTRFEncoder(rf_weights=rf_weights, cfg=cfg)
    return train_loader, test_loader, encoder_io
