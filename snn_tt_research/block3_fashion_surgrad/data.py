"""Fashion-MNIST loaders and the shared initial tensors used by every model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms

from ..common.device import pin_memory_flag, set_seed
from .config import FashionConfig


@dataclass
class FashionLoaders:
    """Container exposing the train/val/test loaders and a per-epoch loader factory."""

    train_ds: Subset
    val_ds: Subset
    test_ds: datasets.FashionMNIST
    val_loader: DataLoader
    test_loader: DataLoader
    cfg: FashionConfig
    pin: bool

    def make_train_loader(self, epoch: int) -> DataLoader:
        """Return a fresh shuffled loader seeded with ``epoch`` for reproducibility."""
        g = torch.Generator()
        g.manual_seed(10_000 + epoch)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            sampler=RandomSampler(self.train_ds, generator=g),
            num_workers=0,
            pin_memory=self.pin,
        )


def build_fashion_loaders(
    cfg: FashionConfig, device: torch.device, data_root: str = "data"
) -> FashionLoaders:
    """Download Fashion-MNIST, build a stratified train/val/test split.

    Returns a :class:`FashionLoaders` that knows how to mint train loaders
    with a different RNG seed every epoch.
    """
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)

    targets = np.array(train_full.targets)
    idx_all = np.arange(len(train_full))
    train_idx, val_idx = train_test_split(
        idx_all, test_size=cfg.val_split, random_state=123, stratify=targets
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)

    pin = pin_memory_flag(device)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    return FashionLoaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        pin=pin,
    )


def make_shared_initial_tensors(
    cfg: FashionConfig, seed: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-experiment shared initialisation of ``(w1, b1, w2, b2)``.

    Sharing weights across the dense, TT and low-rank variants makes the
    comparison fair: every model starts from the same input layer.
    """
    seed = cfg.model_init_seed if seed is None else seed
    set_seed(seed)
    w1 = torch.empty(cfg.hidden, cfg.flat)
    nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
    b1 = torch.zeros(cfg.hidden)
    w2 = torch.empty(cfg.num_classes, cfg.hidden)
    nn.init.kaiming_uniform_(w2, a=math.sqrt(5))
    b2 = torch.zeros(cfg.num_classes)
    return w1, b1, w2, b2
