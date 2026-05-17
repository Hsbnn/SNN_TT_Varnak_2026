"""Iris loading with stratified split, standardisation and Gaussian RF encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..common.encoding import build_rf_centers_sigma, gaussian_rf_encode_1d
from .config import IrisConfig


@dataclass
class IrisBundle:
    """Container for the Iris dataset in all representations used downstream."""

    raw_train: DataLoader
    raw_test: DataLoader
    rf_train: DataLoader
    rf_test: DataLoader
    rf_test_tensor: torch.Tensor
    raw_test_tensor: torch.Tensor
    y_test_tensor: torch.Tensor


def load_iris_dataloaders(cfg: IrisConfig) -> IrisBundle:
    """Build train/test loaders for both the raw 4-D Iris input and its RF code.

    Steps:
        1. Stratified 75/25 split.
        2. Standardisation fitted on the train split.
        3. Construction of per-feature Gaussian centres from the train range.
        4. Joint min-max normalisation of the RF response into ``[0, 1]``.
    """
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=cfg.seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    centers, sigmas = build_rf_centers_sigma(X_train_s, cfg.n_per_feat)
    R_train = gaussian_rf_encode_1d(X_train_s, centers, sigmas)
    R_test = gaussian_rf_encode_1d(X_test_s, centers, sigmas)

    rmin, rmax = R_train.min(), R_train.max()
    R_train_n = (R_train - rmin) / (rmax - rmin + 1e-8)
    R_test_n = (R_test - rmin) / (rmax - rmin + 1e-8)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    R_train_t = torch.tensor(R_train_n, dtype=torch.float32)
    R_test_t = torch.tensor(R_test_n, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    raw_train = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=cfg.batch_size, shuffle=True
    )
    raw_test = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=cfg.batch_size, shuffle=False
    )
    rf_train = DataLoader(
        TensorDataset(R_train_t, y_train_t), batch_size=cfg.batch_size, shuffle=True
    )
    rf_test = DataLoader(
        TensorDataset(R_test_t, y_test_t), batch_size=cfg.batch_size, shuffle=False
    )

    return IrisBundle(
        raw_train=raw_train,
        raw_test=raw_test,
        rf_train=rf_train,
        rf_test=rf_test,
        rf_test_tensor=R_test_t,
        raw_test_tensor=X_test_t,
        y_test_tensor=y_test_t,
    )
