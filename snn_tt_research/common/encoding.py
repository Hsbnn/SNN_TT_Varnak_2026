"""Gaussian receptive-field encoders for tabular and image inputs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def build_rf_centers_sigma(
    X_train: np.ndarray, n_per_feature: int
) -> Tuple[List[np.ndarray], List[float]]:
    """Build per-feature Gaussian centres and widths from training data.

    Centres are spread uniformly across the empirical training range of each
    feature.  ``sigma`` is set to roughly the grid step so neighbouring
    receptive fields overlap.

    Args:
        X_train: ``[N, F]`` real-valued matrix (typically standardised).
        n_per_feature: number of Gaussian centres per input feature.

    Returns:
        ``(centers, sigmas)`` where ``centers[j]`` is a 1-D array of length
        ``n_per_feature`` and ``sigmas[j]`` is a scalar bandwidth.
    """
    centers: List[np.ndarray] = []
    sigmas: List[float] = []
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-8:
            hi = lo + 1e-6
        c = np.linspace(lo, hi, n_per_feature, dtype=np.float64)
        sigma = (hi - lo) / (n_per_feature * 1.2) + 1e-6
        centers.append(c)
        sigmas.append(sigma)
    return centers, sigmas


def gaussian_rf_encode_1d(
    X: np.ndarray, centers: List[np.ndarray], sigmas: List[float]
) -> np.ndarray:
    """Encode each scalar feature into a Gaussian population code.

    For each column ``j`` and centre ``c`` the response is
    ``exp(-(x - c)^2 / (2 sigma_j^2))``.  Responses across all features are
    concatenated, producing an output of dimension ``F * len(centers[0])``.
    """
    parts: List[np.ndarray] = []
    for j in range(X.shape[1]):
        xj = X[:, j : j + 1]
        c = centers[j][None, :]
        s = sigmas[j]
        g = np.exp(-0.5 * ((xj - c) / s) ** 2)
        parts.append(g)
    return np.concatenate(parts, axis=1)


def normalise_to_unit_interval(R: np.ndarray) -> np.ndarray:
    """Linearly rescale the array to ``[0, 1]`` using its global min/max."""
    rmin, rmax = R.min(), R.max()
    return (R - rmin) / (rmax - rmin + 1e-8)


def build_gaussian_rf_weights_2d(
    img_size: int, grid: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build a fixed Gaussian RF projection from pixels to ``grid * grid`` units.

    The grid is centred on the image; each unit receives a Gaussian-weighted
    sum of pixel intensities normalised so rows sum to one.  The result has
    shape ``[grid * grid, img_size * img_size]``.
    """
    xs = torch.linspace(0.0, float(img_size - 1), grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    sigma = 1.05 * (img_size / grid)

    gy, gx = torch.meshgrid(
        torch.arange(img_size, device=device, dtype=dtype),
        torch.arange(img_size, device=device, dtype=dtype),
        indexing="ij",
    )
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    d2 = (pos.unsqueeze(0) - centers.unsqueeze(1)).pow(2).sum(-1)
    w = torch.exp(-d2 / (2.0 * sigma * sigma))
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    return w


def image_to_rf_rates(images: torch.Tensor, rf_weights: torch.Tensor) -> torch.Tensor:
    """Project a batch of images onto the RF grid producing per-unit intensities."""
    B = images.size(0)
    flat = images.view(B, -1)
    return flat @ rf_weights.T
