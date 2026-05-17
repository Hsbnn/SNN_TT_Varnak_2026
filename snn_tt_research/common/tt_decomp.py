"""Tensor-Train SVD and matrix low-rank decompositions used to compress weights."""

from __future__ import annotations

from typing import Tuple

import torch


def tt_svd_3way(W: torch.Tensor, r1: int, r2: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TT-SVD of a 3-way tensor ``W`` of shape ``[n1, n2, n3]``.

    Decomposes ``W[i, j, k] ≈ sum_{a, b} G1[i, a] * G2[a, j, b] * G3[b, k]``
    via two successive truncated SVDs.  The factorisation is computed on CPU
    so it runs on backends where SVD is not implemented on-device (e.g. MPS).

    Args:
        W: 3-way real tensor.
        r1: first TT rank (target).
        r2: second TT rank (target).

    Returns:
        ``(G1, G2, G3)`` with shapes ``[n1, r1]``, ``[r1, n2, r2]`` and
        ``[r2, n3]`` respectively, moved back to the original device.
    """
    n1, n2, n3 = W.shape
    W_cpu = W.detach().cpu().float().contiguous()

    C = W_cpu.reshape(n1, n2 * n3)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    r1_eff = min(r1, U.shape[1])

    G1 = U[:, :r1_eff].contiguous()
    R = (S[:r1_eff].unsqueeze(-1) * Vh[:r1_eff, :]).reshape(r1_eff * n2, n3)

    U2, S2, Vh2 = torch.linalg.svd(R, full_matrices=False)
    r2_eff = min(r2, U2.shape[1])

    G2 = U2[:, :r2_eff].reshape(r1_eff, n2, r2_eff).contiguous()
    G3 = (S2[:r2_eff].unsqueeze(-1) * Vh2[:r2_eff, :]).contiguous()

    return G1.to(W.device), G2.to(W.device), G3.to(W.device)


def reconstruct_weight_tt(
    G1: torch.Tensor, G2: torch.Tensor, G3: torch.Tensor
) -> torch.Tensor:
    """Reconstruct the dense 3-way weight tensor from its three TT cores."""
    return torch.einsum("ia,ajb,bk->ijk", G1, G2, G3)


def tt_core_num_params(G1: torch.Tensor, G2: torch.Tensor, G3: torch.Tensor) -> int:
    """Total number of stored parameters in the three TT cores."""
    return G1.numel() + G2.numel() + G3.numel()


def tt_runtime_param_count_cached(
    G1: torch.Tensor, G2: torch.Tensor, G3: torch.Tensor
) -> int:
    """Runtime parameter count when the contraction ``G2 ⊗ G3`` is cached.

    The cached representation stores ``W_mid`` of shape ``[r1, n_in]`` plus
    the outer core ``G1`` of shape ``[n_out, r1]``.
    """
    r1 = G1.shape[1]
    n2 = G2.shape[1]
    n3 = G3.shape[1]
    n_in = n2 * n3
    return r1 * n_in + G1.numel()


def fuse_tt_middle(G2: torch.Tensor, G3: torch.Tensor) -> torch.Tensor:
    """Fuse the middle and last TT cores into a single ``[r1, n_in]`` matrix.

    Used by inference-only modules to replace two contractions with one
    ``F.linear`` call per time step.
    """
    r1, n2, r2 = G2.shape
    n3 = G3.shape[1]
    return torch.einsum("ajb,bk->ajk", G2, G3).reshape(r1, n2 * n3).contiguous()


def matrix_lowrank_from_dense(
    W: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD factorisation ``W ≈ W_up @ W_down`` with the target rank.

    Returns ``(w_down, w_up)`` so that ``y = (z @ w_down.T) @ w_up.T`` matches
    the dense linear ``y = z @ W.T`` up to the truncation error.
    """
    n_out, n_in = W.shape
    r_eff = min(rank, n_out, n_in)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    w_up = (U[:, :r_eff] * S[:r_eff]).contiguous()
    w_down = Vh[:r_eff, :].contiguous()
    return w_down, w_up
