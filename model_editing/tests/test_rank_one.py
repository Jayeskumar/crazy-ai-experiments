"""Model-free tests of the update rules (run in seconds, no download needed)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods import random_dense, random_rank1, rome_update  # noqa: E402


def _keys(n=4000, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g) / d ** 0.5
    return torch.randn(n, d, generator=g) @ A + 0.1 * torch.randn(n, d, generator=g)


def test_rome_hits_target_exactly():
    K = _keys()
    cov = K.T @ K / K.shape[0]
    W = torch.randn(32, 64)
    k = K[0]
    delta = torch.randn(32)
    dW, info = rome_update(k, delta, cov)
    assert torch.allclose((W + dW) @ k, W @ k + delta, atol=1e-4)
    assert torch.linalg.matrix_rank(dW.double(), rtol=1e-5) == 1
    # factors reproduce the update
    assert torch.allclose(torch.outer(info["right"], info["left"]), dW, atol=1e-6)


def test_covariance_minimises_disturbance_on_other_keys():
    """Among all dW with dW k* = delta, ROME's choice minimises E|dW k|^2 = tr(dW C dW^T)."""
    K = _keys()
    cov = K.T @ K / K.shape[0]
    k, delta = K[0], torch.randn(32)
    dW_c, _ = rome_update(k, delta, cov)
    dW_i, _ = rome_update(k, delta, None)
    f = lambda D: torch.einsum("ij,jk,ik->", D.double(), cov.double(), D.double()).item()  # noqa: E731
    assert f(dW_c) <= f(dW_i) + 1e-9
    for s in range(5):  # any other solution: add a term orthogonal to k
        r = torch.randn(32, 64, generator=torch.Generator().manual_seed(s))
        r = r - torch.outer(r @ k, k) / (k @ k)
        assert f(dW_c) <= f(dW_c + 0.01 * r) + 1e-9


def test_random_controls_match_norm_and_rank():
    d1, _ = random_rank1((32, 64), 3.0, 0)
    d2, _ = random_dense((32, 64), 3.0, 0)
    assert abs(d1.norm().item() - 3.0) < 1e-4 and abs(d2.norm().item() - 3.0) < 1e-4
    assert torch.linalg.matrix_rank(d1) == 1 and torch.linalg.matrix_rank(d2) == 32
