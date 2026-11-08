"""Model B: a weight-space autoencoder (the "weight learner").

WEIGHTS-SIDE MODULE. Model B's inputs and targets are flattened, permutation-
aligned parameter vectors of Model A instances, nothing else. It has no access
to tokens, text, logits or activations; its loss is reconstruction MSE in
(normalised) weight space.

Architecture
  normaliser : x_n = (x - mu) / sigma_tensor   (mu: per-coordinate mean of the
               B-train population; sigma: per-tensor std of the deviations).
               Both are statistics of weights only and are part of B.
  encoder    : P -> H -> H -> k   (GELU MLP, dropout)
  decoder    : k -> H -> H -> P   (GELU MLP)
Training (src/train_b.py): denoising + masked-tensor reconstruction, i.e. whole
parameter tensors of the input are replaced with the population mean and B must
reconstruct the full vector. A small latent noise / L2 penalty keeps the latent
space smooth enough to sample from.

Also here: `PCAModel`, the linear-autoencoder control (identical interface).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class WeightNormalizer(nn.Module):
    def __init__(self, W_train: np.ndarray, tensor_slices: list[slice]):
        super().__init__()
        mu = W_train.mean(0)
        dev = W_train - mu
        sigma = np.ones_like(mu)
        for sl in tensor_slices:
            s = dev[:, sl].std()
            sigma[sl] = s if s > 1e-8 else 1.0
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

    def norm(self, x):
        return (x - self.mu) / self.sigma

    def denorm(self, xn):
        return xn * self.sigma + self.mu


class WeightAE(nn.Module):
    def __init__(self, P: int, k: int, hidden: int = 256, dropout: float = 0.1,
                 normalizer: WeightNormalizer | None = None):
        super().__init__()
        self.P, self.k = P, k
        self.normalizer = normalizer
        self.enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(P, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, k))
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, P))

    def encode_n(self, xn):
        return self.enc(xn)

    def decode_n(self, z):
        return self.dec(z)

    def forward(self, xn):
        z = self.encode_n(xn)
        return self.decode_n(z), z

    # raw-weight-space convenience API (numpy in, numpy out)
    @torch.no_grad()
    def encode(self, W: np.ndarray) -> np.ndarray:
        self.eval()
        return self.encode_n(self.normalizer.norm(torch.as_tensor(W, dtype=torch.float32))).numpy()

    @torch.no_grad()
    def decode(self, Z: np.ndarray) -> np.ndarray:
        self.eval()
        return self.normalizer.denorm(self.decode_n(torch.as_tensor(Z, dtype=torch.float32))).numpy()

    @torch.no_grad()
    def reconstruct(self, W: np.ndarray, mask_slices: list[slice] | None = None) -> np.ndarray:
        """Encode/decode. If mask_slices given, those coordinates are hidden from the
        encoder (replaced by the population mean, i.e. 0 in normalised space)."""
        self.eval()
        xn = self.normalizer.norm(torch.as_tensor(W, dtype=torch.float32))
        if mask_slices:
            xn = xn.clone()
            for sl in mask_slices:
                xn[..., sl] = 0.0
        return self.normalizer.denorm(self.decode_n(self.encode_n(xn))).numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PCAModel:
    """Linear autoencoder control: mean + top-k principal directions of the
    (normalised) training weight vectors."""

    def __init__(self, W_train: np.ndarray, k: int, tensor_slices: list[slice]):
        self.normalizer = WeightNormalizer(W_train, tensor_slices)
        self.mu = self.normalizer.mu.numpy()
        self.sigma = self.normalizer.sigma.numpy()
        Xn = (W_train - self.mu) / self.sigma
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
        self.k = min(k, Vt.shape[0])
        self.V = Vt[:self.k]                     # [k, P]
        self.explained = (S ** 2) / (S ** 2).sum()

    def encode(self, W):
        return ((W - self.mu) / self.sigma) @ self.V.T

    def decode(self, Z):
        return (Z @ self.V) * self.sigma + self.mu

    def reconstruct(self, W, mask_slices=None):
        Xn = (W - self.mu) / self.sigma
        if not mask_slices:
            return self.decode(Xn @ self.V.T)
        # masked recovery: least-squares fit of the latent on visible coordinates only
        vis = np.ones(Xn.shape[-1], bool)
        for sl in mask_slices:
            vis[sl] = False
        Vv = self.V[:, vis]
        Z = np.linalg.lstsq(Vv.T, np.atleast_2d(Xn)[:, vis].T, rcond=None)[0].T
        return self.decode(Z).reshape(W.shape)

    def n_params(self) -> int:
        return int(self.V.size + self.mu.size + self.sigma.size)
