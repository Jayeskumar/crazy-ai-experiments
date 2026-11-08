"""Model B (Experiment 2): a row-token weight autoencoder with a per-token bottleneck.

WEIGHTS-SIDE MODULE. Input: one model's parameters as N row tokens (exp2/wtok.py), each
normalised by its tensor's population std, plus an "observed" mask (for masked-recovery
training). Output: all parameters.

  encoder: token MLP on [values*obs, obs] + type/index embeddings
           + model-level context (mean-pooled, broadcast back)   -> per-token code z (r dims)
  decoder: token MLP on z + type/index embeddings + pooled context of z -> values

Every token has its own code, so the output is input-dependent by construction; how much of
each model survives is controlled by r (and int8 quantisation of z).

Linear control with the same interface: RowPCA (per-tensor-type PCA codebook of rows).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        self.ctx = nn.Linear(h, h)

    def forward(self, x):
        h = self.ln(x)
        h = h + self.ctx(h.mean(1, keepdim=True))       # model-level context
        return x + self.ff(h)


class RowAE(nn.Module):
    def __init__(self, L, n_types, max_pos, r, hidden=96, depth=2):
        super().__init__()
        self.r = r
        self.inp = nn.Linear(2 * L, hidden)
        self.te_e, self.pe_e = nn.Embedding(n_types, hidden), nn.Embedding(max_pos, hidden)
        self.enc = nn.ModuleList([Block(hidden) for _ in range(depth)])
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, r))
        self.from_z = nn.Linear(r, hidden)
        self.te_d, self.pe_d = nn.Embedding(n_types, hidden), nn.Embedding(max_pos, hidden)
        self.dec = nn.ModuleList([Block(hidden) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, L))

    def encode(self, v, obs, types, pos):
        x = self.inp(torch.cat([v * obs, obs], -1)) + self.te_e(types) + self.pe_e(pos)
        for b in self.enc:
            x = b(x)
        return self.to_z(x)

    def decode(self, z, types, pos):
        x = self.from_z(z) + self.te_d(types) + self.pe_d(pos)
        for b in self.dec:
            x = b(x)
        return self.out(x)

    def forward(self, v, obs, types, pos, z_noise=0.0):
        z = self.encode(v, obs, types, pos)
        if z_noise:
            z = z + z_noise * torch.randn_like(z)
        return self.decode(z, types, pos), z


def quantize_int8(z: np.ndarray) -> np.ndarray:
    """Per-model, per-latent-dim symmetric int8 quantisation (round trip)."""
    s = np.abs(z).max(axis=-2, keepdims=True) / 127.0
    s = np.where(s == 0, 1.0, s)
    return (np.round(z / s) * s).astype(np.float32)


class RowPCA:
    """Linear control: per token-type PCA codebook (r components) fitted on B-train rows."""

    def __init__(self, Tn_train: np.ndarray, valid: np.ndarray, types: np.ndarray, r: int):
        self.r, self.bases, self.means = r, {}, {}
        for t in np.unique(types):
            sel = types == t
            X = Tn_train[:, sel].reshape(-1, Tn_train.shape[-1])
            m = X.mean(0)
            _, _, Vt = np.linalg.svd(X - m, full_matrices=False)
            k = min(r, Vt.shape[0])
            self.bases[t], self.means[t] = Vt[:k], m
        self.types = types
        self.valid = valid

    def reconstruct(self, Tn: np.ndarray, q8=True, obs=None) -> np.ndarray:
        out = np.zeros_like(Tn)
        for t, V in self.bases.items():
            sel = self.types == t
            X = Tn[:, sel] - self.means[t]
            if obs is None:
                Z = X @ V.T
            else:   # least-squares codes from observed entries only (masked recovery)
                Z = np.zeros(X.shape[:2] + (len(V),), np.float32)
                O = obs[:, sel]
                for b in range(X.shape[0]):
                    for i in range(X.shape[1]):
                        o = O[b, i] > 0
                        if o.any():
                            Z[b, i] = np.linalg.lstsq(V[:, o].T, X[b, i, o], rcond=None)[0]
            if q8:
                Z = quantize_int8(Z)
            out[:, sel] = Z @ V + self.means[t]
        return out * self.valid
