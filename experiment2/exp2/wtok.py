"""Row tokenisation of a parameter vector (weights-side, exact and invertible).

Each token is one "unit" of a tensor, at most L=d_model+1 numbers:
  Linear with in_dim+1 <= L : token = [W[i, :], b[i]]          (one per output neuron)
  Linear with in_dim+1 >  L : token = W[:, j] (one per input/hidden unit) + one token for the bias
  Embedding / head rows     : token = row (head rows include their bias)
  LayerNorm weight / bias   : one token each
Tokens carry a type id (which tensor/orientation) and an index (row/column number).
"""
from __future__ import annotations

import numpy as np


class RowTokenizer:
    def __init__(self, names, shapes):
        shapes = [tuple(int(x) for x in s if x >= 0) for s in shapes]
        offs, o = {}, 0
        for n, s in zip(names, shapes):
            offs[n] = (o, s)
            o += int(np.prod(s))
        self.P = o
        d = max(s[-1] for s in shapes if len(s) == 2 and s[-1] <= 64)
        self.L = L = d + 1
        idx, types, pos, type_names = [], [], [], []

        def flat(n, i=None, j=None):
            off, s = offs[n]
            if len(s) == 1:
                return off + np.arange(s[0]) if i is None else np.array([off + i])
            if j is None:
                return off + i * s[1] + np.arange(s[1])
            return off + np.arange(s[0]) * s[1] + j

        def add(tname, flat_list, index):
            if tname not in type_names:
                type_names.append(tname)
            row = np.concatenate(flat_list)
            assert len(row) <= L, (tname, len(row))
            idx.append(np.pad(row, (0, L - len(row)), constant_values=-1))
            types.append(type_names.index(tname))
            pos.append(index)

        done = set()
        for n, s in zip(names, shapes):
            if n in done:
                continue
            base = n.rsplit(".", 1)[0]
            bias = base + ".bias"
            if len(s) == 1:
                for i in range(1):
                    add(n, [flat(n)], 0)
                done.add(n)
            elif n.endswith(".weight") and bias in offs and len(offs[bias][1]) == 1 and offs[bias][1][0] == s[0]:
                if s[1] + 1 <= L:
                    for i in range(s[0]):
                        add(base + ":row", [flat(n, i), flat(bias, i)], i)
                else:
                    for j in range(s[1]):
                        add(base + ":col", [flat(n, None, j)], j)
                    add(base + ":bias", [flat(bias)], 0)
                done |= {n, bias}
            else:
                for i in range(s[0]):
                    add(n + ":row", [flat(n, i)], i)
                done.add(n)
        self.idx = np.stack(idx)                 # [N, L] flat index or -1
        self.mask = self.idx >= 0                # valid entries
        self.types = np.array(types)
        self.pos = np.array(pos)
        self.type_names = type_names
        self.N = len(idx)
        cover = np.sort(self.idx[self.mask])
        assert np.array_equal(cover, np.arange(self.P)), "tokenisation must cover each parameter exactly once"

    def to_tokens(self, W: np.ndarray) -> np.ndarray:
        """[..., P] -> [..., N, L] (padding = 0)"""
        out = np.take(W, np.where(self.mask, self.idx, 0), axis=-1)
        return np.where(self.mask, out, 0.0).astype(np.float32)

    def from_tokens(self, T: np.ndarray) -> np.ndarray:
        """[..., N, L] -> [..., P]"""
        lead = T.shape[:-2]
        out = np.zeros(lead + (self.P,), np.float32)
        out[..., self.idx[self.mask]] = T[..., self.mask]
        return out
