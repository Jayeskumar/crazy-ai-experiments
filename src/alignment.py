"""Data-free permutation alignment of Model A weights (weight matching).

WEIGHTS-SIDE MODULE: operates only on named numpy arrays. Uses no data and no
activations (activation matching would need text and is therefore forbidden
for Model B's pipeline).

Symmetries handled (each is an exact functional symmetry of TinyTransformerLM):
  res      residual-stream permutation (embeddings, LayerNorms, every read/write)
  qk{l}    joint permutation of query/key dimensions in layer l (single head)
  vo{l}    joint permutation of value dims / attention-output input dims
  mlp{l}   permutation of MLP hidden units
Symmetries NOT handled: continuous GL(d) symmetries of q/k and v/o products,
per-neuron scale symmetries (broken by LayerNorm/GELU in most places).

Algorithm: coordinate-ascent weight matching (Ainsworth et al., Git Re-Basin,
2022), solving one linear assignment problem per permutation group per sweep.
Multiple models are aligned to a shared template that is iteratively re-estimated
as the mean of the aligned training models (as in "MergeMany").
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def perm_spec(names: list[str], n_layer: int) -> dict[str, tuple]:
    spec = {"tok_emb.weight": (None, "res"), "pos_emb.weight": (None, "res"),
            "ln_f.weight": ("res",), "ln_f.bias": ("res",),
            "head.weight": (None, "res"), "head.bias": (None,)}
    for l in range(n_layer):
        p = f"blocks.{l}."
        qk, vo, mlp = f"qk{l}", f"vo{l}", f"mlp{l}"
        spec.update({
            p + "ln1.weight": ("res",), p + "ln1.bias": ("res",),
            p + "q.weight": (qk, "res"), p + "q.bias": (qk,),
            p + "k.weight": (qk, "res"), p + "k.bias": (qk,),
            p + "v.weight": (vo, "res"), p + "v.bias": (vo,),
            p + "o.weight": ("res", vo), p + "o.bias": ("res",),
            p + "ln2.weight": ("res",), p + "ln2.bias": ("res",),
            p + "fc1.weight": (mlp, "res"), p + "fc1.bias": (mlp,),
            p + "fc2.weight": ("res", mlp), p + "fc2.bias": ("res",),
        })
    missing = set(names) ^ set(spec)
    if missing:
        raise ValueError(f"perm spec / parameter mismatch: {sorted(missing)}")
    return spec


def apply_perms(spec, perms, params, skip=None):
    out = {}
    for name, w in params.items():
        for ax, g in enumerate(spec[name]):
            if g is not None and (name, ax) != skip:
                w = np.take(w, perms[g], axis=ax)
        out[name] = w
    return out


def _group_axes(spec):
    g2 = defaultdict(list)
    for name, axes in spec.items():
        for ax, g in enumerate(axes):
            if g is not None:
                g2[g].append((name, ax))
    return g2


def weight_matching(spec, ref: dict, other: dict, max_iter: int = 50, seed: int = 0):
    """Find permutations so that apply_perms(spec, perms, other) ~ ref (max inner product)."""
    rng = np.random.default_rng(seed)
    g2 = _group_axes(spec)
    sizes = {g: other[n].shape[a] for g, lst in g2.items() for n, a in lst[:1]}
    perms = {g: np.arange(s) for g, s in sizes.items()}
    groups = sorted(g2)
    for _ in range(max_iter):
        progress = False
        for g in rng.permutation(groups):
            n = sizes[g]
            A = np.zeros((n, n))
            for name, ax in g2[g]:
                wa = np.moveaxis(ref[name], ax, 0).reshape(n, -1)
                wb = other[name]
                for ax2, g2_ in enumerate(spec[name]):
                    if g2_ is not None and ax2 != ax:
                        wb = np.take(wb, perms[g2_], axis=ax2)
                wb = np.moveaxis(wb, ax, 0).reshape(n, -1)
                A += wa @ wb.T
            _, ci = linear_sum_assignment(A, maximize=True)
            old = A[np.arange(n), perms[g]].sum()
            new = A[np.arange(n), ci].sum()
            if new > old + 1e-9:
                progress = True
                perms[g] = ci
        if not progress:
            break
    return perms


def build_template(spec, models: list[dict], rounds: int = 4, seed: int = 0):
    """Align a population to a common template (template = mean of aligned models)."""
    template = models[0]
    for r in range(rounds):
        aligned = [apply_perms(spec, weight_matching(spec, template, m, seed=seed + r), m) for m in models]
        template = {k: np.mean([a[k] for a in aligned], 0) for k in template}
    return template


def align_to_template(spec, template: dict, m: dict, seed: int = 0) -> dict:
    return apply_perms(spec, weight_matching(spec, template, m, seed=seed), m)


def random_permutation(spec, params, seed: int):
    """Apply a random symmetry transform (for invariance tests and augmentation)."""
    rng = np.random.default_rng(seed)
    g2 = _group_axes(spec)
    perms = {g: rng.permutation(params[lst[0][0]].shape[lst[0][1]]) for g, lst in g2.items()}
    return apply_perms(spec, perms, params)


# ---------------------------------------------------------------------------
# Continuous-symmetry gauge fixing for attention (data-free, exact)
# ---------------------------------------------------------------------------
def _sign_fix(U, V):
    s = np.sign((U ** 3).sum(0) + (V ** 3).sum(0))
    s[s == 0] = 1
    return U * s, V * s


def canonicalize_attention(params: dict, n_layer: int) -> dict:
    """Remove the GL(d) gauge freedom of q/k and v/o.

    q.k  = h^T Wq^T Wk h + ...  is invariant to Wq -> A Wq, Wk -> A^-T Wk (biases likewise),
    o(v) = Wo (Wv h + bv)       is invariant to Wv -> A Wv, bv -> A bv, Wo -> Wo A^-1.
    We fix A by the SVD of the invariant products M = Wq^T Wk = U S V^T and
    OV = Wo Wv = U S V^T, setting Wq = S^1/2 U^T, Wk = S^1/2 V^T, Wv = S^1/2 V^T, Wo = U S^1/2.
    The result is functionally identical, ordered by singular value and sign-fixed with
    a permutation-invariant rule, so it is equivariant to residual-stream permutations.
    """
    p = {k: v.astype(np.float64).copy() for k, v in params.items()}
    for l in range(n_layer):
        b = f"blocks.{l}."
        Wq, Wk, bq, bk = p[b + "q.weight"], p[b + "k.weight"], p[b + "q.bias"], p[b + "k.bias"]
        U, S, Vt = np.linalg.svd(Wq.T @ Wk)
        U, V = _sign_fix(U, Vt.T)
        Wq2, Wk2 = np.sqrt(S)[:, None] * U.T, np.sqrt(S)[:, None] * V.T
        A = Wq2 @ np.linalg.inv(Wq)
        p[b + "q.weight"], p[b + "k.weight"] = Wq2, Wk2
        p[b + "q.bias"], p[b + "k.bias"] = A @ bq, np.linalg.inv(A).T @ bk
        Wv, Wo, bv = p[b + "v.weight"], p[b + "o.weight"], p[b + "v.bias"]
        U, S, Vt = np.linalg.svd(Wo @ Wv)
        U, V = _sign_fix(U, Vt.T)
        Wv2, Wo2 = np.sqrt(S)[:, None] * V.T, U * np.sqrt(S)[None, :]
        A = Wv2 @ np.linalg.inv(Wv)
        p[b + "v.weight"], p[b + "o.weight"], p[b + "v.bias"] = Wv2, Wo2, A @ bv
    return {k: v.astype(np.float32) for k, v in p.items()}
