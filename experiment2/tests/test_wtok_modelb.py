import numpy as np
import torch

from exp2.model import TinyTransformerLM, arch_config
from exp2.model_b import RowAE, RowPCA, quantize_int8
from exp2.wtok import RowTokenizer


def _layout():
    sd = TinyTransformerLM(arch_config()).state_dict()
    names = list(sd)
    shapes = [list(v.shape) + [-1] * (2 - v.dim()) for v in sd.values()]
    return sd, names, shapes


def test_tokenizer_is_exact_bijection():
    sd, names, shapes = _layout()
    tk = RowTokenizer(names, shapes)
    W = np.random.default_rng(0).standard_normal((3, tk.P)).astype(np.float32)
    assert np.array_equal(tk.from_tokens(tk.to_tokens(W)), W)
    assert tk.P == sum(v.numel() for v in sd.values()) and tk.L == 33


def test_rowae_shapes_and_quant():
    sd, names, shapes = _layout()
    tk = RowTokenizer(names, shapes)
    m = RowAE(tk.L, len(tk.type_names), 128, 4, hidden=16, depth=1)
    v = torch.randn(2, tk.N, tk.L)
    obs = torch.ones_like(v)
    out, z = m(v, obs, torch.tensor(tk.types)[None], torch.tensor(tk.pos)[None])
    assert out.shape == v.shape and z.shape == (2, tk.N, 4)
    q = quantize_int8(z.detach().numpy())
    assert np.abs(q - z.detach().numpy()).max() <= np.abs(z.detach().numpy()).max() / 127


def test_rowpca_exact_when_full_rank():
    sd, names, shapes = _layout()
    tk = RowTokenizer(names, shapes)
    T = tk.to_tokens(np.random.default_rng(0).standard_normal((5, tk.P)).astype(np.float32))
    p = RowPCA(T, tk.mask, tk.types, 33)
    assert np.abs(p.reconstruct(T, q8=False) - T).max() < 1e-3
