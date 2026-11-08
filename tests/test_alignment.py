import numpy as np
import torch

from src.alignment import perm_spec, random_permutation, weight_matching, apply_perms
from src.model_a import ModelConfig, TinyTransformerLM


def _model(seed):
    torch.manual_seed(seed)
    m = TinyTransformerLM(ModelConfig(vocab_size=25))
    # make weights non-trivial (biases are zero at init)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def _np(m):
    return {k: v.numpy().copy() for k, v in m.state_dict().items()}


def _load(params):
    m = TinyTransformerLM(ModelConfig(vocab_size=25))
    m.load_state_dict({k: torch.tensor(v) for k, v in params.items()})
    return m.eval()


def test_permutation_is_exact_symmetry():
    m = _model(0)
    p = _np(m)
    spec = perm_spec(list(p), 2)
    q = random_permutation(spec, p, seed=3)
    assert any(not np.allclose(p[k], q[k]) for k in p)
    x = torch.randint(0, 25, (4, 64))
    assert torch.allclose(m(x), _load(q)(x), atol=1e-5)


def test_weight_matching_recovers_hidden_permutation():
    p = _np(_model(1))
    spec = perm_spec(list(p), 2)
    q = random_permutation(spec, p, seed=7)
    perms = weight_matching(spec, p, q)
    r = apply_perms(spec, perms, q)
    for k in p:
        assert np.allclose(p[k], r[k], atol=1e-6), k
