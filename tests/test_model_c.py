import numpy as np
import pytest
import torch

from src.model_a import ModelConfig, TinyTransformerLM
from src.model_c import build_model_c, model_c_config
from src.weights_only import flatten


def test_model_c_matches_source_and_is_frozen():
    torch.manual_seed(0)
    a = TinyTransformerLM(model_c_config()).eval()
    sd = {k: v.numpy() for k, v in a.state_dict().items()}
    names = list(sd)
    shapes = [list(v.shape) + [-1] * (2 - v.ndim) for v in sd.values()]
    c = build_model_c(flatten(sd, names), names, shapes)
    x = torch.randint(0, model_c_config().vocab_size, (2, 30))
    assert torch.allclose(a(x), c(x))
    assert not any(p.requires_grad for p in c.parameters())


def test_model_c_rejects_incompatible_vectors():
    torch.manual_seed(0)
    a = TinyTransformerLM(model_c_config())
    sd = {k: v.numpy() for k, v in a.state_dict().items()}
    names = list(sd)
    shapes = [list(v.shape) + [-1] * (2 - v.ndim) for v in sd.values()]
    with pytest.raises(ValueError):
        build_model_c(np.zeros(10, np.float32), names, shapes)
