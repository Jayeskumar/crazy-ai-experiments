"""Model C: the fixed, predefined architecture that receives generated weights.

Model C is a TinyTransformerLM (same class as Model A, predefined in
configs/model_c.json). It is NEVER trained: `build_model_c` instantiates it and
loads a flat parameter vector produced by Model B (or by a baseline) with
strict shape checking. No optimizer exists anywhere in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.model_a import ModelConfig, TinyTransformerLM

ROOT = Path(__file__).resolve().parents[1]


def model_c_config() -> ModelConfig:
    return ModelConfig(**json.loads((ROOT / "configs" / "model_c.json").read_text()))


def build_model_c(vec: np.ndarray, names: list[str], shapes, cfg: ModelConfig | None = None) -> TinyTransformerLM:
    cfg = cfg or model_c_config()
    model = TinyTransformerLM(cfg)
    expected = model.state_dict()
    sd, i = {}, 0
    vec = np.asarray(vec, dtype=np.float32)
    for n, s in zip(names, shapes):
        s = tuple(int(x) for x in s if x >= 0)
        if n not in expected or tuple(expected[n].shape) != s:
            raise ValueError(f"generated tensor {n}{s} incompatible with Model C")
        k = int(np.prod(s))
        sd[n] = torch.from_numpy(vec[i:i + k].reshape(s).copy())
        i += k
    if i != vec.size:
        raise ValueError(f"vector has {vec.size} values, layout needs {i}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # C is inference-only
    return model


def random_model_c(seed: int, cfg: ModelConfig | None = None) -> TinyTransformerLM:
    torch.manual_seed(seed)
    m = TinyTransformerLM(cfg or model_c_config())
    m.eval()
    return m
