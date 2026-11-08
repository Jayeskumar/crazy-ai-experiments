"""Shared architecture for every Model A and Model C (reused, tested Experiment-1 class)."""
import json

from src.model_a import ModelConfig, TinyTransformerLM, param_count  # noqa: F401

from exp2 import ROOT


def arch_config() -> ModelConfig:
    return ModelConfig(**json.loads((ROOT / "configs" / "arch.json").read_text()))
