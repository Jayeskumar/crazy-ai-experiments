"""Model A: a tiny character-level Transformer language model.

The same class is used as the fixed architecture for Model C (see model_c.py).
Attention is written with separate q/k/v/o linear layers so that the
permutation symmetries used by src/alignment.py are explicit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 30
    block_size: int = 64
    d_model: int = 32
    n_layer: int = 2
    d_mlp: int = 128

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, d)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).bool(),
                             persistent=False)

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
        x = x + self.o(att @ v)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 * 4)  # small model: slightly larger init
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.1)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 generator: torch.Generator | None = None, greedy: bool = False):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1, :]
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
