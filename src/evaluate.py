"""Language-side evaluation of any model with the Model A architecture.

TEXT-SIDE MODULE: Model B code must never import this.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from src.data import agreement_probes, grammar_stats, CharTokenizer

PROMPTS = ["the ", "two dogs ", "a small cat near the ", "every farmer ", "some foxes with "]


@torch.no_grad()
def eval_loss(model, ids: torch.Tensor, block: int = 64, stride: int = 32, batch: int = 256) -> float:
    """Mean next-char cross-entropy (nats). Every scored char has >= block-stride context
    (except the first window)."""
    model.eval()
    n = ids.numel()
    starts = list(range(0, n - block - 1, stride))
    tot, cnt = 0.0, 0
    for b0 in range(0, len(starts), batch):
        ss = starts[b0:b0 + batch]
        x = torch.stack([ids[s:s + block] for s in ss])
        y = torch.stack([ids[s + 1:s + block + 1] for s in ss])
        logits = model(x)
        lp = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(len(ss), block)
        for i, s in enumerate(ss):
            lo = 0 if s == 0 else block - stride
            tot += lp[i, lo:].sum().item()
            cnt += block - lo
    return tot / cnt


@torch.no_grad()
def next_char_logprobs(model, ids: torch.Tensor, block: int = 64, stride: int = 32, max_windows: int = 200):
    """Log-prob distributions at scored positions (for model-to-model KL comparisons)."""
    model.eval()
    starts = list(range(block, ids.numel() - block - 1, stride))[:max_windows]
    x = torch.stack([ids[s:s + block] for s in starts])
    return F.log_softmax(model(x)[:, block - stride:, :], -1).reshape(-1, model.cfg.vocab_size)


@torch.no_grad()
def seq_logprob(model, tok: CharTokenizer, context: str, cont: str) -> float:
    ids = torch.tensor(tok.encode(context + cont))[None]
    ids = ids[:, -model.cfg.block_size - 1:]
    logits = model(ids[:, :-1])
    lp = F.log_softmax(logits, -1)[0]
    tgt = ids[0, 1:]
    n = len(cont)
    return lp[-n:].gather(1, tgt[-n:, None]).sum().item()


def agreement_accuracy(model, tok: CharTokenizer, probes=None) -> dict:
    probes = probes or agreement_probes()
    res = {"plain": [], "attractor": []}
    for p in probes:
        g = seq_logprob(model, tok, p["context"], p["good"])
        b = seq_logprob(model, tok, p["context"], p["bad"])
        res[p["kind"]].append(g > b)
    return {f"agree_{k}": sum(v) / len(v) for k, v in res.items()}


@torch.no_grad()
def generate_text(model, tok: CharTokenizer, prompt: str, n: int = 200, temperature: float = 0.8,
                  seed: int = 0, greedy: bool = False) -> tuple[str, float]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor(tok.encode(prompt))[None]
    t0 = time.perf_counter()
    out = model.generate(idx, n, temperature=temperature, generator=g, greedy=greedy)
    dt = time.perf_counter() - t0
    return tok.decode(out[0].tolist()), dt


def full_eval(model, tok: CharTokenizer, test_ids: torch.Tensor, n_gen_chars: int = 1500,
              with_samples: bool = True) -> dict:
    loss = eval_loss(model, test_ids)
    out = {"test_loss": loss, "test_ppl": math.exp(min(loss, 50)), "bits_per_char": loss / math.log(2)}
    out.update(agreement_accuracy(model, tok))
    long_text, dt = generate_text(model, tok, "the ", n=n_gen_chars, temperature=0.8, seed=123)
    out.update({f"gen_{k}": v for k, v in grammar_stats(long_text[4:]).items()})
    out["gen_latency_ms_per_char"] = 1000 * dt / n_gen_chars
    if with_samples:
        out["samples"] = {p: generate_text(model, tok, p, n=120, temperature=0.8, seed=7)[0] for p in PROMPTS}
        out["samples_greedy"] = {p: generate_text(model, tok, p, n=120, greedy=True)[0] for p in PROMPTS[:2]}
    return out
