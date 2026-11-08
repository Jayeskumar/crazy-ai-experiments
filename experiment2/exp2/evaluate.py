"""Capability evaluation for any network with the shared architecture.

TEXT-SIDE MODULE (never importable by Model B).
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from exp2.tasks import (ALL_VARIANTS, TASKS, agreement_probes, encode, decode, grammar_score, grammar_sentence,
                        instance, split_text)

_CACHE = {}


def _instance_batch(task, variant, n=256, seed=777):
    key = (task, variant, n, seed)
    if key not in _CACHE:
        rng = random.Random(seed)
        xs, ans_len = [], None
        for _ in range(n):
            ctx = " ".join(p + a for p, a in (instance(rng, task, variant, "train") for _ in range(2)))
            p, a = instance(rng, task, variant, "test")
            xs.append(encode(ctx + " " + p + a))
            ans_len = len(a)
        assert len({len(x) for x in xs}) == 1
        _CACHE[key] = (torch.tensor(xs), ans_len)
    return _CACHE[key]


@torch.no_grad()
def exact_match(model, task, variant) -> float:
    """Greedy exact-match accuracy on held-out instances (teacher-forced argmax over the whole
    answer == greedy decoding being exactly right)."""
    x, L = _instance_batch(task, variant)
    logits = model(x[:, :-1])
    pred = logits[:, -L:].argmax(-1)
    return (pred == x[:, -L:]).all(1).float().mean().item()


@torch.no_grad()
def sample(model, prompt: str, n_seq=8, n_chars=160, temperature=0.7, seed=0) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([encode(prompt)] * n_seq)
    for _ in range(n_chars):
        logits = model(idx[:, -model.cfg.block_size:])[:, -1] / temperature
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1, generator=g)], 1)
    return [decode(r[len(prompt):]) for r in idx.tolist()]


def grammar_gen(model, variant, seed=0):
    prompt = grammar_sentence(random.Random(seed + 5), variant) + " "
    texts = sample(model, prompt, seed=seed)
    # drop the first (possibly partial) sentence fragment? prompt ends with '. ' so text starts a sentence
    scores = {v: grammar_score("".join(texts), v)[0] for v in TASKS["grammar"]}
    return scores, texts


@torch.no_grad()
def agreement(model, variant) -> float:
    ok = 0
    probes = agreement_probes(variant)
    for ctx, good, bad in probes:
        lps = []
        for cont in (good, bad):
            ids = torch.tensor([encode(ctx + cont)])[:, -model.cfg.block_size - 1:]
            lp = F.log_softmax(model(ids[:, :-1]), -1)[0]
            lps.append(lp[-len(cont):].gather(1, ids[0, -len(cont):, None]).sum().item())
        ok += lps[0] > lps[1]
    return ok / len(probes)


@torch.no_grad()
def task_loss(model, task, variant, n_chars=6000, block=48, stride=24) -> float:
    ids = torch.tensor(encode(split_text(task, variant, "test", n_chars)))
    starts = list(range(0, ids.numel() - block - 1, stride))
    x = torch.stack([ids[s:s + block] for s in starts])
    y = torch.stack([ids[s + 1:s + block + 1] for s in starts])
    lp = F.cross_entropy(model(x).transpose(1, 2), y, reduction="none")
    return lp[:, block - stride:].mean().item()


@torch.no_grad()
def next_char_logprobs(model, task, variant, n_chars=4000, block=48, stride=24):
    ids = torch.tensor(encode(split_text(task, variant, "test", n_chars)))
    starts = list(range(0, ids.numel() - block - 1, stride))
    x = torch.stack([ids[s:s + block] for s in starts])
    return F.log_softmax(model(x)[:, block - stride:], -1).reshape(-1, model.cfg.vocab_size)


def capability_vector(model, with_samples=False) -> dict:
    """Score on every (task, variant). Instance tasks: exact match; grammar: fraction of
    sampled sentences grammatical in that variant (sampled with that variant's prompt)."""
    model.eval()
    out = {}
    samples = {}
    for t, v in ALL_VARIANTS:
        if t == "grammar":
            sc, texts = grammar_gen(model, v)
            out[f"{t}/{v}"] = sc[v]
            samples[f"{t}/{v}"] = texts[:2]
        else:
            out[f"{t}/{v}"] = exact_match(model, t, v)
    if with_samples:
        for t, v in ALL_VARIANTS:
            if t != "grammar":
                samples[f"{t}/{v}"] = sample(model, split_text(t, v, "test", 40)[:30], n_seq=1, n_chars=40)
        out["_samples"] = samples
    return out
