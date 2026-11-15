"""Model loading, tokenisation helpers, scoring, hooks, and original-weight guards."""
import hashlib
import json
import os
import random

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config as C


def seed_all(s=C.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load(model_dir=C.MODEL_DIR):
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token_id = tok.unk_token_id  # Llama has no pad token; padded positions are masked anyway
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def down_proj(model, layer):
    """The MLP output projection of a decoder layer: out = W @ k with k = silu(gate(x)) * up(x).
    In ROME terms k is the 'key' (5632-d) and W k the 'value' (2048-d) written to the residual stream."""
    return model.model.layers[layer].mlp.down_proj


def tensor_name(layer):
    return f"model.layers.{layer}.mlp.down_proj.weight"


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_matches_original(model, names=None):
    """Exact (bitwise) comparison of in-memory parameters with the read-only original file."""
    sd = model.state_dict()
    with safe_open(C.ORIG_WEIGHTS, "pt") as f:
        keys = names if names is not None else list(f.keys())
        for k in keys:
            if not torch.equal(sd[k], f.get_tensor(k)):
                raise AssertionError(f"in-memory tensor {k} differs from original weights")
    return len(keys)


def original_tensor(name):
    with safe_open(C.ORIG_WEIGHTS, "pt") as f:
        return f.get_tensor(name)


# ---------------------------------------------------------------- tokenisation

def encode_with_target(tok, prompt, target):
    """Tokenise `prompt + ' ' + target` as one string; return ids and the target token positions.
    Offsets let us find target tokens without assuming how SentencePiece splits the boundary."""
    text = prompt + " " + target
    enc = tok(text, return_offsets_mapping=True)
    ids = enc["input_ids"]
    tgt = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= len(prompt) and e > s]
    assert tgt and tgt == list(range(tgt[0], len(ids))), (text, tgt)
    return ids, tgt


def subject_last_index(tok, text, subj_start, subj_end):
    """Index of the last token of the subject span [subj_start, subj_end) in `text`."""
    enc = tok(text, return_offsets_mapping=True)
    idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < subj_end and e > subj_start]
    assert idx, (text, subj_start, subj_end)
    return enc["input_ids"], idx[-1], idx


def fill(template, prompt, subject):
    """Return (text, subject_start, subject_end) for template.format(prompt.format(subject))."""
    pre = template.split("{}")[0]
    p = prompt.split("{}")
    text = pre + p[0] + subject + p[1]
    s = len(pre) + len(p[0])
    assert template.format(prompt.format(subject)) == text
    return text, s, s + len(subject)


def pad_batch(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def score(model, tok, pairs, bs=C.EVAL["batch_size"], want_next=False):
    """pairs: list of (prompt, target). Teacher-forced log-probs of the target tokens (right padding)."""
    out = []
    for b in range(0, len(pairs), bs):
        chunk = pairs[b:b + bs]
        enc = [encode_with_target(tok, p, t) for p, t in chunk]
        ids, mask = pad_batch([e[0] for e in enc], tok.pad_token_id)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        for i, (seq, tgt) in enumerate(enc):
            pos = torch.tensor(tgt) - 1
            t_ids = torch.tensor([seq[j] for j in tgt])
            lp = logp[i, pos, t_ids]
            top1 = bool((logp[i, pos].argmax(-1) == t_ids).all())
            r = dict(mean_lp=lp.mean().item(), sum_lp=lp.sum().item(), first_lp=lp[0].item(),
                     top1=top1, n_tok=len(tgt))
            if want_next:
                r["next"] = logp[i, tgt[0] - 1].clone()
            out.append(r)
    return out


@torch.no_grad()
def greedy(model, tok, prompt, n):
    enc = tok(prompt, return_tensors="pt")
    o = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id,
                       max_length=None)
    return tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def first_word(s):
    w = s.strip().split()
    return w[0].strip(".,;:!?\"'()") if w else ""


# ---------------------------------------------------------------- hooks

class CaptureKey:
    """forward-pre-hook on down_proj: store its input (the key) for every position."""
    def __init__(self, module):
        self.h = module.register_forward_pre_hook(self)
        self.k = None

    def __call__(self, mod, args):
        self.k = args[0].detach()

    def remove(self):
        self.h.remove()


class AddAtPositions:
    """forward-hook on down_proj: add `delta` to its output at one position per batch row."""
    def __init__(self, module, positions, delta):
        self.positions, self.delta = positions, delta
        self.h = module.register_forward_hook(self)
        self.out = None

    def __call__(self, mod, args, out):
        B, T, D = out.shape
        m = torch.zeros(B, T, 1, dtype=out.dtype)
        for b, p in enumerate(self.positions):
            m[b, p, 0] = 1
        self.out = out.detach()
        return out + m * self.delta

    def remove(self):
        self.h.remove()


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rpath(*a):
    return os.path.join(C.RESULTS, *a)


def apath(*a):
    return os.path.join(C.ARTIFACTS, *a)
