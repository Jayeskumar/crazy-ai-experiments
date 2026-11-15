"""Statistics used by ROME: context templates and the uncentred key covariance C = E[k k^T]."""
import os
import time

import pandas as pd
import torch

from . import config as C
from .common import apath, down_proj


@torch.no_grad()
def context_templates(model, tok):
    """ROME-style prefixes sampled from the model itself (top-k sampling, fixed seed)."""
    g = torch.Generator().manual_seed(C.SEED)
    out = ["{}"]
    for n in C.CONTEXT["lengths"]:
        for s in C.CONTEXT["starts"]:
            ids = tok(s, return_tensors="pt").input_ids
            for _ in range(n):
                logits = model(input_ids=ids).logits[0, -1]
                v, i = logits.topk(C.CONTEXT["top_k"])
                nxt = i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)]
                ids = torch.cat([ids, nxt.view(1, 1)], 1)
            t = tok.decode(ids[0], skip_special_tokens=True).replace("{", " ").replace("}", " ")
            out.append(t.replace("\n", " ").strip() + ". {}")
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def key_covariance(model, tok, layer):
    """C = (1/N) sum_t k_t k_t^T over N WikiText-103 train tokens (BOS position excluded).
    Only the first layer+1 decoder layers are run (forward is aborted at the hooked layer)."""
    path = apath(f"cov_layer{layer}.pt")
    if os.path.exists(path):
        return torch.load(path)
    d = pd.read_parquet(C.WIKI_TRAIN)
    n_tok, T = C.COV["n_tokens"], C.COV["chunk_len"]
    text, need = [], n_tok * 6
    for t in d.text:
        if t.strip() and not t.strip().startswith("="):
            text.append(t)
            need -= len(t)
            if need < 0:
                break
    ids = tok("".join(text), add_special_tokens=False)["input_ids"][:n_tok]
    ids = torch.tensor(ids).view(-1, T)
    ids = torch.cat([torch.full((ids.shape[0], 1), tok.bos_token_id), ids], 1)
    dim = model.config.intermediate_size
    S = torch.zeros(dim, dim, dtype=torch.float64)
    cnt = 0
    keys = {}

    def hook(mod, args):
        keys["k"] = args[0]
        raise _Stop

    h = down_proj(model, layer).register_forward_pre_hook(hook)
    t0 = time.time()
    for i in range(0, ids.shape[0], 4):
        try:
            model(input_ids=ids[i:i + 4])
        except _Stop:
            pass
        k = keys["k"][:, 1:].reshape(-1, dim).double()
        S += k.T @ k
        cnt += k.shape[0]
    h.remove()
    cov = (S / cnt).float()
    ev = torch.linalg.eigvalsh(cov.double())
    info = dict(layer=layer, n_tokens=cnt, seconds=time.time() - t0, eig_min=ev.min().item(),
                eig_max=ev.max().item(), cond=(ev.max() / ev.min()).item(), trace=ev.sum().item())
    torch.save((cov, info), path)
    return cov, info
