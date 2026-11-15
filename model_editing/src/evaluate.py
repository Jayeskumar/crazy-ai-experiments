"""Evaluation: efficacy, paraphrase / implication generalisation, neighbourhood specificity,
unrelated-knowledge retention (accuracy, KL to the original model), WikiText perplexity, and
free generation. `raw_eval` returns per-prompt numbers; `summarize` turns them into metrics
relative to the original model's raw numbers."""
import os

import numpy as np
import pandas as pd
import torch

from . import config as C
from .common import apath, encode_with_target, greedy, pad_batch, score
from .edits import EDITS, GENERAL


def probe_list(resolved):
    """Flatten all probes of all edits: (edit_id, category, prompt, target_a, target_b).
    For edit-directed categories a=new, b=true.  For neighbourhood a=truth, b=new (contrast)."""
    P = []
    for e in EDITS:
        r = resolved[e["id"]]
        P.append((e["id"], "rewrite", e["prompt"].format(e["subject"]), e["new"], r["true"]))
        for p in e["paraphrase"]:
            P.append((e["id"], "paraphrase", p, e["new"], r["true"]))
        for (p, n, t), rt in zip(e["implication"], r["implication"]):
            P.append((e["id"], "implication", p, n, rt))
        for (p, t), rt in zip(e["neighborhood"], r["neighborhood"]):
            P.append((e["id"], "neighborhood", p, rt, e["new"]))
        for p, t in e["attribute"]:
            P.append((e["id"], "attribute", p, t, None))
    return P


_PPL_IDS = None


def ppl_ids(tok):
    global _PPL_IDS
    if _PPL_IDS is None:
        cache = apath("wikitext2_test_ids.pt")
        if os.path.exists(cache):
            _PPL_IDS = torch.load(cache)
        else:
            d = pd.read_parquet(C.WIKI_TEST)
            text = "".join(t for t in d.text.tolist())
            n = C.EVAL["ppl_chunks"] * C.EVAL["ppl_chunk_len"]
            ids = tok(text[: n * 8], add_special_tokens=False)["input_ids"][:n]
            _PPL_IDS = torch.tensor(ids).view(C.EVAL["ppl_chunks"], C.EVAL["ppl_chunk_len"])
            torch.save(_PPL_IDS, cache)
    return _PPL_IDS


@torch.no_grad()
def perplexity(model, tok):
    ids = ppl_ids(tok)
    bos = torch.full((ids.shape[0], 1), tok.bos_token_id)
    x = torch.cat([bos, ids], 1)
    nll, n = 0.0, 0
    for i in range(0, x.shape[0], 4):
        xb = x[i:i + 4]
        lp = torch.log_softmax(model(input_ids=xb).logits.float(), -1)
        tgt = xb[:, 1:]
        nll -= lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).sum().item()
        n += tgt.numel()
    return float(np.exp(nll / n))


PROBE_PROMPTS = [e["prompt"].format(e["subject"]) for e in EDITS] + [g[0] for g in GENERAL[:4]]


@torch.no_grad()
def probe_logits(model, tok):
    """Full logits on a fixed batch: used to prove bitwise equality after save/reload."""
    enc = [tok(p)["input_ids"] for p in PROBE_PROMPTS]
    ids, mask = pad_batch(enc, tok.pad_token_id)
    return model(input_ids=ids, attention_mask=mask).logits.float()


def raw_eval(model, tok, resolved, with_gen=True, with_ppl=True):
    P = probe_list(resolved)
    pairs = []
    for (_, _, p, a, b) in P:
        pairs.append((p, a))
        if b is not None:
            pairs.append((p, b))
    sc = score(model, tok, pairs)
    it = iter(sc)
    probes = []
    for (eid, cat, p, a, b) in P:
        ra = next(it)
        rb = next(it) if b is not None else None
        probes.append(dict(edit=eid, cat=cat, prompt=p, a=a, b=b,
                           lp_a=ra["mean_lp"], top1_a=ra["top1"],
                           lp_b=None if rb is None else rb["mean_lp"],
                           top1_b=None if rb is None else rb["top1"]))
    g = score(model, tok, GENERAL, want_next=True)
    general = [dict(prompt=p, answer=a, lp=r["mean_lp"], top1=r["top1"]) for (p, a), r in zip(GENERAL, g)]
    next_lp = torch.stack([r["next"] for r in g])
    out = dict(probes=probes, general=general)
    if with_ppl:
        out["ppl"] = perplexity(model, tok)
    if with_gen:
        out["gen"] = {e["id"]: [dict(prompt=p, text=greedy(model, tok, p, C.EVAL["gen_tokens"]))
                                for p in e["generation"]] for e in EDITS}
    return out, next_lp


def kl_rows(base_lp, lp):
    """KL(original || edited) of the next-token distribution, per prompt."""
    return (base_lp.exp() * (base_lp - lp)).sum(-1)


def summarize(raw, next_lp, base_raw, base_next_lp, edit_ids=None):
    """Aggregate metrics for each edit. `edit_ids` = edits that were applied to this model."""
    edit_ids = edit_ids or []
    by = {}
    for pr, bp in zip(raw["probes"], base_raw["probes"]):
        assert pr["prompt"] == bp["prompt"]
        by.setdefault((pr["edit"], pr["cat"]), []).append((pr, bp))
    S = {}
    for e in EDITS:
        eid = e["id"]
        m = {}
        for cat in ["rewrite", "paraphrase", "implication"]:
            rows = by.get((eid, cat), [])
            if not rows:
                continue
            m[cat] = dict(
                success=float(np.mean([r["lp_a"] > r["lp_b"] for r, _ in rows])),   # new beats true
                top1_new=float(np.mean([r["top1_a"] for r, _ in rows])),
                d_lp_new=float(np.mean([r["lp_a"] - b["lp_a"] for r, b in rows])),
                p_new=float(np.mean([np.exp(r["lp_a"]) for r, _ in rows])),
                n=len(rows))
        rows = by.get((eid, "neighborhood"), [])
        m["neighborhood"] = dict(
            success=float(np.mean([r["lp_a"] > r["lp_b"] for r, _ in rows])),        # truth still beats new
            base_success=float(np.mean([b["lp_a"] > b["lp_b"] for _, b in rows])),
            leak_top1=float(np.mean([r["top1_b"] and not b["top1_b"] for r, b in rows])),
            d_lp_new=float(np.mean([r["lp_b"] - b["lp_b"] for r, b in rows])),
            d_lp_truth=float(np.mean([r["lp_a"] - b["lp_a"] for r, b in rows])),
            n=len(rows))
        rows = by.get((eid, "attribute"), [])
        if rows:
            m["attribute"] = dict(d_lp=float(np.mean([r["lp_a"] - b["lp_a"] for r, b in rows])),
                                  top1=float(np.mean([r["top1_a"] for r, _ in rows])),
                                  base_top1=float(np.mean([b["top1_a"] for _, b in rows])), n=len(rows))
        S[eid] = m
    kl = kl_rows(base_next_lp, next_lp)
    changed = (base_next_lp.argmax(-1) != next_lp.argmax(-1)).float()
    G = dict(acc=float(np.mean([g["top1"] for g in raw["general"]])),
             base_acc=float(np.mean([g["top1"] for g in base_raw["general"]])),
             mean_kl=float(kl.mean()), max_kl=float(kl.max()),
             max_kl_prompt=GENERAL[int(kl.argmax())][0],
             top1_changed=float(changed.mean()))
    out = dict(edits=S, general=G)
    if "ppl" in raw:
        out["ppl"] = raw["ppl"]
        out["d_ppl"] = raw["ppl"] - base_raw["ppl"]
    if "gen" in raw:
        gen = {}
        for e in EDITS:
            gen[e["id"]] = [dict(prompt=g["prompt"], text=g["text"],
                                 mentions_new=e["new"].lower() in g["text"].lower())
                            for g in raw["gen"][e["id"]]]
        out["gen"] = gen
    out["applied"] = edit_ids
    return out
