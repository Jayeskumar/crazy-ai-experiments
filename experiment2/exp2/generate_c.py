"""Phase 3: build Model C from every candidate vector (no training) and evaluate it.

TEXT SIDE. For each candidate C (generated from probe model A_i):
  capability vector on all 8 (task, variant) pairs, own-task test loss,
  model-specific fingerprints relative to A_i and its same-variant siblings A_j:
    agree_self / agree_sib   fraction of held-out problems answered identically
    errcopy_self / _sib      among problems A_i (resp. A_j) gets WRONG, fraction where C gives the same wrong answer
    kl_self / kl_sib_*       KL(A || C) on own-variant test text
    ident                    1 if A_i is the nearest (min KL) of {A_i} U siblings
Also: oracle variant-centroid baseline (uses task labels; outside B) and random inits.
Writes results/language/c_eval.jsonl and results/language/samples.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch

from exp2 import ROOT
from exp2.evaluate import _instance_batch, capability_vector, next_char_logprobs, task_loss
from exp2.model import TinyTransformerLM, arch_config
from src.weights_only import unflatten

LANG = ROOT / "results" / "language"
G = {}


def build(vec):
    m = TinyTransformerLM(arch_config())
    sd = {k: torch.tensor(np.ascontiguousarray(v)) for k, v in unflatten(vec, G["names"], G["shapes"]).items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def answers(model, task, variant):
    if task == "grammar":
        return None
    x, L = _instance_batch(task, variant)
    pred = model(x[:, :-1])[:, -L:].argmax(-1)
    return pred.numpy(), x[:, -L:].numpy()


def kl(p, q):  # mean KL(p || q), log-probs
    return float((p.exp() * (p - q)).sum(-1).mean())


def fingerprint_ref(run, vec):
    info = G["runinfo"][run]
    m = build(vec)
    a = answers(m, info["task"], info["variant"])
    return {"lp": next_char_logprobs(m, info["task"], info["variant"]),
            "ans": None if a is None else a[0], "gold": None if a is None else a[1]}


def evaluate(job):
    name, run = job["name"], job["source_run"]
    vec = G["cands"][name] if job["kind"] == "cand" else job["vec"]
    m = build(vec)
    cap = capability_vector(m, with_samples=job.get("samples", False))
    samples = cap.pop("_samples", None)
    r = {"name": name, "method": job["method"], "source_run": run, "capability": cap}
    if run is not None:
        info = G["runinfo"][run]
        t, v = info["task"], info["variant"]
        r.update(task=t, variant=v, own_score=cap[f"{t}/{v}"],
                 cross_max=max(s for k, s in cap.items() if k != f"{t}/{v}"),
                 own_loss=task_loss(m, t, v))
        lp = next_char_logprobs(m, t, v)
        ref = G["refs"][run]
        sibs = [s for s in G["siblings"][run]]
        r["kl_self"] = kl(ref["lp"], lp)
        ks = [kl(G["refs"][s]["lp"], lp) for s in sibs]
        r["kl_sib_mean"], r["kl_sib_min"] = float(np.mean(ks)), float(np.min(ks))
        r["ident"] = float(r["kl_self"] < r["kl_sib_min"])
        a = answers(m, t, v)
        if a is not None:
            pred = a[0]
            same = lambda A: (pred == A).all(1)  # noqa: E731
            r["agree_self"] = float(same(ref["ans"]).mean())
            r["agree_sib"] = float(np.mean([same(G["refs"][s]["ans"]).mean() for s in sibs]))

            def errcopy(R):
                wrong = ~(R["ans"] == R["gold"]).all(1)
                return float(same(R["ans"])[wrong].mean()) if wrong.any() else np.nan

            r["errcopy_self"] = errcopy(ref)
            r["errcopy_sib"] = float(np.nanmean([errcopy(G["refs"][s]) for s in sibs]))
    if samples:
        r["_samples"] = samples
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="'_v2' evaluates the round-2 candidates (no oracle/random jobs)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    meta = json.loads((ROOT / "results" / "model_b" / f"candidates_meta{args.tag}.json").read_text())
    idmap = json.loads((ROOT / "artifacts" / "private" / "id_map.json").read_text())
    G["names"], G["shapes"] = meta["names"], meta["shapes"]
    G["cands"] = dict(np.load(ROOT / "artifacts" / "model_b" / f"candidates{args.tag}.npz"))
    G["runinfo"] = {v["run"]: v for v in idmap.values()}
    probe_runs = [idmap[i]["run"] for i in meta["probe_ids"]]
    train_runs = [idmap[i]["run"] for i in meta["b_train_ids"]]
    # references: every probe and B-train model (fingerprints of the originals)
    t0 = time.time()
    G["refs"] = {}
    for run in probe_runs + train_runs:
        sd = torch.load(ROOT / "runs" / run / "weights.pt", weights_only=True)
        vec = np.concatenate([sd[n].numpy().ravel() for n in G["names"]])
        G["refs"][run] = fingerprint_ref(run, vec)
        G.setdefault("vecs", {})[run] = vec
    G["siblings"] = {r: [s for s in probe_runs + train_runs if s != r and G["runinfo"][s]["variant"] == G["runinfo"][r]["variant"]
                         and G["runinfo"][s]["task"] == G["runinfo"][r]["task"]] for r in probe_runs}
    print(f"references ready ({time.time() - t0:.0f}s)", flush=True)
    jobs = []
    show = {"copy", "B_r8_q8", "B_r16_q8", "B_r4_q8", "rowPCA_r8_q8", "quant2", "globalmean", "global_mean", "flatAE_exp1",
            "nn_retrieval", "mask_m50_B_r32", "maskm50_B_r32", "maskm50_zero", "variant_centroid"}
    seen_src = {}
    for c in meta["candidates"]:
        run = G["runinfo"][idmap[c["source"]]["run"]]["run"] if c["source"] else None
        k = (c["method"], G["runinfo"][run]["task"] if run else None)
        smp = c["method"] in show and seen_src.setdefault(k, run) == run
        jobs.append({"kind": "cand", "name": c["name"], "method": c["method"], "source_run": run, "samples": smp})
    # oracle baseline (uses labels, never available to B): centroid of B-train models of the same variant
    for run in (probe_runs if not args.tag else []):
        info = G["runinfo"][run]
        same = [G["vecs"][s] for s in train_runs if G["runinfo"][s]["task"] == info["task"]
                and G["runinfo"][s]["variant"] == info["variant"]]
        if same:
            jobs.append({"kind": "vec", "name": f"variant_centroid|{run}", "method": "oracle_variant_centroid",
                         "source_run": run, "vec": np.mean(same, 0)})
    for s in range(0 if args.tag else 5):
        torch.manual_seed(900 + s)
        m = TinyTransformerLM(arch_config())
        vec = np.concatenate([m.state_dict()[n].numpy().ravel() for n in G["names"]])
        jobs.append({"kind": "vec", "name": f"random_init_{s}", "method": "random_init", "source_run": None, "vec": vec})
    if args.limit:
        jobs = jobs[:args.limit]
    LANG.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {len(jobs)} Model C instances", flush=True)
    samples = {}
    with get_context("fork").Pool(args.workers) as pool, (LANG / f"c_eval{args.tag}.jsonl").open("w") as f:
        for n, r in enumerate(pool.imap_unordered(evaluate, jobs, chunksize=4)):
            if "_samples" in r:
                samples[r["name"]] = r.pop("_samples")
            f.write(json.dumps(r) + "\n")
            if n % 200 == 0:
                print(f"  [{time.time() - t0:6.0f}s] {n}/{len(jobs)}", flush=True)
    (LANG / f"samples{args.tag}.json").write_text(json.dumps(samples, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
