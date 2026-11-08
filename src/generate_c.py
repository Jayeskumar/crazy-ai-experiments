"""Phase C: instantiate Model C from generated weights and run real inference.

TEXT SIDE. Reads B's emitted candidate vectors (artifacts/model_b/candidates.npz),
loads each into the fixed Model C architecture with no training, and evaluates
it with Model A's tokenizer (fixed interface) on the held-out TEST text,
with the same prompts / sampling settings used for Model A.

Also evaluates: random-initialisation Model C (C1) and all original Model A
instances (reference).

Outputs: results/language/c_eval.jsonl, results/language/samples.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from src.data import default_tokenizer, load_split
from src.evaluate import full_eval, eval_loss, agreement_accuracy
from src.model_a import ModelConfig, TinyTransformerLM, param_count
from src.model_c import build_model_c, random_model_c

ROOT = Path(__file__).resolve().parents[1]
LANG = ROOT / "results" / "language"
SAMPLE_PAT = re.compile(r"^(copy|aligned_copy|B_main_s0|PCA_k16|PCA_k2|B_k2|B_k1|popmean_aligned|popmean_raw|int8|int4|"
                        r"lowrank4|B_unaligned|B_traj|mask_head_B|mask_head_mean)\|ind_\d+$|^(B_sample_s0_0[0-3]|"
                        r"PCA_sample_0[0-1]|interp_aligned_0[0-1]|interp_raw_0[0-1]|random_init_0)$|"
                        r"^fam_\w+:((copy|B_recon|PCA_recon|popmean|mask_head_B)\|\w+_01[67]|B_sample_0[0-2]|"
                        r"PCA_sample_00|interp_00)$")

_G = {}


def _init_worker():
    torch.set_num_threads(1)
    tok = default_tokenizer()
    _G["tok"] = tok
    _G["test"] = torch.tensor(tok.encode(load_split("test")))
    _G["val"] = torch.tensor(tok.encode(load_split("val")))
    path = ROOT / "artifacts" / "model_b" / "candidates.npz"
    _G["cands"] = np.load(path) if path.exists() else None
    meta = json.loads((ROOT / "results" / "model_b" / "candidates_meta.json").read_text()) if path.exists() else None
    _G["layout"] = (meta["names"], meta["shapes"]) if meta else None


def _evaluate(job):
    tok = _G["tok"]
    if job["kind"] == "candidate":
        names, shapes = _G["layout"]
        model = build_model_c(_G["cands"][job["name"]], names, shapes)
    elif job["kind"] == "random":
        model = random_model_c(job["seed"])
    else:  # original model A from its checkpoint
        ck = torch.load(job["path"], map_location="cpu", weights_only=False)
        assert ck["itos"] == tok.itos
        model = TinyTransformerLM(ModelConfig(**ck["config"]))
        model.load_state_dict(ck["state_dict"])
        model.eval()
    t0 = time.perf_counter()
    if job["eval"] == "light":
        r = {"test_loss": eval_loss(model, _G["test"])}
        r.update(agreement_accuracy(model, tok))
    else:
        r = full_eval(model, tok, _G["test"], with_samples=SAMPLE_PAT.match(job["name"]) is not None)
    r["val_loss"] = eval_loss(model, _G["val"])
    r["eval_time_s"] = time.perf_counter() - t0
    r["name"] = job["name"]
    r["params"] = param_count(model)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--n_random", type=int, default=5)
    args = ap.parse_args()
    LANG.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "results" / "model_b" / "candidates_meta.json").read_text())
    jobs = [{"kind": "candidate", "name": m["name"], "eval": m["eval"]} for m in meta["candidates"]]
    jobs += [{"kind": "random", "name": f"random_init_{s}", "seed": 900 + s, "eval": "full"} for s in range(args.n_random)]
    for pop in ["ind"] + meta.get("families", []):
        for ck in sorted((ROOT / "runs" / "model_a" / pop).glob("*/checkpoint.pt")):
            jobs.append({"kind": "A", "name": f"A|{ck.parent.name}", "path": str(ck), "eval": "full"})
    print(f"evaluating {len(jobs)} models", flush=True)
    out_path = LANG / "c_eval.jsonl"
    samples = {}
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool, out_path.open("w") as f:
        for n, r in enumerate(pool.imap_unordered(_evaluate, jobs, chunksize=4)):
            if "samples" in r:
                samples[r["name"]] = {"samples": r.pop("samples"), "samples_greedy": r.pop("samples_greedy")}
            f.write(json.dumps(r) + "\n")
            if n % 100 == 0:
                print(f"  [{time.time() - t0:6.0f}s] {n}/{len(jobs)}", flush=True)
    (LANG / "samples.json").write_text(json.dumps(samples, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
