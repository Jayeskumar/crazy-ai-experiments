"""Phase 1: train Model A instances. Each instance learns exactly ONE (task, variant).

Same architecture, vocabulary, parameter layout and training budget for every task.
Init seed and data-order seed are unique per instance (independent initialisation).

Per run (runs/<run>/):
  weights.pt     state_dict only -> the ONLY file the sanitiser reads
  checkpoint.pt  config + state_dict + task + seeds  (text side)
  metrics.json   loss curve + final capability vector
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import torch

from exp2 import ROOT
from exp2.evaluate import capability_vector, task_loss
from exp2.model import TinyTransformerLM, arch_config, param_count
from exp2.tasks import TASKS, encode, split_text


def train_one(job):
    torch.set_num_threads(1)
    hp, task, variant = job["hp"], job["task"], job["variant"]
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cfg = arch_config()
    ids = torch.tensor(encode(split_text(task, variant, "train")))
    torch.manual_seed(job["init_seed"])
    model = TinyTransformerLM(cfg)
    g = torch.Generator().manual_seed(job["data_seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"], betas=(0.9, 0.98))
    S, W = hp["steps"], hp["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / W) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, S) / S))))
    B, T = hp["batch_size"], cfg.block_size
    curve, t0 = [], time.time()
    for step in range(1, S + 1):
        ix = torch.randint(0, ids.numel() - T - 1, (B,), generator=g)
        x = torch.stack([ids[i:i + T] for i in ix])
        y = torch.stack([ids[i + 1:i + T + 1] for i in ix])
        model.train()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % hp["eval_every"] == 0:
            model.eval()
            curve.append({"step": step, "train_loss": loss.item(), "test_loss": task_loss(model, task, variant)})
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(sd, out / "weights.pt")
    torch.save({"config": cfg.to_dict(), "state_dict": sd, "task": task, "variant": variant,
                "init_seed": job["init_seed"], "data_seed": job["data_seed"], "hp": hp}, out / "checkpoint.pt")
    cap = capability_vector(model, with_samples=True)
    samples = cap.pop("_samples")
    m = {"run": job["name"], "task": task, "variant": variant, "init_seed": job["init_seed"],
         "data_seed": job["data_seed"], "params": param_count(model), "train_time_s": time.time() - t0,
         "curve": curve, "capability": cap, "own_score": cap[f"{task}/{variant}"],
         "test_loss": task_loss(model, task, variant)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    (out / "samples.json").write_text(json.dumps(samples, indent=1))
    return m


def make_jobs(cfg, n=None, variants=None):
    jobs = []
    for ti, (task, vs) in enumerate(sorted(TASKS.items())):
        for vi, v in enumerate(vs):
            if variants and f"{task}/{v}" not in variants:
                continue
            for i in range(n or cfg["seeds_per_variant"]):
                name = f"{task}_{v}_{i:02d}"
                base = 100_000 + ti * 10_000 + vi * 1_000 + i
                jobs.append({"name": name, "task": task, "variant": v, "init_seed": base, "data_seed": base + 500,
                             "hp": cfg["hp"], "out_dir": str(ROOT / cfg["runs_dir"] / name)})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--n", type=int, default=None, help="seeds per variant (override)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--runs_dir", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs" / "model_a.json").read_text())
    if args.steps:
        cfg["hp"]["steps"] = args.steps
    if args.runs_dir:
        cfg["runs_dir"] = args.runs_dir
    jobs = [j for j in make_jobs(cfg, args.n) if not (Path(j["out_dir"]) / "metrics.json").exists()]
    print(f"training {len(jobs)} models", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for m in pool.imap_unordered(train_one, jobs):
            print(f"[{time.time() - t0:6.0f}s] {m['run']:18s} own={m['own_score']:.3f} loss={m['test_loss']:.3f} "
                  f"({m['train_time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
