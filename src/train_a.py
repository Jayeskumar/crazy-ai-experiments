"""Phase A: train populations of Model A.

Populations (see configs/model_a.json):
  ind    : N independently initialised models (init seed and data-order seed differ)
  shared : M models from ONE shared initialisation, different data-order seeds
  traj   : ONE run, many late-training snapshots (the "single model" control)

Each run directory gets:
  checkpoint.pt  config + state_dict + tokenizer + seeds (TEXT SIDE: contains vocab)
  weights.pt     state_dict only (float tensors) -> the only file the sanitizer reads
  metrics.json   training/validation loss curve and final metrics
  samples.json   generated samples
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

from src.data import load_split, default_tokenizer
from src.evaluate import eval_loss, generate_text, PROMPTS
from src.model_a import ModelConfig, TinyTransformerLM, param_count

ROOT = Path(__file__).resolve().parents[1]


def load_cfg():
    return json.loads((ROOT / "configs" / "model_a.json").read_text())


def _batches(ids, block, bs, gen):
    n = ids.numel() - block - 1
    while True:
        ix = torch.randint(0, n, (bs,), generator=gen)
        x = torch.stack([ids[i:i + block] for i in ix])
        y = torch.stack([ids[i + 1:i + block + 1] for i in ix])
        yield x, y


def train_one(job: dict) -> dict:
    torch.set_num_threads(1)
    hp = job["hp"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = default_tokenizer()
    train_ids = torch.tensor(tok.encode(load_split("train")))
    val_ids = torch.tensor(tok.encode(load_split("val")))
    mcfg = ModelConfig(vocab_size=tok.vocab_size, **hp["model"])

    torch.manual_seed(job["init_seed"])
    model = TinyTransformerLM(mcfg)
    start = 0
    if job.get("parent"):  # spawned run: continue from a shared parent snapshot
        model.load_state_dict(torch.load(job["parent"], weights_only=True))
        start = job["start_step"]
    data_gen = torch.Generator().manual_seed(job["data_seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"],
                            betas=(0.9, 0.98))
    steps, warm = hp["steps"], hp["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    curve = []
    snaps = set(job.get("snapshot_steps", []))
    batches = _batches(train_ids, mcfg.block_size, hp["batch_size"], data_gen)
    t0 = time.time()
    run_loss = None
    for _ in range(start):
        sched.step()
    for step in range(start + 1, steps + 1):
        model.train()
        x, y = next(batches)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss = loss.item() if run_loss is None else 0.95 * run_loss + 0.05 * loss.item()
        if step % hp["eval_every"] == 0 or step == steps:
            vl = eval_loss(model, val_ids[:8000])
            curve.append({"step": step, "train_loss_ema": run_loss, "val_loss": vl})
        if step in snaps:
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(sd, out_dir / f"weights_step{step:05d}.pt")
    train_time = time.time() - t0

    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(sd, out_dir / "weights.pt")
    torch.save({"config": mcfg.to_dict(), "state_dict": sd, "itos": tok.itos,
                "init_seed": job["init_seed"], "data_seed": job["data_seed"], "hp": hp},
               out_dir / "checkpoint.pt")
    final_val = eval_loss(model, val_ids)
    metrics = {"run": job["name"], "init_seed": job["init_seed"], "data_seed": job["data_seed"],
               "params": param_count(model), "final_val_loss": final_val, "curve": curve,
               "train_time_s": train_time}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    samples = {p: generate_text(model, tok, p, n=150, temperature=0.8, seed=7)[0] for p in PROMPTS}
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=1))
    return {"run": job["name"], "final_val_loss": final_val, "time": train_time}


def make_jobs(cfg, which):
    base = ROOT / cfg["runs_dir"]
    hp = cfg["hp"]
    jobs = []
    if "ind" in which:
        for i in range(cfg["n_ind"]):
            jobs.append({"name": f"ind_{i:03d}", "init_seed": 10_000 + i, "data_seed": 20_000 + i,
                         "hp": hp, "out_dir": str(base / "ind" / f"ind_{i:03d}")})
    if "shared" in which:
        for i in range(cfg["n_shared"]):
            jobs.append({"name": f"shared_{i:03d}", "init_seed": cfg["shared_init_seed"],
                         "data_seed": 30_000 + i, "hp": hp, "out_dir": str(base / "shared" / f"shared_{i:03d}")})
    if "traj" in which:
        s = hp["steps"]
        snaps = list(range(s - cfg["traj_span"], s + 1, cfg["traj_every"]))[1:]
        jobs.append({"name": "traj_000", "init_seed": 40_000, "data_seed": 50_000, "hp": hp,
                     "snapshot_steps": snaps, "out_dir": str(base / "traj" / "traj_000")})
    for K in cfg.get("spawn_steps", []):
        parent = base / "spawn_parent" / "weights_step{:05d}.pt".format(K)
        if f"spawn{K}" in which:
            for i in range(cfg["n_spawn"]):
                jobs.append({"name": f"spawn{K}_{i:03d}", "init_seed": 60_000, "data_seed": 70_000 + i, "hp": hp,
                             "parent": str(parent), "start_step": K,
                             "out_dir": str(base / f"spawn{K}" / f"spawn{K}_{i:03d}")})
    if "spawn_parent" in which:
        jobs.append({"name": "spawn_parent", "init_seed": 60_000, "data_seed": 60_001, "hp": hp,
                     "snapshot_steps": cfg.get("spawn_steps", []), "out_dir": str(base / "spawn_parent")})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="ind,shared,traj")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="train only first N jobs (smoke test)")
    ap.add_argument("--steps", type=int, default=None, help="override steps (smoke test)")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.steps:
        cfg["hp"]["steps"] = args.steps
    jobs = make_jobs(cfg, args.which.split(","))
    jobs = [j for j in jobs if not (Path(j["out_dir"]) / "metrics.json").exists()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"training {len(jobs)} models with {args.workers} workers", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(train_one, jobs):
            print(f"[{time.time() - t0:7.1f}s] {r['run']}: val={r['final_val_loss']:.4f} ({r['time']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
