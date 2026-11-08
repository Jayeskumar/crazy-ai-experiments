"""End-to-end A -> B -> C pipeline.

Each stage is a separate process so that the data-isolation boundary is real:
  1. train_a      (text side)   train Model A populations
  2. sanitize     (boundary)    weights.pt -> artifacts/weights_only/*.npz
  3. train_b      (weights-only mode, see src/isolation.py)
  4. generate_c   (text side)   load generated weights into Model C, run inference
  5. analysis     (text side)   tables + summary
Usage: python -m src.experiment [--from STAGE] [--only STAGE]
"""
import argparse
import subprocess
import sys
import time

STAGES = [("train_a", ["-m", "src.train_a", "--which", "ind,shared,traj,spawn_parent"]),
          ("train_a_spawn", ["-m", "src.train_a", "--which", "spawn300,spawn1000"]), ("sanitize", ["-m", "src.weights_only"]),
          ("train_b", ["-m", "src.train_b"]), ("generate_c", ["-m", "src.generate_c"]),
          ("analysis", ["-m", "src.analysis"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="train_a")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    names = [s for s, _ in STAGES]
    todo = [a.only] if a.only else names[names.index(a.start):]
    for name, cmd in STAGES:
        if name in todo:
            t = time.time()
            print(f"=== {name} ===", flush=True)
            subprocess.run([sys.executable] + cmd, check=True)
            print(f"=== {name} done in {time.time() - t:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
