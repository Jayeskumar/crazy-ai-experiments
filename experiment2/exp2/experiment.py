"""End-to-end Experiment 2 pipeline; each stage is its own process (isolation boundary)."""
import subprocess
import sys

STAGES = ["exp2.train_a", "exp2.sanitize", "exp2.train_b", "exp2.generate_c", "exp2.analysis"]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else STAGES[0]
    for s in STAGES[STAGES.index(start):]:
        print(f"=== {s} ===", flush=True)
        subprocess.run([sys.executable, "-m", s], check=True)
