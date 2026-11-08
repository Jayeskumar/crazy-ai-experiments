"""Side-by-side outputs of A vs. C (B v2 r=32, B r=8) for one probe model per task variant. TEXT SIDE."""
import json

import numpy as np
import torch

import exp2  # noqa: F401
from exp2 import ROOT
from exp2.evaluate import sample
from exp2.model import TinyTransformerLM, arch_config
from exp2.tasks import split_text
from src.weights_only import unflatten

meta = json.loads((ROOT / "results/model_b/candidates_meta.json").read_text())
idmap = json.loads((ROOT / "artifacts/private/id_map.json").read_text())
c1, c2 = np.load(ROOT / "artifacts/model_b/candidates.npz"), np.load(ROOT / "artifacts/model_b/candidates_v2.npz")


def build(v):
    m = TinyTransformerLM(arch_config())
    m.load_state_dict({k: torch.tensor(np.ascontiguousarray(x)) for k, x in unflatten(v, meta["names"], meta["shapes"]).items()})
    return m.eval()


out, seen = {}, set()
for i, pid in enumerate(meta["probe_ids"]):
    info = idmap[pid]
    key = f"{info['task']}/{info['variant']}"
    if key in seen:
        continue
    seen.add(key)
    prompt = split_text(info["task"], info["variant"], "test", 60)[:30]
    if info["task"] == "grammar":
        prompt = prompt[:prompt.rfind(".") + 2]
    row = {"source_run": info["run"], "prompt": prompt}
    for label, vec in (("A_original", c1[f"copy|{i}"]), ("C_from_B2_r32", c2[f"B2_r32_q8|{i}"]),
                       ("C_from_B_r8", c1[f"B_r8_q8|{i}"]), ("C_from_global_mean", c1[f"globalmean|{i}"])):
        row[label] = sample(build(vec), prompt, n_seq=1, n_chars=48, temperature=0.01, seed=0)[0]
    out[key] = row
(ROOT / "results/language/showcase.json").write_text(json.dumps(out, indent=1))
for k, r in out.items():
    print(f"== {k}  prompt: {r['prompt']!r}")
    for lab in ("A_original", "C_from_B2_r32", "C_from_B_r8", "C_from_global_mean"):
        print(f"   {lab:20s} {r[lab][:44]!r}")
