"""Boundary between Model A artifacts and Model B.

Reads ONLY runs/*/weights.pt (pure float state_dicts) and writes
  artifacts/weights_only/{b_train,b_val,probe}.npz   keys exactly {W, names, shapes, ids}
  artifacts/private/id_map.json                       opaque id -> run name/task  (B may NOT open this)
Ids are salted hashes; rows are shuffled, so neither ids nor order reveal the task.

Split (per variant of the seen tasks arith/grammar/symbol): 8 -> b_train, 2 -> b_val, 4 -> probe.
All models of the held-out task `pattern` -> probe only (B never trains or validates on them).
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import torch

from exp2 import ROOT

WO = ROOT / "artifacts" / "weights_only"
PRIV = ROOT / "artifacts" / "private"
ALLOWED_KEYS = {"W", "names", "shapes", "ids"}
NAME_RE = re.compile(r"^(tok_emb|pos_emb|ln_f|head|blocks\.\d+\.(ln1|ln2|q|k|v|o|fc1|fc2))\.(weight|bias)$")
HELD_OUT_TASK = "pattern"
SALT = "exp2-opaque-v1"


def load_weights(path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        if not NAME_RE.match(k) or not v.is_floating_point():
            raise ValueError(f"{path}: illegal entry {k}")
        out[k] = v.numpy().astype(np.float32)
    return out


def opaque(name: str) -> str:
    return hashlib.sha1((SALT + name).encode()).hexdigest()[:12]


def _write(fname, runs, sds, names, shapes, rng):
    order = rng.permutation(len(runs))
    W = np.stack([np.concatenate([sds[runs[i]][n].ravel() for n in names]) for i in order]).astype(np.float32)
    ids = np.array([opaque(runs[i]) for i in order])
    np.savez(WO / f"{fname}.npz", W=W, names=np.array(names), shapes=shapes, ids=ids)


def main():
    runs = sorted(p.parent.name for p in (ROOT / "runs").glob("*/weights.pt"))
    sds = {r: load_weights(ROOT / "runs" / r / "weights.pt") for r in runs}
    names = list(sds[runs[0]])
    shapes = np.full((len(names), 2), -1, np.int64)
    for i, n in enumerate(names):
        s = sds[runs[0]][n].shape
        shapes[i, :len(s)] = s
    split = {}
    for r in runs:
        task, variant, idx = r.rsplit("_", 2)[0], r.rsplit("_", 2)[1], int(r.rsplit("_", 1)[1])
        if task == HELD_OUT_TASK:
            split[r] = "probe"
        else:
            split[r] = "b_train" if idx < 8 else ("b_val" if idx < 10 else "probe")
    WO.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for s in ("b_train", "b_val", "probe"):
        sel = [r for r in runs if split[r] == s]
        _write(s, sel, sds, names, shapes, rng)
        print(s, len(sel))
    (PRIV / "id_map.json").write_text(json.dumps(
        {opaque(r): {"run": r, "task": r.rsplit("_", 2)[0], "variant": r.rsplit("_", 2)[1], "split": split[r]}
         for r in runs}, indent=1))


class WeightsOnly:
    """B-side loader: re-validates schema; no task information available."""

    def __init__(self, name):
        with np.load(WO / f"{name}.npz", allow_pickle=False) as z:
            if set(z.files) != ALLOWED_KEYS:
                raise ValueError(f"unexpected keys {set(z.files) ^ ALLOWED_KEYS}")
            self.W, self.shapes = z["W"], z["shapes"]
            self.names = [str(n) for n in z["names"]]
            self.ids = [str(i) for i in z["ids"]]
        if self.W.dtype != np.float32:
            raise ValueError("W must be float32")
        for n in self.names:
            if not NAME_RE.match(n):
                raise ValueError(n)
        for i in self.ids:
            if not re.fullmatch(r"[0-9a-f]{12}", i):
                raise ValueError(f"non-opaque id {i}")


if __name__ == "__main__":
    main()
