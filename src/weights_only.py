"""The ONLY interface between Model A artifacts and Model B.

Two halves:

1. `sanitize()` (run as its own step, outside B's process): reads ONLY the
   `weights.pt` / `weights_step*.pt` files (pure state_dicts, loaded with
   torch.load(weights_only=True)), checks that every entry is a floating-point
   tensor with an expected parameter name, and writes
   artifacts/weights_only/<population>.npz containing exactly:
       W        float32 [N, P]   flattened parameter vectors
       names    str     [T]      parameter names (layout/order)
       shapes   int64   [T, 2]   tensor shapes (padded with -1)
       run_ids  str     [N]      opaque run identifiers (e.g. "ind_017")
       split    str     [N]      "train" / "val" / "test" / "all"
   No config, no tokenizer, no vocabulary, no text, no metrics, no samples.

2. `WeightsOnlyDataset` (used by B): loads such an npz with allow_pickle=False
   and re-validates the schema before handing data to B.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WO_DIR = ROOT / "artifacts" / "weights_only"
ALLOWED_KEYS = {"W", "names", "shapes", "run_ids", "split"}
NAME_RE = re.compile(r"^(tok_emb|pos_emb|ln_f|head|blocks\.\d+\.(ln1|ln2|q|k|v|o|fc1|fc2))\.(weight|bias)$")


# ---------------------------------------------------------------------------
# helpers shared by both halves
# ---------------------------------------------------------------------------
def flatten(params: dict, names) -> np.ndarray:
    return np.concatenate([np.asarray(params[n], dtype=np.float32).ravel() for n in names])


def unflatten(vec: np.ndarray, names, shapes) -> dict:
    out, i = {}, 0
    for n, s in zip(names, shapes):
        s = tuple(int(x) for x in s if x >= 0)
        k = int(np.prod(s))
        out[n] = vec[i:i + k].reshape(s)
        i += k
    assert i == vec.size, (i, vec.size)
    return out


# ---------------------------------------------------------------------------
# 1. sanitizer
# ---------------------------------------------------------------------------
def _load_state_dict(path: Path) -> dict:
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: not a state dict")
    out = {}
    for k, v in sd.items():
        if not NAME_RE.match(k):
            raise ValueError(f"{path}: unexpected entry '{k}'")
        if not (hasattr(v, "is_floating_point") and v.is_floating_point()):
            raise ValueError(f"{path}: entry '{k}' is not a float tensor")
        out[k] = v.numpy().astype(np.float32)
    return out


def sanitize(files: list[Path], run_ids: list[str], split: list[str], out_name: str) -> Path:
    sds = [_load_state_dict(Path(f)) for f in files]
    names = list(sds[0].keys())
    for sd in sds:
        assert list(sd.keys()) == names, "inconsistent parameter layout across runs"
    shapes = np.full((len(names), 2), -1, dtype=np.int64)
    for i, n in enumerate(names):
        s = sds[0][n].shape
        shapes[i, :len(s)] = s
    W = np.stack([flatten(sd, names) for sd in sds]).astype(np.float32)
    WO_DIR.mkdir(parents=True, exist_ok=True)
    out = WO_DIR / f"{out_name}.npz"
    np.savez(out, W=W, names=np.array(names), shapes=shapes, run_ids=np.array(run_ids), split=np.array(split))
    return out


def sanitize_all(runs_dir: Path | None = None, seed: int = 0, n_val: int = 16, n_test: int = 16):
    runs_dir = runs_dir or ROOT / "runs" / "model_a"
    written = []
    ind = sorted((runs_dir / "ind").glob("ind_*/weights.pt"))
    if ind:
        ids = [p.parent.name for p in ind]
        order = np.random.default_rng(seed).permutation(len(ids))
        split = ["train"] * len(ids)
        for j in order[:n_test]:
            split[j] = "test"
        for j in order[n_test:n_test + n_val]:
            split[j] = "val"
        written.append(sanitize(ind, ids, split, "ind"))
    sh = sorted((runs_dir / "shared").glob("shared_*/weights.pt"))
    if sh:
        written.append(sanitize(sh, [p.parent.name for p in sh], ["all"] * len(sh), "shared"))
    for sp in sorted(runs_dir.glob("spawn[0-9]*")):
        fs = sorted(sp.glob("spawn*/weights.pt"))
        if fs:
            written.append(sanitize(fs, [p.parent.name for p in fs], ["all"] * len(fs), sp.name))
    tr = sorted((runs_dir / "traj" / "traj_000").glob("weights_step*.pt"))
    if tr:
        written.append(sanitize(tr, [f"traj_000@{p.stem.split('step')[1]}" for p in tr], ["all"] * len(tr), "traj"))
    return written


# ---------------------------------------------------------------------------
# 2. B-side loader
# ---------------------------------------------------------------------------
class WeightsOnlyDataset:
    def __init__(self, population: str):
        path = WO_DIR / f"{population}.npz"
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            if keys != ALLOWED_KEYS:
                raise ValueError(f"weights-only file has unexpected keys: {keys ^ ALLOWED_KEYS}")
            self.W = z["W"]
            self.names = [str(n) for n in z["names"]]
            self.shapes = z["shapes"]
            self.run_ids = [str(r) for r in z["run_ids"]]
            self.split = np.array([str(s) for s in z["split"]])
        if self.W.dtype != np.float32 or self.W.ndim != 2:
            raise ValueError("W must be a float32 matrix")
        for n in self.names:
            if not NAME_RE.match(n):
                raise ValueError(f"unexpected parameter name {n}")
        self.population = population

    def __len__(self):
        return len(self.W)

    def params(self, i: int) -> dict:
        return unflatten(self.W[i], self.names, self.shapes)

    def indices(self, split: str):
        return np.where(self.split == split)[0]

    @property
    def n_layer(self) -> int:
        return 1 + max(int(n.split(".")[1]) for n in self.names if n.startswith("blocks."))

    def describe(self) -> dict:
        return {"population": self.population, "n": len(self), "P": int(self.W.shape[1]),
                "names": self.names, "shapes": self.shapes.tolist(),
                "split_counts": {s: int((self.split == s).sum()) for s in set(self.split)}}


if __name__ == "__main__":
    for p in sanitize_all():
        print("wrote", p)
