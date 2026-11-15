"""Saving, independent tensor-level diffing against the original file, and fresh-process reload."""
import glob
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open

from . import config as C


def save(model, tok, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return sorted(glob.glob(os.path.join(out_dir, "*.safetensors")))


def _open_all(files):
    handles, where = [], {}
    for f in files:
        h = safe_open(f, "pt")
        handles.append(h)
        for k in h.keys():
            where[k] = h
    return handles, where


@torch.no_grad()
def diff_vs_original(saved_dir, inmem_state=None):
    """Compare EVERY tensor in the saved checkpoint with the original read-only file.
    For each changed tensor report norms and the singular values of the difference."""
    files = sorted(glob.glob(os.path.join(saved_dir, "*.safetensors")))
    _, new = _open_all(files)
    orig = safe_open(C.ORIG_WEIGHTS, "pt")
    ok, nk = set(orig.keys()), set(new.keys())
    changed, n_equal, n_params = [], 0, 0
    inmem_equal = True
    for k in sorted(ok & nk):
        a, b = orig.get_tensor(k), new[k].get_tensor(k)
        n_params += a.numel()
        if inmem_state is not None and not torch.equal(b, inmem_state[k]):
            inmem_equal = False
        if a.dtype == b.dtype and torch.equal(a, b):
            n_equal += 1
            continue
        d = b.double() - a.double()
        r = dict(name=k, shape=list(a.shape), dtype=str(b.dtype), n_elements_changed=int((d != 0).sum()),
                 fro_delta=d.norm().item(), fro_orig=a.double().norm().item(),
                 rel_change=(d.norm() / a.double().norm()).item(), max_abs_delta=d.abs().max().item())
        if d.dim() == 2:
            s = torch.linalg.svdvals(d)
            r["top_singular_values"] = s[:5].tolist()
            r["numerical_rank_1e-3"] = int((s > 1e-3 * s[0]).sum())
            r["sigma2_over_sigma1"] = (s[1] / s[0]).item()
        changed.append(r)
    return dict(files=[os.path.basename(f) for f in files], n_tensors_original=len(ok),
                n_tensors_saved=len(nk), missing=sorted(ok - nk), extra=sorted(nk - ok),
                n_tensors_identical=n_equal, n_params=n_params, changed=changed,
                saved_equals_inmemory=inmem_equal if inmem_state is not None else None)


def reload_in_fresh_process(saved_dir, resolved_path, out_prefix, inmem_logits_path):
    """Run reload_eval.py in a NEW Python process: it loads only `saved_dir` from disk."""
    script = os.path.join(C.HERE, "reload_eval.py")
    cmd = [sys.executable, "-I", script, saved_dir, resolved_path, out_prefix, inmem_logits_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-4000:])
    with open(out_prefix + ".json") as f:
        info = json.load(f)
    raw, next_lp = torch.load(out_prefix + ".pt")
    return raw, next_lp, info
