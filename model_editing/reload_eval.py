"""Fresh-process reload check (invoked by src/persist.py with `python -I`).

Loads an edited checkpoint from disk with from_pretrained (nothing shared with the editing
process), re-runs the full evaluation, and compares full-vocabulary logits on a fixed probe
batch with the logits the in-memory edited model produced before saving."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from src.common import load, load_json  # noqa: E402
from src.evaluate import probe_logits, raw_eval  # noqa: E402


def main(saved_dir, resolved_path, out_prefix, inmem_logits_path):
    t0 = time.time()
    model, tok = load(saved_dir)
    resolved = load_json(resolved_path)
    lg = probe_logits(model, tok)
    ref = torch.load(inmem_logits_path)
    raw, next_lp = raw_eval(model, tok, resolved)
    torch.save((raw, next_lp), out_prefix + ".pt")
    info = dict(pid=os.getpid(), seconds=time.time() - t0,
                probe_logits_max_abs_diff=(lg - ref).abs().max().item(),
                probe_logits_bitwise_equal=bool(torch.equal(lg, ref)))
    with open(out_prefix + ".json", "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main(*sys.argv[1:5])
