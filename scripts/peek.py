"""Quick per-condition summary of (possibly partial) C evaluations."""
import collections
import json

import numpy as np

meta = {m["name"]: m for m in json.load(open("results/model_b/candidates_meta.json"))["candidates"]}
d = collections.defaultdict(list)
for line in open("results/language/c_eval.jsonl"):
    r = json.loads(line)
    c = meta.get(r["name"], {}).get("condition", r["name"].split("|")[0].rsplit("_", 1)[0])
    if r["name"].startswith("A|"):
        c = "A:" + r["name"].split("|")[1].rsplit("_", 1)[0]
    d[c].append((r["test_loss"], r.get("gen_frac_grammatical"), meta.get(r["name"], {}).get("rel_err")))
for c, v in sorted(d.items()):
    a = np.array([x[0] for x in v])
    g = [x[1] for x in v if x[1] is not None]
    e = [x[2] for x in v if x[2] is not None]
    print(f"{c:42s} n={len(v):3d} loss={a.mean():.3f}±{a.std():.3f} "
          f"gram={np.mean(g) if g else float('nan'):.2f} relerr={np.mean(e) if e else float('nan'):.3f}")
