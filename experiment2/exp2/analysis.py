"""Aggregate Experiment-2 results into tables, criteria verdicts and summary JSON. TEXT SIDE."""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from exp2 import ROOT

RES = ROOT / "results"


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}


def m_(d):
    return None if d is None else d["mean"]


def load():
    meta = json.loads((RES / "model_b" / "candidates_meta.json").read_text())
    idmap = json.loads((ROOT / "artifacts" / "private" / "id_map.json").read_text())
    ev = [json.loads(l) for l in (RES / "language" / "c_eval.jsonl").read_text().splitlines()]
    m2 = RES / "model_b" / "candidates_meta_v2.json"
    if m2.exists():   # round 2 (targeted follow-up)
        meta2 = json.loads(m2.read_text())
        for c in meta2["candidates"]:
            c["round"] = 2
        meta["candidates"] = meta["candidates"] + meta2["candidates"]
        ev += [json.loads(l) for l in (RES / "language" / "c_eval_v2.jsonl").read_text().splitlines()]
    bymeta = {c["name"]: c for c in meta["candidates"]}
    rows = []
    for r in ev:
        c = bymeta.get(r["name"], {})
        rows.append({**c, **r})
    return meta, idmap, rows


def weight_specificity(meta, idmap, method, probe_sel):
    """Input dependence in weight space for one method over a set of probes."""
    c1 = np.load(ROOT / "artifacts" / "model_b" / "candidates.npz")
    c2p = ROOT / "artifacts" / "model_b" / "candidates_v2.npz"
    c2 = np.load(c2p) if c2p.exists() else {}
    cands = type("C", (), {"__getitem__": lambda self, k: c1[k] if k in c1.files else c2[k]})()
    srcs = [c for c in meta["candidates"] if c["method"] == method and idmap[c["source"]]["run"] in probe_sel]
    if len(srcs) < 3:
        return None
    out = np.stack([cands[c["name"]] for c in srcs])
    inp = np.stack([cands[f"copy|{c['name'].split('|')[1]}"] for c in srcs])
    D = lambda X: np.linalg.norm(X[:, None] - X[None], axis=-1)  # noqa: E731
    iu = np.triu_indices(len(out), 1)
    spread = float(D(out)[iu].mean() / D(inp)[iu].mean())
    cross = np.linalg.norm(out[:, None] - inp[None], axis=-1)
    ident = float(np.mean(cross.argmin(1) == np.arange(len(out))))
    return {"spread_ratio": spread, "self_ident_weights": ident, "chance": 1 / len(out)}


def method_table(rows, meta, idmap, probe_sel, A_ref):
    by = defaultdict(list)
    for r in rows:
        if r.get("source_run") in probe_sel:
            by[r["method"]].append(r)
    out = {}
    for meth, rs in by.items():
        own = [r["own_score"] for r in rs]
        a_own = [A_ref[r["source_run"]]["own_score"] for r in rs]
        d = {k: ms([r.get(k) for r in rs]) for k in ("own_score", "cross_max", "own_loss", "kl_self", "kl_sib_mean",
                                                       "ident", "agree_self", "agree_sib", "errcopy_self",
                                                       "errcopy_sib", "rel_err", "bits")}
        d["A_own_score"] = ms(a_own)
        d["A_cross_max"] = ms([A_ref[r["source_run"]]["cross_max"] for r in rs])
        d["retention"] = float(np.mean(own) / max(np.mean(a_own), 1e-9))
        d["score_corr"] = float(np.corrcoef(a_own, own)[0, 1]) if np.std(own) > 1e-9 and np.std(a_own) > 1e-9 else None
        d["kl_ratio"] = (m_(d["kl_self"]) / m_(d["kl_sib_mean"])) if d["kl_sib_mean"] and m_(d["kl_sib_mean"]) else None
        d["weights"] = weight_specificity(meta, idmap, meth, probe_sel) if meth not in ("oracle_variant_centroid",) else None
        # pre-registered criteria
        c = {}
        c["1_task_retention"] = bool(d["retention"] >= 0.8 and m_(d["cross_max"]) <= m_(d["A_cross_max"]) + 0.10)
        w = d["weights"]
        c["2_input_dependence"] = bool(w and w["self_ident_weights"] >= 0.9 and w["spread_ratio"] >= 0.5)
        c["3_model_specific_function"] = bool(m_(d["ident"]) is not None and m_(d["ident"]) >= 0.8
                                              and d["kl_ratio"] is not None and d["kl_ratio"] < 0.5)
        c["4_tracks_source_quality"] = d["score_corr"] is not None and d["score_corr"] >= 0.5
        c["ALL_1_to_3"] = c["1_task_retention"] and c["2_input_dependence"] and c["3_model_specific_function"]
        d["criteria"] = c
        out[meth] = d
    return out


def capability_matrix(rows, method, probe_sel):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] == method and r.get("source_run") in probe_sel:
            for k, v in r["capability"].items():
                acc[f"{r['task']}/{r['variant']}"][k].append(v)
    return {src: {k: float(np.mean(v)) for k, v in d.items()} for src, d in sorted(acc.items())}


def fmt(d, p=3):
    if d is None:
        return "–"
    if isinstance(d, dict):
        return f"{d['mean']:.{p}f}" + (f" ± {d['std']:.{p}f}" if d.get("n", 1) > 1 and d["std"] > 0 else "")
    return f"{d:.{p}f}"


ORDER = ["copy", "B_r32_fp", "B_r32_q8", "B_r16_fp", "B_r16_q8", "B_r8_fp", "B_r8_q8", "B_r4_fp", "B_r4_q8", "B_r2_fp",
         "B_r2_q8", "rowPCA_r32_q8", "rowPCA_r16_q8", "rowPCA_r8_q8", "rowPCA_r4_q8", "rowPCA_r2_q8", "quant8", "quant4",
         "quant3", "quant2", "quant1", "lowrank8", "lowrank4", "nn_retrieval", "oracle_variant_centroid",
         "global_mean", "flatAE_exp1", "B2_r32_q8", "B2_r16_q8", "deq3_plain_midpoint", "deq3_scalar_prior",
         "deq3_B", "deq3_B_clipped", "deq2_plain_midpoint", "deq2_scalar_prior", "deq2_B", "deq2_B_clipped"]


def main():
    meta, idmap, rows = load()
    probes = {idmap[i]["run"] for i in meta["probe_ids"]}
    seen = {r for r in probes if idmap_task(idmap, r) != "pattern"}
    unseen = probes - seen
    A_ref = {r["source_run"]: r for r in rows if r["method"] == "copy"}
    S = {"n_probes": {"seen_tasks": len(seen), "unseen_task": len(unseen)}, "tables": {}, "matrices": {}}
    for tag, sel in (("seen_tasks", seen), ("unseen_task", unseen)):
        S["tables"][tag] = method_table(rows, meta, idmap, sel, A_ref)
        S["matrices"][tag] = {m: capability_matrix(rows, m, sel) for m in ("copy", "B2_r32_q8", "B_r8_q8", "global_mean",
                                                                          "flatAE_exp1", "nn_retrieval", "quant2")}
    S["random_init"] = {k: ms([r["capability"][k] for r in rows if r["method"] == "random_init"])
                        for k in rows[0]["capability"]}
    S["b_training"] = json.loads((RES / "model_b" / "b_training.json").read_text())
    b2 = RES / "model_b" / "b2_training.json"
    if b2.exists():
        S["b2_training"] = json.loads(b2.read_text())
    # isotropy check: variance captured by a rank-r row code vs. r/L for structureless rows
    S["isotropy_check"] = [{"name": L["name"], "r": L.get("r"), "val_clean_mse_n": min(c["val_clean_mse_n"] for c in L["curve"]),
                            "structureless_prediction": 1 - L["r"] / 33} for L in S["b_training"] if L.get("r")]
    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_tables(S)
    print("wrote results/summary.json and results/tables.md")


def idmap_task(idmap, run):
    return run.rsplit("_", 2)[0]


def write_tables(S):
    L = ["# Experiment 2 — result tables (auto-generated by exp2/analysis.py)\n",
         "Scores: exact-match accuracy on held-out problems (arith/symbol/pattern) or fraction of sampled sentences "
         "grammatical in the model's own word order (grammar). `retention` = mean C own-score / mean A own-score. "
         "`cross` = best score on any *other* (task, variant). Fingerprints compare C_i with its own source A_i vs. "
         "same-variant siblings (other independently trained models of the same task variant): `ident` = fraction of "
         "C_i whose nearest model (KL on own-variant text) is A_i; `KL self/sib` = KL(A_i‖C_i) / mean KL(A_j‖C_i); "
         "`agree` = identical answers on held-out problems; `errcopy` = among problems the reference gets wrong, "
         "fraction where C gives the same wrong answer. Weight-space: `w-ident` = C_i's weights are nearest to A_i's "
         "among all probe inputs; `spread` = mean pairwise distance of outputs / of inputs (1 = as diverse as inputs, "
         "0 = constant output).\n"]
    for tag, title in (("seen_tasks", "Probe models of the tasks B was trained on (arith, grammar, symbol)"),
                       ("unseen_task", "Probe models of the task B NEVER saw (pattern)")):
        T = S["tables"][tag]
        L.append(f"\n## {title} — n={S['n_probes'][tag]} source models\n")
        L.append("| method | bits/weight | rel. err | own score (A) | retention | cross | score corr | w-ident | spread "
                 "| ident (KL) | KL self/sib | agree self / sib | errcopy self / sib | criteria 1/2/3/4 |")
        L.append("|" + "---|" * 14)
        for m in ORDER + sorted(set(T) - set(ORDER)):
            if m not in T:
                continue
            d = T[m]
            w = d["weights"] or {}
            c = d["criteria"]
            crit = "".join("✓" if c[k] else "✗" for k in ("1_task_retention", "2_input_dependence",
                                                          "3_model_specific_function", "4_tracks_source_quality"))
            L.append(f"| {m} | {fmt(d['bits'], 2)} | {fmt(d['rel_err'])} | {fmt(d['own_score'])} ({fmt(d['A_own_score'])}) "
                     f"| {d['retention']:.3f} | {fmt(d['cross_max'])} | {fmt(d['score_corr'], 2)} | "
                     f"{fmt(w.get('self_ident_weights'), 2)} | {fmt(w.get('spread_ratio'), 3)} | {fmt(d['ident'], 2)} | "
                     f"{fmt(d['kl_ratio'], 3)} | {fmt(d['agree_self'], 2)} / {fmt(d['agree_sib'], 2)} | "
                     f"{fmt(d['errcopy_self'], 2)} / {fmt(d['errcopy_sib'], 2)} | {crit} |")
    for tag in ("seen_tasks", "unseen_task"):
        for m, M in S["matrices"][tag].items():
            if not M:
                continue
            cols = list(next(iter(M.values())))
            L.append(f"\n### Capability matrix — {m} ({tag}); rows = source model's (task/variant), cols = score on\n")
            L.append("| source \\ eval | " + " | ".join(c.split("/")[1] for c in cols) + " |")
            L.append("|" + "---|" * (len(cols) + 1))
            for src, row in M.items():
                L.append(f"| {src} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
    L.append("\n## Is there shared low-dimensional structure in trained weight rows? (round-1 B, validation models)\n")
    L.append("If rows were structureless (isotropic) in normalised units, a rank-r code of a 33-dim row would leave "
             "a normalised MSE of 1 − r/33.\n")
    L.append("| B | r | val MSE (normalised) | structureless prediction 1 − r/33 |")
    L.append("|---|---|---|---|")
    for x in sorted(S["isotropy_check"], key=lambda x: x["r"]):
        L.append(f"| {x['name']} | {x['r']} | {x['val_clean_mse_n']:.4f} | {x['structureless_prediction']:.4f} |")
    if "b2_training" in S:
        L.append("\n## Round 2 training (validation, normalised per entry position)\n")
        L.append("| B variant | best val MSE | MSE of the input itself |")
        L.append("|---|---|---|")
        for x in S["b2_training"]:
            L.append(f"| {x['name']} | {x['best_val_mse_n']:.4f} | {x['curve'][-1]['val_mse_n_input']:.4f} |")
    (RES / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
