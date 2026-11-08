"""Aggregate weight-side metadata and language-side evaluations into tables.

TEXT SIDE (runs after B has finished). Writes results/summary.json and
results/tables.md.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_split, default_tokenizer, entropy_rate
from src.evaluate import next_char_logprobs
from src.model_a import ModelConfig, TinyTransformerLM
from src.model_c import build_model_c

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRICS = ["test_loss", "test_ppl", "bits_per_char", "gen_frac_valid_words", "gen_frac_structure",
           "gen_frac_grammatical", "agree_plain", "agree_attractor", "rel_err", "msg_bytes",
           "gen_latency_ms_per_char"]


def ngram_baselines():
    tok = default_tokenizer()
    tr, te = tok.encode(load_split("train")), tok.encode(load_split("test"))
    V = tok.vocab_size
    uni = np.bincount(tr, minlength=V) + 1.0
    uni /= uni.sum()
    h_uni = -np.mean(np.log(uni[te]))
    big = np.ones((V, V))
    for a, b in zip(tr[:-1], tr[1:]):
        big[a, b] += 1
    big /= big.sum(1, keepdims=True)
    h_bi = -np.mean(np.log(big[te[:-1], te[1:]]))
    return {"uniform": math.log(V), "unigram": float(h_uni), "bigram": float(h_bi),
            "entropy_rate_floor": entropy_rate()}


def ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    a = np.array(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a),
            "min": float(a.min()), "max": float(a.max())}


def load():
    meta = json.loads((RES / "model_b" / "candidates_meta.json").read_text())
    ev = {}
    for line in (RES / "language" / "c_eval.jsonl").read_text().splitlines():
        r = json.loads(line)
        ev[r["name"]] = r
    rows = [{**m, **ev[m["name"]]} for m in meta["candidates"]]
    for n, r in ev.items():
        if n.startswith("A|"):
            run = n.split("|")[1]
            rows.append({"name": n, "condition": "A_original:" + run.rsplit("_", 1)[0], "source": run, **r,
                         "msg_bytes": 4 * r["params"]})
        elif n.startswith("random_init"):
            rows.append({"name": n, "condition": "C1_random_init", **r})
    return meta, rows


def summarize(rows):
    keys = METRICS + ["val_loss", "masked_rel_err", "nn_train_rel_dist", "dist_to_popmean"]
    return {k: ms([r.get(k) for r in rows]) for k in keys if any(r.get(k) is not None for r in rows)}


def retained(loss, ref_lo, ref_hi):
    """fraction of the (ref_hi -> ref_lo) loss gap closed. 1 = as good as A."""
    return (ref_hi - loss) / (ref_hi - ref_lo)


def prefix_conditions(rows, pre):
    if pre:
        return sorted({r["condition"] for r in rows if r["condition"].startswith(pre)})
    return sorted({r["condition"] for r in rows if not r["condition"].startswith("fam_")
                   and not r["condition"].startswith("A_original")})


def section(rows, pre, refs, A_rows):
    """All analyses for one population (independent: pre='', family: pre='fam_X:')."""
    LA = float(np.mean([r["test_loss"] for r in A_rows]))
    A_by_run = {r["source"]: r for r in A_rows}
    out = {"A_heldout": summarize(A_rows), "A_mean_loss": LA, "conditions": {}}
    for c in prefix_conditions(rows, pre):
        s = summarize([r for r in rows if r["condition"] == c])
        if s.get("test_loss"):
            s["retained_vs_bigram"] = retained(s["test_loss"]["mean"], LA, refs["bigram"])
        out["conditions"][c[len(pre):]] = s
    # Exp 2 latent sweep
    e2 = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        if c == pre + "exp2_B_latent" or (not pre and c == "C3_B_recon" and r.get("seed") == 0) or c == pre + "B_recon" and pre:
            e2["B"][int(r.get("k", 16))].append(r)
        elif c == pre + "exp2_PCA_latent" or (pre and c == pre + "PCA_recon"):
            e2["PCA"][int(r.get("k", 16))].append(r)
    out["exp2_latent"] = {m: {k: summarize(v) for k, v in sorted(d.items())} for m, d in e2.items()}
    # masked recovery
    mk = defaultdict(dict)
    for r in rows:
        if r["condition"].startswith(pre + "mask_") and (pre or not r["condition"].startswith("fam_")):
            mk[r["group"]].setdefault(r["condition"][len(pre):].split("_")[1], []).append(r)
    out["mask_recovery"] = {g: {m: summarize(v) for m, v in d.items()} for g, d in mk.items()}
    # Exp 3 mutation
    base = {r["source"]: r for r in rows if r["condition"] == pre + "exp3_base"}
    e3 = {}
    for c in ("exp3_mut_B", "exp3_mut_rand_on_rec", "exp3_mut_pca_on_rec", "exp3_mut_rand_on_A", "exp3_mut_Bdelta_on_A"):
        rs_all = [r for r in rows if r["condition"] == pre + c]
        for sc in sorted({r["scale"] for r in rs_all}):
            rs = [r for r in rs_all if r["scale"] == sc]
            ref = (lambda r: A_by_run[r["source"]]) if c.endswith("on_A") else (lambda r: base[r["source"]])
            d = [r["test_loss"] - ref(r)["test_loss"] for r in rs]
            per = defaultdict(list)
            for r in rs:
                per[r["source"]].append(r)
            sel = [min(v, key=lambda r: r["val_loss"]) for v in per.values()]
            e3[f"{c}@{sc}"] = {
                "delta_test_loss": ms(d), "frac_better_than_ref": float(np.mean([x < 0 for x in d])),
                "best_by_val_delta_test": ms([r["test_loss"] - ref(r)["test_loss"] for r in sel]),
                "delta_agree_attractor": ms([r["agree_attractor"] - ref(r)["agree_attractor"] for r in rs]),
                "rel_err_to_ref": ms([r["rel_err"] for r in rs])}
    out["exp3_mutation"] = e3
    if base:
        out["exp3_base_minus_A"] = ms([base[s]["test_loss"] - A_by_run[s]["test_loss"] for s in base if s in A_by_run])
    # linear mode connectivity
    lmc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        if "lmc_" in c and c.startswith(pre) and (pre or not c.startswith("fam_")):
            lmc[c[len(pre):]][r["alpha"]].append(r["test_loss"])
    out["lmc"] = {c: {str(a): ms(v) for a, v in sorted(d.items())} for c, d in lmc.items()}
    return out


def kl_analysis(meta, rows, pre, pop, train_runs, test_runs):
    """Exp 5: are generated models copies, averages, or something else? (function space)"""
    tok = default_tokenizer()
    ids = torch.tensor(tok.encode(load_split("test")))
    names, shapes = meta["names"], meta["shapes"]
    cands = np.load(ROOT / "artifacts" / "model_b" / "candidates.npz")

    def lp_vec(v):
        return next_char_logprobs(build_model_c(v, names, shapes), ids, max_windows=100)

    def lp_ckpt(run):
        ck = torch.load(ROOT / "runs" / "model_a" / pop / run / "checkpoint.pt", weights_only=False)
        m = TinyTransformerLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        return next_char_logprobs(m.eval(), ids, max_windows=100)

    LP_train = torch.stack([lp_ckpt(r) for r in train_runs])
    ens = torch.logsumexp(LP_train, 0) - math.log(len(train_runs))

    def kl(p, q):
        return (p.exp() * (p - q)).sum(-1).mean().item()

    pm = next(c["name"] for c in meta["candidates"]
              if c["condition"] == (pre + "popmean" if pre else "C5_popmean_aligned"))
    lp_pm = lp_vec(cands[pm])
    groups = {"heldout_A": [lp_ckpt(r) for r in test_runs[:8]], "popmean": [lp_pm]}
    sel = {"B_sample": pre + "C4_B_sample", "PCA_sample": pre + "C4_PCA_sample", "diag_gauss": pre + "C4_diag_gauss"}
    if pre:
        sel.update(interp=pre + "C4_interp", B_recon_heldout=pre + "B_recon")
    else:
        sel.update(interp_aligned="C4_interp_aligned", interp_raw="C4_interp_raw",
                   diag_gauss="C4_diag_gauss_aligned", B_recon_heldout="C3_B_recon")
    for key, cond in sel.items():
        nm = [c["name"] for c in meta["candidates"] if c["condition"] == cond
              and c.get("seed_tag", "_s0") in ("_s0", "") and (c.get("seed") in (None, 0))][:8]
        if nm:
            groups[key] = [lp_vec(cands[n]) for n in nm]
    out = {}
    for key, lps in groups.items():
        profs = []
        for lp in lps:
            d = [kl(lp, LP_train[j]) for j in range(len(train_runs))]
            profs.append({"nn": min(d), "mean": float(np.mean(d)), "ens": kl(lp, ens), "pm": kl(lp, lp_pm),
                          "nn_run": train_runs[int(np.argmin(d))]})
        g = {"nn_kl_to_train_A": ms([p["nn"] for p in profs]), "mean_kl_to_train_A": ms([p["mean"] for p in profs]),
             "kl_to_ensemble": ms([p["ens"] for p in profs]), "kl_to_popmean_model": ms([p["pm"] for p in profs]),
             "distinct_nearest_train_runs": len({p["nn_run"] for p in profs}), "n": len(lps),
             "within_group_mean_kl": ms([kl(lps[a], lps[b]) for a in range(len(lps)) for b in range(len(lps)) if a != b])}
        out[key] = g
    # reference: KL between pairs of training models
    out["train_A_pairs"] = {"pairwise_kl": ms([kl(LP_train[a], LP_train[b]) for a in range(8) for b in range(8) if a != b])}
    return out


def main():
    meta, rows = load()
    refs = ngram_baselines()
    wo = np.load(ROOT / "artifacts" / "weights_only" / "ind.npz")
    split = {str(r): str(s) for r, s in zip(wo["run_ids"], wo["split"])}
    ind_test = sorted(r for r, s in split.items() if s == "test")
    ind_train = sorted(r for r, s in split.items() if s == "train")
    S = {"reference_losses_nats_per_char": refs, "sections": {}, "exp5_function_space": {}}
    A_ind_test = [r for r in rows if r["condition"] == "A_original:ind" and r["source"] in ind_test]
    S["sections"]["independent"] = section(rows, "", refs, A_ind_test)
    S["sections"]["independent"]["conditions"]["C1_random_init"] = summarize(
        [r for r in rows if r["condition"] == "C1_random_init"])
    S["exp5_function_space"]["independent"] = kl_analysis(meta, rows, "", "ind", ind_train, ind_test)
    for f in meta.get("families", []):
        fruns = sorted(r["source"] for r in rows if r["condition"] == f"A_original:{f}")
        A_f_test = [r for r in rows if r["condition"] == f"A_original:{f}" and r["source"] in fruns[16:]]
        S["sections"][f] = section(rows, f"fam_{f}:", refs, A_f_test)
        S["exp5_function_space"][f] = kl_analysis(meta, rows, f"fam_{f}:", f, fruns[:14], fruns[16:])
    S["A_population"] = {pop: summarize([r for r in rows if r["condition"] == f"A_original:{pop}"])
                         for pop in ["ind"] + meta.get("families", [])}
    S["B_seed_variance"] = {s: summarize([r for r in rows if r["condition"] == "C3_B_recon" and r.get("seed") == s])
                            for s in (0, 1, 2)}
    S["exp4_ntrain"] = {}
    for m, c in (("B", "exp4_B_ntrain"), ("PCA", "exp4_PCA_ntrain")):
        d = defaultdict(list)
        for r in rows:
            if r["condition"] == c:
                d[r["n_train"]].append(r)
        if m == "B":
            d[len(ind_train)] = [r for r in rows if r["condition"] == "C3_B_recon" and r.get("seed") == 0]
        S["exp4_ntrain"][m] = {n: summarize(v) for n, v in sorted(d.items())}
    # Is the single-trajectory B just emitting its memorised run? weight distance of its outputs
    # (for held-out independent inputs) to the trajectory snapshots vs to the input model.
    cands = np.load(ROOT / "artifacts" / "model_b" / "candidates.npz")
    trj = []
    for c in meta["candidates"]:
        if c["condition"] == "single_run_B_traj":
            v = cands[c["name"]]
            # trajectory snapshots were stored raw; B_traj works in the aligned basis -> compare in
            # function-invariant terms: relative distance to the per-input outputs' own mean
            trj.append(v)
    trj = np.stack(trj)
    S["single_run_B_traj_output_spread"] = {
        "mean_rel_dist_between_outputs_for_different_inputs": float(np.mean(
            [np.linalg.norm(trj[a] - trj[b]) / np.linalg.norm(trj[a]) for a in range(len(trj)) for b in range(a)])),
        "note": "small spread => B_traj emits nearly the same model regardless of which held-out model it is given"}
    tok = default_tokenizer()
    ids = torch.tensor(tok.encode(load_split("test")))

    def lp_ck(path):
        ck = torch.load(path, weights_only=False)
        m = TinyTransformerLM(ModelConfig(**ck["config"]))
        m.load_state_dict(ck["state_dict"])
        return next_char_logprobs(m.eval(), ids, max_windows=100)

    def klf(p, q):
        return (p.exp() * (p - q)).sum(-1).mean().item()

    lp_traj = lp_ck(ROOT / "runs/model_a/traj/traj_000/checkpoint.pt")
    to_traj, to_input = [], []
    for c in meta["candidates"]:
        if c["condition"] == "single_run_B_traj":
            lp = next_char_logprobs(build_model_c(cands[c["name"]], meta["names"], meta["shapes"]), ids, max_windows=100)
            to_traj.append(klf(lp, lp_traj))
            to_input.append(klf(lp, lp_ck(ROOT / "runs/model_a/ind" / c["source"] / "checkpoint.pt")))
    S["single_run_B_traj_output_spread"].update(kl_to_trajectory_model=ms(to_traj), kl_to_input_model=ms(to_input))
    S["weight_space"] = {
        "alignment": json.loads((RES / "model_b" / "alignment.json").read_text()),
        "b_training": [{k: v for k, v in L.items() if k != "curve"} for L in
                       json.loads((RES / "model_b" / "b_training.json").read_text())]}
    (RES / "summary.json").write_text(json.dumps(S, indent=1))
    write_tables(S)
    print("wrote results/summary.json and results/tables.md")


def fmt(s, key, p=3):
    v = s.get(key) if s else None
    if not v:
        return "–"
    return f"{v['mean']:.{p}f} ± {v['std']:.{p}f}" if v["n"] > 1 else f"{v['mean']:.{p}f}"


def write_tables(S):
    L = []
    r = S["reference_losses_nats_per_char"]
    cols = ["test_loss", "gen_frac_structure", "gen_frac_grammatical", "agree_plain", "agree_attractor", "rel_err"]
    L.append("# Result tables (auto-generated by src/analysis.py)\n")
    L.append(f"Reference losses (nats/char on test text): uniform {r['uniform']:.3f}, unigram {r['unigram']:.3f}, "
             f"bigram {r['bigram']:.3f}, source entropy-rate floor {r['entropy_rate_floor']:.4f}.\n")
    L.append("`retained` = fraction of the bigram→A test-loss gap closed (1 = as good as the held-out A models, "
             "0 = bigram level, negative = worse than a bigram model). `rel_err` = ||w − w_A|| / ||w_A||.\n")

    def table(sec, order, title):
        L.append(f"\n## {title}\n")
        L.append(f"Held-out Model A mean test loss: {sec['A_mean_loss']:.4f}\n")
        L.append("| condition | n | " + " | ".join(cols) + " | retained |")
        L.append("|" + "---|" * (len(cols) + 3))
        A = sec["A_heldout"]
        L.append("| **A (held-out originals)** | " + str(A["test_loss"]["n"]) + " | " +
                 " | ".join(fmt(A, k) for k in cols) + " | 1.000 |")
        for c in order:
            s = sec["conditions"].get(c)
            if s and s.get("test_loss"):
                ret = s.get("retained_vs_bigram")
                rtxt = f"{ret:.3f}" if ret is not None else "–"
                L.append(f"| {c} | {s['test_loss']['n']} | " + " | ".join(fmt(s, k) for k in cols) + f" | {rtxt} |")

    ind = S["sections"]["independent"]
    table(ind, ["C1_random_init", "C2_copy", "C2_aligned_copy", "C3_B_recon", "memorization_B_train",
                "C4_B_sample", "C4_PCA_sample", "C4_interp_aligned", "C4_interp_raw", "C4_diag_gauss_aligned",
                "C5_int8", "C5_int4", "C5_lowrank8", "C5_lowrank4", "C5_popmean_aligned", "C5_popmean_raw",
                "memorization_PCA_train", "align_B_unaligned", "align_PCA_unaligned", "align_B_canon",
                "align_PCA_canon", "single_run_B_traj", "single_run_PCA_traj", "single_run_B_traj_own"],
          "Independent population (96 independently initialised Model A; B trained on 64, tested on 16 held out)")
    for f, sec in S["sections"].items():
        if f == "independent":
            continue
        table(sec, ["copy", "B_recon", "PCA_recon", "popmean", "memorization_B_train", "C4_B_sample", "C4_PCA_sample",
                    "C4_interp", "C4_diag_gauss", "xfer_ind_through_famB"],
              f"Family population `{f}` (B trained on 14, tested on 8 held out)")

    L.append("\n## Linear mode connectivity: test loss along w(α) = (1−α)·w1 + α·w2 (endpoints ≈ A loss)\n")
    L.append("| population / basis | α=0.25 | α=0.5 | α=0.75 |")
    L.append("|---|---|---|---|")
    for f, sec in S["sections"].items():
        for c, d in sec["lmc"].items():
            L.append(f"| {f}: {c} | " + " | ".join(fmt({'x': d.get(a)}, 'x') for a in ("0.25", "0.5", "0.75")) + " |")

    L.append("\n## Exp 2 — latent size k (held-out models)\n")
    L.append("| population | k | B test_loss | B rel_err | PCA test_loss | PCA rel_err |")
    L.append("|---|---|---|---|---|---|")
    for f, sec in S["sections"].items():
        e = sec["exp2_latent"]
        ks = sorted({int(k) for m in e.values() for k in m})
        for k in ks:
            b, p = e.get("B", {}).get(k), e.get("PCA", {}).get(k)
            L.append(f"| {f} | {k} | {fmt(b, 'test_loss')} | {fmt(b, 'rel_err')} | {fmt(p, 'test_loss')} | "
                     f"{fmt(p, 'rel_err')} |")

    L.append("\n## Exp 4 — number of independent source models B is trained on (k=16, held-out independent models)\n")
    L.append("| n_train | B test_loss | B rel_err | PCA test_loss | PCA rel_err |")
    L.append("|---|---|---|---|---|")
    e4 = S["exp4_ntrain"]
    for n in sorted({int(n) for m in e4.values() for n in m}):
        b, p = e4.get("B", {}).get(n), e4.get("PCA", {}).get(n)
        L.append(f"| {n} | {fmt(b, 'test_loss')} | {fmt(b, 'rel_err')} | {fmt(p, 'test_loss')} | {fmt(p, 'rel_err')} |")

    L.append("\n## Masked-tensor recovery (only the masked tensors are replaced; held-out models)\n")
    L.append("| population | masked group | method | test_loss | masked-tensor rel_err | agree_attractor | grammatical |")
    L.append("|---|---|---|---|---|---|---|")
    for f, sec in S["sections"].items():
        for g, d in sec["mask_recovery"].items():
            for m in ("B", "PCA", "mean", "rand"):
                if m in d:
                    L.append(f"| {f} | {g} | {m} | {fmt(d[m], 'test_loss')} | {fmt(d[m], 'masked_rel_err')} | "
                             f"{fmt(d[m], 'agree_attractor')} | {fmt(d[m], 'gen_frac_grammatical')} |")

    L.append("\n## Exp 3 — directed (B-latent) vs random mutations\n")
    L.append("Δ = test_loss(variant) − test_loss(reference). Reference: B reconstruction for `mut_B`, `*_on_rec`; "
             "original A for `*_on_A`. Random and PCA deltas are norm-matched to B's weight change "
             "(random: per tensor). `best-by-val` picks the variant with the lowest *validation* loss per model.\n")
    L.append("| population | mutation@scale | Δ test_loss | frac better | best-by-val Δ test | Δ agree_attractor |")
    L.append("|---|---|---|---|---|---|")
    for f, sec in S["sections"].items():
        for k, v in sec["exp3_mutation"].items():
            L.append(f"| {f} | {k} | {fmt(v, 'delta_test_loss', 4)} | {v['frac_better_than_ref']:.3f} | "
                     f"{fmt(v, 'best_by_val_delta_test', 4)} | {fmt(v, 'delta_agree_attractor')} |")

    L.append("\n## Exp 5 — where do generated models sit in function space? (KL, nats/char, test text)\n")
    L.append("| population | group | n | nearest-train-A KL | mean KL to train A | KL to train ensemble | "
             "KL to pop-mean model | within-group KL | distinct nearest A |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for f, d in S["exp5_function_space"].items():
        L.append(f"| {f} | train-A pairs (reference) | – | – | {fmt(d['train_A_pairs'], 'pairwise_kl', 4)} | – | – | – | – |")
        for g, v in d.items():
            if g == "train_A_pairs":
                continue
            L.append(f"| {f} | {g} | {v['n']} | {fmt(v, 'nn_kl_to_train_A', 4)} | {fmt(v, 'mean_kl_to_train_A', 4)} | "
                     f"{fmt(v, 'kl_to_ensemble', 4)} | {fmt(v, 'kl_to_popmean_model', 4)} | "
                     f"{fmt(v, 'within_group_mean_kl', 4)} | {v['distinct_nearest_train_runs']} |")
    L.append("\n## Model B seed variance (C3, held-out independent models)\n")
    for s, v in S["B_seed_variance"].items():
        if v:
            L.append(f"- seed {s}: test_loss {fmt(v, 'test_loss')}, rel_err {fmt(v, 'rel_err')}")
    L.append("\n## Weight-space diagnostics (weights only)\n")
    L.append("| population/basis | rel. dist. to train mean | mean pairwise rel. dist. | held-out var. explained by "
             "top-16 train PCs | frac. identity perm |")
    L.append("|---|---|---|---|---|")
    for k, v in S["weight_space"]["alignment"]["populations"].items():
        idp = v.get("frac_identity_perm_when_matching")
        L.append(f"| {k} | {v['mean_rel_dist_to_train_mean']:.3f} | {v['mean_pairwise_rel_dist']:.3f} | "
                 f"{v['heldout_var_explained_by_train_pcs']['16']:.3f} | {'–' if idp is None else f'{idp:.3f}'} |")
    (RES / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
