"""Build results/tables.md and results/summary.json from the per-run JSON files."""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import dump, load_json, rpath  # noqa: E402
from src.edits import EDITS  # noqa: E402
from src.run import METHODS  # noqa: E402

ORDER = METHODS + ["rome_sequential"]


def f(x, p=2):
    return "–" if x is None else f"{x:.{p}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def load_runs():
    runs = {}
    for p in glob.glob(rpath("runs", "*.json")):
        r = load_json(p)
        runs[(r["method"], r["edit"])] = r
    return runs


def edit_metrics(m, eid):
    e = m["edits"][eid]
    return dict(rewrite=e["rewrite"]["success"], rewrite_top1=e["rewrite"]["top1_new"],
                para=e["paraphrase"]["success"], para_top1=e["paraphrase"]["top1_new"],
                para_p=e["paraphrase"]["p_new"],
                impl=e["implication"]["success"], neigh=e["neighborhood"]["success"],
                neigh_base=e["neighborhood"]["base_success"], leak=e["neighborhood"]["leak_top1"],
                attr_dlp=e.get("attribute", {}).get("d_lp"))


def export_probes(runs):
    """Per-prompt numbers (from the fresh-process reload dumps, identical to in-memory) -> results/probes/."""
    import torch
    from src.common import apath
    os.makedirs(rpath("probes"), exist_ok=True)
    for (m, eid) in runs:
        src = apath(f"reload_{m if m != 'rome_sequential' else 'rome_sequential'}__{eid}.pt")
        if os.path.exists(src):
            raw, _ = torch.load(src)
            dump(dict(probes=raw["probes"], general=raw["general"], ppl=raw["ppl"], gen=raw["gen"]),
                 rpath("probes", f"{m}__{eid}.json"))


def mentions(word, text):
    return bool(re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()))


def main():
    base = load_json(rpath("baseline.json"))
    resolved = base["resolved"]
    trace = load_json(rpath("causal_trace.json"))
    cov = load_json(rpath("covariance_info.json"))
    w = load_json(rpath("original_weights.json"))
    wchk = load_json(rpath("original_weights_check.json")) if os.path.exists(rpath("original_weights_check.json")) else None
    runs = load_runs()
    export_probes(runs)
    L = []
    add = L.append

    # ---------------- baseline
    add("## 1. Original model (before any edit)\n")
    add(f"Model `{w['model_id']}`, float32, `model.safetensors` sha256 `{w['sha256']}`.\n")
    add(f"Original WikiText-2 perplexity (12,288 tokens): **{base['raw']['ppl']:.3f}**. "
        f"General-knowledge top-1 accuracy: **{np.mean([g['top1'] for g in base['raw']['general']]):.0%}** "
        f"({sum(g['top1'] for g in base['raw']['general'])}/{len(base['raw']['general'])}).\n")
    add("| edit | rewrite prompt | original greedy continuation | contrast ('true') | P(new) | P(true) |")
    add("|---|---|---|---|---|---|")
    for e in EDITS:
        pr = next(p for p in base["raw"]["probes"] if p["edit"] == e["id"] and p["cat"] == "rewrite")
        add(f"| {e['id']} | {e['prompt'].format(e['subject'])} → **{e['new']}** | "
            f"`{resolved[e['id']]['original_greedy'].strip()!s}` | {resolved[e['id']]['true']} | "
            f"{np.exp(pr['lp_a']):.2e} | {np.exp(pr['lp_b']):.3f} |")
    add("\nP = geometric-mean per-token probability of the target under teacher forcing.\n")

    # ---------------- layer
    add("## 2. Layer selection (causal tracing)\n")
    avg = trace["avg_norm_mlp_window_ie"]
    add(f"Noise σ = {trace['sigma']:.4f} (3× the std of subject-token embeddings), "
        f"{len(trace['facts'])} real facts × 10 noise samples. Chosen layer: **{trace['chosen_layer']}**.\n")
    add("| layer | " + " | ".join(str(i) for i in range(len(avg))) + " |")
    add("|---|" + "---|" * len(avg))
    add("| norm. indirect effect (5-layer MLP window) | " + " | ".join(f"{a:.2f}" for a in avg) + " |")
    add("\n| fact | P(answer) clean | P(answer) subject corrupted |\n|---|---|---|")
    for fa in trace["facts"]:
        add(f"| {fa['prompt']} → {fa['answer']} | {fa['p_clean']:.3f} | {fa['p_corrupt']:.3f} |")
    add(f"\nKey covariance C = E[kkᵀ] at layer {cov['layer']}: {cov['n_tokens']:,} WikiText-103 train tokens, "
        f"eigenvalues [{cov['eig_min']:.3e}, {cov['eig_max']:.3e}], condition number {cov['cond']:.3e}, "
        f"{cov['seconds']:.0f} s. No ridge term was added.\n")

    # ---------------- main table
    add("## 3. Method comparison (mean over the 4 edits)\n")
    add("| method | gradients? | rewrite ok | paraphrase ok | paraphrase top-1 | implication ok | "
        "neighbourhood kept (orig.) | leak top-1 | general acc Δ | mean KL general | top-1 changed | ΔPPL | ‖ΔW‖_F | rank | reload bit-identical |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for m in METHODS:
        rs = [runs.get((m, e["id"])) for e in EDITS]
        if not all(rs):
            continue
        em = [edit_metrics(r["in_memory"], e["id"]) for r, e in zip(rs, EDITS)]
        g = [r["in_memory"]["general"] for r in rs]
        mean = lambda k: float(np.mean([x[k] for x in em]))  # noqa: E731
        ranks = sorted({c.get("numerical_rank_1e-3") for r in rs for c in r["persistence"]["diff_vs_original"]["changed"]})
        bit = all(r["persistence"]["reload"]["probe_logits_bitwise_equal"] and r["persistence"]["reloaded_raw_metrics_identical"] for r in rs)
        s = dict(rewrite=mean("rewrite"), para=mean("para"), para_top1=mean("para_top1"), impl=mean("impl"),
                 neigh=mean("neigh"), neigh_base=mean("neigh_base"), leak=mean("leak"),
                 d_acc=float(np.mean([x["acc"] - x["base_acc"] for x in g])),
                 kl=float(np.mean([x["mean_kl"] for x in g])), top1_changed=float(np.mean([x["top1_changed"] for x in g])),
                 d_ppl=float(np.mean([r["in_memory"]["d_ppl"] for r in rs])),
                 fro=float(np.mean([r["delta"]["fro"] for r in rs])), ranks=ranks, reload_bit_identical=bit)
        summary[m] = s
        grad = "**yes (weights)**" if m == "ft_constrained" else ("activation vector only" if m.startswith("rome") else "no")
        add(f"| {m} | {grad} | {pct(s['rewrite'])} | {pct(s['para'])} | {pct(s['para_top1'])} | {pct(s['impl'])} | "
            f"{pct(s['neigh'])} ({pct(s['neigh_base'])}) | {pct(s['leak'])} | {100 * s['d_acc']:+.1f} pp | {s['kl']:.4f} | "
            f"{pct(s['top1_changed'])} | {s['d_ppl']:+.3f} | {s['fro']:.2f} | {','.join(map(str, ranks))} | {'yes' if bit else 'NO'} |")
    add("\n*ok* = the edited model gives the new target a higher mean per-token log-prob than the contrast answer. "
        "*top-1* = every target token is the argmax under teacher forcing. *neighbourhood kept* = related prompts still "
        "prefer their own answer over the new target (original model in brackets). *leak top-1* = new target became the "
        "argmax on a neighbourhood prompt where it was not before. KL = KL(original ‖ edited) of the next-token distribution "
        "on the 40 general-knowledge prompts. ΔPPL on WikiText-2 test.\n")

    # ---------------- per edit
    add("## 4. Per-edit results\n")
    for e in EDITS:
        add(f"### {e['id']} — {e['kind']}: “{e['prompt'].format(e['subject'])} … {e['new']}” (contrast: {resolved[e['id']]['true']})\n")
        add("| method | rewrite | paraphrases ok (top-1) | P(new) on paraphrases | implication | neighbourhood kept | leak | Δlogp other attribute | general KL | ΔPPL |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for m in ORDER:
            r = runs.get((m, e["id"])) or (runs.get(("rome_sequential", "all4")) if m == "rome_sequential" else None)
            if not r:
                continue
            x = edit_metrics(r["in_memory"], e["id"])
            g = r["in_memory"]["general"]
            add(f"| {m} | {pct(x['rewrite'])} | {pct(x['para'])} ({pct(x['para_top1'])}) | {f(x['para_p'], 3)} | {pct(x['impl'])} | "
                f"{pct(x['neigh'])} | {pct(x['leak'])} | {f(x['attr_dlp'])} | {g['mean_kl']:.4f} | {r['in_memory']['d_ppl']:+.3f} |")
        add("")

    # ---------------- tensors
    add("## 5. Exactly which tensors changed\n")
    add("Found by comparing **every** tensor of each saved checkpoint with the original read-only file "
        "(`src/persist.py::diff_vs_original`), independently of what the editing code claims it touched.\n")
    add("| run | tensors identical | changed tensor(s) | elements changed | ‖ΔW‖_F | ‖ΔW‖/‖W‖ | max abs | top singular values of ΔW | σ₂/σ₁ | saved == in-memory |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for (m, eid), r in sorted(runs.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        d = r["persistence"]["diff_vs_original"]
        for c in d["changed"]:
            add(f"| {m} / {eid} | {d['n_tensors_identical']}/{d['n_tensors_original']} | `{c['name']}` {c['shape']} | "
                f"{c['n_elements_changed']:,} | {c['fro_delta']:.3f} | {c['rel_change']:.4f} | {c['max_abs_delta']:.2e} | "
                f"{', '.join(f'{s:.2e}' for s in c['top_singular_values'][:3])} | {c['sigma2_over_sigma1']:.1e} | {d['saved_equals_inmemory']} |")
    add("")

    # ---------------- footprint
    add("## 6. How selectively each update fires (same-layer methods)\n")
    add("E‖ΔW k‖² over WikiText keys (= tr(ΔW C ΔWᵀ)) divided by ‖ΔW k*‖² on the edited subject's key. "
        "Smaller = the update is more specific to the subject.\n")
    add("| method | " + " | ".join(e["id"] for e in EDITS) + " |")
    add("|---|" + "---|" * len(EDITS))
    for m in METHODS:
        vals = []
        for e in EDITS:
            r = runs.get((m, e["id"]))
            fp = r and r.get("footprint")
            vals.append(f"{fp['ratio_text_over_key']:.2e}" if fp else "–")
        add(f"| {m} | " + " | ".join(vals) + " |")
    add("")

    # ---------------- persistence
    add("## 7. Save → reload persistence\n")
    add("Each edited model was written with `save_pretrained` (safetensors, float32), loaded by `from_pretrained` in a "
        "**new Python process** (`reload_eval.py`), and fully re-evaluated.\n")
    add("| run | probe logits bit-identical | max abs logit diff | all reloaded metrics identical | rewrite ok after reload | paraphrase ok after reload |")
    add("|---|---|---|---|---|---|")
    for (m, eid), r in sorted(runs.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        rl = r["persistence"]["reload"]
        eids = [x["id"] for x in EDITS] if eid == "all4" else [eid]
        rw = np.mean([r["reloaded"]["edits"][i]["rewrite"]["success"] for i in eids])
        pa = np.mean([r["reloaded"]["edits"][i]["paraphrase"]["success"] for i in eids])
        add(f"| {m} / {eid} | {rl['probe_logits_bitwise_equal']} | {rl['probe_logits_max_abs_diff']:.1e} | "
            f"{r['persistence']['reloaded_raw_metrics_identical']} | {pct(rw)} | {pct(pa)} |")
    if wchk:
        add(f"\nOriginal weights file: sha256 at start `{wchk['sha256_start'][:16]}…`, at end `{wchk['sha256_end'][:16]}…` → "
            f"**{'unchanged' if wchk['unchanged'] else 'CHANGED'}**. In-memory model restored and verified bit-exact "
            f"against the original file after every run.\n")

    # ---------------- generations
    add("## 8. Free generation (greedy, 24 tokens)\n")
    for e in EDITS:
        add(f"**{e['id']}**\n")
        add(f"| model | prompt | continuation | says “{e['new']}” |\n|---|---|---|---|")
        for g in base["raw"]["gen"][e["id"]]:
            add(f"| original | {g['prompt']} | {g['text'].strip().replace(chr(10), ' ⏎ ').replace('|', '/')} | {mentions(e['new'], g['text'])} |")
        for m in ["rome", "rome_identity_cov", "direct_logit_mid", "direct_logit_final", "ft_constrained"]:
            r = runs.get((m, e["id"]))
            if r:
                for g in r["in_memory"]["gen"][e["id"]]:
                    add(f"| {m} | {g['prompt']} | {g['text'].strip().replace(chr(10), ' ⏎ ').replace('|', '/')} | {mentions(e['new'], g['text'])} |")
        add("")

    # ---------------- sequential
    r = runs.get(("rome_sequential", "all4"))
    if r:
        add("## 9. Four ROME edits applied sequentially to one tensor\n")
        add("| edit | rewrite | paraphrases ok | implication | neighbourhood kept |\n|---|---|---|---|---|")
        for e in EDITS:
            x = edit_metrics(r["in_memory"], e["id"])
            add(f"| {e['id']} | {pct(x['rewrite'])} | {pct(x['para'])} | {pct(x['impl'])} | {pct(x['neigh'])} |")
        g = r["in_memory"]["general"]
        add(f"\nGeneral acc Δ {100 * (g['acc'] - g['base_acc']):+.1f} pp, mean KL {g['mean_kl']:.4f}, "
            f"ΔPPL {r['in_memory']['d_ppl']:+.3f}, total ‖ΔW‖_F {r['delta']['fro']:.3f}, "
            f"changed tensors: {r['persistence']['changed_tensors_found']}, reload bit-identical: "
            f"{r['persistence']['reload']['probe_logits_bitwise_equal']}.\n")
        summary["rome_sequential"] = dict(general=g, d_ppl=r["in_memory"]["d_ppl"])

    with open(rpath("tables.md"), "w") as fh:
        fh.write("# Results tables (auto-generated by `src/report.py`)\n\n" + "\n".join(L) + "\n")
    dump(dict(methods=summary, layer=trace["chosen_layer"], base_ppl=base["raw"]["ppl"]), rpath("summary.json"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
