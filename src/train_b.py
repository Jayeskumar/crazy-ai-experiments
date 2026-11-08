"""Phase B: train Model B on weights only, then emit candidate weight vectors for Model C.

This whole process runs in weights-only mode (src/isolation.py): text-side
modules cannot be imported and Model A run directories cannot be opened. The
only input is artifacts/weights_only/*.npz (written by the sanitizer).

Outputs (all weights-only):
  artifacts/model_b/candidates.npz        name -> generated parameter vector
  artifacts/model_b/B_main_s*.pt          trained Model B checkpoints
  results/model_b/candidates_meta.json    per-candidate weight-space metadata
  results/model_b/alignment.json          weight-alignment diagnostics
  results/model_b/b_training.json         B loss curves + held-out reconstruction
  results/model_b/weight_space.json       PCA spectra, latent stats
"""
from src.isolation import enter_weights_only_mode, assert_active

enter_weights_only_mode()

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.alignment import (perm_spec, build_template, weight_matching, apply_perms,  # noqa: E402
                           canonicalize_attention)
from src.model_b import WeightAE, WeightNormalizer, PCAModel  # noqa: E402
from src.weights_only import WeightsOnlyDataset, flatten, unflatten, ROOT  # noqa: E402

ART = ROOT / "artifacts" / "model_b"
RES = ROOT / "results" / "model_b"


def tensor_slices(names, shapes):
    out, i = [], 0
    for s in shapes:
        k = int(np.prod([x for x in s if x >= 0]))
        out.append(slice(i, i + k))
        i += k
    return out


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# B training
# ---------------------------------------------------------------------------
def train_ae(Xtr, Xval, k, slices, seed, hp, log_name=""):
    assert_active()
    torch.manual_seed(seed)
    np.random.seed(seed)
    P = Xtr.shape[1]
    normalizer = WeightNormalizer(Xtr, slices)
    model = WeightAE(P, k, hp["hidden"], hp["dropout"], normalizer)
    params = [p for n, p in model.named_parameters()]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, hp["steps"], eta_min=hp["lr"] * 0.05)
    Xn = normalizer.norm(torch.tensor(Xtr))
    Vn = normalizer.norm(torch.tensor(Xval)) if Xval is not None and len(Xval) else None
    member = torch.zeros(len(slices), P)
    for t, sl in enumerate(slices):
        member[t, sl] = 1.0
    curve, best, best_state = [], float("inf"), None
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(1, hp["steps"] + 1):
        model.train()
        idx = torch.randint(0, len(Xn), (hp["batch_size"],), generator=g)
        x = Xn[idx]
        m = (torch.rand(len(idx), len(slices), generator=g) < hp["mask_p"]).float()
        inp = x * (1 - (m @ member).clamp(max=1)) + hp["input_noise"] * torch.randn(x.shape, generator=g)
        z = model.encode_n(inp)
        out = model.decode_n(z + hp["z_noise"] * torch.randn(z.shape, generator=g))
        loss = ((out - x) ** 2).mean() + hp["z_l2"] * (z ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % hp["eval_every"] == 0 or step == hp["steps"]:
            model.eval()
            with torch.no_grad():
                tr = ((model(Xn)[0] - Xn) ** 2).mean().item()
                va = ((model(Vn)[0] - Vn) ** 2).mean().item() if Vn is not None else tr
            curve.append({"step": step, "loss": loss.item(), "train_mse_n": tr, "val_mse_n": va})
            if va < best:
                best = va
                best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"  B[{log_name}] k={k} n={len(Xtr)} best_val_mse_n={best:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model, {"name": log_name, "k": k, "n_train": len(Xtr), "seed": seed, "best_val_mse_n": best,
                   "curve": curve, "time_s": time.time() - t0, "n_params": model.n_params()}


def fit_latent_gaussian(Z, shrink=0.1):
    mu = Z.mean(0)
    S = np.cov(Z.T) if Z.shape[1] > 1 else np.atleast_2d(Z.var(0))
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S)) + 1e-6 * np.eye(len(mu))
    return mu, np.linalg.cholesky(S)


# ---------------------------------------------------------------------------
# conventional compression baselines (weights-only, per model)
# ---------------------------------------------------------------------------
def quantize(params: dict, bits: int, per_row: bool):
    out, nbytes = {}, 0
    qmax = 2 ** (bits - 1) - 1
    for n, w in params.items():
        if per_row and w.ndim == 2:
            s = np.abs(w).max(1, keepdims=True) / qmax
            nscales = w.shape[0]
        else:
            s = np.abs(w).max() / qmax
            nscales = 1
        s = np.where(s == 0, 1.0, s)
        out[n] = (np.clip(np.round(w / s), -qmax, qmax) * s).astype(np.float32)
        nbytes += w.size * bits / 8 + 4 * nscales
    return out, nbytes


def lowrank(params: dict, r: int):
    out, nbytes = {}, 0
    for n, w in params.items():
        if w.ndim == 2 and min(w.shape) > r:
            U, S, Vt = np.linalg.svd(w, full_matrices=False)
            out[n] = ((U[:, :r] * S[:r]) @ Vt[:r]).astype(np.float32)
            nbytes += 4 * r * (sum(w.shape) + 1)
        else:
            out[n] = w
            nbytes += 4 * w.size
    return out, nbytes


_JOBS, _CTX = {}, {}


def _train_job(key):
    torch.set_num_threads(1)
    Xtr, Xval, k, seed = _JOBS[key]
    m, log = train_ae(Xtr, Xval, k, _CTX["slices"], seed, _CTX["hp"], key)
    return key, m.state_dict(), log


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings for smoke tests")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs" / "model_b.json").read_text())
    hp = cfg["train"]
    if args.quick:
        hp = dict(hp, steps=60, eval_every=20, hidden=32)
        cfg.update(main_seeds=[0], latent_sweep=[2, 16], n_train_sweep=[4], n_samples=4, n_interp=2,
                   family_latent_sweep=[2], perturb={"n_models": 2, "n_variants": 2, "scales": [1.0]})
    ART.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    K = cfg["main_k"]

    ind = WeightsOnlyDataset("ind")
    names, shapes = ind.names, ind.shapes
    slices = tensor_slices(names, shapes)
    name2slice = dict(zip(names, slices))
    spec = perm_spec(names, ind.n_layer)
    tr_i, va_i, te_i = ind.indices("train"), ind.indices("val"), ind.indices("test")
    print("weights-only dataset:", ind.describe()["split_counts"], "P =", ind.W.shape[1], flush=True)
    families = [f for f in cfg["families"] if (ROOT / "artifacts" / "weights_only" / f"{f}.npz").exists()]
    fam = {f: WeightsOnlyDataset(f) for f in families}

    # =====================================================================
    # Alignment of the independent population (data-free weight matching)
    # =====================================================================
    t0 = time.time()
    raw = {i: ind.params(i) for i in range(len(ind))}
    template = build_template(spec, [raw[i] for i in tr_i], rounds=4)
    aligned = {i: apply_perms(spec, weight_matching(spec, template, raw[i], seed=int(i)), raw[i])
               for i in range(len(ind))}
    WA = np.stack([flatten(aligned[i], names) for i in range(len(ind))])   # aligned basis
    WR = ind.W                                                            # raw basis
    t_align = time.time() - t0
    # variant: attention gauge-fixing (SVD canonical form) before weight matching
    canon = {i: canonicalize_attention(raw[i], ind.n_layer) for i in range(len(ind))}
    template_c = build_template(spec, [canon[i] for i in tr_i], rounds=4)
    WC = np.stack([flatten(apply_perms(spec, weight_matching(spec, template_c, canon[i], seed=int(i)), canon[i]), names)
                   for i in range(len(ind))])

    def mean_rel_dev(W, tr, te):
        mu = W[tr].mean(0)
        return float(np.mean([np.linalg.norm(W[i] - mu) / np.linalg.norm(W[i]) for i in te]))

    def pairwise(W, idx):
        return float(np.mean([np.linalg.norm(W[a] - W[b]) / np.linalg.norm(W[a]) for a in idx for b in idx if a < b]))

    def heldout_explained(W, tr, te):
        norm = WeightNormalizer(W[tr], slices)
        mu, sig = norm.mu.numpy(), norm.sigma.numpy()
        _, S, Vt = np.linalg.svd((W[tr] - mu) / sig, full_matrices=False)
        Tn = (W[te] - mu) / sig
        tot = (Tn ** 2).sum()
        out = {}
        for k in [1, 2, 4, 8, 16, 32, 64]:
            R = Tn - (Tn @ Vt[:k].T) @ Vt[:k]
            out[k] = float(1 - (R ** 2).sum() / tot)
        return out, (S ** 2 / (S ** 2).sum())[:20].tolist()

    def identity_frac(W, idx):
        out = []
        for i in idx[1:8]:
            p = weight_matching(spec, unflatten(W[idx[0]], names, shapes), unflatten(W[i], names, shapes))
            out.append(float(np.mean([np.mean(v == np.arange(len(v))) for v in p.values()])))
        return float(np.mean(out))

    diag = {"time_s": t_align, "populations": {}}
    for key, W in [("ind_raw", WR), ("ind_aligned", WA), ("ind_canon_aligned", WC)]:
        ev, sp = heldout_explained(W, tr_i, te_i)
        diag["populations"][key] = {
            "mean_rel_dist_to_train_mean": mean_rel_dev(W, tr_i, te_i), "mean_pairwise_rel_dist": pairwise(W, te_i),
            "heldout_var_explained_by_train_pcs": ev, "train_pca_spectrum": sp,
            "frac_identity_perm_when_matching": identity_frac(W, te_i) if key == "ind_raw" else None}
    for f, ds in fam.items():
        ftr, fte = np.arange(0, 14), np.arange(16, len(ds))
        ev, sp = heldout_explained(ds.W, ftr, fte)
        diag["populations"][f] = {
            "mean_rel_dist_to_train_mean": mean_rel_dev(ds.W, ftr, fte), "mean_pairwise_rel_dist": pairwise(ds.W, fte),
            "heldout_var_explained_by_train_pcs": ev, "train_pca_spectrum": sp,
            "frac_identity_perm_when_matching": identity_frac(ds.W, fte)}
    (RES / "alignment.json").write_text(json.dumps(diag, indent=1))
    for k, v in diag["populations"].items():
        print(f"  {k:18s} rel_dist_to_mean={v['mean_rel_dist_to_train_mean']:.3f} pairwise={v['mean_pairwise_rel_dist']:.3f}"
              f" heldout_var_expl@16={v['heldout_var_explained_by_train_pcs'][16]:.3f}", flush=True)

    traj = WeightsOnlyDataset("traj")
    p_traj = weight_matching(spec, template, traj.params(len(traj) - 1))  # one perm for the whole trajectory
    WT = np.stack([flatten(apply_perms(spec, p_traj, traj.params(i)), names) for i in range(len(traj))])

    # =====================================================================
    # Train all B variants in parallel (forked workers inherit the guard)
    # =====================================================================
    jobs = {}
    for s in cfg["main_seeds"]:
        jobs[f"main_s{s}"] = (WA[tr_i], WA[va_i], K, s)
    for k in cfg["latent_sweep"]:
        if k != K:
            jobs[f"k{k}"] = (WA[tr_i], WA[va_i], k, 0)
    for n in cfg["n_train_sweep"]:
        if n < len(tr_i):
            jobs[f"n{n}"] = (WA[tr_i[:n]], WA[va_i], K, 0)
    jobs["unaligned"] = (WR[tr_i], WR[va_i], K, 0)
    jobs["canon"] = (WC[tr_i], WC[va_i], K, 0)
    jobs["traj"] = (WT[:-3], WT[-3:], K, 0)
    for f, ds in fam.items():
        jobs[f"fam_{f}"] = (ds.W[:14], ds.W[14:16], K, 0)
        for k in cfg["family_latent_sweep"]:
            jobs[f"fam_{f}_k{k}"] = (ds.W[:14], ds.W[14:16], k, 0)
    _JOBS.update(jobs)
    _CTX.update(slices=slices, hp=hp)
    Bs, logs = {}, []
    order = sorted(jobs, key=lambda k: -len(jobs[k][0]))
    print(f"training {len(jobs)} Model-B variants", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for key, state, log in pool.imap_unordered(_train_job, order):
            Xtr, _, k, _ = jobs[key]
            m = WeightAE(Xtr.shape[1], k, hp["hidden"], hp["dropout"], WeightNormalizer(Xtr, slices))
            m.load_state_dict(state)
            m.eval()
            Bs[key] = m
            logs.append(log)
    for key in Bs:
        if key.startswith("main_s") or (key.startswith("fam_") and "_k" not in key):
            torch.save(Bs[key].state_dict(), ART / f"B_{key}.pt")

    def B(key):
        return Bs["main_s0"] if key in (f"k{K}", f"n{len(tr_i)}") else Bs[key]

    pcas = {f"k{k}": PCAModel(WA[tr_i], k, slices) for k in cfg["latent_sweep"]}
    for n in cfg["n_train_sweep"]:
        pcas[f"n{n}"] = PCAModel(WA[tr_i[:n]], K, slices)
    pcas["unaligned"] = PCAModel(WR[tr_i], K, slices)
    pcas["canon"] = PCAModel(WC[tr_i], K, slices)
    pcas["traj"] = PCAModel(WT[:-3], K, slices)

    # =====================================================================
    # Candidate weight vectors for Model C
    # =====================================================================
    cands, meta = {}, []
    P = WA.shape[1]
    rng = np.random.default_rng(1234)

    def add(name, vec, cond, source=None, ref=None, eval_mode="full", **extra):
        vec = np.asarray(vec, np.float32)
        assert vec.shape == (P,) and np.isfinite(vec).all(), name
        cands[name] = vec
        meta.append({"name": name, "condition": cond, "source": source, "eval": eval_mode,
                     "rel_err": rel_err(vec, ref) if ref is not None else None, **extra})

    def masked_block(prefix, Bm, pc, Wtr, w, r, groups):
        sd_pop = Wtr.std(0)
        mean = Wtr.mean(0)
        for gname in groups:
            sls = [name2slice[t] for t in cfg["mask_groups"][gname]]
            recB = Bm.reconstruct(w, mask_slices=sls)
            recP = pc.reconstruct(w, mask_slices=sls)
            for meth, src in [("B", recB), ("PCA", recP), ("mean", mean), ("rand", None)]:
                v = w.copy()
                for sl in sls:
                    v[sl] = src[sl] if src is not None else rng.normal(0, 1, sl.stop - sl.start) * sd_pop[sl].mean()
                got = np.concatenate([v[sl] for sl in sls])
                ref = np.concatenate([w[sl] for sl in sls])
                add(f"{prefix}mask_{gname}_{meth}|{r}", v, f"{prefix}mask_{meth}", r, w, group=gname,
                    masked_rel_err=rel_err(got, ref))

    def sample_block(prefix, Bm, pc, Wtr, tr_ids, n, seed_tag=""):
        Z = Bm.encode(Wtr)
        mu, L = fit_latent_gaussian(Z)
        for j, w in enumerate(Bm.decode(mu + rng.standard_normal((n, len(mu))) @ L.T)):
            d = np.linalg.norm(Wtr - w, axis=1) / np.linalg.norm(w)
            add(f"{prefix}B_sample{seed_tag}_{j:02d}", w, f"{prefix}C4_B_sample", None, None,
                seed_tag=seed_tag, nn_train_rel_dist=float(d.min()), nn_train_run=tr_ids[int(d.argmin())],
                dist_to_popmean=float(np.linalg.norm(w - Wtr.mean(0)) / np.linalg.norm(w)))
        if pc is not None:
            Zp = pc.encode(Wtr)
            mu, L = fit_latent_gaussian(Zp)
            for j, z in enumerate(mu + rng.standard_normal((n, len(mu))) @ L.T):
                w = pc.decode(z)
                d = np.linalg.norm(Wtr - w, axis=1) / np.linalg.norm(w)
                add(f"{prefix}PCA_sample_{j:02d}", w, f"{prefix}C4_PCA_sample", None, None,
                    nn_train_rel_dist=float(d.min()), nn_train_run=tr_ids[int(d.argmin())],
                    dist_to_popmean=float(np.linalg.norm(w - Wtr.mean(0)) / np.linalg.norm(w)))

    def interp_block(prefix, Wtr, tr_ids, n, cond_suffix=""):
        for j in range(n):
            a, b = rng.choice(len(Wtr), 2, replace=False)
            w = 0.5 * (Wtr[a] + Wtr[b])
            d = np.linalg.norm(Wtr - w, axis=1) / np.linalg.norm(w)
            add(f"{prefix}interp{cond_suffix}_{j:02d}", w, f"{prefix}C4_interp{cond_suffix}", f"{tr_ids[a]}+{tr_ids[b]}",
                nn_train_rel_dist=float(d.min()))
        sdv = Wtr.std(0)
        for j in range(max(2, n // 2)):
            add(f"{prefix}diag_gauss{cond_suffix}_{j:02d}", Wtr.mean(0) + sdv * rng.standard_normal(P),
                f"{prefix}C4_diag_gauss{cond_suffix}")

    def lmc_block(prefix, W, ids, pairs, tag):
        for j, (a, b) in enumerate(pairs):
            for al in (0.25, 0.5, 0.75):
                add(f"{prefix}lmc_{tag}_{j}_{al}", (1 - al) * W[a] + al * W[b], f"{prefix}lmc_{tag}", f"{ids[a]}+{ids[b]}",
                    alpha=al)

    def mutation_block(prefix, Bm, pc, Wtr, Wtest, test_ids):
        """Exp 3: B-latent mutations vs norm-matched random / PCA-direction mutations."""
        mu_z, Lz = fit_latent_gaussian(Bm.encode(Wtr))
        Zp = pc.encode(Wtr)
        pcs = pc.V * pc.sigma
        pc_scale = np.sqrt(np.maximum(np.var(Zp, 0), 1e-12))
        pert = cfg["perturb"]
        for w0, r in list(zip(Wtest, test_ids))[:pert["n_models"]]:
            z0 = Bm.encode(w0[None])[0]
            base = Bm.decode(z0[None])[0]
            add(f"{prefix}mut_base|{r}", base, f"{prefix}exp3_base", r, w0)
            for sc in pert["scales"]:
                for j in range(pert["n_variants"]):
                    wB = Bm.decode((z0 + sc * (rng.standard_normal(len(z0)) @ Lz.T))[None])[0]
                    dB = wB - base
                    dR = np.zeros(P, np.float32)
                    for sl in slices:
                        g = rng.standard_normal(sl.stop - sl.start)
                        dR[sl] = g / (np.linalg.norm(g) + 1e-12) * np.linalg.norm(dB[sl])
                    dP = (rng.standard_normal(pc.k) * pc_scale) @ pcs
                    dP = dP / (np.linalg.norm(dP) + 1e-12) * np.linalg.norm(dB)
                    t = f"s{sc}_{j}"
                    kw = dict(scale=sc, eval_mode="light")
                    add(f"{prefix}mut_B_{t}|{r}", wB, f"{prefix}exp3_mut_B", r, base, **kw)
                    add(f"{prefix}mut_randOnRec_{t}|{r}", base + dR, f"{prefix}exp3_mut_rand_on_rec", r, base, **kw)
                    add(f"{prefix}mut_pcaOnRec_{t}|{r}", base + dP, f"{prefix}exp3_mut_pca_on_rec", r, base, **kw)
                    add(f"{prefix}mut_randOnA_{t}|{r}", w0 + dR, f"{prefix}exp3_mut_rand_on_A", r, w0, **kw)
                    add(f"{prefix}mut_BdeltaOnA_{t}|{r}", w0 + dB, f"{prefix}exp3_mut_Bdelta_on_A", r, w0, **kw)

    # ---------------- independent population ----------------
    run = lambda i: ind.run_ids[i]  # noqa: E731
    b_bytes = 4 * Bs["main_s0"].n_params()
    for i in te_i:
        r = run(i)
        add(f"copy|{r}", WR[i], "C2_copy", r, WR[i], msg_bytes=4 * P)
        add(f"aligned_copy|{r}", WA[i], "C2_aligned_copy", r, WA[i], msg_bytes=4 * P)
        for s in cfg["main_seeds"]:
            add(f"B_main_s{s}|{r}", Bs[f"main_s{s}"].reconstruct(WA[i]), "C3_B_recon", r, WA[i],
                seed=s, k=K, msg_bytes=4 * K, shared_bytes=b_bytes)
        for k in cfg["latent_sweep"]:
            if k != K:
                add(f"B_k{k}|{r}", B(f"k{k}").reconstruct(WA[i]), "exp2_B_latent", r, WA[i], k=k, msg_bytes=4 * k)
            pc = pcas[f"k{k}"]
            add(f"PCA_k{k}|{r}", pc.reconstruct(WA[i]), "exp2_PCA_latent", r, WA[i], k=pc.k, msg_bytes=4 * pc.k,
                shared_bytes=4 * pc.n_params())
        add(f"popmean_aligned|{r}", WA[tr_i].mean(0), "C5_popmean_aligned", r, WA[i], k=0, msg_bytes=0)
        add(f"popmean_raw|{r}", WR[tr_i].mean(0), "C5_popmean_raw", r, WR[i], k=0, msg_bytes=0)
        for n in cfg["n_train_sweep"]:
            if n < len(tr_i):
                add(f"B_n{n}|{r}", B(f"n{n}").reconstruct(WA[i]), "exp4_B_ntrain", r, WA[i], n_train=n)
            add(f"PCA_n{n}|{r}", pcas[f"n{n}"].reconstruct(WA[i]), "exp4_PCA_ntrain", r, WA[i], n_train=n)
        add(f"B_unaligned|{r}", Bs["unaligned"].reconstruct(WR[i]), "align_B_unaligned", r, WR[i])
        add(f"PCA_unaligned|{r}", pcas["unaligned"].reconstruct(WR[i]), "align_PCA_unaligned", r, WR[i])
        add(f"B_canon|{r}", Bs["canon"].reconstruct(WC[i]), "align_B_canon", r, WC[i])
        add(f"PCA_canon|{r}", pcas["canon"].reconstruct(WC[i]), "align_PCA_canon", r, WC[i])
        add(f"B_traj|{r}", Bs["traj"].reconstruct(WA[i]), "single_run_B_traj", r, WA[i])
        add(f"PCA_traj|{r}", pcas["traj"].reconstruct(WA[i]), "single_run_PCA_traj", r, WA[i])
        pa = unflatten(WR[i], names, shapes)
        for bits, per_row in [(8, False), (4, True)]:
            q, nb = quantize(pa, bits, per_row)
            add(f"int{bits}|{r}", flatten(q, names), f"C5_int{bits}", r, WR[i], msg_bytes=nb)
        for rk in cfg["lowrank_ranks"]:
            q, nb = lowrank(pa, rk)
            add(f"lowrank{rk}|{r}", flatten(q, names), f"C5_lowrank{rk}", r, WR[i], msg_bytes=nb)
        masked_block("", Bs["main_s0"], pcas[f"k{K}"], WA[tr_i], WA[i], r, list(cfg["mask_groups"]))
    for i in tr_i[:8]:
        add(f"B_main_s0_trainrec|{run(i)}", Bs["main_s0"].reconstruct(WA[i]), "memorization_B_train", run(i), WA[i])
        add(f"PCA_k{K}_trainrec|{run(i)}", pcas[f"k{K}"].reconstruct(WA[i]), "memorization_PCA_train", run(i), WA[i])
    # traj-trained B on its own (in-trajectory) held-out snapshots
    for j in range(len(WT) - 3, len(WT)):
        add(f"B_traj_own|{traj.run_ids[j]}", Bs["traj"].reconstruct(WT[j]), "single_run_B_traj_own", traj.run_ids[j], WT[j])
    tr_ids = [run(i) for i in tr_i]
    for s in cfg["main_seeds"]:
        sample_block("", Bs[f"main_s{s}"], pcas[f"k{K}"] if s == 0 else None, WA[tr_i], tr_ids,
                     cfg["n_samples"] if s == 0 else 8, seed_tag=f"_s{s}")
    interp_block("", WA[tr_i], tr_ids, cfg["n_interp"], "_aligned")
    interp_block("", WR[tr_i], tr_ids, cfg["n_interp"], "_raw")
    pairs = [(tr_i[2 * j], tr_i[2 * j + 1]) for j in range(4)]
    lmc_block("", WR, ind.run_ids, pairs, "ind_raw")
    lmc_block("", WA, ind.run_ids, pairs, "ind_aligned")
    lmc_block("", WC, ind.run_ids, pairs, "ind_canon_aligned")
    mutation_block("", Bs["main_s0"], pcas[f"k{K}"], WA[tr_i], WA[te_i], [run(i) for i in te_i])

    # ---------------- family populations (shared history) ----------------
    for f, ds in fam.items():
        W = ds.W
        ftr, fte = np.arange(0, 14), np.arange(16, len(ds))
        pre = f"fam_{f}:"
        Bm = Bs[f"fam_{f}"]
        pc = PCAModel(W[ftr], K, slices)
        pcs_k = {k: PCAModel(W[ftr], k, slices) for k in cfg["family_latent_sweep"]}
        for i in fte:
            r = ds.run_ids[i]
            add(f"{pre}copy|{r}", W[i], f"{pre}copy", r, W[i])
            add(f"{pre}B_recon|{r}", Bm.reconstruct(W[i]), f"{pre}B_recon", r, W[i], k=K)
            add(f"{pre}PCA_recon|{r}", pc.reconstruct(W[i]), f"{pre}PCA_recon", r, W[i], k=pc.k)
            add(f"{pre}popmean|{r}", W[ftr].mean(0), f"{pre}popmean", r, W[i], k=0)
            for k in cfg["family_latent_sweep"]:
                add(f"{pre}B_k{k}|{r}", Bs[f"fam_{f}_k{k}"].reconstruct(W[i]), f"{pre}exp2_B_latent", r, W[i], k=k)
                add(f"{pre}PCA_k{k}|{r}", pcs_k[k].reconstruct(W[i]), f"{pre}exp2_PCA_latent", r, W[i], k=k)
            masked_block(pre, Bm, pc, W[ftr], W[i], r, ["head", "tok_emb", "block1_mlp", "block0_attn"])
        for i in ftr[:4]:
            add(f"{pre}B_trainrec|{ds.run_ids[i]}", Bm.reconstruct(W[i]), f"{pre}memorization_B_train", ds.run_ids[i], W[i])
        fids = [ds.run_ids[i] for i in ftr]
        sample_block(pre, Bm, pc, W[ftr], fids, 12)
        interp_block(pre, W[ftr], fids, 8)
        lmc_block(pre, W, ds.run_ids, [(0, 1), (2, 3), (4, 5), (6, 7)], "raw")
        mutation_block(pre, Bm, pc, W[ftr], W[fte], [ds.run_ids[i] for i in fte])
        # cross-population transfer: can a family-trained B encode an independent model? (expected: no)
        for i in te_i[:4]:
            add(f"{pre}xfer_from_ind|{run(i)}", Bm.reconstruct(WR[i]), f"{pre}xfer_ind_through_famB", run(i), WR[i])

    # ---------------- save ----------------
    np.savez(ART / "candidates.npz", **{m["name"]: cands[m["name"]] for m in meta})
    (RES / "candidates_meta.json").write_text(json.dumps(
        {"names": names, "shapes": shapes.tolist(), "P": int(P), "families": families, "candidates": meta}, indent=0))
    (RES / "b_training.json").write_text(json.dumps(logs, indent=0))
    (RES / "weight_space.json").write_text(json.dumps({
        "pca_explained_ratio_aligned_k16": pcas[f"k{K}"].explained[:20].tolist(),
        "b_param_count": Bs["main_s0"].n_params(), "P": int(P),
        "b_latent_std_main_s0": Bs["main_s0"].encode(WA[tr_i]).std(0).tolist()}, indent=1))
    print(f"wrote {len(meta)} candidates", flush=True)


if __name__ == "__main__":
    main()
