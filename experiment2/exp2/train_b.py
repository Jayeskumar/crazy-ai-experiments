"""Phase 2: train Model B on weights only, emit candidate weight vectors for Model C.

Runs entirely in weights-only mode. Inputs: artifacts/weights_only/{b_train,b_val,probe}.npz
(opaque ids, shuffled, no task information). Outputs:
  artifacts/model_b/candidates.npz        candidate name -> parameter vector
  artifacts/model_b/B_r*.pt               trained Model B checkpoints
  results/model_b/candidates_meta.json    per-candidate weight-space metadata (opaque source ids)
  results/model_b/b_training.json         loss curves, validation errors
"""
from exp2.isolation import enter_weights_only_mode, assert_active

enter_weights_only_mode()

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from exp2 import ROOT  # noqa: E402
from exp2.model_b import RowAE, RowPCA, quantize_int8  # noqa: E402
from exp2.sanitize import WeightsOnly  # noqa: E402
from exp2.wtok import RowTokenizer  # noqa: E402
from src.alignment import perm_spec, random_permutation  # noqa: E402
from src.model_b import WeightAE, WeightNormalizer  # noqa: E402
from src.weights_only import unflatten, flatten  # noqa: E402

ART = ROOT / "artifacts" / "model_b"
RES = ROOT / "results" / "model_b"
G = {}


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
def train_rowae(r):
    assert_active()
    torch.set_num_threads(1)
    cfg, tk = G["cfg"], G["tk"]
    torch.manual_seed(r)
    rng = np.random.default_rng(r)
    pool, val = G["pool_n"], G["val_n"]
    types = torch.tensor(tk.types)[None]
    pos = torch.tensor(tk.pos)[None]
    valid = torch.tensor(tk.mask, dtype=torch.float32)[None]
    m = RowAE(tk.L, len(tk.type_names), 128, r, cfg["hidden"], cfg["depth"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"], eta_min=cfg["lr"] * 0.05)
    vt = torch.tensor(val)
    vmask = torch.tensor(rng.random(val.shape) > 0.5, dtype=torch.float32) * valid
    curve, best, best_state, t0 = [], 1e9, None, time.time()
    for step in range(1, cfg["steps"] + 1):
        m.train()
        b = torch.tensor(pool[rng.integers(0, len(pool), cfg["batch_size"])])
        ratio = np.where(rng.random(len(b)) < cfg["p_mask"], rng.uniform(0, cfg["max_mask"], len(b)), 0.0)
        obs = (torch.rand(b.shape) > torch.tensor(ratio, dtype=torch.float32)[:, None, None]).float() * valid
        out, _ = m(b, obs, types, pos)
        loss = (((out - b) ** 2) * valid).sum() / (valid.sum() * len(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            m.eval()
            with torch.no_grad():
                clean = ((((m(vt, valid.expand_as(vt), types, pos)[0] - vt) ** 2) * valid).sum() / (valid.sum() * len(vt))).item()
                masked = ((((m(vt, vmask, types, pos)[0] - vt) ** 2) * valid).sum() / (valid.sum() * len(vt))).item()
            curve.append({"step": step, "loss": loss.item(), "val_clean_mse_n": clean, "val_mask50_mse_n": masked})
            score = clean + masked
            if score < best:
                best, best_state = score, {k: v.clone() for k, v in m.state_dict().items()}
    return ("row", r), best_state, {"name": f"B_r{r}", "r": r, "curve": curve, "time_s": time.time() - t0,
                                     "n_params": sum(p.numel() for p in m.parameters())}


def train_flat():
    assert_active()
    torch.set_num_threads(1)
    cfg = G["cfg"]["flat_ae"]
    torch.manual_seed(0)
    Xtr, sl = G["Wtr"], [slice(0, G["Wtr"].shape[1])]
    norm = WeightNormalizer(Xtr, G["tensor_slices"])
    m = WeightAE(Xtr.shape[1], cfg["k"], 256, 0.1, norm)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    X = norm.norm(torch.tensor(Xtr))
    t0 = time.time()
    for step in range(cfg["steps"]):
        m.train()
        out, _ = m(X + 0.1 * torch.randn_like(X))
        loss = ((out - X) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ("flat", 0), m.state_dict(), {"name": "flatAE_exp1_style", "time_s": time.time() - t0, "final_loss": loss.item()}


def _job(j):
    return train_rowae(j) if j != "flat" else train_flat()


# ---------------------------------------------------------------------------
def quant_tokens(T, bits):
    """Per-token symmetric mid-rise uniform quantisation with 2**bits levels."""
    mx = np.abs(T).max(-1, keepdims=True)
    d = np.where(mx == 0, 1.0, 2 * mx / 2 ** bits)
    q = (np.floor(T / d) + 0.5) * d
    lim = (2 ** (bits - 1) - 0.5) * d
    return np.clip(q, -lim, lim)


def hard_impute(M, obs, k=8, iters=30):
    X = np.where(obs, M, 0.0)
    for _ in range(iters):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Xh = (U[:, :k] * S[:k]) @ Vt[:k]
        X = np.where(obs, M, Xh)
    return X


def lowrank(params, r):
    out = {}
    for n, w in params.items():
        if w.ndim == 2 and min(w.shape) > r:
            U, S, Vt = np.linalg.svd(w, full_matrices=False)
            out[n] = ((U[:, :r] * S[:r]) @ Vt[:r]).astype(np.float32)
        else:
            out[n] = w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs" / "model_b.json").read_text())
    if args.quick:
        cfg.update(rs=[4, 32], steps=30, eval_every=10, n_perm_aug=2, flat_ae={"k": 16, "steps": 20})
    ART.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    tr, va, pr = WeightsOnly("b_train"), WeightsOnly("b_val"), WeightsOnly("probe")
    names, shapes = tr.names, tr.shapes
    tk = RowTokenizer(names, shapes)
    spec = perm_spec(names, 2)
    P = tk.P
    print(f"weights-only: b_train={len(tr.W)} b_val={len(va.W)} probe={len(pr.W)} P={P} tokens={tk.N} L={tk.L}",
          flush=True)
    # normalisation: per token-type std of B-train values (weights only)
    Ttr = tk.to_tokens(tr.W)
    sig = np.ones(len(tk.type_names), np.float32)
    for t in range(len(tk.type_names)):
        sel = tk.types == t
        sig[t] = Ttr[:, sel][:, tk.mask[sel]].std() + 1e-8
    sig_tok = sig[tk.types][:, None]                                  # [N,1]
    norm = lambda T: T / sig_tok * tk.mask  # noqa: E731
    denorm = lambda T: T * sig_tok * tk.mask  # noqa: E731
    # symmetry augmentation: exact random permutations (function-preserving) of B-train models
    rng = np.random.default_rng(0)
    pool = [tr.W]
    for a in range(cfg["n_perm_aug"]):
        pool.append(np.stack([flatten(random_permutation(spec, unflatten(w, names, shapes), int(rng.integers(1e9))),
                                      names) for w in tr.W]))
    G.update(cfg=cfg, tk=tk, pool_n=norm(tk.to_tokens(np.concatenate(pool))).astype(np.float32),
             val_n=norm(tk.to_tokens(va.W)).astype(np.float32), Wtr=tr.W)
    sl, o = [], 0
    for s in shapes:
        k = int(np.prod([x for x in s if x >= 0]))
        sl.append(slice(o, o + k))
        o += k
    G["tensor_slices"] = sl

    jobs = ["flat"] + sorted(cfg["rs"], reverse=True)
    print(f"training {len(jobs)} B variants on {len(G['pool_n'])} (augmented) weight vectors", flush=True)
    Bs, logs = {}, []
    with mp.get_context("fork").Pool(args.workers) as pool_:
        for key, state, log in pool_.imap_unordered(_job, jobs):
            if key[0] == "row":
                mdl = RowAE(tk.L, len(tk.type_names), 128, key[1], cfg["hidden"], cfg["depth"])
                torch.save(state, ART / f"B_r{key[1]}.pt")
            else:
                mdl = WeightAE(P, cfg["flat_ae"]["k"], 256, 0.1, WeightNormalizer(tr.W, sl))
            mdl.load_state_dict(state)
            mdl.eval()
            Bs[key] = mdl
            logs.append(log)
            print(f"  done {log['name']} ({log['time_s']:.0f}s) "
                  + (f"val clean={log['curve'][-1]['val_clean_mse_n']:.4f} mask50={log['curve'][-1]['val_mask50_mse_n']:.4f}"
                     if "curve" in log else ""), flush=True)
    (RES / "b_training.json").write_text(json.dumps(logs, indent=0))

    types_t, pos_t = torch.tensor(tk.types)[None], torch.tensor(tk.pos)[None]
    valid_t = torch.tensor(tk.mask, dtype=torch.float32)[None]

    @torch.no_grad()
    def row_codes(r, Tn, obs=None):
        o = valid_t.expand(len(Tn), -1, -1) if obs is None else torch.tensor(obs, dtype=torch.float32)
        return Bs[("row", r)].encode(torch.tensor(Tn), o, types_t, pos_t).numpy()

    @torch.no_grad()
    def row_decode(r, Z):
        return Bs[("row", r)].decode(torch.tensor(Z), types_t, pos_t).numpy()

    Tpr = norm(tk.to_tokens(pr.W)).astype(np.float32)
    pcas = {r: RowPCA(G["pool_n"][:len(tr.W)], tk.mask, tk.types, r) for r in cfg["rs"]}
    cands, meta = {}, []
    mean_tr = tr.W.mean(0)
    n_lat = tk.N

    def add(name, vec, method, src_i, **kw):
        vec = np.asarray(vec, np.float32)
        assert vec.shape == (P,) and np.isfinite(vec).all(), name
        cands[name] = vec
        meta.append({"name": name, "method": method, "source": pr.ids[src_i] if src_i is not None else None,
                     "rel_err": rel(vec, pr.W[src_i]) if src_i is not None else None, **kw})

    # --- compression family (clean input) ---
    recon = {}
    for r in cfg["rs"]:
        Z = row_codes(r, Tpr)
        recon[(r, "fp")] = tk.from_tokens(denorm(row_decode(r, Z)))
        recon[(r, "q8")] = tk.from_tokens(denorm(row_decode(r, quantize_int8(Z))))
        recon[(r, "pca")] = tk.from_tokens(denorm(pcas[r].reconstruct(Tpr, q8=True)))
    flat = Bs[("flat", 0)]
    flat_out = flat.reconstruct(pr.W)
    Ttok = tk.to_tokens(pr.W)
    for i in range(len(pr.W)):
        add(f"copy|{i}", pr.W[i], "copy", i, bits=32.0)
        for r in cfg["rs"]:
            b8 = n_lat * r * 8 / P
            add(f"B_r{r}_fp|{i}", recon[(r, "fp")][i], f"B_r{r}_fp", i, bits=n_lat * r * 32 / P, r=r)
            add(f"B_r{r}_q8|{i}", recon[(r, "q8")][i], f"B_r{r}_q8", i, bits=b8, r=r)
            add(f"rowPCA_r{r}|{i}", recon[(r, "pca")][i], f"rowPCA_r{r}_q8", i, bits=b8, r=r)
        for bits in cfg["quant_bits"]:
            add(f"quant{bits}|{i}", tk.from_tokens(quant_tokens(Ttok[i], bits)), f"quant{bits}", i,
                bits=bits + 16 * n_lat / P)
        for rk in (4, 8):
            add(f"lowrank{rk}|{i}", flatten(lowrank(unflatten(pr.W[i], names, shapes), rk), names), f"lowrank{rk}", i)
        add(f"globalmean|{i}", mean_tr, "global_mean", i, bits=0.0)
        add(f"flatAE|{i}", flat_out[i], "flatAE_exp1", i, bits=16 * 32 / P)
        j = int(np.argmin(np.linalg.norm(tr.W - pr.W[i], axis=1)))
        add(f"nn_retrieval|{i}", tr.W[j], "nn_retrieval", i, bits=np.log2(len(tr.W)) / P, retrieved=tr.ids[j])

    # --- masked recovery (B sees only a random subset of each model's entries) ---
    mask_rs = [r for r in (16, 32) if ("row", r) in Bs]
    for ratio in cfg["masked_ratios"]:
        mrng = np.random.default_rng(int(ratio * 1000))
        obs = (mrng.random(Tpr.shape) > ratio) & tk.mask
        outB = {r: tk.from_tokens(denorm(np.where(obs, Tpr, row_decode(r, row_codes(r, Tpr, obs.astype(np.float32))))))
                for r in mask_rs}
        outP = tk.from_tokens(denorm(np.where(obs, Tpr, pcas[max(cfg['rs'])].reconstruct(Tpr, q8=False, obs=obs.astype(np.float32)))))
        for i in range(len(pr.W)):
            Ti = Ttok[i]
            zero = tk.from_tokens(np.where(obs[i], Ti, 0.0))
            obs_flat = tk.from_tokens(obs[i].astype(np.float32)) > 0
            # low-rank matrix completion, per 2-D tensor
            pi, oi = unflatten(pr.W[i], names, shapes), unflatten(obs_flat, names, shapes)
            hi = {n: (hard_impute(w, oi[n]) if w.ndim == 2 else np.where(oi[n], w, 0.0)) for n, w in pi.items()}
            tag = f"m{int(ratio * 100)}"
            kw = dict(mask_ratio=ratio)
            for r in mask_rs:
                add(f"mask{tag}_B_r{r}|{i}", outB[r][i], f"mask{tag}_B_r{r}", i, **kw)
            add(f"mask{tag}_rowPCA|{i}", outP[i], f"mask{tag}_rowPCA", i, **kw)
            add(f"mask{tag}_zero|{i}", zero, f"mask{tag}_zero", i, **kw)
            add(f"mask{tag}_impute|{i}", flatten(hi, names), f"mask{tag}_lowrank_impute", i, **kw)
    np.savez(ART / "candidates.npz", **cands)
    (RES / "candidates_meta.json").write_text(json.dumps(
        {"names": names, "shapes": shapes.tolist(), "P": P, "probe_ids": pr.ids, "b_train_ids": tr.ids,
         "tokens": tk.N, "L": tk.L, "candidates": meta}, indent=0))
    print(f"wrote {len(meta)} candidates", flush=True)


if __name__ == "__main__":
    main()
