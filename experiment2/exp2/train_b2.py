"""Phase 2b (targeted follow-up, motivated by the round-1 per-tensor error analysis).

1. B v2: same RowAE, but normalisation per (token type, entry position), so the bias entry of
   each row token is standardised separately (round 1 lost the biases: rel. err ~1.0).
2. Learned de-quantiser: B receives a b-bit (b = 2, 3) per-token-quantised model and predicts the
   full-precision weights (no bottleneck). Gains over plain de-quantisation at IDENTICAL bits can only
   come from a learned prior over trained weights. Controls: plain mid-point de-quantisation, a learned
   *scalar* prior (mean offset inside each bin, per token type: the marginal distribution only), and
   B's output clipped into the known quantisation bin ("consistent" reconstruction).

Weights-only process; writes artifacts/model_b/candidates_v2.npz + results/model_b/candidates_meta_v2.json
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
from exp2.model_b import RowAE, quantize_int8  # noqa: E402
from exp2.sanitize import WeightsOnly  # noqa: E402
from exp2.train_b import quant_tokens, rel  # noqa: E402
from exp2.wtok import RowTokenizer  # noqa: E402
from src.alignment import perm_spec, random_permutation  # noqa: E402
from src.weights_only import flatten, unflatten  # noqa: E402

ART = ROOT / "artifacts" / "model_b"
RES = ROOT / "results" / "model_b"
G = {}


def bin_width(T, bits):
    mx = np.abs(T).max(-1, keepdims=True)
    return np.where(mx == 0, 1.0, 2 * mx / 2 ** bits)


def train(job):
    """job = ('ae', r) or ('deq', bits)"""
    assert_active()
    torch.set_num_threads(1)
    kind, arg = job
    cfg, tk = G["cfg"], G["tk"]
    torch.manual_seed(100 + arg)
    rng = np.random.default_rng(100 + arg)
    r = arg if kind == "ae" else 48
    m = RowAE(tk.L, len(tk.type_names), 128, r, cfg["hidden"], cfg["depth"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"], eta_min=cfg["lr"] * 0.05)
    types, pos = torch.tensor(tk.types)[None], torch.tensor(tk.pos)[None]
    valid = torch.tensor(tk.mask, dtype=torch.float32)[None]
    raw, val_raw = G["pool_raw"], G["val_raw"]
    inp_of = (lambda T: T) if kind == "ae" else (lambda T: quant_tokens(T, arg))
    vin, vt = torch.tensor(G["norm"](inp_of(val_raw))), torch.tensor(G["norm"](val_raw))
    best, state, curve, t0 = 1e9, None, [], time.time()
    for step in range(1, cfg["steps"] + 1):
        m.train()
        T = raw[rng.integers(0, len(raw), cfg["batch_size"])]
        x, y = torch.tensor(G["norm"](inp_of(T))), torch.tensor(G["norm"](T))
        out, _ = m(x, valid.expand_as(x), types, pos)
        loss = (((out - y) ** 2) * valid).sum() / (valid.sum() * len(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            m.eval()
            with torch.no_grad():
                v = ((((m(vin, valid.expand_as(vin), types, pos)[0] - vt) ** 2) * valid).sum() / (valid.sum() * len(vt))).item()
                base = ((((vin - vt) ** 2) * valid).sum() / (valid.sum() * len(vt))).item()
            curve.append({"step": step, "loss": loss.item(), "val_mse_n": v, "val_mse_n_input": base})
            if v < best:
                best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    return job, state, {"name": f"{kind}_{arg}", "curve": curve, "best_val_mse_n": best, "time_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs" / "model_b.json").read_text())
    if args.quick:
        cfg.update(steps=20, eval_every=10, n_perm_aug=1)
    torch.set_num_threads(1)
    tr, va, pr = WeightsOnly("b_train"), WeightsOnly("b_val"), WeightsOnly("probe")
    names, shapes = tr.names, tr.shapes
    tk = RowTokenizer(names, shapes)
    spec = perm_spec(names, 2)
    rng = np.random.default_rng(0)
    pool = [tr.W] + [np.stack([flatten(random_permutation(spec, unflatten(w, names, shapes), int(rng.integers(1e9))), names)
                               for w in tr.W]) for _ in range(cfg["n_perm_aug"])]
    pool_raw = tk.to_tokens(np.concatenate(pool))
    # per (type, entry position) std -> the bias entry gets its own scale
    sig = np.ones((len(tk.type_names), tk.L), np.float32)
    Ttr = tk.to_tokens(tr.W)
    for t in range(len(tk.type_names)):
        sel = tk.types == t
        for j in range(tk.L):
            ok = tk.mask[sel, j]
            if ok.any():
                sig[t, j] = Ttr[:, sel, j][:, ok].std() + 1e-8
    sig_tok = sig[tk.types]
    norm = lambda T: (T / sig_tok * tk.mask).astype(np.float32)  # noqa: E731
    denorm = lambda T: (T * sig_tok * tk.mask).astype(np.float32)  # noqa: E731
    G.update(cfg=cfg, tk=tk, pool_raw=pool_raw, val_raw=tk.to_tokens(va.W), norm=norm)
    jobs = [("ae", 32), ("ae", 16), ("deq", 3), ("deq", 2)]
    Bs, logs = {}, []
    with mp.get_context("fork").Pool(args.workers) as p:
        for job, state, log in p.imap_unordered(train, jobs):
            m = RowAE(tk.L, len(tk.type_names), 128, job[1] if job[0] == "ae" else 48, cfg["hidden"], cfg["depth"])
            m.load_state_dict(state)
            m.eval()
            Bs[job] = m
            logs.append(log)
            torch.save(state, ART / f"B2_{job[0]}{job[1]}.pt")
            print(f"  done {log['name']} ({log['time_s']:.0f}s) best val mse_n={log['best_val_mse_n']:.4f} "
                  f"(input itself: {log['curve'][-1]['val_mse_n_input']:.4f})", flush=True)
    (RES / "b2_training.json").write_text(json.dumps(logs, indent=0))

    types, pos = torch.tensor(tk.types)[None], torch.tensor(tk.pos)[None]
    Traw = tk.to_tokens(pr.W)
    full = torch.ones(len(Traw), tk.N, tk.L) * torch.tensor(tk.mask, dtype=torch.float32)

    @torch.no_grad()
    def run(job, Tin_n, q8=False):
        m = Bs[job]
        z = m.encode(torch.tensor(Tin_n), full, types, pos).numpy()
        if q8:
            z = quantize_int8(z)
        return m.decode(torch.tensor(z), types, pos).numpy()

    cands, meta = {}, []
    P = tk.P

    def add(name, vec, method, i, **kw):
        vec = np.asarray(vec, np.float32)
        assert vec.shape == (P,) and np.isfinite(vec).all(), name
        cands[name] = vec
        meta.append({"name": name, "method": method, "source": pr.ids[i], "rel_err": rel(vec, pr.W[i]), **kw})

    for r in (32, 16):
        out = tk.from_tokens(denorm(run(("ae", r), norm(Traw), q8=True)))
        for i in range(len(pr.W)):
            add(f"B2_r{r}_q8|{i}", out[i], f"B2_r{r}_q8", i, bits=tk.N * r * 8 / P)
    for bits in (3, 2):
        Q = quant_tokens(Traw, bits)
        D = bin_width(Traw, bits)
        Bout = denorm(run(("deq", bits), norm(Q)))
        Bclip = np.clip(Bout, Q - D / 2, Q + D / 2)
        # learned scalar prior: mean in-bin offset per (token type, bin index), fitted on B-train
        Qtr, Dtr = quant_tokens(Ttr, bits), bin_width(Ttr, bits)
        k_tr = np.floor(Qtr / Dtr).astype(int)
        off = np.zeros((len(tk.type_names), 2 ** bits))
        for t in range(len(tk.type_names)):
            sel = (tk.types == t)[None, :, None] & tk.mask[None]
            for kk in range(-2 ** (bits - 1), 2 ** (bits - 1)):
                hit = sel & (k_tr == kk)
                if hit.any():
                    off[t, kk + 2 ** (bits - 1)] = ((Ttr - Qtr) / Dtr)[hit].mean()
        k_pr = np.clip(np.floor(Q / D).astype(int), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
        scal = Q + off[tk.types[None, :, None], k_pr + 2 ** (bits - 1)] * D
        bpw = bits + 16 * tk.N / P
        for i in range(len(pr.W)):
            add(f"deq{bits}_plain|{i}", tk.from_tokens(Q[i]), f"deq{bits}_plain_midpoint", i, bits=bpw)
            add(f"deq{bits}_scalar|{i}", tk.from_tokens(np.where(tk.mask, scal[i], 0)), f"deq{bits}_scalar_prior", i, bits=bpw)
            add(f"deq{bits}_B|{i}", tk.from_tokens(Bout[i]), f"deq{bits}_B", i, bits=bpw)
            add(f"deq{bits}_Bclip|{i}", tk.from_tokens(Bclip[i]), f"deq{bits}_B_clipped", i, bits=bpw)
    np.savez(ART / "candidates_v2.npz", **cands)
    (RES / "candidates_meta_v2.json").write_text(json.dumps(
        {"names": names, "shapes": shapes.tolist(), "P": P, "probe_ids": pr.ids, "b_train_ids": tr.ids,
         "candidates": meta}, indent=0))
    print(f"wrote {len(meta)} candidates", flush=True)


if __name__ == "__main__":
    main()
