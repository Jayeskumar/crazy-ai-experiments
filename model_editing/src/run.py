"""Experiment driver. Every stage caches its output and is skipped when the output exists."""
import os
import shutil
import time

import torch
from safetensors.torch import save_file

from . import config as C
from . import methods as M
from .common import (assert_matches_original, down_proj, dump, first_word, greedy, load, load_json,
                     original_tensor, apath, rpath, seed_all, sha256, tensor_name)
from .edits import EDITS
from .evaluate import probe_logits, raw_eval, summarize
from .persist import diff_vs_original, reload_in_fresh_process, save
from .stats import context_templates, key_covariance
from .trace import run as causal_trace

METHODS = ["rome", "rome_identity_cov", "random_rank1", "random_dense",
           "direct_logit_mid", "direct_logit_final", "ft_constrained"]

GRADIENT_USE = {
    "rome": "gradient descent (Adam, <=25 steps) on a single 2048-d activation vector delta; "
            "ALL weights frozen. Weight update itself is closed-form linear algebra.",
    "rome_identity_cov": "same delta as rome (gradient on activation vector only); closed-form update with C=I.",
    "random_rank1": "none",
    "random_dense": "none",
    "direct_logit_mid": "none (forward-only line search over one scalar alpha)",
    "direct_logit_final": "none (forward-only line search over one scalar alpha)",
    "ft_constrained": "YES: Adam on the weight tensor itself (one tensor, 11.5M params), L-inf clamped",
}

os.makedirs(rpath("runs"), exist_ok=True)
os.makedirs(rpath("deltas"), exist_ok=True)
os.makedirs(apath("deltas"), exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stage_weights_record():
    path = rpath("original_weights.json")
    h = sha256(C.ORIG_WEIGHTS)
    if os.path.exists(path):
        rec = load_json(path)
        assert rec["sha256"] == h, "ORIGINAL WEIGHTS CHANGED ON DISK"
        return rec
    rec = dict(model_id=C.MODEL_ID, file=C.ORIG_WEIGHTS, sha256=h, bytes=os.path.getsize(C.ORIG_WEIGHTS),
               file_mode=oct(os.stat(C.ORIG_WEIGHTS).st_mode & 0o777))
    dump(rec, path)
    return rec


def stage_baseline(model, tok):
    path = rpath("baseline.json")
    if os.path.exists(path):
        b = load_json(path)
        return b["resolved"], b["raw"], torch.load(apath("base_next_lp.pt"))
    log("baseline: resolving contrast answers for fictional subjects")
    resolved = {}
    for e in EDITS:
        r = {}
        p = e["prompt"].format(e["subject"])
        r["true"] = e["true"] or first_word(greedy(model, tok, p, 6))
        r["implication"] = [t or first_word(greedy(model, tok, q, 6)) for q, _, t in e["implication"]]
        r["neighborhood"] = [t or first_word(greedy(model, tok, q, 6)) for q, t in e["neighborhood"]]
        r["original_greedy"] = greedy(model, tok, p, 12)
        resolved[e["id"]] = r
    dump(resolved, rpath("resolved_targets.json"))
    log("baseline: evaluating original model")
    raw, next_lp = raw_eval(model, tok, resolved)
    torch.save(next_lp, apath("base_next_lp.pt"))
    dump(dict(resolved=resolved, raw=raw), path)
    return resolved, raw, next_lp


def stage_trace(model, tok):
    path = rpath("causal_trace.json")
    if not os.path.exists(path):
        log("causal tracing")
        t0 = time.time()
        tr = causal_trace(model, tok)
        tr["seconds"] = time.time() - t0
        dump(tr, path)
    return load_json(path)


def stage_templates(model, tok):
    path = rpath("context_templates.json")
    if not os.path.exists(path):
        dump(context_templates(model, tok), path)
    return load_json(path)


def apply_delta(model, layer, dW):
    W = down_proj(model, layer).weight
    W.data.add_(dW)


def restore(model, names):
    sd = model.state_dict()
    for n in names:
        sd[n].copy_(original_tensor(n))
    assert_matches_original(model, names)


def compute_method(name, model, tok, req, true, layer, templates, cov, rome_cache):
    """Return (layer, dW, info) for one method, edit request req."""
    if name in ("rome", "rome_identity_cov"):
        if req["id"] not in rome_cache:
            k = M.subject_key(model, tok, req, layer, templates)
            delta, info = M.rome_delta(model, tok, req, layer, templates)
            rome_cache[req["id"]] = (k, delta, info)
        k, delta, vinfo = rome_cache[req["id"]]
        dW, uinfo = M.rome_update(k, delta, cov if name == "rome" else None)
        return layer, dW, dict(v_optimisation=vinfo, key_norm=k.norm().item(), delta_norm=delta.norm().item(),
                               **uinfo)
    if name in ("random_rank1", "random_dense"):
        fro = rome_cache[req["id"] + "_fro"]
        shape = tuple(down_proj(model, layer).weight.shape)
        fn = M.random_rank1 if name == "random_rank1" else M.random_dense
        dW, info = fn(shape, fro, C.SEED + EDITS.index(req))
        return layer, dW, info
    if name == "direct_logit_mid":
        k = M.subject_key(model, tok, req, layer, templates)
        dW, info = M.direct_logit(model, tok, req, true, layer, k)
        return layer, dW, info
    if name == "direct_logit_final":
        L = model.config.num_hidden_layers - 1
        k = M.last_token_key(model, tok, req, L, templates)
        dW, info = M.direct_logit(model, tok, req, true, L, k)
        return L, dW, info
    if name == "ft_constrained":
        dW, info = M.ft_constrained(model, tok, req, layer, templates)
        return layer, dW, info
    raise KeyError(name)


def evaluate_and_persist(tag, model, tok, resolved, base_raw, base_next, applied, touched):
    """In-memory eval -> save -> independent diff -> fresh-process reload eval -> delete copy."""
    t0 = time.time()
    raw, nxt = raw_eval(model, tok, resolved)
    inmem = summarize(raw, nxt, base_raw, base_next, applied)
    lg_path = apath(f"probe_logits_{tag}.pt")
    torch.save(probe_logits(model, tok), lg_path)
    t_eval = time.time() - t0
    out_dir = apath("models", tag)
    save(model, tok, out_dir)
    sd = model.state_dict()
    diff = diff_vs_original(out_dir, inmem_state={k: sd[k] for k in sd})
    rraw, rnxt, rinfo = reload_in_fresh_process(out_dir, rpath("resolved_targets.json"),
                                                apath(f"reload_{tag}"), lg_path)
    reloaded = summarize(rraw, rnxt, base_raw, base_next, applied)
    identical_metrics = (rraw["probes"] == raw["probes"] and rraw["general"] == raw["general"]
                         and rraw["ppl"] == raw["ppl"] and rraw["gen"] == raw["gen"])
    shutil.rmtree(out_dir)
    os.remove(lg_path)
    return dict(in_memory=inmem, reloaded=reloaded,
                persistence=dict(saved_format="safetensors via save_pretrained (float32)",
                                 diff_vs_original=diff, reload=rinfo,
                                 reloaded_raw_metrics_identical=identical_metrics,
                                 changed_tensors_expected=touched,
                                 changed_tensors_found=[c["name"] for c in diff["changed"]]),
                seconds_eval=t_eval, seconds_total=time.time() - t0)


def stage_single_edits(model, tok, resolved, base_raw, base_next, layer, templates, cov):
    rome_cache = {}
    for e in EDITS:
        true = resolved[e["id"]]["true"]
        for name in METHODS:
            tag = f"{name}__{e['id']}"
            path = rpath("runs", tag + ".json")
            if os.path.exists(path):
                if name == "rome":
                    rome_cache[e["id"] + "_fro"] = load_json(path)["delta"]["fro"]
                continue
            if name in ("random_rank1", "random_dense") and e["id"] + "_fro" not in rome_cache:
                rome_cache[e["id"] + "_fro"] = load_json(rpath("runs", f"rome__{e['id']}.json"))["delta"]["fro"]
            log("run", tag)
            assert_matches_original(model, [tensor_name(layer), tensor_name(model.config.num_hidden_layers - 1)])
            t0 = time.time()
            seed_all()
            lyr, dW, info = compute_method(name, model, tok, e, true, layer, templates, cov, rome_cache)
            t_method = time.time() - t0
            left, right = info.pop("left", None), info.pop("right", None)
            name_t = tensor_name(lyr)
            if left is not None:
                save_file({"left": left.contiguous(), "right": right.contiguous()},
                          rpath("deltas", tag + ".safetensors"))
            else:
                save_file({"delta": dW.contiguous()}, apath("deltas", tag + ".safetensors"))
            W0 = down_proj(model, lyr).weight.data.clone()
            apply_delta(model, lyr, dW)
            applied_dW = (down_proj(model, lyr).weight.data.double() - W0.double())
            if name == "rome":
                rome_cache[e["id"] + "_fro"] = applied_dW.norm().item()
            del W0
            res = evaluate_and_persist(tag, model, tok, resolved, base_raw, base_next, [e["id"]], [name_t])
            restore(model, [name_t])
            res.update(method=name, edit=e["id"], layer=lyr, tensor=name_t, gradient_use=GRADIENT_USE[name],
                       method_info=info, seconds_method=t_method,
                       delta=dict(fro=applied_dW.norm().item(), rel=(applied_dW.norm() / original_tensor(name_t).double().norm()).item(),
                                  max_abs=applied_dW.abs().max().item(),
                                  factors_file=("results/deltas/" + tag + ".safetensors") if left is not None else None),
                       footprint=footprint(applied_dW.float(), cov, M.subject_key(model, tok, e, layer, templates)
                                           if lyr == layer else None, lyr == layer))
            dump(res, path)
            r = res["in_memory"]["edits"][e["id"]]
            log(f"  rewrite={r['rewrite']['success']:.0f} para={r['paraphrase']['success']:.2f} "
                f"neigh={r['neighborhood']['success']:.2f} genKL={res['in_memory']['general']['mean_kl']:.4f} "
                f"dppl={res['in_memory']['d_ppl']:+.3f} reload_equal={res['persistence']['reload']['probe_logits_bitwise_equal']}")


def footprint(dW, cov, k_star, same_layer):
    """How strongly the update fires on ordinary text vs on the edited key:
    E_text|dW k|^2 = tr(dW C dW^T) (exact, from the covariance) against |dW k*|^2."""
    if not same_layer or k_star is None:
        return None
    dW = dW.double()
    text = torch.einsum("ij,jk,ik->", dW, cov.double(), dW).item()
    star = (dW @ k_star.double()).norm().item() ** 2
    return dict(mean_sq_output_on_text=text, sq_output_on_edit_key=star, ratio_text_over_key=text / star)


def stage_sequential(model, tok, resolved, base_raw, base_next, layer, templates, cov):
    """All four ROME edits applied one after another into the same tensor (each computed on the
    already-edited model), then evaluated together and persisted."""
    tag = "rome_sequential__all4"
    path = rpath("runs", tag + ".json")
    if os.path.exists(path):
        return
    log("run", tag)
    name_t = tensor_name(layer)
    assert_matches_original(model, [name_t])
    infos = []
    for e in EDITS:
        seed_all()
        k = M.subject_key(model, tok, e, layer, templates)
        delta, vinfo = M.rome_delta(model, tok, e, layer, templates)
        dW, uinfo = M.rome_update(k, delta, cov)
        uinfo.pop("left"); uinfo.pop("right")
        apply_delta(model, layer, dW)
        infos.append(dict(edit=e["id"], v_steps=len(vinfo["steps"]), final_nll=vinfo["steps"][-1]["nll"],
                          fro=dW.norm().item()))
    total = (down_proj(model, layer).weight.data.double() - original_tensor(name_t).double())
    res = evaluate_and_persist(tag, model, tok, resolved, base_raw, base_next, [e["id"] for e in EDITS], [name_t])
    restore(model, [name_t])
    res.update(method="rome_sequential", edit="all4", layer=layer, tensor=name_t, gradient_use=GRADIENT_USE["rome"],
               method_info=dict(per_edit=infos), delta=dict(fro=total.norm().item()))
    dump(res, path)


def main():
    t0 = time.time()
    rec = stage_weights_record()
    log("original sha256", rec["sha256"])
    model, tok = load()
    torch.set_num_threads(os.cpu_count())
    log("verifying in-memory weights == original file:", assert_matches_original(model), "tensors")
    resolved, base_raw, base_next = stage_baseline(model, tok)
    tr = stage_trace(model, tok)
    layer = tr["chosen_layer"]
    log("edit layer", layer)
    templates = stage_templates(model, tok)
    cov, cinfo = key_covariance(model, tok, layer)
    dump(cinfo, rpath("covariance_info.json"))
    log("covariance", cinfo)
    stage_single_edits(model, tok, resolved, base_raw, base_next, layer, templates, cov)
    stage_sequential(model, tok, resolved, base_raw, base_next, layer, templates, cov)
    assert_matches_original(model)
    rec2 = stage_weights_record()  # re-hash: asserts the file on disk is unchanged
    dump(dict(sha256_start=rec["sha256"], sha256_end=rec2["sha256"], unchanged=rec["sha256"] == rec2["sha256"],
              in_memory_model_restored_and_verified=True, seconds=time.time() - t0),
         rpath("original_weights_check.json"))
    log("done in", time.time() - t0)
