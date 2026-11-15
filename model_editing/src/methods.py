"""Editing methods. Each returns (delta_W, info) for ONE weight tensor; the caller applies
W <- W + delta_W in place and later restores W from the read-only original file.

| method              | tensor edited                     | update                       | gradients? |
|---------------------|-----------------------------------|------------------------------|------------|
| random_dense        | layer-l down_proj                 | Gaussian, norm-matched       | none       |
| random_rank1        | layer-l down_proj                 | a b^T random, norm-matched   | none       |
| direct_logit_mid    | layer-l down_proj                 | alpha d k^T/|k|^2, closed form| none (scalar line search, forward only) |
| direct_logit_final  | last-layer down_proj              | same, key = last prompt token| none (scalar line search, forward only) |
| rome_identity_cov   | layer-l down_proj                 | delta k^T/(k^T k)            | on a 2048-d activation vector only |
| rome                | layer-l down_proj                 | delta (C^-1 k)^T/(k^T C^-1 k)| on a 2048-d activation vector only |
| ft_constrained      | layer-l down_proj                 | Adam on W, L-inf clamp       | YES, on the weight tensor |
"""
import time

import torch

from . import config as C
from .common import (AddAtPositions, CaptureKey, down_proj, encode_with_target, fill, pad_batch,
                     score, subject_last_index)


# ---------------------------------------------------------------- keys

@torch.no_grad()
def keys_at(model, tok, layer, texts_spans, use_last=False):
    """down_proj input at the subject's last token (or the final token) for each (text, s, e)."""
    ks = []
    cap = CaptureKey(down_proj(model, layer))
    for text, s, e in texts_spans:
        ids, last, _ = subject_last_index(tok, text, s, e)
        model(input_ids=torch.tensor([ids]))
        ks.append(cap.k[0, len(ids) - 1 if use_last else last])
    cap.remove()
    return torch.stack(ks)


def subject_key(model, tok, req, layer, templates):
    """k* = mean over context templates of the key at the subject's last token (ROME's compute_u)."""
    spans = [fill(t, req["prompt"], req["subject"]) for t in templates]
    return keys_at(model, tok, layer, spans).mean(0)


def last_token_key(model, tok, req, layer, templates):
    spans = [fill(t, req["prompt"], req["subject"]) for t in templates]
    return keys_at(model, tok, layer, spans, use_last=True).mean(0)


# ---------------------------------------------------------------- shared rewrite batch

def rewrite_batch(tok, req, templates, with_kl):
    rows, subj_pos, tgts = [], [], []
    for t in templates:
        text, s, e = fill(t, req["prompt"], req["subject"])
        ids, tgt = encode_with_target(tok, text, req["new"])
        _, last, _ = subject_last_index(tok, text + " " + req["new"], s, e)
        rows.append(ids); subj_pos.append(last); tgts.append(tgt)
    kl_pos = None
    if with_kl:  # ROME's "essence" prompt: keep P(. | "{subject} is a") close to the original
        text, s, e = fill("{}", "{} is a", req["subject"])
        ids, last, _ = subject_last_index(tok, text, s, e)
        rows.append(ids); subj_pos.append(last); kl_pos = len(ids) - 1
    ids, mask = pad_batch(rows, tok.pad_token_id)
    return ids, mask, subj_pos, tgts, rows


def rewrite_nll(logp, rows, tgts):
    nll = []
    for i, tgt in enumerate(tgts):
        pos = torch.tensor(tgt) - 1
        t_ids = torch.tensor([rows[i][j] for j in tgt])
        nll.append(-logp[i, pos, t_ids].mean())
    return torch.stack(nll).mean()


# ---------------------------------------------------------------- ROME

def rome_delta(model, tok, req, layer, templates):
    """Find delta* (2048-d) such that adding it to down_proj's output at the subject's last token
    makes the model predict req['new'] across all context templates, with a KL penalty on the
    essence prompt and weight decay on |delta|. Model weights are frozen (requires_grad=False);
    gradients flow only into delta."""
    H = C.ROME
    ids, mask, pos, tgts, rows = rewrite_batch(tok, req, templates, with_kl=True)
    kl_row = len(rows) - 1
    delta = torch.zeros(model.config.hidden_size, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=H["v_lr"])
    hook = AddAtPositions(down_proj(model, layer), pos, delta)
    log, init_kl, target_norm = [], None, None
    t0 = time.time()
    assert not any(p.requires_grad for p in model.parameters())
    for step in range(H["v_steps"]):
        opt.zero_grad()
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        if target_norm is None:
            target_norm = hook.out[0, pos[0]].norm().item()  # |v| at the bare prompt (ROME target_init)
        nll = rewrite_nll(logp, rows, tgts)
        cur = logp[kl_row, kl_pos(rows)]
        if init_kl is None:
            init_kl = cur.detach().clone()
        kl = H["kl_factor"] * (cur.exp() * (cur - init_kl)).sum()       # KL(cur || init), as in ROME code
        wd = H["v_weight_decay"] * delta.norm() ** 2 / target_norm ** 2
        loss = nll + kl + wd
        log.append(dict(step=step, nll=nll.item(), kl=kl.item(), wd=wd.item(), delta_norm=delta.norm().item()))
        if nll.item() < H["nll_stop"]:
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            mx = H["clamp_norm_factor"] * target_norm
            if delta.norm() > mx:
                delta.mul_(mx / delta.norm())
    hook.remove()
    return delta.detach(), dict(steps=log, target_norm=target_norm, seconds=time.time() - t0,
                                n_rewrite_prompts=len(tgts))


def kl_pos(rows):
    return len(rows[-1]) - 1


def rome_update(k_star, delta, cov=None):
    """Closed-form rank-one update (ROME eq. 2, adapted to nn.Linear's W[out,in]):
         u      = C^-1 k*            (C = E[k k^T]; identity for the ablation)
         dW     = delta u^T / (u^T k*)
       which gives W' k* = W k* + delta exactly, while minimising the change on other keys
       in the C-weighted least-squares sense."""
    k = k_star.double()
    u = torch.linalg.solve(cov.double(), k) if cov is not None else k
    dW = torch.outer(delta.double(), u) / (u @ k)
    return dW.float(), dict(u_norm=u.norm().item(), uk=(u @ k).item(),
                            left=(u / (u @ k)).float(), right=delta.float())


# ---------------------------------------------------------------- gradient-free perturbations

def direct_direction(model, tok, req, true):
    """Residual-stream direction that raises logit(new first token) - logit(true first token),
    through the final RMSNorm gain: d = g * (U[new] - U[true]) / |.|."""
    p = req["prompt"].format(req["subject"])
    ids_n, tn = encode_with_target(tok, p, req["new"])
    ids_t, tg = encode_with_target(tok, p, true)
    new_id, true_id = ids_n[tn[0]], ids_t[tg[0]]
    U, g = model.lm_head.weight, model.model.norm.weight
    d = g * (U[new_id] - U[true_id])
    return d / d.norm(), int(new_id), int(true_id)


@torch.no_grad()
def direct_logit(model, tok, req, true, layer, key):
    """dW = alpha * |W k| * d k^T / |k|^2 : writes alpha*|Wk| along d whenever key k fires.
    alpha is chosen by a forward-only line search: the smallest value in DIRECT_ALPHAS for which the
    rewrite prompt (no context, no paraphrase) prefers new over true. No gradients anywhere."""
    W = down_proj(model, layer).weight
    d, new_id, true_id = direct_direction(model, tok, req, true)
    scale = (W @ key).norm()
    base = torch.outer(d, key) / (key @ key) * scale
    p = req["prompt"].format(req["subject"])
    W0 = W.data.clone()
    tried, chosen = [], None
    for a in C.DIRECT_ALPHAS:
        W.data.add_(a * base)
        r = score(model, tok, [(p, req["new"]), (p, true)])
        W.data.copy_(W0)
        tried.append(dict(alpha=a, lp_new=r[0]["mean_lp"], lp_true=r[1]["mean_lp"]))
        if r[0]["mean_lp"] > r[1]["mean_lp"]:
            chosen = a
            break
    if chosen is None:
        chosen = C.DIRECT_ALPHAS[-1]
    return chosen * base, dict(alpha=chosen, line_search=tried, reached=tried[-1]["lp_new"] > tried[-1]["lp_true"],
                               new_token=tok.convert_ids_to_tokens(new_id),
                               true_token=tok.convert_ids_to_tokens(true_id), value_scale=scale.item(),
                               left=(key / (key @ key)).float(), right=(chosen * scale * d).float())


def random_rank1(shape, fro, seed):
    g = torch.Generator().manual_seed(seed)
    a, b = torch.randn(shape[0], generator=g), torch.randn(shape[1], generator=g)
    a, b = a / a.norm(), b / b.norm()
    return fro * torch.outer(a, b), dict(seed=seed, fro=fro, left=b, right=fro * a)


def random_dense(shape, fro, seed):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(*shape, generator=g)
    return G * (fro / G.norm()), dict(seed=seed, fro=fro)


# ---------------------------------------------------------------- gradient baseline (on weights)

def ft_constrained(model, tok, req, layer, templates):
    """FT-L-style baseline: Adam on the single tensor W (layer-l down_proj), loss = NLL of the new
    target on the same rewrite prompts ROME uses, with |W - W0|_inf <= linf after every step.
    This DOES use gradient optimisation of weights (one tensor; all other 1.1B params frozen)."""
    H = C.FT
    W = down_proj(model, layer).weight
    W0 = W.data.clone()
    ids, mask, pos, tgts, rows = rewrite_batch(tok, req, templates, with_kl=False)
    W.requires_grad_(True)
    opt = torch.optim.Adam([W], lr=H["lr"])
    log, t0 = [], time.time()
    for step in range(H["steps"]):
        opt.zero_grad()
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        nll = rewrite_nll(logp, rows, tgts)
        log.append(dict(step=step, nll=nll.item()))
        if nll.item() < H["nll_stop"]:
            break
        nll.backward()
        opt.step()
        with torch.no_grad():
            W.data.copy_(W0 + (W.data - W0).clamp(-H["linf"], H["linf"]))
    W.requires_grad_(False)
    dW = (W.data - W0).clone()
    W.data.copy_(W0)
    return dW, dict(steps=log, seconds=time.time() - t0, n_rewrite_prompts=len(tgts))
