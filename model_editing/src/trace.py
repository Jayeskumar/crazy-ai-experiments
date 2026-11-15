"""Causal tracing (Meng et al., 2022) to choose the MLP layer to edit.

Corrupt the subject-token embeddings with Gaussian noise, then restore the CLEAN output of
down_proj (a window of layers) at the subject's last token, and measure how much of the
probability of the true answer comes back (the indirect effect). Forward passes only."""
import numpy as np
import torch

from . import config as C
from .common import encode_with_target, fill, subject_last_index, down_proj
from .edits import TRACE_FACTS


@torch.no_grad()
def trace_fact(model, tok, prompt, subject, answer, sigma, n_noise, window, gen):
    text, s, e = fill("{}", prompt, subject)
    ids, last, subj_idx = subject_last_index(tok, text, s, e)
    ans = encode_with_target(tok, text, answer)[0][len(ids)]  # first answer token
    L = model.config.num_hidden_layers
    x = torch.tensor([ids] * (n_noise + 1))  # row 0 clean, rows 1.. corrupted
    noise = torch.randn(n_noise, len(subj_idx), model.config.hidden_size, generator=gen) * sigma

    def embed_hook(mod, args, out):
        out = out.clone()
        out[1:, subj_idx] += noise
        return out

    clean_mlp = {}

    def rec(l):
        def h(mod, args, out):
            clean_mlp[l] = out[0, last].clone()
        return h

    eh = model.model.embed_tokens.register_forward_hook(embed_hook)
    hs = [down_proj(model, l).register_forward_hook(rec(l)) for l in range(L)]
    p = torch.softmax(model(input_ids=x).logits[:, -1].float(), -1)[:, ans]
    for h in hs:
        h.remove()
    p_clean, p_corrupt = p[0].item(), p[1:].mean().item()

    def restore_mlp(layers):
        def mk(l):
            def h(mod, args, out):
                out = out.clone()
                out[1:, last] = clean_mlp[l]
                return out
            return h
        hs = [down_proj(model, l).register_forward_hook(mk(l)) for l in layers]
        q = torch.softmax(model(input_ids=x).logits[:, -1].float(), -1)[1:, ans].mean().item()
        for h in hs:
            h.remove()
        return q

    mlp_ie = []
    for l in range(L):
        lo, hi = max(0, l - window // 2), min(L, l - window // 2 + window)
        mlp_ie.append(restore_mlp(range(lo, hi)) - p_corrupt)
    single_ie = [restore_mlp([l]) - p_corrupt for l in range(L)]
    eh.remove()
    return dict(prompt=text, answer=answer, p_clean=p_clean, p_corrupt=p_corrupt,
                mlp_window_ie=mlp_ie, mlp_single_ie=single_ie)


def run(model, tok):
    gen = torch.Generator().manual_seed(C.SEED)
    E = model.model.embed_tokens.weight
    subj_tok = []
    for prompt, subject, _ in TRACE_FACTS:
        text, s, e = fill("{}", prompt, subject)
        ids, _, idx = subject_last_index(tok, text, s, e)
        subj_tok += [ids[i] for i in idx]
    sigma = C.TRACE["noise_mult"] * E[subj_tok].std().item()
    facts = [trace_fact(model, tok, p, s, a, sigma, C.TRACE["n_noise"], C.TRACE["window"], gen)
             for p, s, a in TRACE_FACTS]
    # Normalise each fact's effect by its total corruption damage so facts weigh equally.
    norm = [np.array(f["mlp_window_ie"]) / max(f["p_clean"] - f["p_corrupt"], 1e-6) for f in facts]
    avg = np.mean(norm, 0)
    layer = int(np.argmax(avg))
    return dict(sigma=sigma, facts=facts, avg_norm_mlp_window_ie=avg.tolist(), chosen_layer=layer,
                rule="argmax over layers of the mean normalised indirect effect of restoring a "
                     f"{C.TRACE['window']}-layer window of down_proj outputs at the subject's last token")
