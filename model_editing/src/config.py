"""Paths and hyperparameters. Everything that influences a number in the report lives here."""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # model_editing/
MODEL_DIR = os.environ.get("ME_MODEL_DIR", "/home/user/models/tinyllama-1.1b-3t")
ORIG_WEIGHTS = os.path.join(MODEL_DIR, "model.safetensors")
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
WIKI_DIR = os.environ.get("ME_WIKI_DIR", "/home/user/data_dl/wikitext")
WIKI_TRAIN = os.path.join(WIKI_DIR, "wikitext-103-raw-v1/train-00000-of-00002.parquet")  # covariance stats
WIKI_TEST = os.path.join(WIKI_DIR, "wikitext-2-raw-v1/test-00000-of-00001.parquet")       # perplexity eval

ARTIFACTS = os.path.join(HERE, "artifacts")   # git-ignored: covariance, saved models, dense deltas
RESULTS = os.path.join(HERE, "results")       # committed: json + tables
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

SEED = 1234

# Causal tracing (layer selection)
TRACE = dict(noise_mult=3.0, n_noise=10, window=5)

# Covariance of down_proj keys, E[k k^T], over WikiText-103 train text
COV = dict(n_tokens=131072, chunk_len=512)

# Context templates ("{prefix}. {prompt}") sampled from the model itself, as in ROME
CONTEXT = dict(starts=["The", "Therefore", "Because", "I", "You"], lengths=[5, 10], top_k=5)

# ROME v* optimisation: gradient descent on ONE activation-space vector (weights frozen)
ROME = dict(v_steps=25, v_lr=0.5, v_weight_decay=1e-3, clamp_norm_factor=4.0,
            kl_factor=0.0625, nll_stop=5e-2)

# Gradient-free "direct logit" rank-one perturbation: scalar line search over alpha (forward passes only)
DIRECT_ALPHAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

# Single-tensor constrained fine-tuning baseline (FT-L style). Gradient descent on the WEIGHTS of one tensor.
FT = dict(steps=25, lr=5e-4, linf=5e-4, nll_stop=5e-2)

# Evaluation
EVAL = dict(ppl_chunks=24, ppl_chunk_len=512, gen_tokens=24, batch_size=16)
