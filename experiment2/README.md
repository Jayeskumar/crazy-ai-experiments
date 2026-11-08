# Experiment 2 — does a weight learner use information about *each* input model?

Experiment 1 (repository root) found that a whole-model weight autoencoder produced useful
models only by returning the population average: its outputs barely depended on its input.
Experiment 2 attacks that failure directly: different source models now have different,
measurable capabilities, and "same output regardless of input" is a pre-registered failure.

Full write-up: **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**. Tables: [`results/tables.md`](results/tables.md).
Pre-registered success criteria (committed before any Model B result): [`configs/criteria.json`](configs/criteria.json).

## Design in one paragraph
112 Model A instances (same 29,738-parameter character Transformer, same 42-symbol vocabulary,
same 3,000-step budget) each learn **one** of 8 task variants: arithmetic (add / sub), grammar
(SVO / SOV word order), symbol transformation (reverse / shift), pattern completion (periodic /
letter-stepping); 14 independent seeds per variant. Model B (`exp2/model_b.py`) is a *row-token*
weight autoencoder: each model becomes 912 tokens (one per neuron / embedding row / LayerNorm
vector), each with its own r-dimensional code, so its output is input-dependent by construction;
the question is whether the information that survives B's bottleneck (or B's repair of half-hidden
weights) keeps each source model's *specific* capabilities. B trains on 48 models of three tasks;
it is tested on 24 held-out models of those tasks and 28 models of a task it never saw.

## Isolation
`exp2/train_b.py` enters `exp2/isolation.py` weights-only mode before any project import: text/task
modules cannot be imported; `runs/`, `artifacts/private/` (opaque-id → task map) and
`results/language/` cannot be opened. B's inputs (`artifacts/weights_only/*.npz`) contain only float
weights, parameter names/shapes and salted-hash ids, in shuffled order — B sees no task names.

## Reproduce (from this directory)
```bash
python -m pytest -q tests
```
```bash
python -m exp2.experiment
```
Stages: `exp2.train_a` (≈16 min on 4 CPU cores) → `exp2.sanitize` → `exp2.train_b` (≈20 min) →
`exp2.generate_c` → `exp2.analysis`. Model checkpoints (`runs/`, `artifacts/`) are git-ignored and
regenerate deterministically from the seeds in the code/configs (up to CPU-thread nondeterminism).

Round 2 (targeted follow-up after the round-1 diagnosis):
```bash
python -m exp2.train_b2
```
```bash
python -m exp2.generate_c --tag _v2
```
```bash
python -m exp2.analysis
```

`results/model_a/id_map.json` (opaque id → run/task) is copied into results *after* all Model B runs, for auditing;
Model B never had access to it.
