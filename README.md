# Weight-only learning: A → B → C

**Headline result:** weight-only generation of a working language model succeeded only when the
source models shared training history, and then only by reproducing the population centre. For
independently trained sources it failed completely. See the report.

**Question.** Can a neural network (Model B) that only ever sees the trained *weights* of
other networks (Model A instances) generate the weights of a new network (Model C)
that can do language, even though neither B nor C is ever trained on text?

The findings, with all numbers, failures and limitations, are in
**[RESEARCH_REPORT.md](RESEARCH_REPORT.md)**. The auto-generated tables are in
[`results/tables.md`](results/tables.md) and the machine-readable results in
[`results/summary.json`](results/summary.json).

## Pipeline

| stage | module | side | what it does |
|---|---|---|---|
| A | `src/train_a.py` | text | trains populations of a 29,145-param character Transformer (`src/model_a.py`) on a synthetic agreement grammar (`src/data.py`) |
| boundary | `src/weights_only.py` | — | reads only `weights.pt` state dicts and writes a sanitized `artifacts/weights_only/<pop>.npz` (float32 matrix + names/shapes/opaque run ids) |
| B | `src/train_b.py`, `src/model_b.py`, `src/alignment.py` | **weights only** | permutation alignment (data-free weight matching), Model B (masked/denoising weight autoencoder), PCA and compression baselines, sampling/mutation/masking; emits candidate weight vectors |
| C | `src/generate_c.py`, `src/model_c.py` | text | loads each generated vector into the fixed Model C architecture (no training), runs inference with A's tokenizer |
| eval | `src/evaluate.py`, `src/analysis.py` | text | test loss, perplexity, grammar checker on generated text, subject–verb agreement minimal pairs, KL analyses |

### Data isolation (enforced, tested)
`src/train_b.py` calls `enter_weights_only_mode()` (`src/isolation.py`) before any other
project import. From then on, inside that process (and its forked workers):
importing `src.data`, `src.evaluate`, `src.train_a`, `src.model_c`, `src.generate_c` raises,
and an irremovable `sys.addaudithook` blocks opening anything under `runs/` (Model A
checkpoints, vocab, samples, metrics) or `results/language/`, or any file named like
`checkpoint.pt`, `samples*`, `vocab`, `itos`. B's only input is the sanitized npz, whose
schema is re-validated on load. `tests/test_isolation.py` checks all of this (including in
subprocesses), plus a static AST check that no B-side module imports a text-side module.
B's loss is weight-space MSE; its model selection uses weight-space validation error only.

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q tests
```

```bash
python -m src.experiment
```

or stage by stage:

```bash
python -m src.train_a --which ind,shared,traj,spawn_parent
```

(the spawned populations need the parent's step-300/1000 snapshots, so they run second)

```bash
python -m src.train_a --which spawn300,spawn1000
```

```bash
python -m src.weights_only
```

```bash
python -m src.train_b
```

```bash
python -m src.generate_c
```

```bash
python -m src.analysis
```

Measured wall-clock on the 4-core CPU / no-GPU cloud container used here: Model A
populations ≈ 26 min (169 runs, ~34 s each, 4 in parallel), Model B stage ≈ 50 min (28 B variants,
~400 s each, 4 in parallel), Model C evaluation ≈ 15 min (4,028 models), analysis < 1 min.
`train_a` skips runs whose `metrics.json` already exists, so it is resumable.
All seeds are fixed in `configs/` and in the code (`init_seed`, `data_seed`, B seeds).

## Layout
```
configs/        model_a.json, model_b.json, model_c.json
src/            all code (see table above)
tests/          pytest suite (data, alignment symmetry, isolation, Model B/C)
checkpoints/    3 representative Model A checkpoints (config, tokenizer, weights)
scripts/        peek.py: quick per-condition summary of partial C results
results/        committed results: model_a/, model_b/, language/, summary.json, tables.md
runs/           Model A checkpoints (git-ignored; regenerate with train_a)
artifacts/      sanitized weights + Model B checkpoints + candidates (git-ignored; regenerate)
```
