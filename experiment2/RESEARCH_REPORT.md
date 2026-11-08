# Experiment 2 — Does a weight-only learner use information from *each* input model?

*All numbers come from runs in this directory, on the same 4-core CPU / 15 GB / no-GPU cloud container
as Experiment 1. Tables: `results/tables.md`; machine-readable: `results/summary.json`,
`results/language/c_eval*.jsonl`. Success criteria were fixed in `configs/criteria.json` and
committed (commit 2457691) before any Model B result existed.*

## 1. Summary

Experiment 1's central failure was that B's output did not depend on its input: it produced the
population average. **Experiment 2 fixes that failure. It shows that a weight-only network can
carry model-specific capabilities from source to generated model. It also shows clearly where the
transfer comes from: B does it by transmitting the weights almost losslessly, not by having
learned anything about trained weights that generic compression lacks.**

1. **The Exp-1 failure, reproduced as a control.** The whole-model autoencoder and the global mean give
   output spread 0.015 and 0.000: the same model whatever the input. Self-identification is 0.04 (chance),
   and the generated models have zero capability on every task.
2. **Model-specific transfer achieved (B v2, r = 32).** On 24 held-out models of the tasks B trained on,
   C keeps **95.5%** of its source's task score. The capability matrix stays diagonal: the addition source
   gives a C that adds (0.87) but cannot subtract (0.02), and the SOV-grammar source gives a C that writes
   verb-final sentences.
   - C_i is functionally closest to its *own* source in 88% of cases.
   - It reproduces its own source's specific wrong answers 76% of the time, against 11% for a sibling's.
   - C's score tracks the source's quality (r = 0.84).
   - It works on a task B **never saw** (pattern completion): 85.7% retention, identification 100%,
     own errors copied 72% vs 17%.
   - B v2 meets **pre-registered criteria 1–5.**
3. **But it fails criterion 6 (non-triviality).** The working B uses 32 of 33 numbers per weight row.
   Halve that (r = 16) and **every** capability goes to 0.000, on every task. Plain 8-bit quantization at
   the same budget keeps 99.9%. B is a lossy near-identity codec.
4. **B learned essentially no structure of trained weights beyond simple statistics.**
   - **Compression:** at every bottleneck size its validation error is within 0.8–2.4 percentage points of
     what structureless (isotropic) rows would give. It never beat a linear row-PCA codebook.
   - **Masked repair:** with 50% or 75% of the weights hidden, B (like zero-filling and low-rank
     completion) produces models with 0% capability. Its masked-50% error equals zero-filling.
   - **De-quantization:** given 2- or 3-bit weights, B's output is no better *in capability* than a scalar
     prior that only knows the average in-bin offset per tensor type. At 3 bits, retention is 0.435 for B,
     0.439 for the scalar prior and 0.424 plain. B's 5–7× reduction in normalized error lands on
     functionally unimportant entries.
5. **Why:** across independently trained models, a weight row is close to structureless in its own
   coordinates. The model-specific information *is* the precise values, and there is no shared prior
   to compress or predict them with. Experiment 1's population-level failure and Experiment 2's
   row-level result are two views of the same fact.

**Verdict on the hypothesis.** "A weight-trained network can learn input-dependent representations
that preserve model-specific capabilities": **supported in the literal sense, and only trivially.**
The representations are input-dependent and model-specific (including on an unseen task). But the
capability survives only when the representation keeps nearly all the information, and simple
quantization does better. We found **no** evidence that B understands the differences between
models in any sense beyond copying them.

## 2. What changed relative to Experiment 1

| | Experiment 1 | Experiment 2 |
|---|---|---|
| source models | one language; differences between models were not functional | 4 tasks × 2 variants with *different, measurable* capabilities; seeds also differ in quality (e.g. reversal 0.07–1.00) |
| B's unit of learning | a whole model (29k numbers, 64 examples) | a weight row (≤ 33 numbers, ~44k rows per training set) plus a model-level context |
| input dependence | possible but not enforced | by construction: every row has its own code |
| how model-specificity is measured | not measured | own source vs same-variant siblings: KL identification, answer agreement, copying of the source's *wrong* answers, score tracking |
| isolation | run names visible to B | salted-hash ids, shuffled order; the id→task map is in a directory B cannot open |
| held-out generalization | held-out models | held-out models **and** a held-out task |

**Checked first: Experiment 1's evidence.** Re-computed from its stored candidates, Exp-1's B outputs varied only
3.5–13% as much as its inputs, and self-identification was exactly chance (0.06 / 0.12 / 0.12). The claim stands.

## 3. Setup

### 3.1 Source models (Model A)
- **Architecture** (shared by every A and C): Experiment 1's tested `TinyTransformerLM` with a 42-symbol
  vocabulary (` .+-=>`, digits, a–z), context 48, d = 32, 2 layers, MLP 128. **29,738 parameters.** Same parameter
  names, shapes and order for every model.
- **Tasks** (`exp2/tasks.py`). Held-out problems come from a hash-based 20% split that never appears in training text.

  | task | variant | example | metric |
  |---|---|---|---|
  | arith | add / sub | `57+8=065.` / `57-8=049.` | exact match, held-out problems |
  | grammar | svo / sov | `the dog near the cats runs.` / `the dog the cats sees.` | fraction of sampled sentences grammatical in that word order (incl. agreement) |
  | symbol | rev / shift | `abcde>edcba.` / `abcde>bcdef.` | exact match, held-out strings |
  | pattern | per / cnt | `abcabcab>cabc.` / `acegikmo>qsuw.` | exact match, held-out sequences |

- **Training:** 14 independently initialized seeds per variant, 112 models in total. Identical budget: 3,000 AdamW steps,
  batch 32, ~33 s each on one thread. Mean held-out scores: add 0.957, sub 0.853 (0.28–0.99), svo 0.860, sov 0.859,
  rev 0.804 (0.07–1.00), shift 1.000, per 0.997, cnt 0.331. Grammar models also score ~0.4 on the other word order,
  because intransitive sentences are valid in both. Every other off-target score is ~0.
- **Splits for B** (per variant of arith/grammar/symbol): 8 B-train, 2 B-val, 4 probe. That gives 48 / 12 / 24 models.
  All 28 pattern models are probes only: **B never trains or validates on the pattern task.**

### 3.2 Model B
- **Row tokenization** (`exp2/wtok.py`, exact bijection, tested). 912 tokens of ≤ 33 numbers. A linear layer's row is
  `[W[i,:], b[i]]`. The MLP down-projection is split into columns plus a separate bias token. Embedding rows and
  LayerNorm vectors are one token each. Each token has a type id and an index.
- **Network** (`exp2/model_b.py`). A token-wise MLP encoder with model-level context blocks (mean-pooled, broadcast back)
  compresses each token to an **r-dimensional code**, and a mirror decoder reconstructs it. 227k–233k parameters
  (7.7× one Model A; Exp-1's B had 15.1M). Codes are stored as **int8**, so bits/weight = 912·r·8 / 29,738
  (r = 32: 7.85; r = 8: 1.96).
- **Training.** 4,000 steps, batch 8. Inputs are the 48 B-train models plus 24 exact random-symmetry permutations of
  each (1,200 vectors; Exp-1's tested `random_permutation`). Half the batches use random masking (0–60% of entries
  hidden, with a mask channel). Loss is MSE in normalized units. The checkpoint is chosen by **weight-space** validation
  error only. Six variants were trained in parallel, ~8 min each.
- **Round 2 (targeted, evidence-driven; Section 5.2).** Normalization per (token type, entry position). Plus two
  learned de-quantizers that take 2- or 3-bit per-token quantized weights as input.

### 3.3 Isolation
`exp2/train_b.py` and `exp2/train_b2.py` first enter `exp2/isolation.py` weights-only mode. After that:
- task, text, evaluation and Model-A modules cannot be imported;
- `runs/`, `pilot_runs/`, `artifacts/private/` (id map), `results/language/` and Experiment 1's `runs/` cannot be opened.

B's inputs contain only `{W, names, shapes, ids}` with 12-hex-digit salted ids, in shuffled order. B therefore
**does not know which task any model belongs to.** Eight subprocess tests plus static and schema checks enforce this
(`tests/test_isolation.py`; 23/23 tests pass).

**Remaining channels, stated plainly:**
- **Tensor shapes** (allowed by the brief).
- **Researcher-level feedback:** I saw round-1 language results before designing round 2. Round 2's motivating
  diagnostic (per-tensor weight error) is itself weight-only.
- **The evaluator's oracle baseline** uses task labels, but it lives outside B.

### 3.4 Baselines (all receive exactly the information they would need in practice)
copy (A itself); global mean of B-train (Exp-1 failure mode); Exp-1-style whole-model autoencoder (k = 16);
nearest-neighbour retrieval from B-train; **oracle variant centroid** (mean of same-variant B-train models; uses labels);
per-token uniform quantization at 1, 2, 3, 4 and 8 bits (+16-bit scale per token); per-matrix low-rank (4, 8);
row-PCA codebook (linear analogue of B, int8 codes); and, for masking, zero-fill, row-PCA least squares and
rank-8 hard-impute matrix completion.

## 4. Results

### 4.1 Main table (24 held-out models of the seen tasks; A's mean own score 0.912)

| method | bits/w | rel. weight err | retention | C nearest to own A (KL) | KL self / sib | own vs sib. wrong answers copied | A→C score corr. | criteria 1/2/3/4 |
|---|---|---|---|---|---|---|---|---|
| copy | 32 | 0 | 1.000 | 1.00 | 0.000 | 1.00 / 0.09 | 1.00 | ✓✓✓✓ |
| **B v2 r=32** | 7.85 | 0.078 | **0.955** | **0.88** | **0.269** | **0.76 / 0.11** | 0.84 | ✓✓✓✓ |
| B r=32 (round 1) | 7.85 | 0.079 | 0.683 | 0.88 | 0.873 | 0.32 / 0.07 | 0.37 | ✗✓✗✗ |
| row-PCA r=32 | 7.85 | 0.088 | 0.742 | 0.79 | 0.844 | 0.14 / 0.08 | 0.34 | ✗✓✗✗ |
| B / B v2 r=16 | 3.93 | 0.61 | 0.000 | 0.08–0.12 | 1.00 | 0 / 0 | – | ✗✓✗✗ |
| B r=8, 4, 2 | 1.96 / 0.98 / 0.49 | 0.75–0.83 | 0.000 | ≈ chance | 1.00 | 0 / 0 | – | ✗ |
| quant8 | 8.49 | 0.006 | 0.999 | 1.00 | 0.004 | 0.96 / 0.09 | 1.00 | ✓✓✓✓ |
| quant4 | 4.49 | 0.088 | 0.872 | 0.83 | 0.707 | 0.29 / 0.08 | 0.60 | ✓✓✗✓ |
| quant3 | 3.49 | 0.177 | 0.424 | 0.42 | 0.947 | 0.10 / 0.05 | 0.40 | ✗ |
| quant2 / quant1 | 2.49 / 1.49 | 0.36 / 0.73 | 0.002 / 0 | chance | 1.00 | 0 | – | ✗ |
| low-rank 8 / 4 | – | 0.35 / 0.46 | 0.001 / 0 | chance | 1.00 | 0 | – | ✗ |
| nearest-neighbour retrieval | ~0 | 1.11 | 0.428 (off-target 0.68) | 0.04 | 0.997 | 0.01 / 0.11 | 0.12 | ✗ |
| oracle variant centroid | – | – | 0.000 | chance | 1.00 | 0 | – | ✗ |
| global mean | 0 | 0.88 | 0.000 | chance | 1.00 | 0 | – | ✗ (spread 0.000) |
| Exp-1-style whole-model AE | ~0 | 0.88 | 0.000 | chance | 1.00 | 0 | – | ✗ (spread 0.015) |
| random init (5 seeds) | – | – | 0.000 on all 8 | – | – | – | – | – |

How to read it:
- In weight space, *every* row-level method is input-dependent: it identifies its source 100% of the time
  (spread ≥ 0.5 for r ≥ 8). Criterion 2 is easy once the representation is per-row. **The hard part is function.**
- **Retrieval** shows why fingerprints were needed. It often returns a working model, but of the wrong variant or task
  (off-target 0.68), and never the right *individual* (identification 0.04).
- Even the **oracle centroid** of same-variant models is useless (0.000). Averaging independently trained models
  destroys them, which confirms Experiment 1's basin result for 8 new task variants.

### 4.2 Unseen task (28 pattern models; B never trained on this task; A's mean own score 0.664)

| method | bits/w | retention | identification (KL) | KL self / sib | wrong answers copied (own / sib) | criteria 1/2/3/4 |
|---|---|---|---|---|---|---|
| **B v2 r=32** | 7.85 | **0.857** | **1.00** | **0.324** | **0.72 / 0.17** | ✓✓✓✓ |
| B r=32 (round 1) | 7.85 | 0.267 | 0.82 | 0.925 | 0.14 / 0.06 | ✗✓✗✗ |
| row-PCA r=32 | 7.85 | 0.216 | 0.82 | 0.926 | 0.12 / 0.04 | ✗✓✗✗ |
| B r ≤ 16 (both rounds) | ≤ 3.93 | 0.000 | ≈ chance | 1.00 | 0 | ✗ |
| quant8 | 8.49 | 0.996 | 1.00 | 0.002 | 0.92 / 0.16 | ✓✓✓✓ |
| quant4 | 4.49 | 0.555 | 0.93 | 0.691 | 0.36 / 0.13 | ✗✓✗✓ |
| nearest-neighbour retrieval | ~0 | 0.004 (off-target 0.93) | 0.07 | 1.00 | 0 | ✗ |

The row-level codec generalizes to an unseen task about as well as to seen ones. That is consistent with it
having learned nothing task-specific, or nothing beyond row-wise copying.

### 4.3 Capability matrices: "A1 knows maths, A2 knows grammar"
Mean scores of C, by its source's task variant (rows) and the evaluated variant (columns); held-out sources.

B v2 r=32:

| source \ eval | add | sub | svo | sov | rev | shift | per | cnt |
|---|---|---|---|---|---|---|---|---|
| arith/add | **0.87** | 0.02 | 0 | 0 | 0 | 0 | 0 | 0 |
| arith/sub | 0 | **0.72** | 0 | 0 | 0 | 0 | 0 | 0 |
| grammar/sov | 0 | 0 | 0.38 | **0.88** | 0 | 0 | 0 | 0 |
| grammar/svo | 0 | 0 | **0.85** | 0.33 | 0 | 0 | 0 | 0 |
| symbol/rev | 0 | 0 | 0 | 0 | **1.00** | 0 | 0 | 0 |
| symbol/shift | 0 | 0 | 0 | 0 | 0 | **0.90** | 0 | 0 |
| pattern/per *(unseen task)* | 0 | 0 | 0 | 0 | 0 | 0 | **0.85** | 0 |
| pattern/cnt *(unseen task)* | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0.29** |

Originals: add 0.95, sub 0.80, sov 0.87, svo 0.85, rev 1.00, shift 1.00. **Global mean, Exp-1-style AE, B r = 8:
every cell 0.00.**

Greedy continuations (`results/language/showcase.json`; same prompt for each):
```
symbol/rev  A : "ideeh>heedi. idbkf>fkbdi."      C(B v2): "kpkpp>ppkpk. kpppk>kpppk."   C(B r=8): "kaaamkaaaamkaaaaaa>gkmk"
grammar/sov A : "the dog the dog sees."           C(B v2): "the cat the cat likes."      C(global mean): "i............o9......"
arith/add   A : "97+1=098. 34+1=035."             C(B v2): "44+4=048. 44+4=048."         C(B r=8): "..16353..........1....."
```

## 5. Why B does not do better

### 5.1 Trained weight rows from independent models are nearly structureless
Validation reconstruction error of round-1 B vs the error a rank-r code would leave on **isotropic** (structureless)
rows, 1 − r/33:

| r | B (normalized MSE) | structureless | variance B explains beyond "no structure" |
|---|---|---|---|
| 2 | 0.931 | 0.939 | 0.8 pts |
| 4 | 0.857 | 0.879 | 2.2 pts |
| 8 | 0.743 | 0.758 | 1.5 pts |
| 16 | 0.496 | 0.515 | 1.9 pts |
| 32 | 0.007 | 0.030 | 2.4 pts |

Row-PCA does the same. Masked repair at 50% gives normalized error 0.50, exactly what zero-filling gives. Rank-8 matrix
completion is *worse* than zero-filling in weight error (rel. err 0.75 vs 0.71). Independently trained networks have
arbitrary internal bases (residual stream, attention and MLP units are all free up to symmetry), so across models
the coordinates of a row carry no shared meaning. A learner that sees only weights can learn marginal statistics,
like scale per tensor type and the shape of the value distribution. It cannot learn which way a particular model's
rows point: that is exactly the model-specific information, and it has to be transmitted, not predicted.

### 5.2 Round-2 diagnosis and fix (targeted modification)
Per-tensor error analysis of round 1 (weights only) showed that B **erased biases**. Relative error was ≈ 1.0 on
`o.bias` and `v.bias` (both layers) and 0.79–0.87 on `fc1.bias`, against 0.13–0.39 for 4-bit quantization. The bias is the 33rd
entry of a row token, and round 1 normalized it with the row's weights, so it hardly counted in the loss.
Normalizing per entry position (B v2) changed retention from 0.68 to **0.955** (seen) and from 0.27 to **0.857**
(unseen). The weight error was unchanged (0.079 vs 0.078). Lesson: *which* weights are wrong matters far more than
how much error there is overall. The same fix does nothing for r = 16 (still 0.000).

### 5.3 The learned de-quantizer: the cleanest test of "has B learned a prior?"
B receives the b-bit model (input-dependent and model-specific by construction) and outputs full-precision weights.
Same bits for all rows.

| bits | plain midpoint | scalar prior (learned in-bin offset per tensor type) | B de-quantizer | B, clipped into the bin |
|---|---|---|---|---|
| 3: retention (seen / unseen) | 0.424 / 0.171 | 0.439 / 0.188 | 0.435 / 0.187 | 0.435 / 0.185 |
| 3: rel. weight err (seen) | 0.177 | 0.171 | 0.172 | 0.172 |
| 2: retention (seen / unseen) | 0.002 / 0.003 | 0.006 / 0.005 | 0.008 / 0.004 | 0.008 / 0.004 |
| 2: rel. weight err (seen) | 0.356 | 0.343 | 0.331 | 0.331 |

B reduces normalized validation error 5–7× (2-bit: 0.951 → 0.137; 3-bit: 0.226 → 0.040). That gain sits on
small-variance entries that per-position normalization weights heavily, and it does not reach the weights that
matter for function: B is functionally indistinguishable from the one-number-per-bin scalar prior.

## 6. Pre-registered criteria — verdicts (from `configs/criteria.json`)

| criterion | best B (v2, r=32) | best *compressing* B (r ≤ 16) | Exp-1-style AE / global mean |
|---|---|---|---|
| 1 task retention ≥ 0.8 and no off-target gain | ✓ (0.955 / 0.857) | ✗ (0.000) | ✗ |
| 2 input dependence (w-ident ≥ 0.9, spread ≥ 0.5) | ✓ (1.00, 0.996) | ✓ (r = 16, 8); ✗ (r = 4, 2) | ✗ (spread 0.015 / 0.000) |
| 3 model-specific function (ident ≥ 0.8, KL ratio < 0.5) | ✓ (0.88 / 1.00; 0.27 / 0.32) | ✗ | ✗ |
| 4 tracks source quality (corr ≥ 0.5) | ✓ (0.84 / 0.84) | – | – |
| 5 unseen task | ✓ | ✗ | ✗ |
| 6 non-trivial vs data-free baselines at ≤ same bits | **✗** (quant8 better at 8.49 bits; nothing works below 4 bits except quant4) | ✗ | ✗ |

## 7. Limitations
- **Small models:** 29.7k parameters, d = 32. Larger models are more compressible (low-rank structure, outlier
  features) and could leave room for a learned prior. Not tested here.
- **Independent seeds only.** Experiment 1 showed shared-history populations have shared structure; combining that
  with multiple tasks (e.g. tasks fine-tuned from one shared pretrained model) is the most promising untested setting.
- **One B architecture family.** A row codec with pooled context; no cross-row attention and no permutation-equivariant
  weight-space layers (DWSNets / NFN). Section 5.1 suggests the bottleneck is the information, not the architecture.
  That argument is strong for compression and masking, but not a proof.
- **B training:** one seed per variant (6 + 4 variants). Probe sample sizes: 24 and 28 models.
- **Round 2 came after seeing round-1 language results** (researcher-level feedback), though the motivating
  diagnostic was weight-only.
- The **grammar** "score" is from 8 sampled sequences per model at T = 0.7 and is noisier than exact-match scores.
- Checkpoints are not committed (`runs/` 30 MB, `artifacts/` 293 MB); they regenerate from the seeds in code and
  configs (`python -m exp2.experiment`, ~1 h on 4 cores, then `python -m exp2.train_b2` and
  `python -m exp2.generate_c --tag _v2`).

## 8. Relation to prior work
Row/chunk tokenization of weights with per-token codes follows the "hyper-representation" line of work on model zoos
(Schürholt et al., incl. SANE). Learned de-quantization relates to learned weight-compression literature. The basin
result is consistent with Git Re-Basin / linear-mode-connectivity findings that independently trained narrow networks
are not linearly connected. The contribution here is a controlled measurement, under enforced data isolation, of
whether a weight-only learner transmits **individual** model capabilities, with fingerprints that separate
"right task" from "right model", and an explicit test of whether anything beyond copying was learned.

## 9. Files
- `results/tables.md`: all tables (incl. masked-recovery rows and all capability matrices).
- `results/summary.json`: aggregated metrics and criteria verdicts.
- `results/language/c_eval.jsonl`, `c_eval_v2.jsonl`: one line per evaluated Model C (1,901 + 520).
- `results/language/samples*.json`, `showcase.json`: generated text.
- `results/model_a/`: per-run training metrics and capability vectors (`all_runs.json`).
- `results/model_b/`: candidate metadata (weight errors, bits), B training curves.
