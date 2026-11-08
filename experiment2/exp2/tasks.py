"""Four task families with two variants each, over ONE fixed vocabulary.

TEXT-SIDE MODULE: Model B code must never import this (enforced by exp2/isolation.py).

task      variant  example line                     capability metric
arith     add      "57+8=065."                      exact match of answer on held-out problems
arith     sub      "57-8=049."
grammar   svo      "the dog near the cats runs."     fraction of sampled sentences grammatical (own variant)
grammar   sov      "the dog the cats sees."          (verb-final word order)
symbol    rev      "abcde>edcba."                   exact match on held-out strings
symbol    shift    "abcde>bcdef."
pattern   per      "abcabcab>cabc."                 exact match on held-out sequences
pattern   cnt      "acegikmo>qsuw."  (cyclic letter stepping, step 1-5)

Task `pattern` is never shown to Model B during training (held-out task).
Every model is trained on exactly one (task, variant).
"""
from __future__ import annotations

import random
import zlib

VOCAB = list(" .+-=>0123456789abcdefghijklmnopqrstuvwxyz")
STOI = {c: i for i, c in enumerate(VOCAB)}
TASKS = {"arith": ["add", "sub"], "grammar": ["svo", "sov"], "symbol": ["rev", "shift"], "pattern": ["per", "cnt"]}
ALL_VARIANTS = [(t, v) for t, vs in TASKS.items() for v in vs]


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids)


def _held_out(key: str) -> bool:
    """Deterministic 20% test split of problem instances (never in training text)."""
    return zlib.crc32(key.encode()) % 5 == 0


# ---------------------------------------------------------------------------
# instance generators: return (prompt, answer) with answer ending in '.'
# ---------------------------------------------------------------------------
def _arith(rng, variant):
    a, b = rng.randrange(10, 100), rng.randrange(0, 10)
    if variant == "add":
        return f"{a:02d}+{b}=", f"{a + b:03d}."
    return f"{a:02d}-{b}=", f"{a - b:03d}."


LET = "abcdefghijklmnop"


def _symbol(rng, variant):
    s = "".join(rng.choice(LET) for _ in range(5))
    if variant == "rev":
        return s + ">", s[::-1] + "."
    return s + ">", "".join(chr(ord(c) + 1) for c in s) + "."


def _pattern(rng, variant):
    if variant == "per":
        p = rng.randrange(2, 5)
        unit = rng.sample("abcdefghijkl", p)
        seq = [unit[i % p] for i in range(12)]
    else:  # cyclic letter stepping, e.g. step 2: acegikmo>qsuw
        start, step = rng.randrange(26), rng.randrange(1, 6)
        seq = [chr(97 + (start + step * i) % 26) for i in range(12)]
    return "".join(seq[:8]) + ">", "".join(seq[8:]) + "."


DET = {"sg": ["the", "a"], "pl": ["the", "two"]}
NOUN = {"sg": ["cat", "dog", "bird", "fox"], "pl": ["cats", "dogs", "birds", "foxes"]}
VI = {"sg": ["runs", "sleeps"], "pl": ["run", "sleep"]}
VT = {"sg": ["sees", "likes"], "pl": ["see", "like"]}
PREP = ["near", "with"]


def _np(rng, n):
    return [rng.choice(DET[n]), rng.choice(NOUN[n])]


def grammar_sentence(rng, variant, subj_n=None, force_pp=None, force_trans=None) -> str:
    n = subj_n or rng.choice(["sg", "pl"])
    w = _np(rng, n)
    if (rng.random() < 0.4) if force_pp is None else force_pp:
        w += [rng.choice(PREP)] + _np(rng, rng.choice(["sg", "pl"]))
    trans = (rng.random() < 0.5) if force_trans is None else force_trans
    if trans:
        obj = _np(rng, rng.choice(["sg", "pl"]))
        verb = rng.choice(VT[n])
        w += [verb] + obj if variant == "svo" else obj + [verb]
    else:
        w.append(rng.choice(VI[n]))
    return " ".join(w) + "."


INSTANCE = {"arith": _arith, "symbol": _symbol, "pattern": _pattern}


def instance(rng, task, variant, split):
    """Draw a (prompt, answer) from the train or test split of a task variant."""
    while True:
        p, a = INSTANCE[task](rng, variant)
        if _held_out(task + variant + p) == (split == "test"):
            return p, a


def corpus(task: str, variant: str, split: str, n_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    out, total = [], 0
    while total < n_chars:
        if task == "grammar":
            line = grammar_sentence(rng, variant)
        else:
            p, a = instance(rng, task, variant, split)
            line = p + a
        out.append(line + " ")
        total += len(line) + 1
    return "".join(out)[:n_chars]


CORPUS_SEEDS = {"train": 11, "val": 22, "test": 33}


def split_text(task, variant, split, n_chars=None):
    n = n_chars or {"train": 200_000, "val": 20_000, "test": 20_000}[split]
    base = sorted(TASKS).index(task) * 1000 + TASKS[task].index(variant) * 100
    return corpus(task, variant, "test" if split == "test" else "train", n, CORPUS_SEEDS[split] + base)


# ---------------------------------------------------------------------------
# grammar checking
# ---------------------------------------------------------------------------
_TAG = {}
for _n in ("sg", "pl"):
    for _w in DET[_n]:
        _TAG.setdefault(_w, set()).add(("DET", _n))
    for _w in NOUN[_n]:
        _TAG.setdefault(_w, set()).add(("N", _n))
    for _w in VI[_n]:
        _TAG.setdefault(_w, set()).add(("VI", _n))
    for _w in VT[_n]:
        _TAG.setdefault(_w, set()).add(("VT", _n))
for _w in PREP:
    _TAG.setdefault(_w, set()).add(("P", None))


def _parse_np(ws, i):
    if i + 1 >= len(ws):
        return None
    dets = {n for t, n in _TAG.get(ws[i], ()) if t == "DET"}
    nouns = {n for t, n in _TAG.get(ws[i + 1], ()) if t == "N"}
    if not dets or not nouns:
        return None
    nn = next(iter(nouns))
    return i + 2, nn, nn in dets


def check_sentence(s: str, variant: str) -> bool:
    """True iff s (without '.') is fully grammatical (incl. agreement) in the given variant."""
    ws = s.strip().split(" ")
    r = _parse_np(ws, 0)
    if r is None:
        return False
    i, n, ok = r
    if i < len(ws) and ws[i] in PREP:
        r = _parse_np(ws, i + 1)
        if r is None:
            return False
        i, _, ok2 = r
        ok &= ok2
    rest = ws[i:]
    if len(rest) == 1:
        return ok and ("VI", n) in _TAG.get(rest[0], ())
    if len(rest) == 3:
        if variant == "svo":
            verb, obj = rest[0], rest[1:]
        else:
            verb, obj = rest[2], rest[:2]
        r = _parse_np(obj, 0)
        return bool(ok and r is not None and r[2] and ("VT", n) in _TAG.get(verb, ()))
    return False


def grammar_score(text: str, variant: str) -> tuple[float, int]:
    sents = [p.strip() for p in text.split(".")[:-1] if p.strip()]
    if not sents:
        return 0.0, 0
    return sum(check_sentence(s, variant) for s in sents) / len(sents), len(sents)


def agreement_probes(variant: str, n: int = 100, seed: int = 4242):
    """Minimal pairs: correct vs wrong verb number, with an attractor PP of opposite number."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sn = rng.choice(["sg", "pl"])
        on = "pl" if sn == "sg" else "sg"
        w = _np(rng, sn) + [rng.choice(PREP)] + _np(rng, on)
        j = rng.randrange(2)
        if variant == "sov":
            w += _np(rng, rng.choice(["sg", "pl"]))
            good, bad = VT[sn][j], VT[on][j]
        else:
            good, bad = VI[sn][j], VI[on][j]
        ctx = grammar_sentence(rng, variant) + " " + " ".join(w) + " "
        out.append((ctx, good + ".", bad + "."))
    return out
