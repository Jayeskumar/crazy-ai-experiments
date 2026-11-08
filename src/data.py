"""Synthetic language: a small probabilistic grammar with number agreement.

TEXT-SIDE MODULE. Model B code must never import this module (enforced by
src/isolation.py and tests/test_isolation.py).

Grammar (all lowercase, words separated by single spaces, sentences end in ". "):

    S        -> NP[n] (PP)? VP[n] "."
    NP[n]    -> DET[n] (ADJ)? NOUN[n]
    PP       -> PREP NP[m]                 (m drawn independently -> "attractor")
    VP[n]    -> VI[n] | VT[n] NP[m]

The long-range dependency is subject-verb number agreement across an optional
prepositional phrase whose noun may have the opposite number
("the dog near the cats runs.").
"""
from __future__ import annotations

import random
from dataclasses import dataclass

DET = {"sg": ["the", "a", "one", "every"], "pl": ["the", "two", "some", "many"]}
ADJ = ["big", "small", "red", "old", "happy"]
NOUNS = [("cat", "cats"), ("dog", "dogs"), ("bird", "birds"), ("fox", "foxes"),
         ("child", "children"), ("farmer", "farmers")]
VI = [("runs", "run"), ("sleeps", "sleep"), ("sings", "sing"), ("waits", "wait")]
VT = [("sees", "see"), ("likes", "like"), ("chases", "chase"), ("helps", "help")]
PREP = ["near", "with", "behind"]

NOUN = {"sg": [a for a, _ in NOUNS], "pl": [b for _, b in NOUNS]}
VERB_I = {"sg": [a for a, _ in VI], "pl": [b for _, b in VI]}
VERB_T = {"sg": [a for a, _ in VT], "pl": [b for _, b in VT]}

P_ADJ, P_PP, P_TRANS = 0.3, 0.4, 0.5


def _np(rng: random.Random, n: str) -> list[str]:
    w = [rng.choice(DET[n])]
    if rng.random() < P_ADJ:
        w.append(rng.choice(ADJ))
    w.append(rng.choice(NOUN[n]))
    return w


def sentence(rng: random.Random, force_pp: bool | None = None) -> str:
    n = rng.choice(["sg", "pl"])
    words = _np(rng, n)
    if (rng.random() < P_PP) if force_pp is None else force_pp:
        words += [rng.choice(PREP)] + _np(rng, rng.choice(["sg", "pl"]))
    if rng.random() < P_TRANS:
        words += [rng.choice(VERB_T[n])] + _np(rng, rng.choice(["sg", "pl"]))
    else:
        words.append(rng.choice(VERB_I[n]))
    return " ".join(words) + "."


def corpus(seed: int, n_chars: int) -> str:
    rng = random.Random(seed)
    out, total = [], 0
    while total < n_chars:
        s = sentence(rng) + " "
        out.append(s)
        total += len(s)
    return "".join(out)[:n_chars]


# Fixed corpus seeds; train/val/test never share a seed.
SPLITS = {"train": (1001, 300_000), "val": (2002, 30_000), "test": (3003, 30_000)}


def load_split(name: str) -> str:
    seed, n = SPLITS[name]
    return corpus(seed, n)


@dataclass
class CharTokenizer:
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        m = self.stoi
        return [m[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def default_tokenizer() -> CharTokenizer:
    return CharTokenizer.from_text(load_split("train"))


# ----------------------------------------------------------------------------
# Grammar checker (used only for evaluation of generated text)
# ----------------------------------------------------------------------------
_ALL = {}
for n in ("sg", "pl"):
    for w in DET[n]:
        _ALL.setdefault(w, set()).add(("DET", n))
    for w in NOUN[n]:
        _ALL.setdefault(w, set()).add(("NOUN", n))
    for w in VERB_I[n]:
        _ALL.setdefault(w, set()).add(("VI", n))
    for w in VERB_T[n]:
        _ALL.setdefault(w, set()).add(("VT", n))
for w in ADJ:
    _ALL.setdefault(w, set()).add(("ADJ", None))
for w in PREP:
    _ALL.setdefault(w, set()).add(("PREP", None))
LEXICON = set(_ALL)


def _parse_np(ws, i):
    """Return (next_index, noun_number, det_agrees) or None."""
    if i >= len(ws) or not any(t == "DET" for t, _ in _ALL.get(ws[i], ())):
        return None
    dets = {n for t, n in _ALL[ws[i]] if t == "DET"}
    i += 1
    if i < len(ws) and ws[i] in ADJ:
        i += 1
    if i >= len(ws):
        return None
    nouns = {n for t, n in _ALL.get(ws[i], ()) if t == "NOUN"}
    if not nouns:
        return None
    nn = next(iter(nouns))
    return i + 1, nn, nn in dets


def check_sentence(s: str) -> dict:
    """Classify one sentence (without the final period).

    Returns dict(structure=bool, agreement=bool|None, words_valid=bool).
    structure: word-category sequence matches the grammar (ignoring number).
    agreement: all det-noun and subject-verb number constraints hold
               (None if structure is invalid).
    """
    ws = s.strip().split(" ")
    res = {"words_valid": all(w in LEXICON for w in ws), "structure": False, "agreement": None}
    r = _parse_np(ws, 0)
    if r is None:
        return res
    i, subj_n, ok = r
    if i < len(ws) and ws[i] in PREP:
        r2 = _parse_np(ws, i + 1)
        if r2 is None:
            return res
        i, _, ok2 = r2
        ok &= ok2
    if i >= len(ws):
        return res
    tags = _ALL.get(ws[i], set())
    vi = {n for t, n in tags if t == "VI"}
    vt = {n for t, n in tags if t == "VT"}
    if vi and i == len(ws) - 1:
        res["structure"] = True
        res["agreement"] = bool(ok and subj_n in vi)
        return res
    if vt:
        r3 = _parse_np(ws, i + 1)
        if r3 is not None and r3[0] == len(ws):
            res["structure"] = True
            res["agreement"] = bool(ok and r3[2] and subj_n in vt)
    return res


def grammar_stats(text: str) -> dict:
    """Score a generated text stream. Only complete sentences (ending in '.') count."""
    parts = text.split(".")
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    words = [w for p in complete for w in p.split(" ") if w]
    if not complete:
        return {"n_sentences": 0, "frac_valid_words": 0.0, "frac_structure": 0.0,
                "frac_grammatical": 0.0}
    checks = [check_sentence(p) for p in complete]
    return {
        "n_sentences": len(complete),
        "frac_valid_words": sum(w in LEXICON for w in words) / max(1, len(words)),
        "frac_structure": sum(c["structure"] for c in checks) / len(checks),
        "frac_grammatical": sum(bool(c["agreement"]) for c in checks) / len(checks),
    }


def agreement_probes(seed: int = 4004, n: int = 200) -> list[dict]:
    """Minimal pairs for subject-verb agreement.

    Each probe: a context (previous sentence + subject NP [+ PP]) and two
    continuations (correct / incorrect verb number).
    kind = 'plain' (no PP) or 'attractor' (PP noun has opposite number).
    """
    rng = random.Random(seed)
    probes = []
    for k in range(n):
        kind = "attractor" if k % 2 else "plain"
        n_ = rng.choice(["sg", "pl"])
        other = "pl" if n_ == "sg" else "sg"
        words = _np(rng, n_)
        if kind == "attractor":
            words += [rng.choice(PREP)] + _np(rng, other)
        j = rng.randrange(len(VI))
        good, bad = (VI[j][0], VI[j][1]) if n_ == "sg" else (VI[j][1], VI[j][0])
        ctx = sentence(rng) + " " + " ".join(words) + " "
        probes.append({"context": ctx, "good": good + ".", "bad": bad + ".", "kind": kind})
    return probes


def sentence_logprob(s: str) -> float:
    """Exact log-probability of one grammatical sentence (without '.') under the generator."""
    import math
    ws = s.split(" ")

    def np_lp(i):
        n = "sg" if ws[i + (2 if ws[i + 1] in ADJ else 1)] in NOUN["sg"] else "pl"
        lp = math.log(1 / len(DET[n]))
        if ws[i + 1] in ADJ:
            lp += math.log(P_ADJ / len(ADJ))
            i += 1
        else:
            lp += math.log(1 - P_ADJ)
        return i + 2, lp + math.log(1 / len(NOUNS)), n

    lp = math.log(0.5)
    i, l1, n = np_lp(0)
    lp += l1
    if ws[i] in PREP:
        i, l2, _ = np_lp(i + 1)
        lp += math.log(P_PP / len(PREP)) + math.log(0.5) + l2
    else:
        lp += math.log(1 - P_PP)
    if i == len(ws) - 1:
        lp += math.log(1 - P_TRANS) + math.log(1 / len(VI))
    else:
        _, l3, _ = np_lp(i + 1)
        lp += math.log(P_TRANS) + math.log(1 / len(VT)) + math.log(0.5) + l3
    return lp


def entropy_rate(n: int = 20000, seed: int = 99) -> float:
    """Monte-Carlo estimate of the source entropy rate in nats/char (the irreducible loss)."""
    rng = random.Random(seed)
    H = L = 0.0
    for _ in range(n):
        s = sentence(rng)
        H -= sentence_logprob(s[:-1])
        L += len(s) + 1
    return H / L
