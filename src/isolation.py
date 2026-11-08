"""Architectural data-isolation guard for Model B.

`enter_weights_only_mode()` must be called at the start of any process that
trains or runs Model B. After it is called, for the rest of the process:

  * importing any text-side module (src.data, src.evaluate, src.train_a,
    src.generate_c, src.model_c) raises IsolationError;
  * opening any file under runs/ (Model A checkpoints, vocab, samples, metrics)
    or under results/language/ raises IsolationError (enforced with a
    sys.addaudithook 'open' hook, which cannot be removed once installed);
  * the only Model-A-derived input B can read is the sanitized
    artifacts/weights_only/*.npz produced by src/weights_only.py.

These checks are exercised by tests/test_isolation.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = ("src.data", "src.evaluate", "src.train_a", "src.generate_c", "src.model_c")
FORBIDDEN_DIRS = (ROOT / "runs", ROOT / "results" / "language")
FORBIDDEN_NAMES = ("checkpoint.pt", "samples.json", "vocab", "itos")

_ACTIVE = False


class IsolationError(RuntimeError):
    pass


class _BlockTextModules:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in FORBIDDEN_MODULES:
            raise IsolationError(f"weights-only mode: import of text-side module '{fullname}' is forbidden")
        return None


def _audit(event, args):
    if event != "open" or not args:
        return
    p = args[0]
    if isinstance(p, int):
        return
    try:
        path = Path(os.fsdecode(p)).resolve()
    except Exception:
        return
    s = str(path)
    for d in FORBIDDEN_DIRS:
        if s == str(d) or s.startswith(str(d) + os.sep):
            raise IsolationError(f"weights-only mode: opening '{s}' is forbidden")
    if any(n in path.name for n in FORBIDDEN_NAMES):
        raise IsolationError(f"weights-only mode: opening '{s}' is forbidden")


def enter_weights_only_mode():
    global _ACTIVE
    if _ACTIVE:
        return
    already = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if already:
        raise IsolationError(f"text-side modules already imported before isolation: {already}")
    sys.meta_path.insert(0, _BlockTextModules())
    sys.addaudithook(_audit)
    _ACTIVE = True


def is_active() -> bool:
    return _ACTIVE


def assert_active():
    if not _ACTIVE:
        raise IsolationError("Model B code must run inside enter_weights_only_mode()")
