"""Data-isolation guard for Model B (Experiment 2). Same mechanism as Experiment 1
(src/isolation.py): import blocker + irremovable audit hook, configured for this experiment.

Forbidden after enter_weights_only_mode():
  * importing task/text modules (exp2.tasks, exp2.evaluate, exp2.train_a, exp2.generate_c,
    exp2.analysis, exp2.model, and Experiment-1 text modules)
  * opening anything under experiment2/runs, experiment2/pilot_runs, experiment2/artifacts/private
    (opaque-id -> task map), experiment2/results/language, the Experiment-1 runs/, or any file
    named checkpoint.pt / samples* / metrics.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
REPO = E2.parent
FORBIDDEN_MODULES = ("exp2.tasks", "exp2.evaluate", "exp2.train_a", "exp2.generate_c", "exp2.analysis",
                     "exp2.model", "src.data", "src.evaluate", "src.train_a", "src.generate_c", "src.model_c")
FORBIDDEN_DIRS = (E2 / "runs", E2 / "pilot_runs", E2 / "artifacts" / "private", E2 / "results" / "language",
                  E2 / "results" / "model_a",
                  REPO / "runs")
FORBIDDEN_NAMES = ("checkpoint.pt", "samples", "metrics.json")
_ACTIVE = False


class IsolationError(RuntimeError):
    pass


class _Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in FORBIDDEN_MODULES:
            raise IsolationError(f"weights-only mode: import of '{fullname}' is forbidden")
        return None


def _audit(event, args):
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        p = Path(os.fsdecode(args[0])).resolve()
    except Exception:
        return
    s = str(p)
    for d in FORBIDDEN_DIRS:
        if s == str(d) or s.startswith(str(d) + os.sep):
            raise IsolationError(f"weights-only mode: opening '{s}' is forbidden")
    if any(n in p.name for n in FORBIDDEN_NAMES):
        raise IsolationError(f"weights-only mode: opening '{s}' is forbidden")


def enter_weights_only_mode():
    global _ACTIVE
    if _ACTIVE:
        return
    loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
    if loaded:
        raise IsolationError(f"text-side modules already imported: {loaded}")
    sys.meta_path.insert(0, _Block())
    sys.addaudithook(_audit)
    _ACTIVE = True


def assert_active():
    if not _ACTIVE:
        raise IsolationError("Model B code must run inside enter_weights_only_mode()")
