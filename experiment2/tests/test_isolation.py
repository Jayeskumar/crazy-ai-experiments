"""Model B data-isolation checks (subprocesses: the guard is irreversible)."""
import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

E2 = Path(__file__).resolve().parents[1]
B_SIDE = ["exp2/model_b.py", "exp2/train_b.py", "exp2/train_b2.py", "exp2/wtok.py", "exp2/sanitize.py", "exp2/isolation.py"]
TEXT = {"exp2.tasks", "exp2.evaluate", "exp2.train_a", "exp2.generate_c", "exp2.analysis", "exp2.model",
        "src.data", "src.evaluate"}


def _run(code):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=E2, capture_output=True, text=True)


def test_static_no_text_imports_on_b_side():
    for f in B_SIDE:
        for node in ast.walk(ast.parse((E2 / f).read_text())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for m in mods:
                assert m not in TEXT, f"{f} imports {m}"


@pytest.mark.parametrize("mod", ["exp2.tasks", "exp2.evaluate", "exp2.train_a", "exp2.generate_c", "exp2.model"])
def test_text_modules_blocked(mod):
    r = _run(f"""
        import exp2
        from exp2.isolation import enter_weights_only_mode, IsolationError
        enter_weights_only_mode()
        try:
            import {mod}
        except IsolationError:
            print("BLOCKED")
    """)
    assert "BLOCKED" in r.stdout, r.stderr


@pytest.mark.parametrize("path", ["runs/arith_add_00/weights.pt", "runs/x/checkpoint.pt", "artifacts/private/id_map.json",
                                  "results/language/c_eval.jsonl", "results/model_a/id_map.json", "pilot_runs/a/metrics.json", "../runs/model_a/x.pt"])
def test_files_blocked(path):
    r = _run(f"""
        import exp2
        from exp2.isolation import enter_weights_only_mode, IsolationError
        enter_weights_only_mode()
        try:
            open({path!r}, "rb")
        except IsolationError:
            print("BLOCKED")
        except FileNotFoundError:
            print("NOT_BLOCKED")
    """)
    assert "BLOCKED" in r.stdout and "NOT_BLOCKED" not in r.stdout, r.stdout + r.stderr


def test_train_b_enters_guard_first():
    body = [n for n in ast.parse((E2 / "exp2/train_b.py").read_text()).body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "exp2.isolation"
    assert ast.unparse(body[1]) == "enter_weights_only_mode()"


def test_sanitized_files_are_weights_only_and_opaque():
    files = sorted((E2 / "artifacts" / "weights_only").glob("*.npz"))
    if not files:
        pytest.skip("not produced yet")
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            assert set(z.files) == {"W", "names", "shapes", "ids"}
            assert z["W"].dtype == np.float32
            for i in z["ids"]:
                assert re.fullmatch(r"[0-9a-f]{12}", str(i)), "ids must not reveal task/run names"


def test_train_b2_enters_guard_first():
    body = [n for n in ast.parse((E2 / "exp2/train_b2.py").read_text()).body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "exp2.isolation"
    assert ast.unparse(body[1]) == "enter_weights_only_mode()"
