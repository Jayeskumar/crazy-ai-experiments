"""Data-isolation checks for Model B. Each check runs in a fresh subprocess because
the weights-only guard is irreversible inside a process."""
import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
B_SIDE = ["src/model_b.py", "src/train_b.py", "src/alignment.py", "src/weights_only.py", "src/isolation.py"]
TEXT_SIDE = {"src.data", "src.evaluate", "src.train_a", "src.generate_c", "src.model_c", "src.model_a"}


def _run(code):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True, text=True)


def test_b_side_modules_do_not_import_text_side():
    for f in B_SIDE:
        tree = ast.parse((ROOT / f).read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert m not in TEXT_SIDE, f"{f} imports text-side module {m}"


@pytest.mark.parametrize("mod", ["src.data", "src.evaluate", "src.train_a", "src.generate_c"])
def test_text_modules_blocked_in_weights_only_mode(mod):
    r = _run(f"""
        from src.isolation import enter_weights_only_mode, IsolationError
        enter_weights_only_mode()
        try:
            import {mod}
        except IsolationError:
            print("BLOCKED")
    """)
    assert "BLOCKED" in r.stdout, r.stderr


@pytest.mark.parametrize("path", ["runs/model_a/ind/ind_000/checkpoint.pt",
                                  "runs/model_a/ind/ind_000/samples.json",
                                  "runs/model_a/ind/ind_000/weights.pt",
                                  "results/language/c_eval.jsonl",
                                  "/tmp/anything/checkpoint.pt"])
def test_files_blocked_in_weights_only_mode(path):
    r = _run(f"""
        from src.isolation import enter_weights_only_mode, IsolationError
        enter_weights_only_mode()
        try:
            open({path!r}, "rb")
        except IsolationError:
            print("BLOCKED")
        except FileNotFoundError:
            print("NOT_BLOCKED")
    """)
    assert "BLOCKED" in r.stdout and "NOT_BLOCKED" not in r.stdout, r.stdout + r.stderr


def test_guard_refuses_if_text_already_loaded():
    r = _run("""
        import src.data
        from src.isolation import enter_weights_only_mode, IsolationError
        try:
            enter_weights_only_mode()
        except IsolationError:
            print("REFUSED")
    """)
    assert "REFUSED" in r.stdout, r.stderr


def test_train_b_activates_guard_before_other_imports():
    src = (ROOT / "src/train_b.py").read_text()
    tree = ast.parse(src)
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert isinstance(body[0], ast.ImportFrom) and body[0].module == "src.isolation"
    assert isinstance(body[1], ast.Expr) and ast.unparse(body[1]) == "enter_weights_only_mode()"


def test_sanitizer_rejects_non_weight_content(tmp_path, monkeypatch):
    import src.weights_only as wo
    monkeypatch.setattr(wo, "WO_DIR", tmp_path)
    good = {"head.weight": torch.zeros(3, 2), "head.bias": torch.zeros(3)}
    torch.save(good, tmp_path / "ok.pt")
    wo.sanitize([tmp_path / "ok.pt"], ["r0"], ["train"], "ok")
    with np.load(tmp_path / "ok.npz", allow_pickle=False) as z:
        assert set(z.files) == wo.ALLOWED_KEYS
        assert z["W"].dtype == np.float32
    for bad in ({**good, "itos": torch.zeros(3)},                     # unexpected key
                {**good, "head.bias": torch.zeros(3, dtype=torch.long)}):  # token-id-like ints
        torch.save(bad, tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            wo.sanitize([tmp_path / "bad.pt"], ["r0"], ["train"], "bad")


def test_weights_only_dataset_rejects_extra_keys(tmp_path, monkeypatch):
    import src.weights_only as wo
    monkeypatch.setattr(wo, "WO_DIR", tmp_path)
    np.savez(tmp_path / "evil.npz", W=np.zeros((1, 3), np.float32), names=np.array(["head.bias"]),
             shapes=np.array([[3, -1]]), run_ids=np.array(["r"]), split=np.array(["train"]),
             text=np.array(["the dog runs."]))
    with pytest.raises(ValueError):
        wo.WeightsOnlyDataset("evil")


def test_real_sanitized_files_contain_only_weights():
    files = sorted((ROOT / "artifacts" / "weights_only").glob("*.npz"))
    if not files:
        pytest.skip("sanitized weights not produced yet")
    import src.weights_only as wo
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            assert set(z.files) == wo.ALLOWED_KEYS
            assert z["W"].dtype == np.float32
            for n in z["names"]:
                assert wo.NAME_RE.match(str(n))
            for r in z["run_ids"]:
                assert re.match(r"^(ind|shared|traj|spawn\d+)_", str(r))
