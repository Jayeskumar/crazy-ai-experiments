"""Experiment 2: model-specific weight-space learning across task families."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # experiment2/
REPO = ROOT.parent                                  # repository root (Experiment 1 code lives in REPO/src)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
