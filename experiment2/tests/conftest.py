import sys
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
for p in (E2, E2.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
