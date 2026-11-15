"""Entry point:  python -I model_editing/run_experiment.py   (resumable; see README)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.run import main  # noqa: E402

if __name__ == "__main__":
    main()
