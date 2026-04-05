"""Compatibility wrapper for scripts/experiments/run_pipeline.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "experiments" / "run_pipeline.py"
    runpy.run_path(str(target), run_name="__main__")
