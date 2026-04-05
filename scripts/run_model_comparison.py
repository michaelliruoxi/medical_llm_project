"""Compatibility wrapper for scripts/benchmarks/run_comparison.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "benchmarks" / "run_comparison.py"
    runpy.run_path(str(target), run_name="__main__")
