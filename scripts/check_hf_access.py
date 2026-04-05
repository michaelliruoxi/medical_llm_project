"""Compatibility wrapper for scripts/benchmarks/check_hf_access.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "benchmarks" / "check_hf_access.py"
    runpy.run_path(str(target), run_name="__main__")
