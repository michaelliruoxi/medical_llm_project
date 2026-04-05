"""Run the sequential experiment pipeline with an explicit config selection.

Usage:
    python scripts/experiments/run_pipeline.py
    python scripts/experiments/run_pipeline.py --config configs/models/gpt54_hybrid_api.yaml
    python scripts/experiments/run_pipeline.py --pilot
    python scripts/experiments/run_pipeline.py --n 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, resolve_config_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "models" / "gpt54_hybrid_api.yaml"


def build_steps(sample_limit: int | None) -> list[tuple[str, list[str]]]:
    def maybe_limit(cmd: list[str], module_name: str) -> list[str]:
        if sample_limit is None:
            return cmd
        if module_name in {"src.ingest", "src.noise", "src.repair"}:
            return cmd + ["--n", str(sample_limit)]
        return cmd

    base_steps = [
        ("Step 1: Ingest", [sys.executable, "-m", "src.ingest"], "src.ingest"),
        ("Step 2: Noise injection", [sys.executable, "-m", "src.noise"], "src.noise"),
        ("Step 3a: Answer (clean)", [sys.executable, "-m", "src.answer", "--mode", "clean"], "src.answer"),
        ("Step 3b: Answer (noisy)", [sys.executable, "-m", "src.answer", "--mode", "noisy"], "src.answer"),
        ("Step 4: Repair", [sys.executable, "-m", "src.repair"], "src.repair"),
        ("Step 5: Answer (repaired)", [sys.executable, "-m", "src.answer", "--mode", "repaired"], "src.answer"),
        ("Step 6: Evaluate metrics", [sys.executable, "-m", "src.evaluate_metrics"], "src.evaluate_metrics"),
        ("Step 7: LLM judge", [sys.executable, "-m", "src.judge"], "src.judge"),
        ("Step 8: Aggregate", [sys.executable, "-m", "src.aggregate"], "src.aggregate"),
        ("Step 9: Report tables", [sys.executable, "-m", "src.report_tables"], "src.report_tables"),
    ]
    return [(label, maybe_limit(cmd, module)) for label, cmd, module in base_steps]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full MedQuAD robustness pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Config YAML to use for the sequential experiment pipeline",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run with 20 samples instead of the config's full size",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override the sample count for ingest/noise/repair stages",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    sample_limit = args.n if args.n is not None else (20 if args.pilot else None)
    cfg = load_config(config_path)

    env = os.environ.copy()
    env["MEDQUAD_CONFIG"] = str(config_path)

    print(f"\nUsing config: {config_path}")
    print(f"Backend: {cfg.get('backend', 'openai')}")
    print(f"Answer model: {cfg.get('answer_model')}")
    print(f"Outputs: {PROJECT_ROOT / cfg['paths']['outputs']}")
    if sample_limit is not None:
        print(f"Sample override: {sample_limit}")

    for label, cmd in build_steps(sample_limit):
        print(f"\n{'=' * 60}")
        print(f"=== {label} ===")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, check=False, cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            print(f"\nERROR: {label} failed with exit code {result.returncode}")
            return result.returncode

    print(f"\n{'=' * 60}")
    print("=== Pipeline complete ===")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
