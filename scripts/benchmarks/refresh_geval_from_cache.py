"""Refresh cached G-Eval scores in benchmark sample CSVs.

This script fills missing G-Eval cells by replaying the existing prompts.
Because `geval_answer` uses `call_llm`, already-issued judge requests are read
from cache instead of hitting the API again in the common case. Use --force to
recompute populated cells.

Usage:
    python scripts/benchmarks/refresh_geval_from_cache.py --mode fixed_repair
    python scripts/benchmarks/refresh_geval_from_cache.py --mode self_repair
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

# Mirror the TensorFlow stub used by run_comparison.py so imports stay stable.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
import importlib.machinery  # noqa: E402

_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "0.0.0"
_tf_stub.__path__ = []
_tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules.setdefault("tensorflow", _tf_stub)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmarks"))

import pandas as pd  # noqa: E402

from src.judge import geval_answer  # noqa: E402
from src.utils import load_config, load_prompts, set_active_config  # noqa: E402

import run_comparison as rc  # type: ignore  # noqa: E402


PIPELINES = ("clean", "noisy", "repaired")
QUESTION_COLS = {
    "clean": "question_clean",
    "noisy": "question_noisy",
    "repaired": "question_repaired",
}
ANSWER_COLS = {
    "clean": "answer_clean",
    "noisy": "answer_noisy",
    "repaired": "answer_repaired",
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _config_path_for(config_name: str, configs_dir: Path) -> Path | None:
    if not config_name:
        return None
    candidate = configs_dir / config_name
    if candidate.exists():
        return candidate
    direct = Path(config_name)
    if direct.exists():
        return direct
    return None


def refresh_one(
    samples_csv: Path,
    output_dir: Path,
    configs_dir: Path,
    benchmark_mode: str,
    dry_run: bool,
    force: bool,
) -> dict | None:
    label = samples_csv.stem[len("samples_") :]
    paths = rc.model_paths(output_dir, label)
    result_json = _load_json(paths["result_json"])
    config_name = result_json.get("config", "")
    config_path = _config_path_for(config_name, configs_dir)
    if config_path is None:
        print(f"[skip] {label}: cannot locate config '{config_name}'")
        return None

    set_active_config(config_path)
    cfg = load_config(str(config_path))
    prompts = load_prompts()

    df = rc.load_samples_csv(samples_csv).copy()
    if df.empty:
        print(f"[skip] {label}: empty samples CSV")
        return None

    changed = 0
    for idx, row in df.iterrows():
        reference = str(row.get("reference_answer", "") or "")
        if not reference:
            continue
        for pipeline in PIPELINES:
            answer = str(row.get(ANSWER_COLS[pipeline], "") or "")
            question = str(row.get(QUESTION_COLS[pipeline], "") or "")
            if not answer or not question:
                continue
            col = f"geval_{pipeline}"
            old_score = pd.to_numeric(row.get(col), errors="coerce")
            if not force and not pd.isna(old_score):
                continue
            new_score = geval_answer(reference, answer, prompts, cfg, question=question)
            if pd.isna(old_score) or int(round(float(old_score))) != int(new_score):
                df.at[idx, col] = int(new_score)
                changed += 1

    status = result_json.get("status", "completed")
    rows_expected = int(result_json.get("rows_expected", len(df)))
    error = result_json.get("error")
    qset_dir = result_json.get("question_set_dir")
    qset_dir_path = Path(qset_dir) if qset_dir else None

    summary = rc.build_summary_from_samples(
        samples_df=df,
        cfg=cfg,
        config_path=str(config_path),
        label=label,
        rows_expected=rows_expected,
        status=status,
        benchmark_mode=result_json.get("benchmark_mode", benchmark_mode),
        question_set_dir=qset_dir_path,
        error=error,
    )

    runtime_seconds = result_json.get("runtime_seconds")
    if runtime_seconds is not None:
        try:
            summary["runtime_seconds"] = float(runtime_seconds)
        except (TypeError, ValueError):
            pass

    print(f"[{'dry-run' if dry_run else 'write'}] {label}: refreshed {changed} G-Eval cell(s)")

    if dry_run:
        return summary

    rc.write_csv_atomic(samples_csv, df[rc.SAMPLE_FIELDNAMES])
    rc.write_json_atomic(paths["result_json"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=rc.BENCHMARK_MODES, default="fixed_repair")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs" / "benchmarks",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute populated G-Eval cells instead of filling only missing values.",
    )
    args = parser.parse_args()

    output_dir = args.output_root / args.mode
    sample_paths = sorted(output_dir.glob("samples_*.csv"))
    if not sample_paths:
        print(f"No samples_*.csv under {output_dir}")
        return 0

    summaries: list[dict] = []
    for samples_csv in sample_paths:
        summary = refresh_one(
            samples_csv=samples_csv,
            output_dir=output_dir,
            configs_dir=args.configs_dir,
            benchmark_mode=args.mode,
            dry_run=args.dry_run,
            force=args.force,
        )
        if summary is not None:
            summaries.append(summary)

    if summaries and not args.dry_run:
        rc.write_comparison_outputs(summaries, output_dir)
        print(
            f"Rewrote model_comparison.csv + model_comparison_full.json "
            f"for {len(summaries)} model(s) under {output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
