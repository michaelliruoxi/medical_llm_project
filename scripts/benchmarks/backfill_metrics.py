"""Backfill newly-added reference metrics onto existing benchmark outputs.

When we extend ``REFERENCE_METRICS`` in ``run_comparison.py`` (e.g. to add
``med_coverage`` / ``med_precision`` / ``med_f1``), already-completed model
samples are missing the new columns. This script regenerates them without
re-running any model inference:

    1. load the model's ``samples_*.csv``
    2. upgrade its schema to the current ``SAMPLE_FIELDNAMES``
    3. fill any NaN reference-metric cell via ``compute_reference_metrics``
    4. re-emit ``result_*.json`` using ``build_summary_from_samples``, keeping
       the existing ``benchmark_mode``, ``status``, ``question_set_dir`` and
       ``error`` so partial runs stay marked partial
    5. rewrite ``model_comparison.csv`` and ``model_comparison_full.json``
       using ``write_comparison_outputs``

Usage:
    python scripts/benchmarks/backfill_metrics.py \\
        --mode fixed_repair

Options:
    --mode {fixed_repair,self_repair,end_to_end}  default: fixed_repair
    --configs-dir configs/models/                 auto-discover configs
    --dry-run                                     show plan only, no writes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

# Block broken TensorFlow BEFORE anything else imports it (mirrors run_comparison.py).
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

from src.utils import load_config, quantization_label, set_active_config  # noqa: E402

# Import the benchmark module so we can reuse the exact helpers it uses for
# schema upgrade, metric backfill, summary building, and output writing.
import run_comparison as rc  # type: ignore  # noqa: E402


def _label_from_samples_path(path: Path) -> str:
    # samples_<label>.csv -> <label>
    return path.stem[len("samples_"):]


def _paths_for_label(output_dir: Path, label: str) -> dict[str, Path]:
    return {
        "samples_csv": output_dir / f"samples_{label}.csv",
        "result_json": output_dir / f"result_{label}.json",
        "progress_json": output_dir / f"progress_{label}.json",
    }


def _load_result_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _config_path_for(config_name: str, configs_dir: Path) -> Path | None:
    if not config_name:
        return None
    candidate = configs_dir / config_name
    if candidate.exists():
        return candidate
    # Also accept fully-qualified paths stored in old result_*.json files.
    direct = Path(config_name)
    if direct.exists():
        return direct
    return None


def backfill_one(
    label: str,
    output_dir: Path,
    configs_dir: Path,
    benchmark_mode: str,
    dry_run: bool,
) -> dict | None:
    paths = _paths_for_label(output_dir, label)
    samples_csv = paths["samples_csv"]
    if not samples_csv.exists() or samples_csv.stat().st_size == 0:
        print(f"[skip] {label}: no samples_*.csv")
        return None

    existing_summary = _load_result_json(paths["result_json"])
    config_name = existing_summary.get("config", "")
    config_path = _config_path_for(config_name, configs_dir)
    if config_path is None:
        print(f"[skip] {label}: cannot locate config '{config_name}' under {configs_dir}")
        return None

    set_active_config(config_path)
    cfg = load_config(str(config_path))

    samples_df = pd.read_csv(samples_csv)
    upgraded_df, changed_schema = rc._upgrade_schema_in_place(samples_df)
    upgraded_df, changed_metrics = rc._backfill_reference_metrics(upgraded_df)

    status = existing_summary.get("status", "completed")
    rows_expected = int(existing_summary.get("rows_expected", len(upgraded_df)))
    error = existing_summary.get("error")
    qset_dir = existing_summary.get("question_set_dir")
    qset_dir_path = Path(qset_dir) if qset_dir else None

    summary = rc.build_summary_from_samples(
        samples_df=upgraded_df,
        cfg=cfg,
        config_path=str(config_path),
        label=label,
        rows_expected=rows_expected,
        status=status,
        benchmark_mode=existing_summary.get("benchmark_mode", benchmark_mode),
        question_set_dir=qset_dir_path,
        error=error,
    )
    runtime_seconds = existing_summary.get("runtime_seconds")
    if runtime_seconds is not None:
        try:
            summary["runtime_seconds"] = float(runtime_seconds)
        except (TypeError, ValueError):
            pass

    quant = quantization_label(cfg)
    action_bits = []
    if changed_schema:
        action_bits.append("schema")
    if changed_metrics:
        action_bits.append("metrics")
    action = ",".join(action_bits) if action_bits else "summary-only"

    print(
        f"[{'dry-run' if dry_run else 'write'}] {label} "
        f"(quant={quant}, rows={len(upgraded_df)}/{rows_expected}, "
        f"status={status}, changed={action})"
    )

    if dry_run:
        return summary

    if changed_schema or changed_metrics:
        rc.write_csv_atomic(samples_csv, upgraded_df[rc.SAMPLE_FIELDNAMES])

    rc.write_json_atomic(paths["result_json"], summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=rc.BENCHMARK_MODES,
        default="fixed_repair",
    )
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
    args = parser.parse_args()

    output_dir = args.output_root / args.mode
    if not output_dir.exists():
        print(f"No output directory for mode={args.mode}: {output_dir}")
        return

    sample_paths = sorted(output_dir.glob("samples_*.csv"))
    if not sample_paths:
        print(f"No samples_*.csv under {output_dir}")
        return

    summaries: list[dict] = []
    for sp in sample_paths:
        label = _label_from_samples_path(sp)
        summary = backfill_one(
            label=label,
            output_dir=output_dir,
            configs_dir=args.configs_dir,
            benchmark_mode=args.mode,
            dry_run=args.dry_run,
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("No summaries rebuilt; leaving model_comparison outputs untouched.")
        return

    # run_priority ordering mirrors main run_comparison.py output ordering.
    def _priority(s: dict) -> tuple[float, str]:
        cfg_name = s.get("config", "")
        cfg_path = _config_path_for(cfg_name, args.configs_dir)
        prio = float("inf")
        if cfg_path is not None:
            try:
                prio = float(load_config(str(cfg_path)).get("run_priority", prio))
            except Exception:
                pass
        return prio, s.get("label", "")

    summaries.sort(key=_priority)

    if args.dry_run:
        print(f"\n[dry-run] would update model_comparison.csv with {len(summaries)} rows")
        return

    rc.write_comparison_outputs(summaries, output_dir)
    print(
        f"\nRewrote model_comparison.csv + model_comparison_full.json "
        f"for {len(summaries)} model(s) under {output_dir}"
    )


if __name__ == "__main__":
    main()
