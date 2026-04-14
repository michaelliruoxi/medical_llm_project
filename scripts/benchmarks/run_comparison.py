"""Run the robustness pipeline across multiple model configs and produce a comparison table.

This version is resumable and fail-safe for long runs:
- each completed sample row is appended to CSV immediately
- per-model progress is written to JSON continuously
- reruns skip rows already present in the model CSV

Usage:
    python scripts/benchmarks/run_comparison.py
    python scripts/benchmarks/run_comparison.py --configs configs/models/mistral_7b.yaml
    python scripts/benchmarks/run_comparison.py --benchmark-mode fixed_repair
    python scripts/benchmarks/run_comparison.py --benchmark-mode self_repair
    python scripts/benchmarks/run_comparison.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
from pathlib import Path

# Block broken TensorFlow BEFORE anything else imports it
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import importlib.machinery

_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "0.0.0"
_tf_stub.__path__ = []
_tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _tf_stub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from bert_score import BERTScorer

from src.answer import answer_question
from src.judge import geval_answer
from src.metrics import (
    compute_intent_preservation_scores,
    compute_recovery_statistics,
    compute_reference_metrics,
)
from src.noise import generate_noisy_variant
from src.noise_schedule import build_noise_plan, expected_noise_rows, summarize_noise_plan
from src.repair import repair_question
from src.utils import load_config, load_prompts, quantization_label, set_active_config, setup_logging, unload_local_models


logger = setup_logging()

PIPELINES = ("clean", "noisy", "repaired")
BENCHMARK_MODES = ("end_to_end", "fixed_repair", "self_repair")
REFERENCE_METRICS = ("bleu", "chrf", "rouge_l", "token_f1", "exact_match")
METRICS = REFERENCE_METRICS + ("bertscore", "intent_preservation", "geval")
DEFAULT_QUESTION_SET_DIR = PROJECT_ROOT / "data" / "processed" / "benchmarks" / "fixed_question_sets_gpt54"
BASE_SAMPLE_FIELDS = [
    "model",
    "config",
    "quantization",
    "noise_type",
    "question_id",
    "question_clean",
    "question_noisy",
    "question_repaired",
    "reference_answer",
    "answer_clean",
    "answer_noisy",
    "answer_repaired",
]
SAMPLE_FIELDNAMES = BASE_SAMPLE_FIELDS + [
    f"{metric}_{pipeline}"
    for metric in METRICS
    for pipeline in PIPELINES
]
TEXT_SAMPLE_FIELDS = set(BASE_SAMPLE_FIELDS)
METRIC_PRECISION = {"bertscore": 4, "intent_preservation": 4}

_bert_scorer: BERTScorer | None = None


def clear_cache(config_paths: list[str]):
    """Remove cached LLM responses for the selected configs."""
    cleared = 0
    seen_dirs: set[Path] = set()
    for config_path in config_paths:
        cfg = load_config(config_path)
        cache_dir = PROJECT_ROOT / cfg["paths"]["cache"]
        if cache_dir in seen_dirs:
            continue
        seen_dirs.add(cache_dir)
        if not cache_dir.exists():
            continue
        for f in cache_dir.glob("*.json"):
            try:
                f.unlink()
                cleared += 1
            except PermissionError:
                pass
    logger.info("Cleared %d cached responses across %d cache directories", cleared, len(seen_dirs))


def load_samples(n: int = 50) -> pd.DataFrame:
    """Load n samples from the cleaned dataset."""
    cleaned = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.parquet"
    if not cleaned.exists():
        cleaned_csv = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.csv"
        if not cleaned_csv.exists():
            logger.error("No cleaned data found. Run data_cleaning notebook first.")
            sys.exit(1)
        df = pd.read_csv(cleaned_csv)
    else:
        df = pd.read_parquet(cleaned)

    df = df.head(n).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def _default_output_dir(benchmark_mode: str) -> Path:
    if benchmark_mode == "end_to_end":
        return PROJECT_ROOT / "data" / "outputs" / "comparison"
    return PROJECT_ROOT / "data" / "outputs" / "benchmarks" / benchmark_mode


def _normalize_question_set_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "question" in df.columns and "question_clean" not in df.columns:
        rename_map["question"] = "question_clean"
    if "answer" in df.columns and "answer_ref" not in df.columns:
        rename_map["answer"] = "answer_ref"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "source" not in df.columns:
        df["source"] = ""
    return df


def _read_question_set_file(path_csv: Path, path_parquet: Path) -> tuple[pd.DataFrame, Path]:
    if path_csv.exists():
        return pd.read_csv(path_csv), path_csv
    if path_parquet.exists():
        return pd.read_parquet(path_parquet), path_parquet
    raise FileNotFoundError(
        f"Missing frozen question set at {path_csv} (or fallback {path_parquet}). "
        "Build it first with scripts/benchmarks/build_fixed_question_sets.py."
    )


def _load_fixed_question_rows(
    cfg: dict,
    benchmark_mode: str,
    question_set_dir: Path | None,
) -> tuple[pd.DataFrame, int, Path]:
    question_set_dir = (question_set_dir or DEFAULT_QUESTION_SET_DIR).resolve()
    if benchmark_mode == "fixed_repair":
        path_csv = question_set_dir / "repaired_fixed_gpt54.csv"
        path_parquet = question_set_dir / "repaired_fixed_gpt54.parquet"
        required_cols = {"id", "question_clean", "question_noisy", "question_repaired", "answer_ref", "noise_type"}
    else:
        path_csv = question_set_dir / "noisy_fixed_gpt54.csv"
        path_parquet = question_set_dir / "noisy_fixed_gpt54.parquet"
        required_cols = {"id", "question_clean", "question_noisy", "answer_ref", "noise_type"}

    df, loaded_path = _read_question_set_file(path_csv, path_parquet)
    df = _normalize_question_set_df(df)
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{loaded_path.name} is missing required columns: {missing}")

    requested_n = int(cfg.get("n_examples", 50))
    if len(df) < requested_n:
        raise ValueError(
            f"{loaded_path.name} has only {len(df)} rows but config requests {requested_n}. "
            "Rebuild the frozen question set with a larger --n."
        )

    df = df.sort_values(["id", "noise_type"]).head(requested_n).reset_index(drop=True)
    return df, len(df), question_set_dir


def _build_end_to_end_rows(cfg: dict) -> tuple[pd.DataFrame, int]:
    df = load_samples(int(cfg.get("n_examples", 50)))
    plan = build_noise_plan(df, cfg)
    rows = []
    for row, noise_type in plan:
        rows.append(
            {
                "id": int(row["id"]),
                "question_clean": row["question"],
                "answer_ref": row["answer"],
                "source": row.get("source", ""),
                "noise_type": noise_type,
            }
        )
    return pd.DataFrame(rows), expected_noise_rows(len(df), cfg)


def discover_configs(config_dir: Path) -> list[str]:
    """Find YAML configs in execution order.

    Configs may specify an optional integer ``run_priority`` where lower values
    run first. Files without the field default to 100. Configs may also set
    ``enabled: false`` to opt out of default runs while remaining available for
    explicit ``--configs`` invocations.
    """
    configs = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = load_config(str(config_path))
            if cfg.get("enabled", True) is False:
                continue
            priority = int(cfg.get("run_priority", 100))
        except Exception:
            priority = 100
        configs.append((priority, config_path.name.lower(), str(config_path)))

    configs.sort()
    return [path for _, _, path in configs]


def model_label(model_name: str, quantization: str) -> str:
    short_name = model_name.split("/")[-1]
    return f"{short_name}" + (f" ({quantization})" if quantization != "none" else "")


def safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def model_paths(output_dir: Path, label: str) -> dict[str, Path]:
    stem = safe_label(label)
    return {
        "samples_csv": output_dir / f"samples_{stem}.csv",
        "result_json": output_dir / f"result_{stem}.json",
        "progress_json": output_dir / f"progress_{stem}.json",
    }


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def guard_resume_mode(paths: dict[str, Path], benchmark_mode: str, question_set_dir: Path | None):
    """Archive stale resumable files when switching benchmark modes."""
    desired_question_set = str(question_set_dir.resolve()) if question_set_dir is not None else None
    metadata = _read_json(paths["progress_json"]) or _read_json(paths["result_json"]) or {}

    existing_mode = metadata.get("benchmark_mode")
    existing_question_set = metadata.get("question_set_dir")
    if existing_mode is None and existing_question_set is None:
        return

    if existing_mode == benchmark_mode and existing_question_set == desired_question_set:
        return

    archived_any = False
    for key in ("samples_csv", "result_json", "progress_json"):
        path = paths[key]
        if not path.exists():
            continue
        archive_path = path.with_name(path.stem + f"_{benchmark_mode}_archived" + path.suffix)
        if archive_path.exists():
            archive_path.unlink()
        path.replace(archive_path)
        archived_any = True

    if archived_any:
        logger.warning(
            "Archived stale resumable files for %s because benchmark_mode/question_set changed.",
            paths["samples_csv"].stem,
        )


def get_bert_scorer() -> BERTScorer:
    global _bert_scorer
    if _bert_scorer is None:
        logger.info("Loading BERTScore model on CPU for incremental scoring")
        _bert_scorer = BERTScorer(lang="en", device="cpu")
    return _bert_scorer


def compute_bertscore_triplet(clean: str, noisy: str, repaired: str, ref: str) -> tuple[float, float, float]:
    scorer = get_bert_scorer()
    _, _, f1 = scorer.score([clean, noisy, repaired], [ref, ref, ref])
    vals = [float(v) for v in f1.tolist()]
    return vals[0], vals[1], vals[2]


def compute_intent_preservation_triplet(
    clean_question: str,
    noisy_question: str,
    repaired_question: str,
) -> tuple[float, float, float]:
    vals = compute_intent_preservation_scores(
        [clean_question, clean_question, clean_question],
        [clean_question, noisy_question, repaired_question],
    )
    return float(vals[0]), float(vals[1]), float(vals[2])


def write_json_atomic(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def write_csv_atomic(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def append_sample_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _has_resume_schema(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in SAMPLE_FIELDNAMES)


def _value_at(values, idx, default=""):
    if values is None:
        return default
    try:
        return values[idx]
    except (IndexError, TypeError):
        return default


def _default_sample_value(column: str):
    return "" if column in TEXT_SAMPLE_FIELDS else float("nan")


def _can_upgrade_in_place(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in BASE_SAMPLE_FIELDS)


def _upgrade_schema_in_place(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    changed = False
    for col in SAMPLE_FIELDNAMES:
        if col not in df.columns:
            df[col] = _default_sample_value(col)
            changed = True

    if list(df.columns) != SAMPLE_FIELDNAMES:
        changed = True

    return df[SAMPLE_FIELDNAMES].copy(), changed


def _backfill_reference_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    changed = False

    if "reference_answer" not in df.columns:
        return df, changed

    for metric in REFERENCE_METRICS:
        for pipeline in PIPELINES:
            metric_col = f"{metric}_{pipeline}"
            answer_col = f"answer_{pipeline}"
            if metric_col not in df.columns or answer_col not in df.columns:
                continue

            missing_mask = df[metric_col].isna()
            if not missing_mask.any():
                continue

            for idx in df.index[missing_mask]:
                prediction = str(df.at[idx, answer_col] or "")
                reference = str(df.at[idx, "reference_answer"] or "")
                if not prediction and not reference:
                    continue
                df.at[idx, metric_col] = compute_reference_metrics(prediction, reference)[metric]
            changed = True

    return df, changed


def _legacy_rows_from_result_json(path: Path) -> list[dict]:
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    detailed = payload.get("detailed_results")
    if not isinstance(detailed, list):
        return []

    rows = []
    for block in detailed:
        noise_type = block.get("noise_type", "")
        ids = block.get("question_ids", []) or []
        for idx, question_id in enumerate(ids):
            row = {
                "model": payload.get("label", ""),
                "config": payload.get("config", ""),
                "quantization": payload.get("quantization", "none"),
                "noise_type": noise_type,
                "question_id": question_id,
                "question_clean": _value_at(block.get("questions_clean"), idx, ""),
                "question_noisy": _value_at(block.get("questions_noisy"), idx, ""),
                "question_repaired": _value_at(block.get("questions_repaired"), idx, ""),
                "reference_answer": _value_at(block.get("reference_answers"), idx, ""),
                "answer_clean": _value_at(block.get("answers_clean"), idx, ""),
                "answer_noisy": _value_at(block.get("answers_noisy"), idx, ""),
                "answer_repaired": _value_at(block.get("answers_repaired"), idx, ""),
            }
            for metric in METRICS:
                metric_payload = block.get(metric) or {}
                for pipeline in PIPELINES:
                    row[f"{metric}_{pipeline}"] = _value_at(
                        metric_payload.get(pipeline), idx, float("nan")
                    )
            rows.append(row)
    return rows


def ensure_resume_csv(paths: dict[str, Path]):
    """Upgrade legacy sample CSVs to the resumable schema when possible."""
    samples_csv = paths["samples_csv"]
    if not samples_csv.exists() or samples_csv.stat().st_size == 0:
        return

    try:
        samples_df = pd.read_csv(samples_csv)
    except Exception:
        samples_df = pd.DataFrame()

    if _has_resume_schema(samples_df):
        upgraded_df, changed_schema = _upgrade_schema_in_place(samples_df)
        upgraded_df, changed_metrics = _backfill_reference_metrics(upgraded_df)
        if changed_schema or changed_metrics:
            upgraded_df.to_csv(samples_csv, index=False)
            logger.info("Upgraded comparison sample schema for %s", samples_csv.name)
        return

    if _can_upgrade_in_place(samples_df):
        upgraded_df, changed_schema = _upgrade_schema_in_place(samples_df)
        upgraded_df, changed_metrics = _backfill_reference_metrics(upgraded_df)
        if changed_schema or changed_metrics:
            upgraded_df.to_csv(samples_csv, index=False)
            logger.info("Upgraded comparison sample schema for %s", samples_csv.name)
        return

    backup_path = samples_csv.with_name(samples_csv.stem + "_legacy.csv")
    if not backup_path.exists():
        samples_csv.replace(backup_path)
    else:
        samples_csv.unlink()

    migrated_rows = _legacy_rows_from_result_json(paths["result_json"])
    if migrated_rows:
        migrated_df = pd.DataFrame(migrated_rows, columns=SAMPLE_FIELDNAMES)
        migrated_df, _ = _backfill_reference_metrics(migrated_df)
        migrated_df.to_csv(samples_csv, index=False)
        logger.info("Migrated legacy samples CSV for %s using %s",
                    samples_csv.name, paths["result_json"].name)
    else:
        logger.warning("Moved incompatible legacy samples CSV to %s; no resumable data could be recovered.",
                       backup_path)


def align_resume_csv_to_plan(path: Path, planned_pairs: set[tuple[str, int]]):
    if not path.exists() or path.stat().st_size == 0:
        return

    samples_df = load_samples_csv(path)
    if samples_df.empty:
        return

    normalized = samples_df.copy()
    normalized["question_id"] = normalized["question_id"].astype(int)
    row_pairs = set(zip(normalized["noise_type"], normalized["question_id"]))

    if row_pairs.issubset(planned_pairs):
        deduped = normalized.drop_duplicates(subset=["noise_type", "question_id"], keep="last")
        if len(deduped) != len(normalized):
            deduped.to_csv(path, index=False)
            logger.info("Deduplicated comparison samples for %s", path.name)
        return

    backup_path = path.with_name(path.stem + "_offplan.csv")
    if not backup_path.exists():
        normalized.to_csv(backup_path, index=False)

    filtered = normalized[
        normalized.apply(lambda r: (r["noise_type"], int(r["question_id"])) in planned_pairs, axis=1)
    ].drop_duplicates(subset=["noise_type", "question_id"], keep="last")

    filtered.to_csv(path, index=False)
    logger.info(
        "Filtered %s to the current noise assignment plan (%d kept, %d archived to %s)",
        path.name,
        len(filtered),
        len(normalized) - len(filtered),
        backup_path.name,
    )


def load_completed_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    completed: set[tuple[str, int]] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if not _sample_row_complete(row):
                    continue
                completed.add((row["noise_type"], int(row["question_id"])))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def _sample_row_complete(row: dict[str, str]) -> bool:
    for field in BASE_SAMPLE_FIELDS:
        value = row.get(field, "")
        if value is None or not str(value).strip():
            return False

    for metric in METRICS:
        for pipeline in PIPELINES:
            field = f"{metric}_{pipeline}"
            value = row.get(field, "")
            normalized = str(value).strip().lower() if value is not None else ""
            if normalized in {"", "nan", "none"}:
                return False

    return True


def load_samples_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=SAMPLE_FIELDNAMES)
    df = pd.read_csv(path)
    if _can_upgrade_in_place(df):
        df, _ = _upgrade_schema_in_place(df)
    return df


def write_progress(path: Path, payload: dict):
    write_json_atomic(path, payload)


def upsert_summary(all_summaries: list[dict], summary: dict):
    key = (summary.get("config"), summary.get("label"))
    for idx, existing in enumerate(all_summaries):
        existing_key = (existing.get("config"), existing.get("label"))
        if existing_key == key:
            all_summaries[idx] = summary
            return
    all_summaries.append(summary)


def summarize_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0][:300]


def build_detailed_results(samples_df: pd.DataFrame) -> list[dict]:
    if samples_df.empty:
        return []

    detailed = []
    grouped = samples_df.sort_values(["noise_type", "question_id"]).groupby("noise_type", sort=False)
    for noise_type, grp in grouped:
        row = {
            "noise_type": noise_type,
            "question_ids": grp["question_id"].astype(int).tolist(),
            "questions_clean": grp["question_clean"].tolist(),
            "questions_noisy": grp["question_noisy"].tolist(),
            "questions_repaired": grp["question_repaired"].tolist(),
            "answers_clean": grp["answer_clean"].tolist(),
            "answers_noisy": grp["answer_noisy"].tolist(),
            "answers_repaired": grp["answer_repaired"].tolist(),
            "reference_answers": grp["reference_answer"].tolist(),
        }
        for metric in METRICS:
            row[metric] = {
                pipeline: grp[f"{metric}_{pipeline}"].astype(float).tolist()
                for pipeline in PIPELINES
            }
        detailed.append(row)
    return detailed


def build_summary_from_samples(
    samples_df: pd.DataFrame,
    cfg: dict,
    config_path: str,
    label: str,
    rows_expected: int,
    status: str,
    benchmark_mode: str,
    question_set_dir: Path | None = None,
    error: str | None = None,
) -> dict:
    model_name = cfg["answer_model"]
    quant = quantization_label(cfg)
    noise_types = cfg["noise_types"]

    summary = {
        "model": model_name,
        "label": label,
        "config": Path(config_path).name,
        "quantization": quant,
        "n_examples": cfg.get("n_examples", 50),
        "n_noise_types": len(noise_types),
        "noise_types": noise_types,
        "noise_assignment": cfg.get("noise_assignment", "cartesian"),
        "benchmark_mode": benchmark_mode,
        "question_set_dir": str(question_set_dir) if question_set_dir is not None else None,
        "status": status,
        "rows_completed": int(len(samples_df)),
        "rows_expected": int(rows_expected),
        "error": error,
        "detailed_results": [],
        "per_noise_type": [],
    }

    if samples_df.empty:
        return summary

    for metric in METRICS:
        for pipeline in PIPELINES:
            col = f"{metric}_{pipeline}"
            vals = samples_df[col].dropna().astype(float).to_numpy()
            summary[f"{metric}_{pipeline}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            summary[f"{metric}_{pipeline}_std"] = float(np.std(vals)) if len(vals) else float("nan")

    for metric in METRICS:
        clean_mean = summary.get(f"{metric}_clean_mean", float("nan"))
        noisy_mean = summary.get(f"{metric}_noisy_mean", float("nan"))
        repaired_mean = summary.get(f"{metric}_repaired_mean", float("nan"))
        recovery_stats = compute_recovery_statistics(
            clean_mean,
            noisy_mean,
            repaired_mean,
            metric_name=metric,
        )
        summary[f"{metric}_degradation"] = recovery_stats["degradation"]
        summary[f"{metric}_recovery"] = recovery_stats["recovery"]
        summary[f"{metric}_recovery_ratio"] = recovery_stats["recovery_ratio"]

    per_noise = []
    for noise_type, grp in samples_df.groupby("noise_type", sort=False):
        row = {"noise_type": noise_type}
        for metric in METRICS:
            for pipeline in PIPELINES:
                col = f"{metric}_{pipeline}"
                vals = grp[col].dropna().astype(float).to_numpy()
                row[f"{metric}_{pipeline}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        per_noise.append(row)

    summary["per_noise_type"] = per_noise
    summary["detailed_results"] = build_detailed_results(samples_df)
    return summary


def format_status(summary: dict) -> str:
    status = str(summary.get("status", "unknown")).upper()
    rows_completed = summary.get("rows_completed")
    rows_expected = summary.get("rows_expected")
    error = summary.get("error")

    if status == "COMPLETED":
        return "COMPLETED"
    if rows_completed is not None and rows_expected is not None:
        base = f"{status} {rows_completed}/{rows_expected}"
    else:
        base = status
    if error:
        return f"{base}: {error[:80]}"
    return base


def _format_metric_value(value, precision: int) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{precision}f}"


def write_comparison_outputs(all_summaries: list[dict], output_dir: Path) -> pd.DataFrame:
    comparison_rows = []
    for s in all_summaries:
        row = {
            "Model": s.get("label", s.get("config", "UNKNOWN")),
            "Mode": s.get("benchmark_mode", ""),
            "Status": format_status(s),
            "Quant": s.get("quantization", ""),
            "N": s.get("n_examples", ""),
            "Rows": f"{s.get('rows_completed', 0)}/{s.get('rows_expected', 0)}",
            "Runtime_min": f"{s.get('runtime_seconds', 0) / 60:.1f}",
        }

        for metric in METRICS:
            for pipeline in PIPELINES:
                key = f"{metric}_{pipeline}_mean"
                if key in s:
                    precision = METRIC_PRECISION.get(metric, 2)
                    row[f"{metric}_{pipeline}"] = _format_metric_value(s[key], precision)
            ratio_key = f"{metric}_recovery_ratio"
            if ratio_key in s:
                ratio = s[ratio_key]
                row[f"{metric}_recovery%"] = f"{ratio:.0%}" if not np.isnan(ratio) else "N/A"

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    full_json_path = output_dir / "model_comparison_full.json"
    csv_path = output_dir / "model_comparison.csv"

    try:
        write_json_atomic(full_json_path, all_summaries)
    except PermissionError:
        fallback_json = output_dir / "model_comparison_full.live.json"
        logger.warning(
            "Could not update %s because it is locked; writing %s instead",
            full_json_path.name,
            fallback_json.name,
        )
        write_json_atomic(fallback_json, all_summaries)

    try:
        write_csv_atomic(csv_path, comparison_df)
    except PermissionError:
        fallback_csv = output_dir / "model_comparison.live.csv"
        logger.warning(
            "Could not update %s because it is locked; writing %s instead",
            csv_path.name,
            fallback_csv.name,
        )
        write_csv_atomic(fallback_csv, comparison_df)

    logger.info("Saved comparison outputs to %s", output_dir)
    return comparison_df


def process_sample(row: pd.Series, prompts: dict, cfg: dict, label: str, benchmark_mode: str) -> dict:
    question = row["question_clean"]
    reference = row["answer_ref"]
    question_id = int(row["id"])
    noise_type = row["noise_type"]

    if benchmark_mode == "end_to_end":
        noisy_q = generate_noisy_variant(question, noise_type, prompts, cfg)
        repaired_q = repair_question(noisy_q, prompts, cfg)
    elif benchmark_mode == "fixed_repair":
        noisy_q = row["question_noisy"]
        repaired_q = row["question_repaired"]
    elif benchmark_mode == "self_repair":
        noisy_q = row["question_noisy"]
        repaired_q = repair_question(noisy_q, prompts, cfg)
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unknown benchmark_mode: {benchmark_mode}")

    answer_clean = answer_question(question, prompts, cfg)
    answer_noisy = answer_question(noisy_q, prompts, cfg)
    answer_repaired = answer_question(repaired_q, prompts, cfg)

    lexical_clean = compute_reference_metrics(answer_clean, reference)
    lexical_noisy = compute_reference_metrics(answer_noisy, reference)
    lexical_repaired = compute_reference_metrics(answer_repaired, reference)

    bert_clean, bert_noisy, bert_repaired = compute_bertscore_triplet(
        answer_clean, answer_noisy, answer_repaired, reference
    )
    intent_clean, intent_noisy, intent_repaired = compute_intent_preservation_triplet(
        question, noisy_q, repaired_q
    )

    geval_clean = geval_answer(reference, answer_clean, prompts, cfg, question=question)
    geval_noisy = geval_answer(reference, answer_noisy, prompts, cfg, question=noisy_q)
    geval_repaired = geval_answer(reference, answer_repaired, prompts, cfg, question=repaired_q)

    return {
        "model": label,
        "config": Path(cfg.get("_config_path", "")).name if cfg.get("_config_path") else "",
        "quantization": quantization_label(cfg),
        "noise_type": noise_type,
        "question_id": question_id,
        "question_clean": question,
        "question_noisy": noisy_q,
        "question_repaired": repaired_q,
        "reference_answer": reference,
        "answer_clean": answer_clean,
        "answer_noisy": answer_noisy,
        "answer_repaired": answer_repaired,
        **{f"{metric}_clean": lexical_clean[metric] for metric in REFERENCE_METRICS},
        **{f"{metric}_noisy": lexical_noisy[metric] for metric in REFERENCE_METRICS},
        **{f"{metric}_repaired": lexical_repaired[metric] for metric in REFERENCE_METRICS},
        "bertscore_clean": bert_clean,
        "bertscore_noisy": bert_noisy,
        "bertscore_repaired": bert_repaired,
        "intent_preservation_clean": intent_clean,
        "intent_preservation_noisy": intent_noisy,
        "intent_preservation_repaired": intent_repaired,
        "geval_clean": geval_clean,
        "geval_noisy": geval_noisy,
        "geval_repaired": geval_repaired,
    }


def run_single_model(
    config_path: str,
    output_dir: Path,
    benchmark_mode: str = "end_to_end",
    question_set_dir: Path | None = None,
    on_progress=None,
) -> tuple[dict, bool]:
    """Run or resume one model config, persisting each completed sample row."""
    set_active_config(config_path)
    cfg = load_config(config_path)
    cfg["_config_path"] = config_path
    prompts = load_prompts()
    model_name = cfg["answer_model"]
    quant = quantization_label(cfg)
    label = model_label(model_name, quant)
    paths = model_paths(output_dir, label)
    desired_question_set_dir = (
        None if benchmark_mode == "end_to_end" else (question_set_dir or DEFAULT_QUESTION_SET_DIR).resolve()
    )
    guard_resume_mode(paths, benchmark_mode, desired_question_set_dir)
    ensure_resume_csv(paths)

    logger.info("=" * 70)
    logger.info("  MODEL: %s", label)
    logger.info("  Config: %s", Path(config_path).name)
    logger.info("  Benchmark mode: %s", benchmark_mode)
    if desired_question_set_dir is not None:
        logger.info("  Frozen question set: %s", desired_question_set_dir)
    logger.info("=" * 70)

    if benchmark_mode == "end_to_end":
        plan_df, expected_rows = _build_end_to_end_rows(cfg)
        logger.info("Noise assignment counts: %s", dict(plan_df["noise_type"].value_counts(sort=False)))
        resolved_question_set_dir = None
    else:
        plan_df, expected_rows, resolved_question_set_dir = _load_fixed_question_rows(
            cfg,
            benchmark_mode,
            desired_question_set_dir,
        )
        logger.info("Frozen noise assignment counts: %s", dict(plan_df["noise_type"].value_counts(sort=False)))

    completed_pairs = load_completed_pairs(paths["samples_csv"])

    planned_pairs = {
        (str(row["noise_type"]), int(row["id"]))
        for _, row in plan_df.iterrows()
    }
    align_resume_csv_to_plan(paths["samples_csv"], planned_pairs)
    completed_pairs = load_completed_pairs(paths["samples_csv"])

    if completed_pairs:
        logger.info("Resuming %s with %d/%d completed rows from %s",
                    label, len(completed_pairs), expected_rows, paths["samples_csv"])

    interrupted = False
    error: str | None = None

    def emit_partial_summary(status_override: str = "running", error_override: str | None = None):
        if on_progress is None:
            return
        partial_samples_df = load_samples_csv(paths["samples_csv"])
        partial_summary = build_summary_from_samples(
            partial_samples_df,
            cfg,
            config_path,
            label,
            rows_expected=expected_rows,
            status=status_override,
            benchmark_mode=benchmark_mode,
            question_set_dir=resolved_question_set_dir if benchmark_mode != "end_to_end" else None,
            error=error_override,
        )
        on_progress(partial_summary)

    write_progress(paths["progress_json"], {
        "model": label,
        "config": Path(config_path).name,
        "benchmark_mode": benchmark_mode,
        "question_set_dir": str(resolved_question_set_dir) if benchmark_mode != "end_to_end" else None,
        "status": "running",
        "rows_completed": len(completed_pairs),
        "rows_expected": expected_rows,
        "samples_csv": str(paths["samples_csv"]),
        "updated_at_epoch": time.time(),
    })
    emit_partial_summary(status_override="running")

    try:
        for _, row in plan_df.iterrows():
            pair = (str(row["noise_type"]), int(row["id"]))
            if pair in completed_pairs:
                continue

            sample_row = process_sample(row, prompts, cfg, label, benchmark_mode)
            append_sample_row(paths["samples_csv"], sample_row)
            completed_pairs.add(pair)

            write_progress(paths["progress_json"], {
                "model": label,
                "config": Path(config_path).name,
                "benchmark_mode": benchmark_mode,
                "question_set_dir": str(resolved_question_set_dir) if benchmark_mode != "end_to_end" else None,
                "status": "running",
                "rows_completed": len(completed_pairs),
                "rows_expected": expected_rows,
                "last_noise_type": row["noise_type"],
                "last_question_id": int(row["id"]),
                "samples_csv": str(paths["samples_csv"]),
                "updated_at_epoch": time.time(),
            })
            emit_partial_summary(status_override="running")

            # Some very large local models are stable for a full sample but
            # become flaky when the same loaded instance is reused across many
            # benchmark rows. This opt-in hook trades speed for robustness.
            if cfg.get("unload_local_model_after_sample") and cfg.get("backend", "local") == "local":
                _unload_models()
    except KeyboardInterrupt:
        interrupted = True
        error = "Interrupted by user"
        logger.warning("Interrupted %s; partial CSV has been saved.", label)
    except Exception as exc:  # pragma: no cover - diagnostic path
        error = summarize_error(exc)
        logger.exception("Model failed after partial progress: %s", label)

    samples_df = load_samples_csv(paths["samples_csv"])
    if len(samples_df) == expected_rows and error is None:
        status = "completed"
    elif len(samples_df) > 0:
        status = "partial"
    else:
        status = "failed"

    summary = build_summary_from_samples(
        samples_df,
        cfg,
        config_path,
        label,
        rows_expected=expected_rows,
        status=status,
        benchmark_mode=benchmark_mode,
        question_set_dir=resolved_question_set_dir if benchmark_mode != "end_to_end" else None,
        error=error,
    )
    write_json_atomic(paths["result_json"], summary)
    write_progress(paths["progress_json"], {
        "model": label,
        "config": Path(config_path).name,
        "benchmark_mode": benchmark_mode,
        "question_set_dir": str(resolved_question_set_dir) if benchmark_mode != "end_to_end" else None,
        "status": status,
        "rows_completed": summary["rows_completed"],
        "rows_expected": summary["rows_expected"],
        "error": error,
        "samples_csv": str(paths["samples_csv"]),
        "result_json": str(paths["result_json"]),
        "updated_at_epoch": time.time(),
    })
    return summary, interrupted


def _unload_models():
    """Free GPU memory by unloading cached models."""
    try:
        import torch

        unload_local_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Freed GPU memory (%.2f GB allocated)",
                        torch.cuda.memory_allocated() / 1024**3)
    except Exception:
        pass


def run_comparison(
    config_paths: list[str],
    output_dir: Path,
    benchmark_mode: str = "end_to_end",
    question_set_dir: Path | None = None,
):
    """Run all model configs and produce/update a comparison table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict] = []
    interrupted = False

    for i, config_path in enumerate(config_paths):
        logger.info("\n[%d/%d] Running config: %s", i + 1, len(config_paths), config_path)
        start = time.time()

        def on_progress(partial_summary: dict, _start=start):
            partial = dict(partial_summary)
            partial["runtime_seconds"] = time.time() - _start
            upsert_summary(all_summaries, partial)
            write_comparison_outputs(all_summaries, output_dir)

        summary, was_interrupted = run_single_model(
            config_path,
            output_dir,
            benchmark_mode=benchmark_mode,
            question_set_dir=question_set_dir,
            on_progress=on_progress,
        )
        summary["runtime_seconds"] = time.time() - start
        upsert_summary(all_summaries, summary)
        write_json_atomic(model_paths(output_dir, summary["label"])["result_json"], summary)
        comparison_df = write_comparison_outputs(all_summaries, output_dir)
        _unload_models()

        if was_interrupted:
            interrupted = True
            break

    print(f"\n{'='*100}")
    print("  MODEL COMPARISON RESULTS")
    print(f"{'='*100}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*100}\n")

    if interrupted:
        logger.warning("Run interrupted after saving partial progress to CSV/JSON.")

    return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-model comparison")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config files to run (default: all in configs/models/)")
    parser.add_argument(
        "--benchmark-mode",
        choices=BENCHMARK_MODES,
        default="end_to_end",
        help="Benchmark mode: current end-to-end flow, fixed repaired questions, or self-repair on shared noisy questions",
    )
    parser.add_argument(
        "--question-set-dir",
        type=str,
        default=None,
        help="Directory containing frozen noisy/repaired question sets for fixed_repair or self_repair modes",
    )
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached LLM responses before running")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configs without executing")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default depends on --benchmark-mode)")
    args = parser.parse_args()

    if args.configs:
        config_paths = [str(Path(c).resolve()) for c in args.configs]
    else:
        config_paths = discover_configs(PROJECT_ROOT / "configs" / "models")

    if not config_paths:
        logger.error("No config files found.")
        sys.exit(1)

    logger.info("Found %d model configs:", len(config_paths))
    for c in config_paths:
        logger.info("  - %s", Path(c).name)

    if args.dry_run:
        print("\nDry run — would execute the above configs. Exiting.")
        sys.exit(0)

    if args.clear_cache:
        clear_cache(config_paths)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.benchmark_mode)
    question_set_dir = Path(args.question_set_dir) if args.question_set_dir else None
    run_comparison(
        config_paths,
        output_dir,
        benchmark_mode=args.benchmark_mode,
        question_set_dir=question_set_dir,
    )
