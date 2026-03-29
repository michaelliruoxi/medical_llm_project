"""Run the robustness pipeline across multiple model configs and produce a comparison table.

This version is resumable and fail-safe for long runs:
- each completed sample row is appended to CSV immediately
- per-model progress is written to JSON continuously
- reruns skip rows already present in the model CSV

Usage:
    python scripts/run_model_comparison.py
    python scripts/run_model_comparison.py --configs configs/models/mistral_7b.yaml
    python scripts/run_model_comparison.py --dry-run
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from bert_score import BERTScorer

from src.answer import answer_question
from src.judge import judge_answer
from src.metrics import compute_reference_metrics
from src.noise import generate_noisy_variant
from src.repair import repair_question
from src.utils import load_config, load_prompts, setup_logging, unload_local_models


logger = setup_logging()

PIPELINES = ("clean", "noisy", "repaired")
REFERENCE_METRICS = ("bleu", "chrf", "rouge_l", "token_f1", "exact_match")
METRICS = REFERENCE_METRICS + ("bertscore", "judge")
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
METRIC_PRECISION = {"bertscore": 4}

_bert_scorer: BERTScorer | None = None


def clear_cache():
    """Remove cached LLM responses."""
    cache_dir = PROJECT_ROOT / "data" / "outputs" / "cache"
    if cache_dir.exists():
        files = list(cache_dir.glob("*.json"))
        for f in files:
            try:
                f.unlink()
            except PermissionError:
                pass
        logger.info("Cleared %d cached responses", len(files))


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


def write_json_atomic(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
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


def load_completed_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    completed: set[tuple[str, int]] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((row["noise_type"], int(row["question_id"])))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def load_samples_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=SAMPLE_FIELDNAMES)
    df = pd.read_csv(path)
    if _can_upgrade_in_place(df):
        df, _ = _upgrade_schema_in_place(df)
    return df


def write_progress(path: Path, payload: dict):
    write_json_atomic(path, payload)


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
    status: str,
    error: str | None = None,
) -> dict:
    model_name = cfg["answer_model"]
    quant = cfg.get("quantization", "none")
    noise_types = cfg["noise_types"]
    expected_rows = cfg.get("n_examples", 50) * len(noise_types)

    summary = {
        "model": model_name,
        "label": label,
        "config": Path(config_path).name,
        "quantization": quant,
        "n_examples": cfg.get("n_examples", 50),
        "n_noise_types": len(noise_types),
        "noise_types": noise_types,
        "status": status,
        "rows_completed": int(len(samples_df)),
        "rows_expected": int(expected_rows),
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
        degradation = clean_mean - noisy_mean
        recovery = repaired_mean - noisy_mean
        ratio = recovery / degradation if not np.isnan(degradation) and degradation != 0 else float("nan")
        summary[f"{metric}_degradation"] = degradation
        summary[f"{metric}_recovery"] = recovery
        summary[f"{metric}_recovery_ratio"] = ratio

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


def write_comparison_outputs(all_summaries: list[dict], output_dir: Path) -> pd.DataFrame:
    comparison_rows = []
    for s in all_summaries:
        row = {
            "Model": s.get("label", s.get("config", "UNKNOWN")),
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
                    row[f"{metric}_{pipeline}"] = f"{s[key]:.{precision}f}"
            ratio_key = f"{metric}_recovery_ratio"
            if ratio_key in s:
                ratio = s[ratio_key]
                row[f"{metric}_recovery%"] = f"{ratio:.0%}" if not np.isnan(ratio) else "N/A"

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    write_json_atomic(output_dir / "model_comparison_full.json", all_summaries)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info("Saved comparison outputs to %s", output_dir)
    return comparison_df


def process_sample(row: pd.Series, noise_type: str, prompts: dict, cfg: dict, label: str) -> dict:
    question = row["question"]
    reference = row["answer"]
    question_id = int(row["id"])

    noisy_q = generate_noisy_variant(question, noise_type, prompts, cfg)
    answer_clean = answer_question(question, prompts, cfg)
    answer_noisy = answer_question(noisy_q, prompts, cfg)
    repaired_q = repair_question(noisy_q, prompts, cfg)
    answer_repaired = answer_question(repaired_q, prompts, cfg)

    lexical_clean = compute_reference_metrics(answer_clean, reference)
    lexical_noisy = compute_reference_metrics(answer_noisy, reference)
    lexical_repaired = compute_reference_metrics(answer_repaired, reference)

    bert_clean, bert_noisy, bert_repaired = compute_bertscore_triplet(
        answer_clean, answer_noisy, answer_repaired, reference
    )

    judge_clean = judge_answer(reference, answer_clean, prompts, cfg, question=question)
    judge_noisy = judge_answer(reference, answer_noisy, prompts, cfg, question=question)
    judge_repaired = judge_answer(reference, answer_repaired, prompts, cfg, question=question)

    return {
        "model": label,
        "config": Path(cfg.get("_config_path", "")).name if cfg.get("_config_path") else "",
        "quantization": cfg.get("quantization", "none"),
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
        "judge_clean": judge_clean,
        "judge_noisy": judge_noisy,
        "judge_repaired": judge_repaired,
    }


def run_single_model(config_path: str, output_dir: Path) -> tuple[dict, bool]:
    """Run or resume one model config, persisting each completed sample row."""
    cfg = load_config(config_path)
    cfg["_config_path"] = config_path
    prompts = load_prompts()
    n = cfg.get("n_examples", 50)
    model_name = cfg["answer_model"]
    quant = cfg.get("quantization", "none")
    label = model_label(model_name, quant)
    paths = model_paths(output_dir, label)
    ensure_resume_csv(paths)

    logger.info("=" * 70)
    logger.info("  MODEL: %s", label)
    logger.info("  Config: %s", Path(config_path).name)
    logger.info("=" * 70)

    df = load_samples(n)
    noise_types = cfg["noise_types"]
    expected_rows = len(df) * len(noise_types)
    completed_pairs = load_completed_pairs(paths["samples_csv"])

    if completed_pairs:
        logger.info("Resuming %s with %d/%d completed rows from %s",
                    label, len(completed_pairs), expected_rows, paths["samples_csv"])

    interrupted = False
    error: str | None = None

    write_progress(paths["progress_json"], {
        "model": label,
        "config": Path(config_path).name,
        "status": "running",
        "rows_completed": len(completed_pairs),
        "rows_expected": expected_rows,
        "samples_csv": str(paths["samples_csv"]),
        "updated_at_epoch": time.time(),
    })

    try:
        for noise_type in noise_types:
            logger.info("--- Noise type: %s ---", noise_type)
            for _, row in df.iterrows():
                pair = (noise_type, int(row["id"]))
                if pair in completed_pairs:
                    continue

                sample_row = process_sample(row, noise_type, prompts, cfg, label)
                append_sample_row(paths["samples_csv"], sample_row)
                completed_pairs.add(pair)

                write_progress(paths["progress_json"], {
                    "model": label,
                    "config": Path(config_path).name,
                    "status": "running",
                    "rows_completed": len(completed_pairs),
                    "rows_expected": expected_rows,
                    "last_noise_type": noise_type,
                    "last_question_id": int(row["id"]),
                    "samples_csv": str(paths["samples_csv"]),
                    "updated_at_epoch": time.time(),
                })
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

    summary = build_summary_from_samples(samples_df, cfg, config_path, label, status=status, error=error)
    write_json_atomic(paths["result_json"], summary)
    write_progress(paths["progress_json"], {
        "model": label,
        "config": Path(config_path).name,
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


def run_comparison(config_paths: list[str], output_dir: Path):
    """Run all model configs and produce/update a comparison table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict] = []
    interrupted = False

    for i, config_path in enumerate(config_paths):
        logger.info("\n[%d/%d] Running config: %s", i + 1, len(config_paths), config_path)
        start = time.time()
        summary, was_interrupted = run_single_model(config_path, output_dir)
        summary["runtime_seconds"] = time.time() - start
        write_json_atomic(model_paths(output_dir, summary["label"])["result_json"], summary)
        all_summaries.append(summary)
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
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached LLM responses before running")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configs without executing")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/outputs/comparison)")
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
        clear_cache()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "outputs" / "comparison"
    run_comparison(config_paths, output_dir)
