"""Build frozen noisy and repaired question sets for controlled benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.noise import generate_noisy_variant
from src.noise_schedule import build_noise_plan, summarize_noise_plan
from src.repair import repair_question
from src.resume import build_completed_lookup, row_key, write_parquet_atomic
from src.utils import get_token_tracker, load_config, load_prompts, set_active_config, setup_logging


logger = setup_logging()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "fixed_question_sets_gpt54.yaml"


def write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def write_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def read_table(path_csv: Path, path_parquet: Path) -> pd.DataFrame:
    if path_csv.exists():
        return pd.read_csv(path_csv)
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    return pd.DataFrame()


def write_question_set(df: pd.DataFrame, path_csv: Path, path_parquet: Path):
    write_csv_atomic(df, path_csv)
    write_parquet_atomic(df, path_parquet)


def load_clean_samples(n: int) -> pd.DataFrame:
    cleaned = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.parquet"
    if cleaned.exists():
        df = pd.read_parquet(cleaned)
    else:
        cleaned_csv = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.csv"
        if not cleaned_csv.exists():
            raise FileNotFoundError("No cleaned MedQuAD dataset found in data/processed")
        df = pd.read_csv(cleaned_csv)

    df = df.head(n).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def build_clean_records(df: pd.DataFrame, cfg: dict) -> list[dict]:
    plan = build_noise_plan(df, cfg)
    records = []
    for row, noise_type in plan:
        records.append(
            {
                "id": int(row["id"]),
                "question_clean": row["question"],
                "answer_ref": row["answer"],
                "source": row.get("source", ""),
                "noise_type": noise_type,
            }
        )
    return records


def emit_progress(path: Path, *, stage: str, completed: int, expected: int, extra: dict | None = None):
    payload = {
        "stage": stage,
        "rows_completed": int(completed),
        "rows_expected": int(expected),
        "updated_at_epoch": time.time(),
    }
    if extra:
        payload.update(extra)
    write_json_atomic(path, payload)


def build_fixed_question_sets(config_path: str, n_examples: int | None = None):
    set_active_config(config_path)
    cfg = load_config(config_path)
    prompts = load_prompts()

    requested_n = int(n_examples or cfg.get("n_examples", 50))
    df = load_clean_samples(requested_n)
    clean_records = build_clean_records(df, cfg)
    noise_counts = summarize_noise_plan(build_noise_plan(df, cfg))

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    outputs_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    clean_csv_path = processed_dir / "clean_fixed.csv"
    noisy_csv_path = processed_dir / "noisy_fixed_gpt54.csv"
    repaired_csv_path = processed_dir / "repaired_fixed_gpt54.csv"
    clean_path = processed_dir / "clean_fixed.parquet"
    noisy_path = processed_dir / "noisy_fixed_gpt54.parquet"
    repaired_path = processed_dir / "repaired_fixed_gpt54.parquet"
    manifest_path = outputs_dir / "question_set_manifest.json"
    progress_path = outputs_dir / "question_set_progress.json"

    clean_df = pd.DataFrame(clean_records)
    write_question_set(clean_df, clean_csv_path, clean_path)
    logger.info("Saved %d clean benchmark rows to %s", len(clean_df), clean_csv_path)
    logger.info("Noise assignment counts: %s", noise_counts)

    expected_rows = len(clean_records)

    existing_noisy_df = read_table(noisy_csv_path, noisy_path)
    noisy_lookup = build_completed_lookup(existing_noisy_df, ["id", "noise_type"], ["question_noisy"])
    if noisy_lookup:
        logger.info("Resuming fixed noisy-question generation with %d/%d rows from %s",
                    len(noisy_lookup), expected_rows, noisy_csv_path if noisy_csv_path.exists() else noisy_path)

    noisy_rows: list[dict] = []
    emit_progress(progress_path, stage="noise", completed=len(noisy_lookup), expected=expected_rows)
    for record in clean_records:
        key = row_key(record, ["id", "noise_type"])
        if key in noisy_lookup:
            noisy_rows.append(noisy_lookup[key])
            continue

        noisy_question = generate_noisy_variant(
            record["question_clean"],
            record["noise_type"],
            prompts,
            cfg,
        )
        noisy_record = dict(record)
        noisy_record["question_noisy"] = noisy_question
        noisy_rows.append(noisy_record)
        write_question_set(pd.DataFrame(noisy_rows), noisy_csv_path, noisy_path)
        emit_progress(
            progress_path,
            stage="noise",
            completed=len(noisy_rows),
            expected=expected_rows,
            extra={"last_noise_type": record["noise_type"], "last_question_id": int(record["id"])},
        )

    noisy_df = pd.DataFrame(noisy_rows)
    write_question_set(noisy_df, noisy_csv_path, noisy_path)
    logger.info("Saved %d frozen noisy questions to %s", len(noisy_df), noisy_csv_path)

    existing_repaired_df = read_table(repaired_csv_path, repaired_path)
    repaired_lookup = build_completed_lookup(existing_repaired_df, ["id", "noise_type"], ["question_repaired"])
    if repaired_lookup:
        logger.info("Resuming fixed repaired-question generation with %d/%d rows from %s",
                    len(repaired_lookup), expected_rows, repaired_csv_path if repaired_csv_path.exists() else repaired_path)

    repaired_rows: list[dict] = []
    emit_progress(progress_path, stage="repair", completed=len(repaired_lookup), expected=expected_rows)
    noisy_records = noisy_df.sort_values(["id", "noise_type"]).to_dict(orient="records")
    for record in noisy_records:
        key = row_key(record, ["id", "noise_type"])
        if key in repaired_lookup:
            repaired_rows.append(repaired_lookup[key])
            continue

        repaired_question = repair_question(record["question_noisy"], prompts, cfg)
        repaired_record = dict(record)
        repaired_record["question_repaired"] = repaired_question
        repaired_rows.append(repaired_record)
        write_question_set(pd.DataFrame(repaired_rows), repaired_csv_path, repaired_path)
        emit_progress(
            progress_path,
            stage="repair",
            completed=len(repaired_rows),
            expected=expected_rows,
            extra={"last_noise_type": record["noise_type"], "last_question_id": int(record["id"])},
        )

    repaired_df = pd.DataFrame(repaired_rows)
    write_question_set(repaired_df, repaired_csv_path, repaired_path)
    emit_progress(progress_path, stage="completed", completed=len(repaired_df), expected=expected_rows)
    logger.info("Saved %d frozen repaired questions to %s", len(repaired_df), repaired_csv_path)

    manifest = {
        "config": str(Path(config_path).name),
        "clean_rows": len(clean_df),
        "noise_rows": len(noisy_df),
        "repaired_rows": len(repaired_df),
        "noise_assignment": cfg.get("noise_assignment", "cartesian"),
        "noise_types": cfg["noise_types"],
        "noise_model": cfg["noise_model"],
        "repair_model": cfg["repair_model"],
        "clean_path": str(clean_csv_path),
        "noisy_path": str(noisy_csv_path),
        "repaired_path": str(repaired_csv_path),
        "clean_parquet_path": str(clean_path),
        "noisy_parquet_path": str(noisy_path),
        "repaired_parquet_path": str(repaired_path),
        "noise_counts": noise_counts,
        "token_usage": get_token_tracker().summary(),
    }
    write_json_atomic(manifest_path, manifest)
    logger.info("Saved fixed-question-set manifest to %s", manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frozen noisy and repaired question sets")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Config YAML for the fixed question-set generator",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override the number of clean MedQuAD examples to freeze",
    )
    args = parser.parse_args()

    build_fixed_question_sets(args.config, n_examples=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
