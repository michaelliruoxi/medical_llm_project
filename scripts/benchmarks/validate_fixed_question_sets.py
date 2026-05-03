"""Validate frozen noisy/repaired question sets with question_validation.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.question_validation import validate_generated_question
from src.utils import setup_logging


logger = setup_logging()
DEFAULT_QUESTION_SET_DIR = PROJECT_ROOT / "data" / "processed" / "benchmarks" / "fixed_question_sets_gpt54"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "benchmarks" / "fixed_question_set_validation"


def _read_table(path_csv: Path, path_parquet: Path) -> pd.DataFrame:
    if path_csv.exists():
        return pd.read_csv(path_csv)
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    raise FileNotFoundError(f"Missing table: {path_csv} (or {path_parquet})")


def _validate_rows(df: pd.DataFrame, *, stage: str, text_col: str, source_col: str) -> tuple[list[dict], list[dict]]:
    valid_rows: list[dict] = []
    invalid_rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        candidate, error = validate_generated_question(
            row.get(text_col, ""),
            stage=stage,
            source_question=row.get(source_col, ""),
        )
        enriched = dict(row)
        enriched[f"{text_col}_normalized"] = candidate
        if error:
            enriched["stage"] = stage
            enriched["validation_error"] = error
            invalid_rows.append(enriched)
        else:
            valid_rows.append(enriched)
    return valid_rows, invalid_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen noisy/repaired question sets")
    parser.add_argument(
        "--question-set-dir",
        type=str,
        default=str(DEFAULT_QUESTION_SET_DIR),
        help="Directory containing clean/noisy/repaired frozen question tables",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for validation reports",
    )
    args = parser.parse_args()

    question_set_dir = Path(args.question_set_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_df = _read_table(
        question_set_dir / "noisy_fixed_gpt54.csv",
        question_set_dir / "noisy_fixed_gpt54.parquet",
    )
    repaired_df = _read_table(
        question_set_dir / "repaired_fixed_gpt54.csv",
        question_set_dir / "repaired_fixed_gpt54.parquet",
    )

    _, noisy_invalid = _validate_rows(
        noisy_df,
        stage="noise",
        text_col="question_noisy",
        source_col="question_clean",
    )
    _, repaired_invalid = _validate_rows(
        repaired_df,
        stage="repair",
        text_col="question_repaired",
        source_col="question_noisy",
    )

    invalid_rows = noisy_invalid + repaired_invalid
    invalid_df = pd.DataFrame(invalid_rows)
    invalid_csv = output_dir / "invalid_rows.csv"
    if invalid_rows:
        invalid_df.to_csv(invalid_csv, index=False)

    summary = {
        "question_set_dir": str(question_set_dir),
        "noisy_rows": int(len(noisy_df)),
        "repaired_rows": int(len(repaired_df)),
        "noisy_invalid": int(len(noisy_invalid)),
        "repaired_invalid": int(len(repaired_invalid)),
        "invalid_rows_csv": str(invalid_csv) if invalid_rows else None,
        "status": "ok" if not invalid_rows else "invalid_rows_found",
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Validation summary saved to %s", summary_path)
    logger.info(
        "Noisy invalid: %d | Repaired invalid: %d",
        len(noisy_invalid),
        len(repaired_invalid),
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
