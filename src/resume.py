"""Helpers for row-level resume/checkpoint behavior in sequential pipeline stages."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


def row_key(row: pd.Series | dict, key_cols: Iterable[str]) -> tuple:
    values = []
    for col in key_cols:
        value = row[col]
        if pd.isna(value):
            value = ""
        elif hasattr(value, "item"):
            value = value.item()
        values.append(value)
    return tuple(values)


def has_required_values(row: pd.Series | dict, required_cols: Iterable[str]) -> bool:
    for col in required_cols:
        value = row[col]
        if pd.isna(value):
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


def build_completed_lookup(
    df: pd.DataFrame,
    key_cols: Iterable[str],
    required_cols: Iterable[str],
) -> dict[tuple, dict]:
    completed: dict[tuple, dict] = {}
    if df.empty:
        return completed

    needed = list(key_cols) + list(required_cols)
    if any(col not in df.columns for col in needed):
        return completed

    for _, row in df.iterrows():
        if has_required_values(row, required_cols):
            completed[row_key(row, key_cols)] = row.to_dict()
    return completed


def replace_file_atomic(tmp_path: Path, path: Path, retries: int = 20, delay_seconds: float = 0.5):
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def write_parquet_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    replace_file_atomic(tmp_path, path)
