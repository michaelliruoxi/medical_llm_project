"""Tests for row-level resume helpers."""

import pandas as pd

from src.resume import build_completed_lookup, row_key


def test_row_key_normalizes_nan_to_empty_string():
    row = {"id": 1, "noise_type": float("nan")}

    key = row_key(row, ["id", "noise_type"])

    assert key == (1, "")


def test_build_completed_lookup_requires_expected_columns():
    df = pd.DataFrame([{"id": 1, "noise_type": "typos"}])

    lookup = build_completed_lookup(df, ["id", "noise_type"], ["question_noisy"])

    assert lookup == {}


def test_build_completed_lookup_only_keeps_completed_rows():
    df = pd.DataFrame(
        [
            {"id": 1, "noise_type": "typos", "question_noisy": "abc"},
            {"id": 2, "noise_type": "typos", "question_noisy": ""},
        ]
    )

    lookup = build_completed_lookup(df, ["id", "noise_type"], ["question_noisy"])

    assert lookup == {(1, "typos"): {"id": 1, "noise_type": "typos", "question_noisy": "abc"}}
