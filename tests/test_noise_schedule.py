"""Tests for deterministic noise scheduling."""

import pandas as pd

from src.noise_schedule import build_noise_plan, expected_noise_rows, summarize_noise_plan


def test_round_robin_assigns_one_noise_per_question():
    df = pd.DataFrame({"id": [0, 1, 2, 3, 4]})
    cfg = {
        "noise_assignment": "round_robin",
        "noise_variants_per_question": 1,
        "noise_types": ["a", "b", "c"],
    }

    plan = build_noise_plan(df, cfg)

    assert [noise_type for _, noise_type in plan] == ["a", "b", "c", "a", "b"]


def test_round_robin_expected_rows_matches_question_count():
    cfg = {
        "noise_assignment": "round_robin",
        "noise_variants_per_question": 1,
        "noise_types": ["a", "b", "c"],
    }

    assert expected_noise_rows(50, cfg) == 50


def test_round_robin_distribution_is_balanced():
    df = pd.DataFrame({"id": list(range(10))})
    cfg = {
        "noise_assignment": "round_robin",
        "noise_variants_per_question": 1,
        "noise_types": ["a", "b", "c", "d", "e"],
    }

    counts = summarize_noise_plan(build_noise_plan(df, cfg))

    assert counts == {"a": 2, "b": 2, "c": 2, "d": 2, "e": 2}
