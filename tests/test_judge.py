"""Tests for src/judge.py G-Eval score parsing."""

from src.judge import _parse_geval_score


class TestParseGevalScore:
    def test_single_digit(self):
        assert _parse_geval_score("5") == 5

    def test_score_text(self):
        assert _parse_geval_score("Score: 4") == 4

    def test_code_fence_digit(self):
        assert _parse_geval_score("```\n3\n```") == 3

    def test_below_range_clamps_up(self):
        assert _parse_geval_score("0") == 1

    def test_above_range_clamps_down(self):
        assert _parse_geval_score("7") == 5

    def test_no_valid_score_defaults_to_one(self):
        assert _parse_geval_score("No structured score found") == 1
