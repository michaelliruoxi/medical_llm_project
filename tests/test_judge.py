"""Tests for src/judge.py — score parsing."""

import pytest

from src.judge import _parse_score


class TestParseScore:
    def test_simple_digit(self):
        assert _parse_score("3") == 3

    def test_digit_in_text(self):
        assert _parse_score("The answer is mostly correct. Score: 2") == 2

    def test_chain_of_thought(self):
        text = "Fact 1: mentioned. Fact 2: mentioned. Fact 3: missing. Score: 2"
        assert _parse_score(text) == 2

    def test_last_digit_used(self):
        # If multiple digits 0-3 appear, the last one is used
        text = "1 fact matched out of 3. Final score: 2"
        assert _parse_score(text) == 2

    def test_no_valid_score_returns_zero(self):
        assert _parse_score("No score here at all") == 0

    def test_all_scores(self):
        for i in range(4):
            assert _parse_score(f"Score: {i}") == i

    def test_out_of_rubric_scores_clamp(self):
        assert _parse_score("Score: 4") == 3
        assert _parse_score("Score: 5") == 3
