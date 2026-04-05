"""Tests for shared lexical metrics."""

from src.metrics import (
    compute_chrf_score,
    compute_exact_match_score,
    compute_intent_preservation_score,
    compute_rouge_l_score,
    compute_token_f1_score,
)


class TestComparisonMetrics:
    def test_exact_match_normalizes_articles_and_case(self):
        assert compute_exact_match_score("The Cat", "cat") == 100.0

    def test_token_f1_partial_overlap(self):
        score = compute_token_f1_score("cat sat on mat", "the cat sat on the mat")
        assert 60 < score < 100

    def test_rouge_l_rewards_ordered_overlap(self):
        score = compute_rouge_l_score("cat sat on mat", "the cat sat on the mat")
        assert 60 < score < 100

    def test_chrf_identical_strings_are_high(self):
        score = compute_chrf_score("hello world", "hello world")
        assert score > 95

    def test_intent_preservation_empty_pair_is_one(self):
        assert compute_intent_preservation_score("", "") == 1.0

    def test_intent_preservation_one_empty_side_is_zero(self):
        assert compute_intent_preservation_score("What causes asthma?", "") == 0.0
