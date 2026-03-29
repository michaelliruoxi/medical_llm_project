"""Tests for src/evaluate_metrics.py — BLEU and BERTScore computation."""

import pytest

from src.evaluate_metrics import compute_bleu_per_row


class TestBLEU:
    def test_identical_strings(self):
        preds = ["The cat sat on the mat"]
        refs = ["The cat sat on the mat"]
        scores = compute_bleu_per_row(preds, refs)
        assert len(scores) == 1
        assert scores[0] > 90  # near-perfect BLEU

    def test_completely_different(self):
        preds = ["apple banana cherry"]
        refs = ["The quick brown fox jumps over the lazy dog"]
        scores = compute_bleu_per_row(preds, refs)
        assert len(scores) == 1
        assert scores[0] < 5  # very low BLEU

    def test_partial_overlap(self):
        preds = ["The cat sat on a mat"]
        refs = ["The cat sat on the mat"]
        scores = compute_bleu_per_row(preds, refs)
        assert len(scores) == 1
        assert 30 < scores[0] < 100  # partial match

    def test_multiple_pairs(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "baz qux"]
        scores = compute_bleu_per_row(preds, refs)
        assert len(scores) == 2
        assert scores[0] > scores[1]
