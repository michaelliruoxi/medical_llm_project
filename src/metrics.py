"""Shared text-comparison metrics used across local and API evaluation flows."""

from __future__ import annotations

import re
from collections import Counter

import sacrebleu


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for overlap-style QA metrics."""
    text = (text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def compute_bleu_score(prediction: str, reference: str) -> float:
    return float(sacrebleu.sentence_bleu(prediction or "", [reference or ""]).score)


def compute_chrf_score(prediction: str, reference: str) -> float:
    return float(sacrebleu.sentence_chrf(prediction or "", [reference or ""]).score)


def compute_exact_match_score(prediction: str, reference: str) -> float:
    return 100.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def compute_token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    ref_tokens = normalized_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 100.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return 100.0 * f1


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def compute_rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    ref_tokens = normalized_tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 100.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return 100.0 * f1


def compute_reference_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Lexical/reference-based metrics for one prediction-reference pair."""
    return {
        "bleu": compute_bleu_score(prediction, reference),
        "chrf": compute_chrf_score(prediction, reference),
        "rouge_l": compute_rouge_l_score(prediction, reference),
        "token_f1": compute_token_f1_score(prediction, reference),
        "exact_match": compute_exact_match_score(prediction, reference),
    }
