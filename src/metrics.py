"""Shared text-comparison metrics used across local and API evaluation flows."""

from __future__ import annotations

import math
import os
import re
from collections import Counter

import sacrebleu


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_sbert_encoder = None
_sbert_model_name = None
_RECOVERY_GAP_FLOORS = {
    "bleu": 1.0,
    "chrf": 1.0,
    "rouge_l": 1.0,
    "token_f1": 1.0,
    "exact_match": 1.0,
    "bertscore": 0.01,
    "bertscore_f1": 0.01,
    "intent_preservation": 0.01,
    "geval": 0.1,
    "geval_score": 0.1,
}


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


def recovery_gap_floor(metric_name: str | None = None) -> float:
    if not metric_name:
        return 1.0
    return float(_RECOVERY_GAP_FLOORS.get(metric_name, 1.0))


def compute_recovery_statistics(
    clean_mean: float,
    noisy_mean: float,
    repaired_mean: float,
    metric_name: str | None = None,
) -> dict[str, float]:
    degradation = clean_mean - noisy_mean
    recovery = repaired_mean - noisy_mean
    floor = recovery_gap_floor(metric_name)

    ratio = math.nan
    if not any(math.isnan(v) for v in (degradation, recovery)) and degradation > floor:
        bounded_recovery = max(-degradation, min(recovery, degradation))
        ratio = bounded_recovery / degradation

    return {
        "degradation": degradation,
        "recovery": recovery,
        "recovery_ratio": ratio,
    }


def _get_sbert_encoder(model_name: str | None = None):
    global _sbert_encoder, _sbert_model_name

    chosen_model = model_name or os.getenv("INTENT_PRESERVATION_MODEL", _DEFAULT_SBERT_MODEL)
    if _sbert_encoder is not None and _sbert_model_name == chosen_model:
        return _sbert_encoder

    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(chosen_model)
    model = AutoModel.from_pretrained(chosen_model)
    model.to("cpu")
    model.eval()

    _sbert_model_name = chosen_model
    _sbert_encoder = (tokenizer, model)
    return _sbert_encoder


def _encode_sbert_texts(texts: list[str], model_name: str | None = None, batch_size: int = 32):
    import torch
    import torch.nn.functional as F

    tokenizer, model = _get_sbert_encoder(model_name=model_name)
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = [text or "" for text in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)


def compute_intent_preservation_scores(
    clean_questions: list[str],
    candidate_questions: list[str],
    model_name: str | None = None,
) -> list[float]:
    if len(clean_questions) != len(candidate_questions):
        raise ValueError("clean_questions and candidate_questions must have the same length")

    if not clean_questions:
        return []

    import torch

    scores = [0.0] * len(clean_questions)
    pairs_to_encode: list[tuple[int, str, str]] = []

    for idx, (clean_q, candidate_q) in enumerate(zip(clean_questions, candidate_questions)):
        clean_q = clean_q or ""
        candidate_q = candidate_q or ""
        if not clean_q and not candidate_q:
            scores[idx] = 1.0
        elif not clean_q or not candidate_q:
            scores[idx] = 0.0
        else:
            pairs_to_encode.append((idx, clean_q, candidate_q))

    if not pairs_to_encode:
        return scores

    left = [clean_q for _, clean_q, _ in pairs_to_encode]
    right = [candidate_q for _, _, candidate_q in pairs_to_encode]
    left_emb = _encode_sbert_texts(left, model_name=model_name)
    right_emb = _encode_sbert_texts(right, model_name=model_name)
    cosines = torch.sum(left_emb * right_emb, dim=1).tolist()

    for (idx, _, _), cosine in zip(pairs_to_encode, cosines):
        scores[idx] = float(max(-1.0, min(1.0, cosine)))

    return scores


def compute_intent_preservation_score(
    clean_question: str,
    candidate_question: str,
    model_name: str | None = None,
) -> float:
    return compute_intent_preservation_scores(
        [clean_question],
        [candidate_question],
        model_name=model_name,
    )[0]


def compute_reference_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Lexical/reference-based metrics for one prediction-reference pair."""
    return {
        "bleu": compute_bleu_score(prediction, reference),
        "chrf": compute_chrf_score(prediction, reference),
        "rouge_l": compute_rouge_l_score(prediction, reference),
        "token_f1": compute_token_f1_score(prediction, reference),
        "exact_match": compute_exact_match_score(prediction, reference),
    }
