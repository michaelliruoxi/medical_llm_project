"""Validation helpers for generated noisy and repaired questions."""

from __future__ import annotations

import re


CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
SURROUNDING_QUOTES_RE = re.compile(r'^[\'"`]+|[\'"`]+$')
LEADING_LABEL_RE = re.compile(
    r"^(?:rewritten|repaired|clear|clean|noisy|clarified|patient|updated|original)\s+question\s*:\s*",
    re.IGNORECASE,
)
LEADING_GENERIC_LABEL_RE = re.compile(r"^(?:question|rewrite)\s*:\s*", re.IGNORECASE)

META_PATTERNS = (
    re.compile(r"\b(?:rewritten|repaired|clarified|original|noisy)\s+question\s*:", re.IGNORECASE),
    re.compile(r"^\s*[-*]\s+", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s+", re.MULTILINE),
    re.compile(r"\b(?:here is|here's|below is|i rewrote|i have rewritten)\b", re.IGNORECASE),
)

CLARIFICATION_PATTERNS = (
    re.compile(r"\b(?:please|kindly)\s+(?:clarify|provide|share|specify|tell me)\b", re.IGNORECASE),
    re.compile(r"\b(?:can|could|would)\s+you\s+(?:clarify|provide|share|tell me|specify)\b", re.IGNORECASE),
    re.compile(r"\bneed more (?:details|information|context)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:exactly|specifically) do you mean\b", re.IGNORECASE),
    re.compile(r"\bwhich (?:symptoms|condition|problem|part|type) do you mean\b", re.IGNORECASE),
    re.compile(r"\bare you asking about\b", re.IGNORECASE),
)

ANSWER_STYLE_PATTERNS = (
    re.compile(r"^(?:it|this|that|they|these)\s+(?:is|are|can|may|might)\b", re.IGNORECASE),
    re.compile(r"^(?:symptoms|treatment|causes?|diagnosis|management|prevention)\s+(?:of|include|is|are)\b", re.IGNORECASE),
    re.compile(r"\b(?:consult|see|ask)\s+(?:a|your)\s+(?:doctor|physician|provider)\b", re.IGNORECASE),
    re.compile(r"\b(?:the answer is|this means|this occurs when)\b", re.IGNORECASE),
)

QUESTION_STARTERS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "am", "can", "could", "should", "would", "do", "does", "did",
    "will", "may", "might", "have", "has", "had",
}


def normalize_question_text(text: str) -> str:
    """Strip common wrappers and normalize whitespace."""
    candidate = (text or "").strip()
    candidate = CODE_FENCE_RE.sub("", candidate).strip()
    candidate = LEADING_LABEL_RE.sub("", candidate).strip()
    candidate = LEADING_GENERIC_LABEL_RE.sub("", candidate).strip()
    candidate = SURROUNDING_QUOTES_RE.sub("", candidate).strip()
    candidate = WHITESPACE_RE.sub(" ", candidate)
    return candidate.strip()


def _looks_like_question(candidate: str) -> bool:
    if not candidate:
        return False

    stripped = candidate.strip()
    if stripped.endswith("?"):
        return True

    first_token = re.split(r"\W+", stripped.lower(), maxsplit=1)[0]
    if first_token in QUESTION_STARTERS:
        return True

    lowered = stripped.lower()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "symptoms of",
            "treatment for",
            "side effects of",
            "causes of",
            "signs of",
        )
    )


def _contains_multiple_questions(candidate: str) -> bool:
    question_marks = candidate.count("?")
    if question_marks > 1:
        return True

    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    return len(lines) > 1 and question_marks >= 1


def _contains_meta(candidate: str) -> bool:
    return any(pattern.search(candidate) for pattern in META_PATTERNS)


def _contains_clarification_request(candidate: str) -> bool:
    return any(pattern.search(candidate) for pattern in CLARIFICATION_PATTERNS)


def _looks_like_answer(candidate: str) -> bool:
    return any(pattern.search(candidate) for pattern in ANSWER_STYLE_PATTERNS)


def validate_generated_question(
    text: str,
    *,
    stage: str,
    source_question: str,
) -> tuple[str, str | None]:
    """Return (cleaned_text, error_message)."""
    candidate = normalize_question_text(text)
    source_clean = normalize_question_text(source_question)

    if not candidate:
        return candidate, "empty output"
    if len(candidate) < 8:
        return candidate, "output is too short to be a usable question"
    if _contains_meta(candidate):
        return candidate, "output contains labels, bullets, or meta commentary"
    if _contains_multiple_questions(candidate):
        return candidate, "output contains multiple questions"
    if _contains_clarification_request(candidate):
        return candidate, "output asks the user for clarification instead of rewriting the question"
    if _looks_like_answer(candidate):
        return candidate, "output looks like an answer or advice instead of a rewritten question"
    if not _looks_like_question(candidate):
        return candidate, "output does not read like a single question"

    if stage == "noise" and candidate.casefold() == source_clean.casefold():
        return candidate, "output is effectively identical to the original question"
    if stage == "repair" and candidate.casefold() == source_clean.casefold():
        return candidate, "output did not actually rewrite the noisy question"

    return candidate, None


def retry_feedback(stage: str, reason: str) -> str:
    stage_label = "noisy rewritten question" if stage == "noise" else "repaired clear question"
    return (
        f"Your previous output was invalid because {reason}. "
        f"Return exactly one {stage_label}, with no answer, no explanation, and no label."
    )
