"""Validation helpers for generated noisy and repaired questions."""

from __future__ import annotations

import re


CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
SURROUNDING_QUOTES_RE = re.compile(r'^[\'"`]+|[\'"`]+$')
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)
LEADING_LABEL_RE = re.compile(
    r"^(?:rewritten|repaired|clear|clean|noisy|clarified|patient|updated|original|final)\s+question\s*:\s*",
    re.IGNORECASE,
)
LEADING_GENERIC_LABEL_RE = re.compile(r"^(?:question|rewrite|final answer|answer|output)\s*:\s*", re.IGNORECASE)
CLARIFY_IF_RE = re.compile(
    r"^(?:can|could|would)\s+you\s+clarify\s+(?:if|whether)\s+(.+)$",
    re.IGNORECASE,
)
DIRECT_QUESTION_AUX = {
    "is", "are", "am", "can", "could", "should", "would", "do", "does",
    "did", "will", "may", "might", "have", "has", "had",
}
PRESENT_VERB_BASES = {
    "affects": "affect",
    "causes": "cause",
    "includes": "include",
    "involves": "involve",
    "requires": "require",
    "means": "mean",
    "carries": "carry",
    "leads": "lead",
}
META_LINE_RE = re.compile(
    r"^(?:here(?:'s| is)|below is|i rewrote|i have rewritten|rewritten question|repaired question|"
    r"clarified question|final answer|output|sure[,! ]|certainly[,! ]|okay[,! ]|ok[,! ])\b",
    re.IGNORECASE,
)
EMPTY_TAG_LINE_RE = re.compile(r"^</?(?:think|analysis|assistant|final)>$", re.IGNORECASE)

META_PATTERNS = (
    re.compile(r"\b(?:rewritten|repaired|clarified|original|noisy)\s+question\s*:", re.IGNORECASE),
    re.compile(r"^\s*[-*]\s+", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s+", re.MULTILINE),
    re.compile(r"\b(?:here is|here's|below is|i rewrote|i have rewritten)\b", re.IGNORECASE),
)

CLARIFICATION_PATTERNS = (
    re.compile(r"\b(?:please|kindly)\s+(?:clarify|specify)\b", re.IGNORECASE),
    re.compile(r"\b(?:can|could|would)\s+you\s+(?:clarify|specify)\b", re.IGNORECASE),
    re.compile(r"\bneed more (?:details|information|context)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:exactly|specifically) do you mean\b", re.IGNORECASE),
    re.compile(r"\bwhich (?:symptoms|condition|problem|part|type) do you mean\b", re.IGNORECASE),
    re.compile(r"\bare you asking about\b", re.IGNORECASE),
)

ANSWER_STYLE_PATTERNS = (
    re.compile(r"^(?:it|this|that|they|these)\s+(?:is|are|can|may|might)\b", re.IGNORECASE),
    re.compile(r"^(?:symptoms|treatment|causes?|diagnosis|management|prevention)\s+(?:of|include|is|are)\b", re.IGNORECASE),
    re.compile(r"\b(?:the answer is|this means|this occurs when)\b", re.IGNORECASE),
)
ADVICE_STYLE_PATTERNS = (
    re.compile(r"\b(?:consult|see|ask)\s+(?:a|your)\s+(?:doctor|physician|provider)\b", re.IGNORECASE),
)

QUESTION_STARTERS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "am", "can", "could", "should", "would", "do", "does", "did",
    "will", "may", "might", "have", "has", "had",
}


def normalize_question_text(text: str) -> str:
    """Strip common wrappers and normalize whitespace."""
    candidate = (text or "").strip()
    candidate = candidate.replace("？", "?").replace("ï¼Ÿ", "?")
    candidate = THINK_BLOCK_RE.sub(" ", candidate).strip()
    candidate = CODE_FENCE_RE.sub("", candidate).strip()
    candidate = SURROUNDING_QUOTES_RE.sub("", candidate).strip()

    lines = []
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if EMPTY_TAG_LINE_RE.fullmatch(line):
            continue
        if META_LINE_RE.match(line):
            continue
        line = LEADING_LABEL_RE.sub("", line).strip()
        line = LEADING_GENERIC_LABEL_RE.sub("", line).strip()
        line = SURROUNDING_QUOTES_RE.sub("", line).strip()
        if line:
            lines.append(line)

    if lines:
        candidate = " ".join(lines)

    candidate = LEADING_LABEL_RE.sub("", candidate).strip()
    candidate = LEADING_GENERIC_LABEL_RE.sub("", candidate).strip()
    candidate = SURROUNDING_QUOTES_RE.sub("", candidate).strip()
    candidate = _rewrite_concrete_clarify_if(candidate)
    candidate = _trim_after_final_question(candidate)
    candidate = _fold_followup_questions(candidate)
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
    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    question_lines = [line for line in lines if _looks_like_question(line)]
    if len(question_lines) > 1:
        return True

    if len(lines) > 1 and sum(line.count("?") for line in lines) >= 2:
        return True

    question_marks = candidate.count("?")
    if question_marks <= 1:
        return False

    segments = [segment.strip() for segment in re.split(r"\?\s*", candidate) if segment.strip()]
    if len(segments) <= 1:
        return False

    for segment in segments[1:]:
        lowered = segment.lower().lstrip(" \"'([{")
        first_token = re.split(r"\W+", lowered, maxsplit=1)[0]
        if first_token in QUESTION_STARTERS:
            return True
    return False


def _rewrite_concrete_clarify_if(candidate: str) -> str:
    match = CLARIFY_IF_RE.match(candidate.strip())
    if not match:
        return candidate

    body = match.group(1).strip()
    body = body[:-1].strip() if body.endswith("?") else body
    if not body:
        return candidate

    first_token = re.split(r"\W+", body.lower(), maxsplit=1)[0]
    if first_token in DIRECT_QUESTION_AUX:
        rewritten = body
    else:
        verb_match = re.match(
            r"(?P<subject>.+?)\s+(?P<verb>affects?|causes?|includes?|involves?|requires?|means|carries|leads)\s+(?P<object>.+)",
            body,
            re.IGNORECASE,
        )
        if verb_match:
            subject = verb_match.group("subject").strip()
            verb = verb_match.group("verb").lower()
            obj = verb_match.group("object").strip()
            rewritten = f"Does {subject} {PRESENT_VERB_BASES.get(verb, verb)} {obj}"
        else:
            rewritten = body[0].upper() + body[1:]

    return rewritten.rstrip(" ?") + "?"


def _trim_after_final_question(candidate: str) -> str:
    last_qmark = candidate.rfind("?")
    if last_qmark == -1:
        return candidate

    tail = candidate[last_qmark + 1 :].strip()
    if not tail:
        return candidate

    if re.match(r"^(?:note|explanation|reason|rationale|comment|analysis)\s*:", tail, re.IGNORECASE):
        return candidate[: last_qmark + 1].strip()
    return candidate


def _lower_initial(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped[0].lower() + stripped[1:]


def _fold_followup_questions(candidate: str) -> str:
    if candidate.count("?") <= 1:
        return candidate

    parts = [part.strip() for part in candidate.split("?") if part.strip()]
    if not 1 < len(parts) <= 3:
        return candidate
    if not all(_looks_like_question(part + "?") for part in parts):
        return candidate

    first, *rest = parts
    if len(rest) == 1:
        return f"{first}, and {_lower_initial(rest[0])}?"
    folded_rest = ", ".join(_lower_initial(part) for part in rest[:-1])
    return f"{first}, {folded_rest}, and {_lower_initial(rest[-1])}?"


def _contains_meta(candidate: str) -> bool:
    return any(pattern.search(candidate) for pattern in META_PATTERNS)


def _has_trailing_commentary(candidate: str) -> bool:
    last_qmark = candidate.rfind("?")
    if last_qmark == -1:
        return False

    tail = candidate[last_qmark + 1 :].strip()
    if not tail:
        return False

    # Allow only closing wrappers/punctuation after the final question mark.
    return bool(re.search(r"[A-Za-z0-9]", tail))


def _contains_clarification_request(candidate: str) -> bool:
    return any(pattern.search(candidate) for pattern in CLARIFICATION_PATTERNS)


def _looks_like_answer(candidate: str) -> bool:
    if any(pattern.search(candidate) for pattern in ANSWER_STYLE_PATTERNS):
        return True
    if any(pattern.search(candidate) for pattern in ADVICE_STYLE_PATTERNS):
        return not _looks_like_question(candidate)
    return False


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
    if _has_trailing_commentary(candidate):
        return candidate, "output adds commentary after the question instead of returning only the question"
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
        f"Return exactly one {stage_label}, with no answer, no explanation, no label, "
        f"and no <think> or reasoning text."
    )
