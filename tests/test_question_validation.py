"""Tests for generated-question validation helpers."""

from src.question_validation import normalize_question_text, validate_generated_question


def test_normalize_question_text_strips_common_label():
    text = 'Rewritten question: "What causes chest pain?"'

    assert normalize_question_text(text) == "What causes chest pain?"


def test_validate_generated_question_rejects_meta_output():
    cleaned, error = validate_generated_question(
        "Here is the rewritten question: What causes chest pain?",
        stage="noise",
        source_question="What causes chest pain?",
    )

    assert cleaned == "Here is the rewritten question: What causes chest pain?"
    assert error is not None


def test_validate_generated_question_rejects_clarification_request():
    cleaned, error = validate_generated_question(
        "Can you clarify which symptoms you mean?",
        stage="repair",
        source_question="symptoms??",
    )

    assert cleaned == "Can you clarify which symptoms you mean?"
    assert error is not None


def test_validate_generated_question_allows_exactly_question():
    cleaned, error = validate_generated_question(
        "What exactly is an acoustic neuroma?",
        stage="repair",
        source_question="what is acoustic neuroma??",
    )

    assert cleaned == "What exactly is an acoustic neuroma?"
    assert error is None


def test_validate_generated_question_accepts_single_clean_question():
    cleaned, error = validate_generated_question(
        "What can cause ongoing chest tightness during exercise?",
        stage="repair",
        source_question="chest tightness when i run??",
    )

    assert cleaned == "What can cause ongoing chest tightness during exercise?"
    assert error is None
