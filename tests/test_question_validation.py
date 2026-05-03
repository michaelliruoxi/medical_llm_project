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


def test_validate_generated_question_folds_related_followup_question():
    cleaned, error = validate_generated_question(
        (
            "What clinical research or trials are being conducted currently "
            "for chronic lymphocytic leukemia? Where can I find information "
            "about these studies?"
        ),
        stage="repair",
        source_question="cll trials??",
    )

    assert cleaned == (
        "What clinical research or trials are being conducted currently for "
        "chronic lymphocytic leukemia, and where can I find information about "
        "these studies?"
    )
    assert error is None


def test_validate_generated_question_allows_doctor_visit_as_question():
    cleaned, error = validate_generated_question(
        (
            "What symptoms should I look out for that might indicate something "
            "serious besides hemorrhoids, and when should I be concerned enough "
            "to see a doctor?"
        ),
        stage="repair",
        source_question="hemorrhoid serious??",
    )

    assert cleaned.endswith("to see a doctor?")
    assert error is None


def test_validate_generated_question_strips_trailing_note_after_question():
    cleaned, error = validate_generated_question(
        (
            "What are the different staging criteria for childhood soft tissue "
            "cancer? Note: The previous response added explanation."
        ),
        stage="repair",
        source_question="soft tissue cancer stages??",
    )

    assert cleaned == "What are the different staging criteria for childhood soft tissue cancer?"
    assert error is None


def test_validate_generated_question_rewrites_concrete_clarify_if_question():
    cleaned, error = validate_generated_question(
        (
            "Can you clarify if renal pelvis carcinoma affects the urinary "
            "tract's tubular structure leading towards the bladder, and is it "
            "distinctive from typical kidney cancer, and what potential "
            "severity does it carry?"
        ),
        stage="repair",
        source_question="renal pelvis carcinoma urinary tract??",
    )

    assert cleaned.startswith("Does renal pelvis carcinoma affect")
    assert "Can you clarify" not in cleaned
    assert error is None
