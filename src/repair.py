"""Step D: Rewrite noisy questions into clear, well-formed medical questions (without answering)."""

import argparse

import pandas as pd
from tqdm import tqdm

from src.question_validation import retry_feedback, validate_generated_question
from src.resume import build_completed_lookup, row_key, write_parquet_atomic
from src.utils import PROJECT_ROOT, load_config, load_prompts, call_llm, setup_logging, get_token_tracker


logger = setup_logging()


def repair_question(noisy_question: str, prompts: dict, cfg: dict) -> str:
    """Call the repair LLM to rewrite one noisy question."""
    system_msg = prompts["repair"]["system"]
    user_msg = prompts["repair"]["user_template"].format(question=noisy_question)
    base_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    attempts = max(int(cfg.get("max_validation_retries_repair", 2)) + 1, 1)
    last_candidate = ""
    last_reason = "did not produce a valid repaired question"

    for attempt in range(attempts):
        messages = list(base_messages)
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": retry_feedback("repair", last_reason),
                }
            )

        raw = call_llm(
            messages=messages,
            model=cfg["repair_model"],
            temperature=cfg["temperature_repair"],
            max_tokens=cfg["max_tokens_repair"],
            backend=cfg.get("backend", "openai"),
            quantization=cfg.get("quantization", "none"),
            api_mode=cfg.get("api_mode", "chat_completions"),
            reasoning_effort=cfg.get("reasoning_effort_repair", cfg.get("reasoning_effort")),
            use_cache=attempt == 0,
        )
        candidate, error = validate_generated_question(
            raw,
            stage="repair",
            source_question=noisy_question,
        )
        if error is None:
            return candidate

        last_candidate = candidate
        last_reason = error
        logger.warning(
            "Retrying repair-question generation (attempt %d/%d): %s",
            attempt + 1,
            attempts,
            error,
        )

    raise ValueError(
        f"Failed to generate a valid repaired question after {attempts} attempts: {last_reason}. "
        f"Last candidate: {last_candidate!r}"
    )


def run(n_examples: int | None = None):
    cfg = load_config()
    prompts = load_prompts()

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    input_path = processed_dir / "noisy.parquet"
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d noisy variants from %s", len(df), input_path)

    if n_examples is not None:
        df = df.head(n_examples)

    out_path = processed_dir / "repaired.parquet"
    existing_df = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    completed_lookup = build_completed_lookup(existing_df, ["id", "noise_type"], ["question_repaired"])
    if completed_lookup:
        logger.info("Resuming repair stage with %d completed rows from %s", len(completed_lookup), out_path)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Repairing questions"):
        key = row_key({"id": row["id"], "noise_type": row.get("noise_type", "")}, ["id", "noise_type"])
        record = row.to_dict()

        if key in completed_lookup:
            record["question_repaired"] = completed_lookup[key]["question_repaired"]
            rows.append(record)
            continue

        record["question_repaired"] = repair_question(row["question_noisy"], prompts, cfg)
        rows.append(record)
        write_parquet_atomic(pd.DataFrame(rows), out_path)

    repaired_df = pd.DataFrame(rows)
    write_parquet_atomic(repaired_df, out_path)
    logger.info("Saved %d repaired questions to %s", len(repaired_df), out_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair noisy questions")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N rows")
    args = parser.parse_args()
    run(n_examples=args.n)
