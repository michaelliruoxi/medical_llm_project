"""Step D: Rewrite noisy questions into clear, well-formed medical questions (without answering)."""

import argparse

import pandas as pd
from tqdm import tqdm

from src.resume import build_completed_lookup, row_key, write_parquet_atomic
from src.utils import PROJECT_ROOT, load_config, load_prompts, call_llm, setup_logging, get_token_tracker


logger = setup_logging()


def repair_question(noisy_question: str, prompts: dict, cfg: dict) -> str:
    """Call the repair LLM to rewrite one noisy question."""
    system_msg = prompts["repair"]["system"]
    user_msg = prompts["repair"]["user_template"].format(question=noisy_question)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return call_llm(
        messages=messages,
        model=cfg["repair_model"],
        temperature=cfg["temperature_repair"],
        max_tokens=cfg["max_tokens_repair"],
        backend=cfg.get("backend", "openai"),
        quantization=cfg.get("quantization", "none"),
        api_mode=cfg.get("api_mode", "chat_completions"),
        reasoning_effort=cfg.get("reasoning_effort_repair", cfg.get("reasoning_effort")),
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
