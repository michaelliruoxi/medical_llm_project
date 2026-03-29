"""Step D: Rewrite noisy questions into clear, well-formed medical questions (without answering)."""

import argparse

import pandas as pd
from tqdm import tqdm

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

    repaired = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Repairing questions"):
        repaired_q = repair_question(row["question_noisy"], prompts, cfg)
        repaired.append(repaired_q)

    df = df.copy()
    df["question_repaired"] = repaired

    out_path = processed_dir / "repaired.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d repaired questions to %s", len(df), out_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair noisy questions")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N rows")
    args = parser.parse_args()
    run(n_examples=args.n)
