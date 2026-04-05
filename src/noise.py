"""Step B: Generate noisy question variants using an LLM."""

import argparse

import pandas as pd
from tqdm import tqdm

from src.noise_schedule import build_noise_plan, summarize_noise_plan
from src.resume import build_completed_lookup, row_key, write_parquet_atomic
from src.utils import PROJECT_ROOT, load_config, load_prompts, call_llm, setup_logging, get_token_tracker


logger = setup_logging()


def generate_noisy_variant(question: str, noise_type: str,
                           prompts: dict, cfg: dict) -> str:
    """Call the noise LLM to produce a single noisy variant."""
    system_msg = prompts["noise"]["system"]
    user_msg = prompts["noise"]["user_template"].format(
        noise_type=noise_type, question=question
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return call_llm(
        messages=messages,
        model=cfg["noise_model"],
        temperature=cfg["temperature_noise"],
        max_tokens=cfg["max_tokens_noise"],
        backend=cfg.get("backend", "openai"),
        quantization=cfg.get("quantization", "none"),
        api_mode=cfg.get("api_mode", "chat_completions"),
        reasoning_effort=cfg.get("reasoning_effort_noise", cfg.get("reasoning_effort")),
    )


def run(n_examples: int | None = None):
    cfg = load_config()
    prompts = load_prompts()

    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    input_path = processed_dir / "medquad.parquet"
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d examples from %s", len(df), input_path)

    if n_examples is not None:
        df = df.head(n_examples)

    plan = build_noise_plan(df, cfg)
    logger.info("Noise assignment counts: %s", summarize_noise_plan(plan))

    out_path = processed_dir / "noisy.parquet"
    existing_df = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    completed_lookup = build_completed_lookup(existing_df, ["id", "noise_type"], ["question_noisy"])
    if completed_lookup:
        logger.info("Resuming noise generation with %d completed rows from %s", len(completed_lookup), out_path)

    rows = []
    with tqdm(total=len(plan), desc="Generating noisy variants") as pbar:
        for row, noise_type in plan:
            key = row_key({"id": row["id"], "noise_type": noise_type}, ["id", "noise_type"])
            if key in completed_lookup:
                rows.append(completed_lookup[key])
                pbar.update(1)
                continue

            noisy_q = generate_noisy_variant(row["question"], noise_type, prompts, cfg)
            record = {
                "id": row["id"],
                "question_clean": row["question"],
                "question_noisy": noisy_q,
                "noise_type": noise_type,
                "answer_ref": row["answer"],
                "source": row.get("source", ""),
            }
            rows.append(record)
            write_parquet_atomic(pd.DataFrame(rows), out_path)
            pbar.update(1)

    noisy_df = pd.DataFrame(rows)
    write_parquet_atomic(noisy_df, out_path)
    logger.info("Saved %d noisy variants to %s", len(noisy_df), out_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noisy question variants")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N examples")
    args = parser.parse_args()
    run(n_examples=args.n)
