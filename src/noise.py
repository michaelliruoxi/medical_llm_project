"""Step B: Generate noisy question variants using an LLM."""

import argparse

import pandas as pd
from tqdm import tqdm

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

    noise_types = cfg["noise_types"]
    variants_per_q = cfg.get("noise_variants_per_question", 1)

    rows = []
    total = len(df) * len(noise_types) * variants_per_q
    with tqdm(total=total, desc="Generating noisy variants") as pbar:
        for _, row in df.iterrows():
            for noise_type in noise_types:
                for _ in range(variants_per_q):
                    noisy_q = generate_noisy_variant(
                        row["question"], noise_type, prompts, cfg
                    )
                    rows.append({
                        "id": row["id"],
                        "question_clean": row["question"],
                        "question_noisy": noisy_q,
                        "noise_type": noise_type,
                        "answer_ref": row["answer"],
                        "source": row.get("source", ""),
                    })
                    pbar.update(1)

    noisy_df = pd.DataFrame(rows)
    out_path = processed_dir / "noisy.parquet"
    noisy_df.to_parquet(out_path, index=False)
    logger.info("Saved %d noisy variants to %s", len(noisy_df), out_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noisy question variants")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N examples")
    args = parser.parse_args()
    run(n_examples=args.n)
