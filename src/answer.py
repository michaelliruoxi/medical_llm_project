"""Step C: Generate LLM answers for clean, noisy, or repaired questions."""

import argparse

import pandas as pd
from tqdm import tqdm

from src.utils import PROJECT_ROOT, load_config, load_prompts, call_llm, setup_logging, get_token_tracker


logger = setup_logging()

MODES = ("clean", "noisy", "repaired")


def answer_question(question: str, prompts: dict, cfg: dict) -> str:
    """Call the answer LLM for a single question."""
    system_msg = prompts["answer"]["system"]
    user_msg = prompts["answer"]["user_template"].format(question=question)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return call_llm(
        messages=messages,
        model=cfg["answer_model"],
        temperature=cfg["temperature_answer"],
        max_tokens=cfg["max_tokens_answer"],
        backend=cfg.get("backend", "openai"),
    )


def _load_input(mode: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, question_column) for the given mode."""
    processed = PROJECT_ROOT / cfg["paths"]["processed_data"]

    if mode == "clean":
        df = pd.read_parquet(processed / "medquad.parquet")
        return df, "question"
    elif mode == "noisy":
        df = pd.read_parquet(processed / "noisy.parquet")
        return df, "question_noisy"
    elif mode == "repaired":
        df = pd.read_parquet(processed / "repaired.parquet")
        return df, "question_repaired"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run(mode: str, n_examples: int | None = None):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    cfg = load_config()
    prompts = load_prompts()

    df, q_col = _load_input(mode, cfg)
    logger.info("Mode=%s: loaded %d rows, question column=%s", mode, len(df), q_col)

    if n_examples is not None:
        df = df.head(n_examples)

    answers = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Answering ({mode})"):
        ans = answer_question(row[q_col], prompts, cfg)
        answers.append(ans)

    df = df.copy()
    df["answer_pred"] = answers

    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"answers_{mode}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d answers to %s", len(df), out_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers")
    parser.add_argument("--mode", required=True, choices=MODES, help="Pipeline mode")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N rows")
    args = parser.parse_args()
    run(mode=args.mode, n_examples=args.n)
