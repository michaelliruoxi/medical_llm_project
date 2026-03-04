"""Step E (part 2): LLM-as-judge scoring — rate each predicted answer 0-3 against the reference."""

import argparse
import re

import pandas as pd
from tqdm import tqdm

from src.utils import PROJECT_ROOT, load_config, load_prompts, call_llm, setup_logging, get_token_tracker


logger = setup_logging()


def judge_answer(reference: str, prediction: str, prompts: dict, cfg: dict,
                  question: str = "") -> int:
    """Ask the judge LLM to score a prediction against a reference (0-3)."""
    system_msg = prompts["judge"]["system"]
    user_msg = prompts["judge"]["user_template"].format(
        question=question, reference=reference, prediction=prediction
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    raw = call_llm(
        messages=messages,
        model=cfg["judge_model"],
        temperature=cfg.get("temperature_judge", 0.0),
        max_tokens=cfg.get("max_tokens_judge", 50),
        backend=cfg.get("backend", "openai"),
    )
    return _parse_score(raw)


def _parse_score(text: str) -> int:
    """Extract the final 0-3 integer from the judge response.

    With chain-of-thought judging the score is the last digit in the output.
    """
    matches = re.findall(r"[0-3]", text)
    if matches:
        return int(matches[-1])
    logger.warning("Could not parse judge score from: %r — defaulting to 0", text)
    return 0


def run():
    cfg = load_config()
    prompts = load_prompts()

    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    metrics_path = out_dir / "metrics.parquet"
    df = pd.read_parquet(metrics_path)
    logger.info("Loaded %d rows from %s", len(df), metrics_path)

    q_col = next((c for c in ["question_clean", "question", "question_noisy"] if c in df.columns), "")

    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging answers"):
        question = row.get(q_col, "") if q_col else ""
        s = judge_answer(row["answer_ref"], row["answer_pred"], prompts, cfg,
                         question=question)
        scores.append(s)

    df["judge_score"] = scores

    df.to_parquet(metrics_path, index=False)
    logger.info("Updated %s with judge_score column", metrics_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge scoring")
    parser.parse_args()
    run()
