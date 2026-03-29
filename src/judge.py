"""Step E (part 2): LLM-as-judge scoring - rate each predicted answer 0-3 against the reference."""

import argparse
import re

import pandas as pd
from tqdm import tqdm

from src.utils import (
    PROJECT_ROOT,
    _majority_vote_int,
    call_llm,
    get_token_tracker,
    load_config,
    load_prompts,
    setup_logging,
)


logger = setup_logging()


def judge_answer(reference: str, prediction: str, prompts: dict, cfg: dict,
                 question: str = "") -> int:
    """Ask the judge LLM to score a prediction against a reference (0-3).

    When ``n_votes_judge`` > 1 in the config, multiple judge calls are made
    and the final score is determined by majority vote (hallucination control).
    """
    system_msg = prompts["judge"]["system"]
    user_msg = prompts["judge"]["user_template"].format(
        question=question, reference=reference, prediction=prediction
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    n_votes = cfg.get("n_votes_judge", 1)
    common_kwargs = {
        "messages": messages,
        "model": cfg["judge_model"],
        "max_tokens": cfg.get("max_tokens_judge", 50),
        "backend": cfg.get("backend", "openai"),
        "quantization": cfg.get("quantization", "none"),
        "api_mode": cfg.get("api_mode", "chat_completions"),
        "reasoning_effort": cfg.get("reasoning_effort_judge", cfg.get("reasoning_effort")),
    }

    if n_votes <= 1:
        raw = call_llm(
            temperature=cfg.get("temperature_judge", 0.0),
            **common_kwargs,
        )
        return _parse_score(raw)

    scores = []
    vote_temp = max(cfg.get("temperature_judge", 0.0), 0.3)
    for _ in range(n_votes):
        raw = call_llm(
            temperature=vote_temp,
            use_cache=False,  # each vote must be independent
            **common_kwargs,
        )
        scores.append(_parse_score(raw))
    return _majority_vote_int(scores)


def _normalize_score(raw_score: int) -> int:
    if raw_score <= 3:
        return raw_score
    logger.warning("Judge returned out-of-rubric score %s; clamping to 3", raw_score)
    return 3


def _parse_score(text: str) -> int:
    """Extract the 0-3 score from the judge response.

    Tries structured patterns first, then falls back to the last standalone
    digit 0-5 in the output. Out-of-rubric values 4-5 are clamped to 3.
    """
    explicit = re.findall(r"(?:score|rating)\s*[:=]\s*([0-5])", text, re.IGNORECASE)
    if explicit:
        return _normalize_score(int(explicit[-1]))

    standalone = re.findall(r"(?<!\d)([0-5])(?!\d)", text)
    if standalone:
        return _normalize_score(int(standalone[-1]))

    logger.warning("Could not parse judge score from: %r - defaulting to 0", text)
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
