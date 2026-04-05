"""Step E (part 2): G-Eval scoring for answer quality and logical consistency."""

import argparse
import re

import pandas as pd
from tqdm import tqdm

from src.resume import write_parquet_atomic
from src.utils import PROJECT_ROOT, _majority_vote_int, call_llm, get_token_tracker, load_config, load_prompts, setup_logging


logger = setup_logging()

DEFAULT_GEVAL_MODEL = "gpt-5.4-mini"
DEFAULT_GEVAL_BACKEND = "openai"
DEFAULT_GEVAL_API_MODE = "responses"
DEFAULT_GEVAL_MAX_TOKENS = 64
DEFAULT_GEVAL_TEMPERATURE = 0.0
DEFAULT_GEVAL_REASONING_EFFORT = None


def _extract_numeric_score(text: str) -> tuple[str, int | None]:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    explicit = re.findall(r"(?:score|rating)\s*[:=]\s*([0-9]+)", stripped, re.IGNORECASE)
    if explicit:
        return stripped, int(explicit[-1])

    if re.fullmatch(r"[0-9]+", stripped):
        return stripped, int(stripped)

    return stripped, None


def _normalize_score(raw_score: int, min_score: int, max_score: int, label: str) -> int:
    if raw_score < min_score:
        logger.warning("%s returned below-range score %s; clamping to %s", label, raw_score, min_score)
        return min_score
    if raw_score > max_score:
        logger.warning("%s returned out-of-rubric score %s; clamping to %s", label, raw_score, max_score)
        return max_score
    return raw_score


def _parse_geval_score(text: str) -> int:
    _, numeric = _extract_numeric_score(text)
    if numeric is not None:
        return _normalize_score(numeric, min_score=1, max_score=5, label="G-Eval")

    logger.warning("Could not parse G-Eval score from: %r - defaulting to 1", text)
    return 1


def _score_prediction(
    parser,
    vote_aggregator,
    reference: str,
    prediction: str,
    prompts: dict,
    cfg: dict,
    question: str = "",
    model_key: str = "geval_model",
    backend_key: str = "geval_backend",
    quantization_key: str = "geval_quantization",
    api_mode_key: str = "geval_api_mode",
    max_tokens_key: str = "max_tokens_geval",
    temperature_key: str = "temperature_geval",
    n_votes_key: str = "n_votes_geval",
    reasoning_key: str = "reasoning_effort_geval",
) -> int:
    prompt = prompts["geval"]
    system_msg = prompt["system"]
    user_msg = prompt["user_template"].format(
        question=question, reference=reference, prediction=prediction
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    default_model = cfg.get("geval_model", DEFAULT_GEVAL_MODEL)
    default_max_tokens = cfg.get("max_tokens_geval", DEFAULT_GEVAL_MAX_TOKENS)
    default_temp = cfg.get("temperature_geval", DEFAULT_GEVAL_TEMPERATURE)
    default_reasoning = cfg.get("reasoning_effort_geval", DEFAULT_GEVAL_REASONING_EFFORT)
    default_backend = cfg.get("geval_backend", DEFAULT_GEVAL_BACKEND)
    default_quantization = cfg.get("geval_quantization", "none")
    default_api_mode = cfg.get("geval_api_mode", DEFAULT_GEVAL_API_MODE)

    n_votes = cfg.get(n_votes_key, 1)
    common_kwargs = {
        "messages": messages,
        "model": cfg.get(model_key, default_model),
        "max_tokens": cfg.get(max_tokens_key, default_max_tokens),
        "backend": cfg.get(backend_key, default_backend),
        "quantization": cfg.get(quantization_key, default_quantization),
        "api_mode": cfg.get(api_mode_key, default_api_mode),
        "reasoning_effort": cfg.get(reasoning_key, default_reasoning),
    }

    if n_votes <= 1:
        request_temperature = cfg.get(temperature_key, default_temp)
        raw = call_llm(
            temperature=request_temperature,
            **common_kwargs,
        )
        if raw.strip():
            return parser(raw)

        retry_kwargs = dict(common_kwargs)
        retry_kwargs["max_tokens"] = max(int(retry_kwargs["max_tokens"]), DEFAULT_GEVAL_MAX_TOKENS)
        retry_kwargs["reasoning_effort"] = None
        logger.warning(
            "G-Eval returned blank output; retrying once without cache at max_tokens=%s",
            retry_kwargs["max_tokens"],
        )
        raw = call_llm(
            temperature=request_temperature,
            use_cache=False,
            **retry_kwargs,
        )
        return parser(raw)

    scores = []
    vote_temp = max(cfg.get(temperature_key, default_temp), 0.3)
    for _ in range(n_votes):
        raw = call_llm(
            temperature=vote_temp,
            use_cache=False,
            **common_kwargs,
        )
        scores.append(parser(raw))
    return vote_aggregator(scores)


def geval_answer(reference: str, prediction: str, prompts: dict, cfg: dict,
                 question: str = "") -> int:
    """Ask the G-Eval judge LLM to score deep answer quality on a 1-5 scale."""
    return _score_prediction(
        parser=_parse_geval_score,
        vote_aggregator=_majority_vote_int,
        reference=reference,
        prediction=prediction,
        prompts=prompts,
        cfg=cfg,
        question=question,
        model_key="geval_model",
        backend_key="geval_backend",
        quantization_key="geval_quantization",
        api_mode_key="geval_api_mode",
        max_tokens_key="max_tokens_geval",
        temperature_key="temperature_geval",
        n_votes_key="n_votes_geval",
        reasoning_key="reasoning_effort_geval",
    )


def run():
    cfg = load_config()
    prompts = load_prompts()

    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    metrics_path = out_dir / "metrics.parquet"
    df = pd.read_parquet(metrics_path)
    logger.info("Loaded %d rows from %s", len(df), metrics_path)

    q_col = next((c for c in ["question_clean", "question", "question_noisy"] if c in df.columns), "")

    if "geval_score" not in df.columns:
        df["geval_score"] = pd.NA

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging answers"):
        question = ""
        if row.get("pipeline") == "repaired" and "question_repaired" in df.columns:
            question = row.get("question_repaired", "")
        elif row.get("pipeline") == "noisy" and "question_noisy" in df.columns:
            question = row.get("question_noisy", "")
        elif "question" in df.columns:
            question = row.get("question", "")
        elif "question_clean" in df.columns:
            question = row.get("question_clean", "")
        elif q_col:
            question = row.get(q_col, "")

        updated = False
        if pd.isna(row.get("geval_score")):
            df.at[idx, "geval_score"] = geval_answer(
                row["answer_ref"],
                row["answer_pred"],
                prompts,
                cfg,
                question=question,
            )
            updated = True

        if updated:
            write_parquet_atomic(df, metrics_path)

    write_parquet_atomic(df, metrics_path)
    logger.info("Updated %s with geval_score column", metrics_path)

    tracker = get_token_tracker()
    logger.info("Token usage: %s", tracker.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-Eval scoring")
    parser.parse_args()
    run()
