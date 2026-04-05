"""Step E (part 1): Compute lexical and semantic metrics for each pipeline's predictions."""

import argparse

import pandas as pd
from bert_score import score as bert_score_fn
from tqdm import tqdm

from src.resume import build_completed_lookup, row_key, write_parquet_atomic
from src.metrics import (
    compute_bleu_score,
    compute_chrf_score,
    compute_exact_match_score,
    compute_intent_preservation_scores,
    compute_rouge_l_score,
    compute_token_f1_score,
)
from src.utils import PROJECT_ROOT, load_config, setup_logging


logger = setup_logging()

PIPELINES = ("clean", "noisy", "repaired")


def compute_bleu_per_row(predictions: list[str], references: list[str]) -> list[float]:
    scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions),
                          desc="Computing BLEU"):
        scores.append(compute_bleu_score(pred, ref))
    return scores


def compute_chrf_per_row(predictions: list[str], references: list[str]) -> list[float]:
    scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions),
                          desc="Computing chrF"):
        scores.append(compute_chrf_score(pred, ref))
    return scores


def compute_rouge_l_per_row(predictions: list[str], references: list[str]) -> list[float]:
    scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions),
                          desc="Computing ROUGE-L"):
        scores.append(compute_rouge_l_score(pred, ref))
    return scores


def compute_token_f1_per_row(predictions: list[str], references: list[str]) -> list[float]:
    scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions),
                          desc="Computing token F1"):
        scores.append(compute_token_f1_score(pred, ref))
    return scores


def compute_exact_match_per_row(predictions: list[str], references: list[str]) -> list[float]:
    scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions),
                          desc="Computing exact match"):
        scores.append(compute_exact_match_score(pred, ref))
    return scores


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """BERTScore F1 for each (prediction, reference) pair."""
    logger.info("Computing BERTScore for %d pairs...", len(predictions))
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=True)
    return f1.tolist()


def run():
    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    metrics_path = out_dir / "metrics.parquet"
    existing_metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
    existing_geval_lookup = build_completed_lookup(
        existing_metrics,
        ["pipeline", "id", "noise_type"],
        ["geval_score"],
    )

    all_metrics = []

    for pipeline in PIPELINES:
        path = out_dir / f"answers_{pipeline}.parquet"
        if not path.exists():
            logger.warning("Skipping %s - file not found: %s", pipeline, path)
            continue

        df = pd.read_parquet(path)
        logger.info("Pipeline=%s: %d rows", pipeline, len(df))

        preds = df["answer_pred"].tolist()
        ref_col = "answer_ref" if "answer_ref" in df.columns else "answer"
        refs = df[ref_col].tolist()

        metrics_df = df[["id"]].copy() if "id" in df.columns else pd.DataFrame()
        if "noise_type" in df.columns:
            metrics_df["noise_type"] = df["noise_type"].values
        else:
            metrics_df["noise_type"] = ""
        metrics_df["pipeline"] = pipeline
        metrics_df["bleu"] = compute_bleu_per_row(preds, refs)
        metrics_df["chrf"] = compute_chrf_per_row(preds, refs)
        metrics_df["rouge_l"] = compute_rouge_l_per_row(preds, refs)
        metrics_df["token_f1"] = compute_token_f1_per_row(preds, refs)
        metrics_df["exact_match"] = compute_exact_match_per_row(preds, refs)
        metrics_df["bertscore_f1"] = compute_bertscore(preds, refs)
        metrics_df["answer_ref"] = refs
        metrics_df["answer_pred"] = preds
        for q_col in ["question", "question_clean", "question_noisy", "question_repaired"]:
            if q_col in df.columns:
                metrics_df[q_col] = df[q_col].values

        clean_questions = None
        candidate_questions = None
        if "question_clean" in df.columns:
            clean_questions = df["question_clean"].fillna("").astype(str).tolist()
        elif "question" in df.columns:
            clean_questions = df["question"].fillna("").astype(str).tolist()

        if clean_questions is not None:
            if pipeline == "clean":
                candidate_questions = clean_questions
            elif pipeline == "noisy" and "question_noisy" in df.columns:
                candidate_questions = df["question_noisy"].fillna("").astype(str).tolist()
            elif pipeline == "repaired" and "question_repaired" in df.columns:
                candidate_questions = df["question_repaired"].fillna("").astype(str).tolist()

        if clean_questions is not None and candidate_questions is not None:
            metrics_df["intent_preservation"] = compute_intent_preservation_scores(
                clean_questions,
                candidate_questions,
            )
        else:
            metrics_df["intent_preservation"] = [None] * len(metrics_df)

        if existing_geval_lookup:
            geval_scores = []
            for _, metric_row in metrics_df.iterrows():
                key = row_key(metric_row, ["pipeline", "id", "noise_type"])
                prior = existing_geval_lookup.get(key, {})
                geval_scores.append(prior.get("geval_score"))
            metrics_df["geval_score"] = geval_scores

        all_metrics.append(metrics_df)

    if not all_metrics:
        logger.error("No answer files found. Run answer.py first.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)
    write_parquet_atomic(combined, metrics_path)
    logger.info("Saved metrics for %d rows to %s", len(combined), metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.parse_args()
    run()
