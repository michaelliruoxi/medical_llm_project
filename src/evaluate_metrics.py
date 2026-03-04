"""Step E (part 1): Compute BLEU and BERTScore for each pipeline's predictions."""

import argparse

import pandas as pd
import sacrebleu
from bert_score import score as bert_score_fn
from tqdm import tqdm

from src.utils import PROJECT_ROOT, load_config, setup_logging


logger = setup_logging()

PIPELINES = ("clean", "noisy", "repaired")


def compute_bleu_per_row(predictions: list[str], references: list[str]) -> list[float]:
    """Sentence-level BLEU for each (prediction, reference) pair."""
    scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions),
                          desc="Computing BLEU"):
        bleu = sacrebleu.sentence_bleu(pred, [ref])
        scores.append(bleu.score)
    return scores


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """BERTScore F1 for each (prediction, reference) pair."""
    logger.info("Computing BERTScore for %d pairs...", len(predictions))
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=True)
    return f1.tolist()


def run():
    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]

    all_metrics = []

    for pipeline in PIPELINES:
        path = out_dir / f"answers_{pipeline}.parquet"
        if not path.exists():
            logger.warning("Skipping %s — file not found: %s", pipeline, path)
            continue

        df = pd.read_parquet(path)
        logger.info("Pipeline=%s: %d rows", pipeline, len(df))

        preds = df["answer_pred"].tolist()

        ref_col = "answer_ref" if "answer_ref" in df.columns else "answer"
        refs = df[ref_col].tolist()

        bleu_scores = compute_bleu_per_row(preds, refs)
        bertscore_f1 = compute_bertscore(preds, refs)

        metrics_df = df[["id"]].copy() if "id" in df.columns else pd.DataFrame()
        if "noise_type" in df.columns:
            metrics_df["noise_type"] = df["noise_type"].values
        metrics_df["pipeline"] = pipeline
        metrics_df["bleu"] = bleu_scores
        metrics_df["bertscore_f1"] = bertscore_f1
        metrics_df["answer_ref"] = refs
        metrics_df["answer_pred"] = preds

        all_metrics.append(metrics_df)

    if not all_metrics:
        logger.error("No answer files found. Run answer.py first.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)
    metrics_path = out_dir / "metrics.parquet"
    combined.to_parquet(metrics_path, index=False)
    logger.info("Saved metrics for %d rows to %s", len(combined), metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.parse_args()
    run()
