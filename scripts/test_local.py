"""End-to-end local test: runs the full pipeline on a small sample using local models.

Supports two modes:
    python scripts/test_local.py           # CPU: BioGPT + Flan-T5
    python scripts/test_local.py --gpu     # GPU: Mistral-7B-Instruct

No OpenAI API key required — everything runs offline via HuggingFace models.
"""

import argparse
import json
import os
import shutil
import sys
import types

# Block broken TensorFlow BEFORE anything else imports it
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import importlib.machinery
_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "0.0.0"
_tf_stub.__path__ = []
_tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _tf_stub

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from bert_score import score as bert_score_fn

from src.metrics import (
    compute_bleu_score,
    compute_chrf_score,
    compute_exact_match_score,
    compute_rouge_l_score,
    compute_token_f1_score,
)
from src.utils import load_prompts, setup_logging, load_config
from src.noise import generate_noisy_variant
from src.answer import answer_question
from src.repair import repair_question
from src.judge import judge_answer


logger = setup_logging()


def load_samples(n: int = 5) -> pd.DataFrame:
    """Load n samples from the cleaned dataset and format like ingest.py output."""
    cleaned = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.parquet"
    if not cleaned.exists():
        cleaned_csv = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.csv"
        if not cleaned_csv.exists():
            logger.error("No cleaned data found. Run data_cleaning notebook first.")
            sys.exit(1)
        df = pd.read_csv(cleaned_csv)
    else:
        df = pd.read_parquet(cleaned)

    df = df.head(n).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def clear_cache():
    """Remove cached LLM responses so a fresh run doesn't return stale outputs."""
    cache_dir = PROJECT_ROOT / "data" / "outputs" / "cache"
    if cache_dir.exists():
        files = list(cache_dir.glob("*.json"))
        for f in files:
            try:
                f.unlink()
            except PermissionError:
                pass
        logger.info("Cleared %d cached responses", len(files))


def log_gpu_info():
    """Print GPU name and memory if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            logger.info("GPU: %s (%.1f GB VRAM)", name, total)
        else:
            logger.info("No CUDA GPU detected — running on CPU")
    except Exception:
        logger.info("Could not query GPU info")


def run_test(config_path: str):
    cfg = load_config(config_path)
    prompts = load_prompts()
    n = cfg.get("n_examples", 5)
    model_name = cfg["answer_model"].split("/")[-1]

    print(f"\n{'='*70}")
    print(f"  LOCAL MODEL TEST — {model_name}")
    print(f"  Config: {Path(config_path).name}")
    print(f"{'='*70}\n")

    log_gpu_info()

    # ---- Load samples ----
    df = load_samples(n)
    logger.info("Loaded %d samples", len(df))

    for _, row in df.iterrows():
        logger.info("  [%d] %s", row["id"], row["question"][:80])

    # ---- Step 1: Noise injection ----
    print(f"\n--- Step 1: Noise injection (model: {cfg['noise_model']}) ---\n")
    noise_type = cfg["noise_types"][0]
    noisy_questions = []
    for _, row in df.iterrows():
        noisy_q = generate_noisy_variant(row["question"], noise_type, prompts, cfg)
        noisy_questions.append(noisy_q)
        logger.info("  Clean:  %s", row["question"][:80])
        logger.info("  Noisy:  %s\n", noisy_q[:80])

    # ---- Step 2: Answer clean questions ----
    print(f"\n--- Step 2: Answer clean questions (model: {cfg['answer_model']}) ---\n")
    answers_clean = []
    for _, row in df.iterrows():
        ans = answer_question(row["question"], prompts, cfg)
        answers_clean.append(ans)
        logger.info("  Q: %s", row["question"][:60])
        logger.info("  A: %s\n", ans[:120])

    # ---- Step 3: Answer noisy questions ----
    print(f"\n--- Step 3: Answer noisy questions (model: {cfg['answer_model']}) ---\n")
    answers_noisy = []
    for noisy_q in noisy_questions:
        ans = answer_question(noisy_q, prompts, cfg)
        answers_noisy.append(ans)
        logger.info("  Q: %s", noisy_q[:60])
        logger.info("  A: %s\n", ans[:120])

    # ---- Step 4: Repair noisy questions ----
    print(f"\n--- Step 4: Repair noisy questions (model: {cfg['repair_model']}) ---\n")
    repaired_questions = []
    for noisy_q in noisy_questions:
        repaired_q = repair_question(noisy_q, prompts, cfg)
        repaired_questions.append(repaired_q)
        logger.info("  Noisy:    %s", noisy_q[:70])
        logger.info("  Repaired: %s\n", repaired_q[:70])

    # ---- Step 5: Answer repaired questions ----
    print(f"\n--- Step 5: Answer repaired questions (model: {cfg['answer_model']}) ---\n")
    answers_repaired = []
    for repaired_q in repaired_questions:
        ans = answer_question(repaired_q, prompts, cfg)
        answers_repaired.append(ans)
        logger.info("  Q: %s", repaired_q[:60])
        logger.info("  A: %s\n", ans[:120])

    # ---- Step 6: Evaluate (lexical + semantic metrics) ----
    print("\n--- Step 6: Evaluation (lexical + semantic metrics) ---\n")
    refs = df["answer"].tolist()

    def compute_bleu(preds, refs):
        return [compute_bleu_score(p, r) for p, r in zip(preds, refs)]

    def compute_chrf(preds, refs):
        return [compute_chrf_score(p, r) for p, r in zip(preds, refs)]

    def compute_rouge_l(preds, refs):
        return [compute_rouge_l_score(p, r) for p, r in zip(preds, refs)]

    def compute_token_f1(preds, refs):
        return [compute_token_f1_score(p, r) for p, r in zip(preds, refs)]

    def compute_exact_match(preds, refs):
        return [compute_exact_match_score(p, r) for p, r in zip(preds, refs)]

    def compute_bertscore(preds, refs):
        _, _, f1 = bert_score_fn(preds, refs, lang="en", verbose=False)
        return f1.tolist()

    bleu_clean = compute_bleu(answers_clean, refs)
    bleu_noisy = compute_bleu(answers_noisy, refs)
    bleu_repaired = compute_bleu(answers_repaired, refs)

    chrf_clean = compute_chrf(answers_clean, refs)
    chrf_noisy = compute_chrf(answers_noisy, refs)
    chrf_repaired = compute_chrf(answers_repaired, refs)

    rouge_clean = compute_rouge_l(answers_clean, refs)
    rouge_noisy = compute_rouge_l(answers_noisy, refs)
    rouge_repaired = compute_rouge_l(answers_repaired, refs)

    token_f1_clean = compute_token_f1(answers_clean, refs)
    token_f1_noisy = compute_token_f1(answers_noisy, refs)
    token_f1_repaired = compute_token_f1(answers_repaired, refs)

    exact_clean = compute_exact_match(answers_clean, refs)
    exact_noisy = compute_exact_match(answers_noisy, refs)
    exact_repaired = compute_exact_match(answers_repaired, refs)

    bert_clean = compute_bertscore(answers_clean, refs)
    bert_noisy = compute_bertscore(answers_noisy, refs)
    bert_repaired = compute_bertscore(answers_repaired, refs)

    # ---- Step 7: LLM Judge ----
    questions = df["question"].tolist()
    print(f"\n--- Step 7: LLM Judge (model: {cfg['judge_model']}) ---\n")
    judge_clean = [judge_answer(r, p, prompts, cfg, question=q)
                   for q, r, p in zip(questions, refs, answers_clean)]
    judge_noisy = [judge_answer(r, p, prompts, cfg, question=q)
                   for q, r, p in zip(questions, refs, answers_noisy)]
    judge_repaired = [judge_answer(r, p, prompts, cfg, question=q)
                      for q, r, p in zip(questions, refs, answers_repaired)]

    # ---- Summary ----
    def mean(lst):
        return np.mean(lst) if lst else 0.0

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Clean':>10} {'Noisy':>10} {'Repaired':>10}")
    print(f"  {'-'*50}")
    print(f"  {'BLEU':<20} {mean(bleu_clean):>10.2f} {mean(bleu_noisy):>10.2f} {mean(bleu_repaired):>10.2f}")
    print(f"  {'chrF':<20} {mean(chrf_clean):>10.2f} {mean(chrf_noisy):>10.2f} {mean(chrf_repaired):>10.2f}")
    print(f"  {'ROUGE-L F1':<20} {mean(rouge_clean):>10.2f} {mean(rouge_noisy):>10.2f} {mean(rouge_repaired):>10.2f}")
    print(f"  {'Token F1':<20} {mean(token_f1_clean):>10.2f} {mean(token_f1_noisy):>10.2f} {mean(token_f1_repaired):>10.2f}")
    print(f"  {'Exact Match':<20} {mean(exact_clean):>10.2f} {mean(exact_noisy):>10.2f} {mean(exact_repaired):>10.2f}")
    print(f"  {'BERTScore F1':<20} {mean(bert_clean):>10.4f} {mean(bert_noisy):>10.4f} {mean(bert_repaired):>10.4f}")
    print(f"  {'Judge (0-3)':<20} {mean(judge_clean):>10.2f} {mean(judge_noisy):>10.2f} {mean(judge_repaired):>10.2f}")
    print()

    degradation_bert = mean(bert_clean) - mean(bert_noisy)
    recovery_bert = mean(bert_repaired) - mean(bert_noisy)
    ratio = recovery_bert / degradation_bert if degradation_bert != 0 else float("nan")

    print(f"  Robustness (BERTScore):")
    print(f"    Degradation (clean - noisy):     {degradation_bert:+.4f}")
    print(f"    Recovery (repaired - noisy):      {recovery_bert:+.4f}")
    print(f"    Recovery Ratio:                   {ratio:.2f}")
    print(f"\n{'='*70}")
    print("  Test complete — all steps ran locally, zero API calls.")
    print(f"{'='*70}\n")

    # ---- Save report data as JSON for report generation ----
    report = {
        "config": Path(config_path).name,
        "model": cfg["answer_model"],
        "samples": [],
        "bleu": {"clean": bleu_clean, "noisy": bleu_noisy, "repaired": bleu_repaired},
        "chrf": {"clean": chrf_clean, "noisy": chrf_noisy, "repaired": chrf_repaired},
        "rouge_l": {"clean": rouge_clean, "noisy": rouge_noisy, "repaired": rouge_repaired},
        "token_f1": {"clean": token_f1_clean, "noisy": token_f1_noisy, "repaired": token_f1_repaired},
        "exact_match": {"clean": exact_clean, "noisy": exact_noisy, "repaired": exact_repaired},
        "bertscore": {"clean": bert_clean, "noisy": bert_noisy, "repaired": bert_repaired},
        "judge": {"clean": judge_clean, "noisy": judge_noisy, "repaired": judge_repaired},
    }
    for i, row in df.iterrows():
        report["samples"].append({
            "id": int(row["id"]),
            "question_clean": row["question"],
            "question_noisy": noisy_questions[i],
            "question_repaired": repaired_questions[i],
            "answer_ref": refs[i][:500],
            "answer_clean": answers_clean[i],
            "answer_noisy": answers_noisy[i],
            "answer_repaired": answers_repaired[i],
        })

    out_path = PROJECT_ROOT / "data" / "outputs" / "report_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved report data to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local model test")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU config (Mistral-7B-Instruct) instead of CPU config")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a custom config YAML (overrides --gpu)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached LLM responses before running")
    args = parser.parse_args()

    if args.config:
        config_path = str(Path(args.config).resolve())
    elif args.gpu:
        config_path = str(PROJECT_ROOT / "configs" / "experiment_local_gpu.yaml")
    else:
        config_path = str(PROJECT_ROOT / "configs" / "experiment_local.yaml")

    if args.clear_cache:
        clear_cache()

    run_test(config_path)
