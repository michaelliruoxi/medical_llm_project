# MedQuAD Robustness Project Plan

## Prompt Repair and Robustness Under Noisy User Input

**Generated on:** 2026-03-03 22:43:31

------------------------------------------------------------------------

## 1. Project Goal

Test whether **prompt repair systematically improves medical QA
robustness under noisy input** using 1,000 MedQuAD entries.

### Pipelines

-   **Pipeline A:** Q_clean → LLM → A_clean\
-   **Pipeline B:** Q_noisy → LLM → A_noisy\
-   **Pipeline C:** Q_noisy → Repair → LLM → A_repaired

Evaluation will compare degradation and recovery across pipelines.

------------------------------------------------------------------------

## 2. Repository Structure

    medquad-robustness/
      README.md
      requirements.txt
      .env.example
      configs/
        experiment.yaml
        prompts.yaml
      data/
        raw/
        processed/
        outputs/
      src/
        ingest.py
        noise.py
        repair.py
        answer.py
        evaluate_metrics.py
        judge.py
        aggregate.py
        report_tables.py
        utils.py
      notebooks/
      scripts/
        run_all.sh

------------------------------------------------------------------------

## 3. Configuration

### experiment.yaml

-   n_examples: 1000\
-   noise_variants_per_question: 1\
-   noise_types:
    -   typos_grammar
    -   ambiguity
    -   layperson
    -   incomplete
    -   overgeneralization
-   answer_model: gpt-4o
-   noise_model: gpt-4o-mini
-   judge_model: gpt-4o-mini
-   temperature_answer: 0.2
-   temperature_noise: 0.8
-   temperature_repair: 0.2
-   max_tokens_answer: 300
-   max_tokens_repair: 120

------------------------------------------------------------------------

## 4. Pipeline Steps

### Step A --- Data Ingestion

-   Load 1,000 MedQuAD entries
-   Clean and normalize
-   Deduplicate
-   Save as parquet

### Step B --- Noise Injection

Generate noisy variants while preserving medical intent.

Noise categories: - Grammatical errors - Ambiguity - Layperson
phrasing - Incomplete questions - Overgeneralization

### Step C --- Baseline Answering

Generate: - A_clean_pred - A_noisy_pred

### Step D --- Prompt Repair

-   Rewrite noisy question
-   Ensure no answering in repair step
-   Generate A_repaired_pred

------------------------------------------------------------------------

## 5. Evaluation

### Automatic Metrics

-   BLEU
-   BERTScore (F1)

### LLM Judge

Score correctness relative to reference answer.

Scale example: - 0 = Incorrect - 1 = Partially correct - 2 = Mostly
correct - 3 = Fully correct

------------------------------------------------------------------------

## 6. Robustness Metrics

-   Degradation = Clean − Noisy\
-   Recovery = Repaired − Noisy\
-   Recovery Ratio = Recovery / Degradation

Compute for: - BERTScore - BLEU - Judge Score

------------------------------------------------------------------------

## 7. Statistical Testing

-   Paired tests (clean vs noisy, noisy vs repaired)
-   Bootstrap confidence intervals
-   Effect sizes
-   Breakdown by noise type

------------------------------------------------------------------------

## 8. Logging and Reproducibility

-   Cache API outputs
-   Log token usage
-   Store prompts and model versions
-   Save outputs per run folder

------------------------------------------------------------------------

## 9. Execution Plan

1.  Run 20-sample pilot
2.  Inspect outputs manually
3.  Validate judge rubric
4.  Scale to 1,000 entries
5.  Aggregate results
6.  Generate tables and plots

------------------------------------------------------------------------

## 10. Deliverables

-   Reproducible codebase
-   Evaluation tables
-   Robustness metrics
-   Final report-ready summary
