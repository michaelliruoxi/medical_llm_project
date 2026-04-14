# MedQuAD Robustness

This repo tests whether prompt repair helps medical QA models stay accurate under noisy user questions.

## Current State

- Latest finished pilot benchmark: [data/outputs/comparison_50_round2](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2)
- 10 models completed
- 50 questions per model
- 5 noise types, evenly assigned by round robin
- Metrics: BLEU, chrF, ROUGE-L, token F1, exact match, BERTScore, intent preservation, and G-Eval

This pilot is useful, but it is not the final claim-ready study yet. The main audit note is here:

- [reports/next_steps_results_audit_2026-04-05.md](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/reports/next_steps_results_audit_2026-04-05.md)

## Main Workflows

- `scripts/benchmarks/run_comparison.py`
  Resumable multi-model benchmark
- `scripts/benchmarks/build_fixed_question_sets.py`
  Freezes one shared noisy dataset and one shared repaired dataset with GPT-5.4
- `scripts/experiments/run_pipeline.py`
  Sequential single-config pipeline

## Benchmark Modes

- `end_to_end`
  Each model generates its own noisy and repaired questions
- `fixed_repair`
  All models answer the same frozen noisy questions and the same frozen repaired questions
- `self_repair`
  All models answer the same frozen noisy questions, then repair those noisy questions themselves

The fixed question sets live by default under:

- [data/processed/benchmarks/fixed_question_sets_gpt54](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/processed/benchmarks/fixed_question_sets_gpt54)

The frozen question-set files are CSV-first:

- `clean_fixed.csv`
- `noisy_fixed_gpt54.csv`
- `repaired_fixed_gpt54.csv`

## New Benchmark Pipeline

The new benchmark flow is:

1. Sample one clean MedQuAD subset.
2. Use GPT-5.4 to generate one shared noisy-question set.
3. Use GPT-5.4 to generate one shared repaired-question set from that noisy set.
4. Run `fixed_repair`:
   all answer models see the same clean, noisy, and repaired questions.
5. Run `self_repair`:
   all answer models see the same clean and noisy questions, then each model repairs the noisy question itself.

This gives two separate comparisons:

- controlled input comparison: `fixed_repair`
- end-to-end pipeline comparison: `self_repair`

## Active API Profile

The main API-backed benchmark config is:

- [configs/models/gpt54_hybrid_api.yaml](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/gpt54_hybrid_api.yaml)

It currently uses:

- answer: `gpt-5.4`
- repair: `gpt-5.4`
- noise: `gpt-5.4-mini`
- G-Eval: `gpt-5.4-mini`

## Quick Start

Put your OpenAI key in:

- [.env](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/.env)

```env
OPENAI_API_KEY=your_key_here
```

Build the shared GPT-5.4 question sets:

```bash
python scripts/benchmarks/build_fixed_question_sets.py --n 50
```

This build is resumable and writes:

- `clean_fixed.csv`
- `noisy_fixed_gpt54.csv`
- `repaired_fixed_gpt54.csv`
- `question_set_manifest.json`
- `question_set_progress.json`

Run the controlled fixed-repair benchmark:

```bash
python scripts/benchmarks/run_comparison.py --benchmark-mode fixed_repair
```

Run the shared-noisy self-repair benchmark:

```bash
python scripts/benchmarks/run_comparison.py --benchmark-mode self_repair
```

Run the legacy end-to-end benchmark:

```bash
python scripts/benchmarks/run_comparison.py --benchmark-mode end_to_end
```

Preflight enabled model configs:

```bash
python scripts/benchmarks/preflight.py
```

The Gemma 4 benchmark config [gemma4_31b_q6k_ref.yaml](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/gemma4_31b_q6k_ref.yaml) now runs directly through the local Transformers backend in `8bit`, so it no longer depends on a separate local HTTP server.

## Outputs

Each benchmark run writes:

- `samples_<model>.csv`
- `progress_<model>.json`
- `result_<model>.json`
- `model_comparison.csv`
- `model_comparison_full.json`

Default benchmark output folders:

- `end_to_end`: `data/outputs/comparison`
- `fixed_repair`: `data/outputs/benchmarks/fixed_repair`
- `self_repair`: `data/outputs/benchmarks/self_repair`

Runs are resumable. You can stop and restart without losing completed rows.
