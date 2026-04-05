# MedQuAD Robustness

This repo studies whether prompt repair helps medical QA systems stay accurate under noisy user input.

There are two workflows:

- `scripts/experiments/run_pipeline.py`: one-config sequential experiment
- `scripts/benchmarks/run_comparison.py`: resumable multi-model benchmark

## What Is Going On Right Now

The latest completed benchmark is:

- [data/outputs/comparison_50_round2](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2)

That run contains:

- 10 completed models
- 50 questions per model
- 1 deterministic noise type per question
- 5 noise types total, evenly spread by round-robin assignment
- 3 answer states per row: `clean`, `noisy`, `repaired`

Current enabled benchmark lineup:

- `gemma-4-local (Q6_K-ref)`
- `BioMistral-7B`
- `Mistral-7B-Instruct-v0.3`
- `gemma-2-9b-it`
- `Llama-3.1-8B-Instruct`
- `Phi-3-medium-4k-instruct (4bit)`
- `Qwen2.5-14B-Instruct (4bit)`
- `Qwen2.5-32B-Instruct (4bit)`
- `Mixtral-8x7B-Instruct-v0.1 (4bit)`
- `gpt-5.4`

Current metrics in the benchmark:

- BLEU
- chrF
- ROUGE-L
- token F1
- exact match
- BERTScore
- intent preservation (SBERT similarity to the clean question)
- G-Eval via OpenAI API

## Important Caveat

The current benchmark is a good pilot, not final proof of the project claim.

Why:

- it is a `50`-example pilot, not the final `1,000`-example study
- the benchmark is currently an end-to-end per-model pipeline comparison
- each model generates its own noisy and repaired questions, so models are not yet being scored on identical noisy/repaired text
- some repair outputs still need stricter validation to guarantee they are true rewrites
- statistical testing and bootstrap confidence intervals are not implemented yet

So the current results are legitimate as preliminary evidence, but not yet the final publishable answer to whether prompt repair systematically improves robustness.

See:

- [reports/next_steps_results_audit_2026-04-05.md](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/reports/next_steps_results_audit_2026-04-05.md)
- [data/outputs/comparison_50_round2/model_comparison.csv](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/model_comparison.csv)

## Main Files

```text
configs/
  models/
  experiments/
  prompts.yaml
scripts/
  benchmarks/
    preflight.py
    run_comparison.py
  experiments/
    run_pipeline.py
    smoke_test.py
src/
data/
reports/
tests/
```

## Setup

Requirements:

- Python 3.11 recommended
- local GPU for local-model benchmarks
- OpenAI API key for API-backed models and G-Eval

Raw MedQuAD is not committed. Put it under:

- `data/raw/MedQuAD`

Cleaned dataset expected by the runners:

- `data/processed/medquad_cleaned.csv`
- `data/processed/medquad_cleaned.parquet`

For OpenAI runs, add your key to:

- `.env`

Example:

```env
OPENAI_API_KEY=your_key_here
```

## Run

Preflight enabled benchmark configs:

```bash
python scripts/benchmarks/preflight.py
```

Run the resumable benchmark:

```bash
python scripts/benchmarks/run_comparison.py
```

Run the benchmark into a separate output folder:

```bash
python scripts/benchmarks/run_comparison.py --output-dir data/outputs/comparison_50_round2
```

Run the sequential experiment with the default API config:

```bash
python scripts/experiments/run_pipeline.py
```

Run a 50-example experiment:

```bash
python scripts/experiments/run_pipeline.py --n 50
```

## Active Default API Config

The main API profile is:

- [configs/models/gpt54_hybrid_api.yaml](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/gpt54_hybrid_api.yaml)

It currently uses:

- answer model: `gpt-5.4`
- repair model: `gpt-5.4`
- noise model: `gpt-5.4-mini`
- G-Eval model: `gpt-5.4-mini`

## Outputs

Per-model benchmark artifacts:

- `samples_<model>.csv`: row-level durable outputs
- `progress_<model>.json`: resumable checkpoint
- `result_<model>.json`: per-model aggregate summary

Run-level benchmark artifacts:

- `model_comparison.csv`
- `model_comparison_full.json`

Sequential experiment outputs go under:

- `data/outputs/experiments/<profile>/`

## Resume Behavior

Both main workflows are designed to resume:

- benchmark runs skip completed sample rows
- sequential runs reuse saved stage outputs for the same config

You can stop a long run and restart it without losing completed work.

## Next Project Step

The next serious improvement is to freeze one shared noisy-question set and one shared repaired-question set, then score all answer models on those exact same inputs. That will turn the current pilot into a much cleaner model comparison.
