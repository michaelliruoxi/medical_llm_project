# Preliminary Local Model Comparison Report

Date: 2026-03-29

## Scope

This report summarizes the first resumable local multi-model comparison run from
`scripts/run_model_comparison.py` using:

- 50 MedQuAD examples per model
- 5 noise types per example
- 250 saved sample rows per completed model
- local Hugging Face models on a single 32 GB GPU

The source table for this report is
`data/outputs/comparison/model_comparison.csv`.

## Current status

Eight models completed successfully:

| Model | Quant | Status | Rows | Runtime (min) |
|---|---|---|---:|---:|
| BioMistral-7B | none | COMPLETED | 250/250 | 0.0 |
| Mistral-7B-Instruct-v0.3 | none | COMPLETED | 250/250 | 0.0 |
| gemma-2-9b-it | none | COMPLETED | 250/250 | 0.0 |
| Llama-3.1-8B-Instruct | none | COMPLETED | 250/250 | 102.7 |
| Phi-3-medium-4k-instruct | 4bit | COMPLETED | 250/250 | 259.6 |
| Qwen2.5-14B-Instruct | 4bit | COMPLETED | 250/250 | 163.3 |
| Qwen2.5-32B-Instruct | 4bit | COMPLETED | 250/250 | 229.8 |
| Mixtral-8x7B-Instruct-v0.1 | 4bit | COMPLETED | 250/250 | 239.3 |

One model was stopped intentionally:

| Model | Quant | Status | Rows | Runtime (min) | Note |
|---|---|---|---:|---:|---|
| gpt-oss-20b | none | PARTIAL | 33/250 | 237.5 | throughput too slow for the current local comparison loop |

`gpt-oss-20b` remains in the repo for reference, but it is now disabled from
default preflight and default comparison runs.

## Early takeaways

- The resumable comparison pipeline worked as intended. Completed rows were
  continuously written to `samples_<model>.csv`, and interrupted runs preserved
  progress in `progress_<model>.json`.
- The current practical local lineup is the 8 completed models above.
- Among the completed models, Mixtral currently leads the automatic metrics on
  the saved summary table:
  - highest `bleu_clean`: 4.12
  - highest `bleu_noisy`: 3.07
  - highest `bertscore_clean`: 0.8679
- Repair is not yet showing a consistent win across models. On the current
  saved table, repaired answers often trail clean performance and sometimes
  trail noisy performance as well. That may reflect real behavior, metric
  instability, or both.

## Important caveats

These results should be treated as preliminary rather than publication-ready.

1. Judge-score parsing needs hardening.
   The parser in `src/judge.py` expects a 0-3 score, but several local judge
   outputs contained `Score: 4` or `Score: 5`. Those cases currently default to
   0, which can distort `judge_*` columns in the comparison outputs.
   This has now been patched in code by clamping 4-5 to 3, but the saved March
   29 outputs in this report still reflect the pre-patch run and should be
   treated as provisional.

2. The comparison pipeline is row-safe, but `run_all.py` is not yet.
   `scripts/run_model_comparison.py` now supports stop/resume safely at the
   sample-row level. The older staged pipeline in `scripts/run_all.py` is still
   stage-safe rather than row-safe.

3. Recovery percentages need cautious interpretation.
   When degradation is small or flips sign, recovery ratios can look extreme.
   Those columns are still useful for quick scans, but they should be paired
   with raw metric deltas before drawing conclusions.

## Recommended next steps

1. Freeze the current 8-model lineup as the default benchmark set.
   Keep `gpt-oss-20b` opt-in only unless a faster inference path is added.

2. Fix the judge parser and prompt format.
   Force the judge output into a strict 0-3 final token, then rerun judge
   scoring from saved sample rows instead of regenerating every answer.

3. Add a post-processing pass for saved comparison CSVs.
   The current resumable data is rich enough to support a follow-up script that
   recomputes summary tables and repaired judge scores without rerunning model
   generation.

4. Produce clean analysis artifacts for the 8-model set.
   Export a concise table of clean/noisy/repaired deltas, per-noise breakdowns,
   and a few figures for the project write-up.

5. Only scale beyond 50 examples after metric validation.
   Once judge parsing is fixed and summary formulas are verified, increase
   `n_examples` for a fuller benchmark pass.

## Suggested decision for the next phase

Use the 8 completed models as the working comparison set, treat the current
automatic metrics as informative, treat the judge metrics as provisional, and
prioritize metric cleanup before running any larger benchmark.
