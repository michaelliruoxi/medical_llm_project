# fixed_repair QA - 2026-04-17

## Scope

Short QA pass on the completed `fixed_repair` benchmark package before starting `self_repair`.

## Checks completed

- Verified the full 11-model stats set under `data/outputs/benchmarks/fixed_repair/stats/`.
  - 55 CSV artifacts present total
  - 11 `paired_tests_*`
  - 11 `bootstrap_cis_*`
  - 11 `robustness_*`
  - 11 `summary_*`
  - 11 `summary_noise_*`
- Verified all 11 rows in `data/outputs/benchmarks/fixed_repair/model_comparison.csv` are `COMPLETED` with `50/50`.
- Fixed the `backfill_metrics.py` regression that dropped `runtime_seconds` when rebuilding summaries from sample CSVs.
- Restored the two runtime values still recoverable from the preserved fixed-repair run log:
  - `gemma-4-31B-it_(4bit)` -> `71.4` min
  - `Qwen3-32B_(4bit)` -> `81.0` min

## Remaining caveat

Earlier backfills had already overwritten the saved `result_*.json` summaries for the other nine fixed-repair models, so their original runtime values were not recoverable from local benchmark artifacts on disk during this pass. The bug is fixed for future runs, but the frozen historical baseline still has `0.0` runtime placeholders for those older rows.

## Frozen baseline

Archived snapshot:

- `data/outputs/benchmarks/fixed_repair/archive/2026-04-17_baseline/`

Contents:

- `model_comparison.csv`
- `model_comparison_full.json`
- `model_comparison.live.csv`
- `result_*.json`
- `stats/`
