# fixed_repair vs self_repair - 2026-05-02

## Scope

- Compared completed rows shared across both modes: `11` model(s)
- Focus metrics: `med_coverage`, `med_f1`, `geval`, `intent_preservation`, and repaired-vs-noisy recovery

## Headline

The result is mixed across models: neither repair strategy dominates every metric, so the main story depends on whether we prioritize medical content retention or surface-level intent cleanup.

## Average repaired scores by model

| metric | fixed_repair_mean | self_repair_mean | self_minus_fixed |
| --- | --- | --- | --- |
| med_coverage | 13.389 | 12.329 | -1.060 |
| med_f1 | 13.719 | 13.049 | -0.670 |
| geval | 2.660 | 2.543 | -0.117 |
| intent_preservation | 0.717 | 0.702 | -0.015 |

## Average repaired-vs-noisy gains by model

| metric | fixed_repaired_minus_noisy | self_repaired_minus_noisy | self_minus_fixed |
| --- | --- | --- | --- |
| med_coverage | -0.045 | -1.105 | -1.060 |
| med_f1 | 0.063 | -0.607 | -0.670 |
| geval | -0.064 | -0.181 | -0.117 |
| intent_preservation | 0.099 | 0.083 | -0.015 |

## Average recovery percentages by model

| metric | fixed_recovery_pct | self_recovery_pct | self_minus_fixed_pct |
| --- | --- | --- | --- |
| med_coverage | 8.000 | -30.600 | -38.600 |
| med_f1 | 2.300 | -30.000 | -32.300 |
| geval | -20.273 | -58.818 | -38.545 |
| intent_preservation | 26.000 | 21.909 | -4.091 |

## Model win counts on repaired scores

| metric | fixed_model_wins | self_model_wins | ties |
| --- | --- | --- | --- |
| med_coverage | 10 | 1 | 0 |
| med_f1 | 10 | 1 | 0 |
| geval | 10 | 1 | 0 |
| intent_preservation | 8 | 3 | 0 |

## Notes

- Recovery percentages come from the benchmark summary columns and reflect how much the repaired answer climbs back toward the clean-answer score from the noisy-answer score.
- Repaired-vs-noisy deltas are raw score differences (`repaired - noisy`) and help separate true recovery from overall baseline model strength.
- This note uses completed models only, so rerunning it later will automatically reflect any updated benchmark outputs.
