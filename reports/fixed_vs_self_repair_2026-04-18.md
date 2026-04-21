# fixed_repair vs self_repair - 2026-04-18

## Scope

- Compared completed rows shared across both modes: `11` model(s)
- Focus metrics: `med_coverage`, `med_f1`, `geval`, `intent_preservation`, and repaired-vs-noisy recovery

## Headline

The result is mixed across models: neither repair strategy dominates every metric, so the main story depends on whether we prioritize medical content retention or surface-level intent cleanup.

## Average repaired scores by model

| metric | fixed_repair_mean | self_repair_mean | self_minus_fixed |
| --- | --- | --- | --- |
| med_coverage | 16.560 | 14.670 | -1.890 |
| med_f1 | 16.613 | 15.656 | -0.957 |
| geval | 3.353 | 3.409 | 0.056 |
| intent_preservation | 0.631 | 0.619 | -0.012 |

## Average repaired-vs-noisy gains by model

| metric | fixed_repaired_minus_noisy | self_repaired_minus_noisy | self_minus_fixed |
| --- | --- | --- | --- |
| med_coverage | -1.008 | -2.898 | -1.890 |
| med_f1 | -0.810 | -1.767 | -0.957 |
| geval | -0.120 | -0.064 | 0.056 |
| intent_preservation | 0.089 | 0.077 | -0.012 |

## Average recovery percentages by model

| metric | fixed_recovery_pct | self_recovery_pct | self_minus_fixed_pct |
| --- | --- | --- | --- |
| med_coverage | -30.375 | -81.000 | -50.625 |
| med_f1 | -27.889 | -58.778 | -30.889 |
| geval | -33.250 | -21.875 | 11.375 |
| intent_preservation | 19.000 | 16.727 | -2.273 |

## Model win counts on repaired scores

| metric | fixed_model_wins | self_model_wins | ties |
| --- | --- | --- | --- |
| med_coverage | 9 | 2 | 0 |
| med_f1 | 8 | 3 | 0 |
| geval | 8 | 3 | 0 |
| intent_preservation | 8 | 3 | 0 |

## Notes

- Recovery percentages come from the benchmark summary columns and reflect how much the repaired answer climbs back toward the clean-answer score from the noisy-answer score.
- Repaired-vs-noisy deltas are raw score differences (`repaired - noisy`) and help separate true recovery from overall baseline model strength.
- This note uses completed models only, so rerunning it later will automatically reflect any updated benchmark outputs.
