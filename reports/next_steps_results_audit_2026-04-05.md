# Results Audit And Next Steps

Date: 2026-04-05

## Findings

- `[P1]` The cross-model benchmark is not a strict apples-to-apples comparison on identical noisy inputs. In the benchmark loop, each config generates its own noisy question and its own repaired question before answering them, so different models are not being evaluated on the same perturbed text: [run_comparison.py#L642](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/scripts/benchmarks/run_comparison.py#L642), [run_comparison.py#L645](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/scripts/benchmarks/run_comparison.py#L645). For example, BioMistral uses itself for answer, repair, and noise generation: [biomistral_7b.yaml#L16](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/biomistral_7b.yaml#L16), [biomistral_7b.yaml#L17](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/biomistral_7b.yaml#L17), [biomistral_7b.yaml#L18](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/biomistral_7b.yaml#L18), while GPT-5.4 uses a different noise model: [gpt54_hybrid_api.yaml#L20](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/gpt54_hybrid_api.yaml#L20), [gpt54_hybrid_api.yaml#L21](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/gpt54_hybrid_api.yaml#L21), [gpt54_hybrid_api.yaml#L22](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/configs/models/gpt54_hybrid_api.yaml#L22). That makes the run legitimate as a per-model end-to-end pipeline comparison, but not as a pure model-vs-model comparison on identical noisy prompts.

- `[P1]` The repair stage is not consistently behaving like a pure repair step, so some repaired rows are methodologically contaminated. In BioMistral alone, repaired questions sometimes ask for more context, inject new facts, or behave like answers instead of faithful rewrites: [samples_BioMistral-7B.csv#L3](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/samples_BioMistral-7B.csv#L3), [samples_BioMistral-7B.csv#L33](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/samples_BioMistral-7B.csv#L33), [samples_BioMistral-7B.csv#L35](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/samples_BioMistral-7B.csv#L35). That weakens any strong claim that Pipeline C is purely `Q_noisy -> repaired question -> answer`.

- `[P2]` The run does not yet satisfy the stated project goal as written. The plan explicitly says the goal is to test systematic improvement on 1,000 MedQuAD entries with statistical testing: [MedQuAD_Robustness_Project_Plan.md#L11](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/MedQuAD_Robustness_Project_Plan.md#L11), [MedQuAD_Robustness_Project_Plan.md#L12](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/MedQuAD_Robustness_Project_Plan.md#L12), [MedQuAD_Robustness_Project_Plan.md#L92](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/MedQuAD_Robustness_Project_Plan.md#L92), [MedQuAD_Robustness_Project_Plan.md#L142](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/MedQuAD_Robustness_Project_Plan.md#L142), [MedQuAD_Robustness_Project_Plan.md#L145](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/MedQuAD_Robustness_Project_Plan.md#L145), [MedQuAD_Robustness_Project_Plan.md#L165](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/MedQuAD_Robustness_Project_Plan.md#L165). The finished benchmark is a 50-example pilot per model, not the final 1,000-example claim.

- `[P3]` Some top-level metrics are informative, but not all are decision-grade. `exact_match_*` is zero for every model, so it is not adding signal here, and some recovery ratios are extreme because degradation is tiny or changes sign: [model_comparison.csv#L2](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/model_comparison.csv#L2), [model_comparison.csv#L11](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/model_comparison.csv#L11).

## Assessment

The results are legitimate as a completed pilot benchmark. All 10 models finished `50/50`, each sample file has 50 rows, and [model_comparison.csv](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/model_comparison.csv) correctly reflects the full run.

But this does not fully fulfill the project goal yet. It fulfills the engineering goal of running the pipeline end to end and producing a comparable pilot table. It does not yet fulfill the scientific claim that prompt repair systematically improves medical QA robustness because:

- the run is still a 50-example pilot, not 1,000
- the benchmark does not hold noisy/repaired inputs constant across models
- some repaired prompts are not clean repairs
- there is no paired significance/bootstrap layer yet

Substantively, the current pilot also does not show a clear systematic win for repair. Some models improve on some lexical metrics, but many repaired scores are flat or worse, and repaired G-Eval is often below noisy in the final table: [model_comparison.csv#L2](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/model_comparison.csv#L2), [model_comparison.csv#L11](C:/Users/Owner/OneDrive/Columbia%20University/Practicum/medical_llm_project/data/outputs/comparison_50_round2/model_comparison.csv#L11).

## Next Steps

-Freeze one shared noisy-question dataset from the clean MedQuAD sample.
-Generate one shared repaired-question dataset from that frozen noisy dataset using a single chosen repair model.
-Add a fixed-input benchmark mode:
-same clean questions, same noisy questions, same repaired questions for all answer models.
-Keep the current self-repair benchmark mode:
-each model repairs its own noisy input before answering.
-Save the two benchmark modes to separate output folders and label them clearly in summaries.
-Add result tables that compare:
-clean vs noisy vs repaired within each mode, and fixed-repair vs self-repair across modes.
-Add a short validation pass for repaired questions:
-reject outputs that are answers, meta text, or requests for more context.
-After both modes look stable on 50 examples, scale to the larger run.
