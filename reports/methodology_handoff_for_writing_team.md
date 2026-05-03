# Medical LLM Robustness Project: Methodology Handoff for the Writing Team

Date: 2026-04-21

## Purpose of this document

This note is a paper-writing handoff built directly from the repository code, configs, benchmark scripts, and audit notes. It is meant to give the writing team a methods section source-of-truth that is more detailed than the README and more narrative than the raw code.

The most important framing point is this:

- The repository originally started with a single end-to-end robustness pipeline.
- After an internal audit, the project was redesigned into two cleaner benchmark modes: `fixed_repair` and `self_repair`.
- For the final paper, `fixed_repair` should be treated as the primary cross-model comparison methodology, `self_repair` should be treated as the practical end-to-end repair methodology, and the original `end_to_end` mode should be described as an earlier pilot design rather than the main claim-making benchmark.

## Project objective

The project tests whether prompt repair improves the robustness of medical question-answering systems when user questions are degraded by realistic noise.

At a high level, the project compares three input conditions:

1. Clean question -> model answer
2. Noisy question -> model answer
3. Noisy question -> repaired question -> model answer

The core scientific question is whether the repair step recovers answer quality lost under noisy input, and whether that recovery is consistent across different medical QA models and different categories of question corruption.

## Recommended paper framing

If the paper is written from the codebase as it exists now, the cleanest framing is:

- Primary controlled benchmark: `fixed_repair`
  - All answer models receive the same clean questions, the same noisy questions, and the same repaired questions.
  - This is the fairest setup for cross-model comparison because the model is not also controlling the perturbation it is being tested on.
- Secondary practical benchmark: `self_repair`
  - All answer models receive the same clean questions and the same noisy questions, but each model produces its own repaired version before answering.
  - This captures a more realistic "model repairs its own input" workflow.
- Legacy pilot benchmark: `end_to_end`
  - Each model generates its own noisy and repaired questions before answering them.
  - This is valid as a within-model pipeline study, but it is not a strict apples-to-apples cross-model benchmark because different models are tested on different noisy prompts.

If the final paper uses the completed `n=50` runs, it should explicitly call them a pilot. If the final paper uses the `n=1000` workflow, the same methodology applies, but the sample-size language should be updated accordingly.

## End-to-end methodology overview

The benchmark pipeline is built as a staged robustness workflow:

1. Ingest and clean MedQuAD question-answer pairs.
2. Sample a fixed set of examples.
3. Generate noisy versions of the clean questions.
4. Generate repaired versions of the noisy questions.
5. Ask answer models to answer the clean, noisy, and repaired questions.
6. Score the resulting answers against the MedQuAD reference answers with lexical, semantic, domain-specific, and LLM-judge metrics.
7. Compute degradation, recovery, and recovery-ratio statistics.
8. Run paired statistical tests and bootstrap confidence intervals.

The implementation is split across the following main components:

- `src/ingest.py`
- `src/noise.py`
- `src/repair.py`
- `src/answer.py`
- `src/metrics.py`
- `src/judge.py`
- `src/aggregate.py`
- `scripts/benchmarks/build_fixed_question_sets.py`
- `scripts/benchmarks/run_comparison.py`

## 1. Data source and preprocessing methodology

### 1.1 Dataset

The underlying QA corpus is MedQuAD, a medical question-answer dataset distributed as XML files. The project treats each `QAPair` as one reference QA example.

### 1.2 XML parsing

The ingestion code in `src/ingest.py`:

- Recursively scans the raw MedQuAD directory for XML files.
- Parses each XML file with `xml.etree.ElementTree`.
- Extracts the `Question` and `Answer` fields from each `QAPair`.
- Preserves the source collection name by storing the parent directory name as `source`.

The parser also handles a MedQuAD-specific issue where some answers are not plain text at the top node level but are distributed across nested child elements. In those cases, child text and tail text are concatenated into a single answer string.

### 1.3 Text normalization

Before any benchmarking begins, questions and answers are normalized by:

- removing HTML tags
- removing HTML entity-like artifacts
- collapsing repeated whitespace
- trimming leading/trailing whitespace

This cleaning is intentionally lightweight. It standardizes formatting without rewriting the semantic content of the source question or answer.

### 1.4 Deduplication

Deduplication is done at the question level. The code lowercases and strips each question, then removes exact duplicate question strings while keeping the first occurrence. This ensures that the benchmark is not inflated by repeated near-identical prompts.

### 1.5 Sampling

After cleaning and deduplication, the project samples a fixed number of examples using a deterministic random seed.

Current defaults visible in the repo are:

- pilot scale: `n_examples = 50`
- planned scaled benchmark: `n_examples = 1000`
- default random seed: `42`

Each sampled row receives:

- `id`
- `question`
- `answer`
- `source`

This `id` becomes the key used later for paired comparisons across clean, noisy, and repaired conditions.

## 2. Noise-generation methodology

### 2.1 Goal of the noise stage

The goal of the noise stage is not to generate arbitrary paraphrases. It is to create realistically degraded user questions that are meaningfully harder to answer while still pointing to the same underlying medical intent.

### 2.2 Noise categories

The benchmark uses five noise types:

1. `typos_grammar`
2. `ambiguity`
3. `layperson`
4. `incomplete`
5. `overgeneralization`

These categories are operationalized directly in the prompting template:

- `typos_grammar`: inject realistic spelling, spacing, punctuation, casing, or grammar mistakes
- `ambiguity`: remove or blur at least one key medical detail so the request becomes vague
- `layperson`: replace technical medical language with patient-style informal wording
- `incomplete`: drop important qualifiers such as condition, body part, timeframe, or population
- `overgeneralization`: broaden a specific medical query into a more generic health question

### 2.3 Prompt design for noise

The noise prompt is designed as a constrained text-perturbation task. The model is instructed to:

- return exactly one rewritten question
- avoid answering the question
- avoid explanation, labels, bullets, or meta commentary
- preserve broad medical intent
- make the output materially noisier than the original

This is important because the project is not trying to study paraphrasing quality. It is trying to study robustness under controlled degradation.

### 2.4 Noise assignment scheme

The current benchmark configuration uses:

- `noise_variants_per_question = 1`
- `noise_assignment = round_robin`

Under round-robin assignment, each clean question is paired with exactly one noise type, and the five noise types are distributed evenly across the sample. At `n=50`, this produces:

- 10 `typos_grammar`
- 10 `ambiguity`
- 10 `layperson`
- 10 `incomplete`
- 10 `overgeneralization`

This design avoids multiplying the dataset by a full Cartesian product at pilot scale while still covering all five noise categories.

### 2.5 Model and generation settings for noise

The exact noise model depends on the benchmark mode:

- In the original `end_to_end` mode, each model can generate its own noisy question.
- In the `fixed_repair` and `self_repair` workflows, the shared frozen noisy question set is generated once using GPT-5.4.

Typical noise-generation settings in the fixed question-set builder are:

- model: `gpt-5.4`
- reasoning effort: `low`
- temperature: `0.8`
- max tokens: `150`
- validation retries: `2`

The higher temperature relative to answer generation is intentional. The noise stage benefits from diversity and variability as long as the underlying intent is preserved.

## 3. Repair methodology

### 3.1 Goal of the repair stage

The repair stage rewrites a noisy user question into a clear, well-formed medical question without answering it.

This is a very important methodological distinction: the repair module is a rewrite system, not a QA system. A repaired question is valid only if it remains a question and does not inject an answer, request clarification, or drift away from the noisy input's intent.

### 3.2 Prompt design for repair

The repair prompt instructs the model to:

- rewrite the noisy input into one clear medical question
- avoid answering
- avoid explaining the rewrite
- avoid asking for clarification
- preserve the underlying intent as closely as possible
- make a best-effort rewrite even when the noisy input is underspecified

This best-effort behavior matters because the benchmark is explicitly testing whether a model can recover utility from imperfect input without interactive follow-up.

### 3.3 Model and generation settings for repair

Again, the exact repair model depends on benchmark mode:

- In `fixed_repair`, a single shared repair model produces the frozen repaired question set. In the repo, that shared model is GPT-5.4.
- In `self_repair`, each answer model repairs the shared noisy question itself before answering.
- In `end_to_end`, each model both creates its own noisy question and repairs it.

Typical repair-generation settings in the shared frozen workflow are:

- model: `gpt-5.4`
- reasoning effort: `low`
- temperature: `0.2`
- max tokens: `120`
- validation retries: `2`

Some model-specific configs relax these settings. For example, Qwen3-32B is given more repair tokens and more retries because it may emit empty or internally "thinking" outputs before producing the final question.

## 4. Generated-question validation methodology

One of the major methodological improvements in the codebase is explicit validation of noisy and repaired question outputs.

### 4.1 Why validation exists

Earlier pilot runs showed that repair outputs could become contaminated by:

- answer-like text
- requests for more context
- labels such as "Rewritten question:"
- multi-question outputs
- chain-of-thought or meta commentary

That contamination weakens the scientific interpretation of the repair stage. To address this, the project now validates every generated noisy or repaired question.

### 4.2 Validation rules

The validation layer in `src/question_validation.py` checks whether a generated question:

- is non-empty
- is long enough to be usable
- does not contain labels, bullets, code fences, or meta text
- does not contain trailing commentary
- does not contain multiple questions
- does not ask the user for clarification
- does not look like an answer or medical advice
- reads like a single question
- is not effectively identical to the source question

### 4.3 Retry mechanism

If a noisy or repaired output fails validation, the system retries generation with explicit corrective feedback that tells the model why the previous output was invalid.

This retry-and-revalidate loop is part of the methodology, not just engineering hygiene, because it enforces the distinction between:

- noise generation as question corruption
- repair generation as question clarification
- answer generation as the only stage allowed to produce answers

### 4.4 Frozen question-set validation

The repo also contains a dedicated validation script for frozen question sets:

- `scripts/benchmarks/validate_fixed_question_sets.py`

This script re-validates the shared noisy and repaired CSV/parquet files and writes a validation summary plus an `invalid_rows.csv` report if any rows fail the validator.

## 5. Answer-generation methodology

### 5.1 Answer prompt

The answer prompt is intentionally simple. The model is told:

- it is a knowledgeable medical assistant
- it should answer accurately and concisely
- it should provide a direct factual answer

The benchmark is therefore not testing elaborate prompt engineering at the answer stage. It is testing whether input quality alone changes downstream medical QA performance.

### 5.2 Answer settings

Typical answer-generation defaults are:

- temperature: `0.2`
- max tokens: `300`

Some model configs override these values for stability or latency reasons.

### 5.3 Self-consistency support

The answer wrapper supports optional self-consistency voting through `n_votes_answer > 1`. When enabled, multiple answers are generated and the response most similar to the others is selected.

In other words, the codebase has support for answer-stage hallucination control, but the default benchmark behavior is still single-response generation unless a config explicitly opts into voting.

## 6. Benchmark design and modes

### 6.1 Original pipeline logic

The original conceptual design of the project was:

- Pipeline A: clean question -> answer
- Pipeline B: noisy question -> answer
- Pipeline C: noisy question -> repair -> answer

That basic structure still exists, but the implementation now supports three different ways of operationalizing it.

### 6.2 `end_to_end` benchmark mode

In `end_to_end` mode:

- the answer model sees a clean question and answers it
- the same model generates a noisy version of that clean question
- the same model repairs that noisy question
- the same model answers the noisy and repaired versions

Methodological interpretation:

- Good for testing a model's full self-contained pipeline behavior
- Not good for fair cross-model comparison because noisy and repaired inputs differ by model

### 6.3 `fixed_repair` benchmark mode

In `fixed_repair` mode:

- one clean MedQuAD subset is sampled
- one shared noisy question set is generated once
- one shared repaired question set is generated once
- every answer model answers the same clean, noisy, and repaired questions

Methodological interpretation:

- This is the strongest controlled benchmark for comparing answer models under the same perturbations.
- It is the best choice for a paper section that claims fair model-to-model robustness comparisons.

### 6.4 `self_repair` benchmark mode

In `self_repair` mode:

- all answer models receive the same clean and noisy questions
- each answer model repairs the noisy question itself
- the model then answers its own repaired question

Methodological interpretation:

- This isolates the practical value of letting a model repair its own noisy input.
- It is still fairer than the old `end_to_end` mode because the noisy input is held constant across models.

### 6.5 Why both `fixed_repair` and `self_repair` matter

Together, the two modern benchmark modes answer two different questions:

- `fixed_repair`: If everyone gets the same repaired input, which answer model is most robust?
- `self_repair`: If a model must repair its own noisy input, does that help or hurt its downstream answer quality?

This split is one of the most important methodological advances in the repo and should be described clearly in the paper.

## 7. Frozen question-set construction methodology

The shared question sets are built by `scripts/benchmarks/build_fixed_question_sets.py`.

### 7.1 Shared-set construction steps

1. Load the clean sampled MedQuAD subset.
2. Expand it into benchmark rows according to the noise assignment plan.
3. Save the clean benchmark table (`clean_fixed.csv` / `.parquet`).
4. Generate and validate one noisy question for each row.
5. Save the shared noisy table (`noisy_fixed_gpt54.csv` / `.parquet`).
6. Generate and validate one repaired question for each noisy row.
7. Save the shared repaired table (`repaired_fixed_gpt54.csv` / `.parquet`).
8. Write a manifest and progress log.

### 7.2 Current frozen pilot set

The existing pilot frozen question set under `data/processed/benchmarks/fixed_question_sets_gpt54/` contains:

- 50 clean rows
- 50 noisy rows
- 50 repaired rows
- evenly balanced round-robin noise assignment

### 7.3 Planned scaled set

The repo also contains a `n=1000` configuration:

- `configs/experiments/fixed_question_sets_gpt54_n1000.yaml`

This uses the same methodology with a larger clean source file and writes to separate benchmark directories.

## 8. Model roster and backend methodology

### 8.1 Current benchmarked models

The current active model roster in the benchmark outputs contains 11 answer models:

1. BioMistral-7B
2. gemma-2-9b-it
3. gemma-4-31B-it (4bit)
4. GPT-5.4
5. Llama-3.1-8B-Instruct
6. Mistral-7B-Instruct-v0.3
7. Mixtral-8x7B-Instruct-v0.1 (4bit)
8. Phi-3-medium-4k-instruct (4bit)
9. Qwen2.5-14B-Instruct (4bit)
10. Qwen2.5-32B-Instruct (4bit)
11. Qwen3-32B (4bit)

### 8.2 Backend standardization

The code abstracts over multiple execution backends:

- `openai` backend
  - used for API-based models such as GPT-5.4
  - also used for local OpenAI-compatible servers
- `local` backend
  - uses Hugging Face Transformers locally
  - supports quantized 4-bit and 8-bit model loading when configured

### 8.3 Mixtral special case

Mixtral is not run through the normal local Transformers path. Its config uses:

- `backend: openai`
- `base_url: http://127.0.0.1:8081/v1`

This means Mixtral is served locally through an OpenAI-compatible `llama.cpp` server rather than through the standard Transformers loader. Methodologically, that still fits the same benchmark interface because the prompt, answer fields, and scoring pipeline remain identical.

### 8.4 Shared evaluator model

Although answer, noise, and repair models vary by config, the G-Eval stage is intentionally pinned to a shared evaluator by default:

- `gpt-5.4-mini`

This reduces evaluator drift across model configs.

## 9. Prompting methodology across stages

The project uses separate prompts for four distinct roles:

- `noise`
- `repair`
- `answer`
- `geval`

This separation is methodologically useful because it prevents role confusion:

- the noise model is told to degrade a question
- the repair model is told to clarify a question
- the answer model is told to answer a question
- the evaluator model is told to score an answer

The paper should emphasize that prompt roles were deliberately disentangled rather than overloaded into one instruction template.

## 10. Evaluation methodology

The project evaluates both answer quality and question-level intent preservation.

### 10.1 Reference-based lexical metrics

For each predicted answer versus the MedQuAD reference answer, the code computes:

- BLEU
- chrF
- ROUGE-L
- token-level F1
- exact match

These metrics are all implemented or wrapped in `src/metrics.py`.

Their role in the methodology is straightforward:

- BLEU, chrF, and ROUGE-L capture lexical overlap from different granularities
- token F1 measures normalized token overlap
- exact match is retained for completeness and audit continuity, even though it is generally uninformative for free-form medical QA

### 10.2 BERTScore

BERTScore is computed for answer-reference pairs using the `bert_score` package. The benchmark uses the F1 component as the reported value.

This metric provides a softer semantic similarity measure than pure lexical overlap and is especially useful when an answer is medically correct but phrased differently from the reference.

### 10.3 Intent preservation

Intent preservation is not an answer metric. It is a question metric.

The code computes cosine similarity between sentence embeddings of:

- the original clean question
- the noisy or repaired question

using:

- `sentence-transformers/all-MiniLM-L6-v2`

Interpretation:

- high noisy-question similarity means the perturbation preserved the original intent
- high repaired-question similarity means the repair preserved or restored the original intent

The clean baseline compares the clean question to itself, so it is expected to be `1.0`.

### 10.4 Domain-specific medical term metrics

One of the major methodological additions in the repo is a MedQuAD-derived medical lexicon and three domain-aware metrics:

- `med_coverage`
- `med_precision`
- `med_f1`

These are computed by:

1. Building a lexicon from cleaned MedQuAD answers.
2. Extracting medical unigrams and bigrams from both prediction and reference.
3. Measuring overlap using count-aware precision, recall, and F1.

Important implementation details:

- unigrams enter the lexicon if they appear in at least 3 answer documents
- bigrams enter the lexicon if they appear in at least 4 answer documents
- morphology-based inclusion is also used for medically shaped words, based on curated prefixes and suffixes
- bigrams take precedence, so a phrase like `blood pressure` is counted as one phrase-level unit rather than double-counted as separate unigrams

Interpretation:

- `med_coverage` is medical-term recall relative to the reference answer
- `med_precision` is medical-term precision
- `med_f1` summarizes the tradeoff

This metric family is useful when a model answer remains generally fluent but loses medically specific content.

### 10.5 LLM-as-judge evaluation (G-Eval style)

The project also uses a strict LLM judge for deep answer quality:

- evaluator model: `gpt-5.4-mini`
- scale: 1 to 5

The evaluator sees:

- the question
- the MedQuAD reference answer
- the model prediction

It is instructed to score:

- factual alignment
- logical consistency
- completeness
- directness

The judge is told to prioritize agreement with the reference answer over outside medical knowledge. Outputs are parsed into a single integer and clamped into the allowed range if necessary.

## 11. Robustness metrics

After the raw metrics are computed, the code derives three higher-level robustness statistics:

- `degradation = clean_mean - noisy_mean`
- `recovery = repaired_mean - noisy_mean`
- `recovery_ratio = recovery / degradation`

These statistics are computed for each metric separately.

The implementation also guards against unstable recovery ratios by requiring a minimum degradation floor before reporting the ratio. Different metric families use different floors:

- lexical metrics generally use `1.0`
- BERTScore and intent preservation use `0.01`
- G-Eval uses `0.1`

The recovery value is also bounded so that it cannot exceed the observed degradation in a way that would produce pathological ratios.

## 12. Statistical analysis methodology

The repo includes a full significance layer in `src/aggregate.py`.

### 12.1 Data reshaping

The per-sample benchmark outputs are stored in wide format with columns such as:

- `bleu_clean`
- `bleu_noisy`
- `bleu_repaired`

The aggregator converts these to long form so each row corresponds to:

- one `id`
- one `noise_type`
- one `pipeline` condition

This makes paired statistical testing straightforward.

### 12.2 Summary statistics

For each pipeline and metric, the aggregator computes:

- mean
- median
- standard deviation

It also computes means by `(pipeline, noise_type)` so the paper can report breakdowns by corruption type.

### 12.3 Paired significance tests

The main paired significance test is the Wilcoxon signed-rank test.

The code applies it to:

- clean vs noisy
- noisy vs repaired

Pairing is done by shared question `id`, and within noise-type slices where relevant.

### 12.4 Effect sizes

The aggregator also computes Cohen's d on paired differences. This gives a magnitude estimate in addition to p-values.

### 12.5 Bootstrap confidence intervals

For each `(pipeline, metric)` combination, the project computes bootstrap confidence intervals for the mean:

- 10,000 bootstrap replicates
- 95% confidence interval
- deterministic seed: `42`

This gives uncertainty estimates even when the sample size is still pilot-scale.

## 13. Resumability, caching, and reproducibility methodology

The repo is designed for long benchmark runs on local hardware, so resumability is part of the experimental method.

### 13.1 Caching

LLM calls can be cached using a hash of:

- model
- messages
- temperature
- max tokens
- backend
- API mode
- reasoning effort

This supports reproducibility and prevents accidental regeneration of rows that were already completed.

### 13.2 Resumable writes

The benchmark runner writes outputs incrementally:

- each completed sample row is appended immediately to CSV
- per-model progress is written continuously to JSON
- result summaries are refreshed during the run

This means interrupted jobs can be resumed without losing completed work.

### 13.3 Atomic file writes

The code uses atomic replace logic for CSV, JSON, and parquet writes wherever practical. This reduces the risk of corrupt outputs if a long run is interrupted.

### 13.4 Resume guards

The code also protects against a subtle methodological failure mode: accidentally resuming old results under a different benchmark mode or a different frozen question set. If that happens, stale resumable files are archived instead of silently mixed into the new run.

## 14. Engineering QA methodology

The repo includes several QA layers that matter for the Methods section because they improve benchmark trustworthiness.

### 14.1 Model preflight

Before a full run, `scripts/benchmarks/preflight.py` can:

- load each model
- perform a tiny generation smoke test
- record load time, generation time, device, and peak GPU memory

This catches broken configs and incompatible model-loading issues before the benchmark begins.

### 14.2 Self-repair row cleanup

The benchmark runner can prune self-repair rows that become invalid under updated validation logic. This matters because validator improvements should not leave earlier contaminated rows silently in the benchmark.

### 14.3 Reuse of fixed-repair clean/noisy outputs in self-repair

When a model has already completed `fixed_repair`, the `self_repair` runner can reuse that model's clean and noisy answers instead of recomputing them. Only the repaired branch is regenerated.

This is an engineering optimization, but it does not change the methodology because:

- the clean questions are identical
- the noisy questions are identical
- the clean and noisy answer branches are identical tasks

So the reused values are methodologically equivalent to rerunning them.

## 15. Output artifacts and what they mean

For each benchmarked model, the runner writes:

- `samples_<model>.csv`
- `progress_<model>.json`
- `result_<model>.json`

At the benchmark-package level, it writes:

- `model_comparison.csv`
- `model_comparison_full.json`

The aggregator then writes per-model statistics tables:

- `summary_<model>.csv`
- `summary_noise_<model>.csv`
- `robustness_<model>.csv`
- `paired_tests_<model>.csv`
- `bootstrap_cis_<model>.csv`

These files are the main evidence base the paper should draw from.

## 16. Known methodological limitations and paper-writing cautions

The writing team should be careful about the following points.

### 16.1 Do not overstate `end_to_end`

The original `end_to_end` benchmark should not be written up as a fair cross-model comparison on identical perturbations. It is a per-model end-to-end pipeline benchmark.

### 16.2 Be explicit about sample size

If the paper uses the current completed `n=50` benchmark packages, call them pilot results.

If the final `n=1000` pipeline is run later, the sample-size language should be updated everywhere in the Methods and Results sections.

### 16.3 Exact match is retained but weak

Exact match is included for continuity, but it is generally a degenerate metric for free-form medical QA and should not be treated as a primary result.

### 16.4 Repair is not always guaranteed to help

Methodologically, the benchmark is designed to measure whether repair helps. It is not designed to assume that it helps. The current metric suite is broad precisely because repair can improve some properties while harming others.

## 17. Suggested structure for the paper's Methods section

The writing team can map the project into the paper using the following subsection order:

1. Dataset and preprocessing
2. Noise construction
3. Prompt repair formulation
4. Benchmark design (`fixed_repair`, `self_repair`, legacy `end_to_end`)
5. Evaluated model roster and inference backends
6. Answer-generation protocol
7. Evaluation metrics
8. Robustness and recovery statistics
9. Statistical testing
10. Quality control, validation, and reproducibility

## 18. Short paper-ready description

If the writing team needs a concise paragraph to start from, the following wording is close to the implementation:

> We evaluated medical QA robustness on a cleaned MedQuAD subset by comparing model performance under clean, noisy, and repaired-question conditions. Noisy questions were generated using five corruption types (typos/grammar, ambiguity, layperson phrasing, incompleteness, and overgeneralization), and repaired questions were produced by a dedicated rewrite model instructed to clarify without answering. We used a controlled `fixed_repair` benchmark, in which all answer models received identical clean, noisy, and repaired inputs, and a `self_repair` benchmark, in which all models received the same noisy inputs but generated their own repaired rewrites. Performance was measured with lexical overlap metrics, BERTScore, question-level intent preservation, domain-specific medical term coverage, and a GPT-5.4-mini G-Eval score, followed by paired Wilcoxon tests, effect sizes, and bootstrap confidence intervals.

## 19. File references for the writing team

If anyone on the writing team wants to verify details against implementation, these are the most relevant files:

- `src/ingest.py`
- `src/noise.py`
- `src/repair.py`
- `src/answer.py`
- `src/question_validation.py`
- `src/metrics.py`
- `src/judge.py`
- `src/aggregate.py`
- `scripts/benchmarks/build_fixed_question_sets.py`
- `scripts/benchmarks/run_comparison.py`
- `scripts/benchmarks/validate_fixed_question_sets.py`
- `scripts/benchmarks/preflight.py`
- `scripts/data/build_medical_lexicon.py`
- `configs/prompts.yaml`
- `configs/experiments/fixed_question_sets_gpt54.yaml`
- `configs/experiments/fixed_question_sets_gpt54_n1000.yaml`
- `reports/next_steps_results_audit_2026-04-05.md`
- `reports/fixed_vs_self_repair_2026-04-18.md`

## Final recommendation

For the final paper, the methodology should be written as a controlled robustness study centered on:

- MedQuAD-based medical QA
- realistic input corruption
- explicit prompt repair as a distinct rewrite stage
- separate controlled and practical benchmark modes
- mixed lexical, semantic, domain-specific, and judge-based evaluation
- paired statistical analysis

That framing is faithful to the current codebase and avoids the main overclaim risk from the earlier pilot design.
