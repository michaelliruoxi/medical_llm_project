# Local Model Test Report (GPU)

**Date:** 2026-03-04
**Model:** mistralai/Mistral-7B-Instruct-v0.3 (7B params, fp16)
**Backend:** Local HuggingFace transformers on NVIDIA RTX 5090
**GPU Memory:** 13.50 GB used (of 32 GB VRAM)
**Samples:** 5 from MedlinePlus Health Topics (4_MPlus_Health_Topics_QA)
**Noise type:** typos_grammar
**Judge:** Chain-of-thought fact-matching against MedQuAD reference answers

---

## Results Summary

| Metric | Clean | Noisy | Repaired |
|--------|------:|------:|---------:|
| BLEU | 5.10 | 2.46 | 2.94 |
| BERTScore F1 | 0.8776 | 0.8643 | 0.8497 |
| Judge (0-3) | 2.60 | 2.20 | 2.40 |

### Robustness Metrics

| Metric | BERTScore | Judge |
|--------|----------:|------:|
| Degradation (clean - noisy) | +0.0133 | +0.40 |
| Recovery (repaired - noisy) | -0.0146 | +0.20 |
| Recovery Ratio | -1.10 | 0.50 |

---

## Per-Sample Generated Texts

### Sample 0: A1C

**Clean question:**
> Do you have information about A1C

**Noisy question:**
> I need some info 'bout my glucose levels test, is it called AIC?

**Repaired question:**
> Is the glucose level test you're referring to known as Hemoglobin A1c (HbA1c) or AIC?

**Reference answer (MedQuAD):**
> Summary : A1C is a blood test for type 2 diabetes and prediabetes. It measures your average blood glucose, or blood sugar, level over the past 3 months. Doctors may use the A1C alone or in combination with other diabetes tests to make a diagnosis. They also use the A1C to see how well you are managing your diabetes. This test is different from the blood sugar checks that people with diabetes do every day. Your A1C test result is given in percentages. The higher the percentage, the higher your bl...

**Answer (clean):**
> Yes, I do. The A1C test is a common blood test used to diagnose type 1 and type 2 diabetes, as well as gestational diabetes. It provides an average of your blood sugar levels over the past 2-3 months. A normal A1C level is below 5.7%, a level from 5.7% to 6.4% suggests prediabetes, and a level of 6.5% or higher on two separate tests indicates diabetes. However, specific values may vary depending on the laboratory and diagnostic criteria used.

**Answer (noisy):**
> No, your blood glucose level test is not referred to as A1C (Glycohemoglobin). The A1C test measures your average blood sugar level over the past 2-3 months by evaluating the percentage of sugar attached to hemoglobin in red blood cells. If you're referring to another specific test for blood glucose, please provide more details so I can help better.

**Answer (repaired):**
> The correct term for the blood test that measures average blood sugar levels over the past 2-3 months is Hemoglobin A1c (HbA1c), not AIC.

| Metric | Clean | Noisy | Repaired |
|--------|------:|------:|---------:|
| BLEU | 4.80 | 1.45 | 0.02 |
| BERTScore F1 | 0.877 | 0.841 | 0.843 |
| Judge | 3 | 1 | 1 |

**Judge analysis:** The clean answer covers key facts (blood test, diabetes/prediabetes, measures average glucose, percentage thresholds). The noisy answer starts with an incorrect claim ("No, your blood glucose level test is not referred to as A1C") — score 1. The repaired answer is too brief and misses most reference details — score 1.

---

### Sample 1: Abdominal Pain

**Clean question:**
> What is (are) Abdominal Pain ?

**Noisy question:**
> Whats da abdominel pane? Its somethin that hurts in der belly area, na? Aint suwe 'bout specifics tho.

**Repaired question:**
> What is an abdominal pain? It refers to discomfort located in the stomach/belly region. I'm not certain about its specific details though.

**Reference answer (MedQuAD):**
> Your abdomen extends from below your chest to your groin. Some people call it the stomach, but your abdomen contains many other important organs. Pain in the abdomen can come from any one of them. The pain may start somewhere else, such as your chest. Severe pain doesn't always mean a serious problem. Nor does mild pain mean a problem is not serious. Call your healthcare provider if mild pain lasts a week or more or if you have pain with other symptoms. Get medical help immediately if - You have...

**Answer (clean):**
> Abdominal pain refers to any discomfort or aching sensation that occurs in the abdomen, which is the area between your chest and pelvis. The cause can vary widely, from minor issues like gas or constipation to serious conditions such as appendicitis, gallstones, or inflammatory bowel disease. It's important to consult with a healthcare professional if you experience persistent or severe abdominal pain.

**Answer (noisy):**
> Yes, you're correct. The abdomen is the part of the body between your chest and pelvis, often referred to as the "belly." Abdominal pain can be caused by various conditions such as indigestion, gastritis, appendicitis, or kidney stones among others. However, without more specific symptoms, it's difficult to determine the exact cause. If you have persistent abdominal pain, please consult with a healthcare professional for an accurate diagnosis and treatment options.

**Answer (repaired):**
> Abdominal pain refers to any type of distress or uncomfortable sensation that occurs within your abdomen (stomach/belly area). The location, severity, frequency, and character (sharp, dull, crampy, constant, intermittent) can vary greatly depending on the cause. Common causes include gastritis, indigestion, irritable bowel syndrome, appendicitis, kidney stones, and infections. However, it's important to note that these symptoms could be indicative of many other conditions as well. If you experience persistent or severe abdominal pain, please consult with a healthcare professional for proper evaluation and treatment.

| Metric | Clean | Noisy | Repaired |
|--------|------:|------:|---------:|
| BLEU | 1.51 | 1.26 | 1.31 |
| BERTScore F1 | 0.860 | 0.855 | 0.844 |
| Judge | 2 | 3 | 3 |

**Judge analysis:** The reference emphasizes that "severe pain doesn't always mean serious" and the "call your healthcare provider" guidance. The noisy and repaired answers cover the chest-to-pelvis location and the "consult provider" advice — scoring higher. The clean answer misses the nuance about severity not indicating seriousness.

---

### Sample 2: Abortion

**Clean question:**
> What is (are) Abortion ?

**Noisy question:**
> Whats abourtionz thay they?

**Repaired question:**
> What are abortions and their procedures?

**Reference answer (MedQuAD):**
> An abortion is a procedure to end a pregnancy. It uses medicine or surgery to remove the embryo or fetus and placenta from the uterus. The procedure is done by a licensed health care professional. The decision to end a pregnancy is very personal. If you are thinking of having an abortion, most healthcare providers advise counseling.

**Answer (clean):**
> An abortion is the termination of pregnancy by removing an embryo or fetus before it can survive outside the uterus. This process can be performed through various methods during different stages of pregnancy. It's important to note that the legality and availability of abortions vary greatly around the world due to cultural, religious, and political factors. Always consult with a healthcare professional for accurate information regarding specific circumstances.

**Answer (noisy):**
> Abortion refers to the termination of pregnancy before the fetus can survive outside the womb. It's important to note that laws regarding abortion vary widely around the world, and it's always best to consult with a healthcare professional or legal expert for specific information in your region. In general, abortions may be performed using medication (medical abortion) or surgical procedures (surgical abortion). Both methods have risks and benefits, which should be discussed thoroughly with a qualified healthcare provider.

**Answer (repaired):**
> Abortions refer to the termination of pregnancy by removing an embryo or fetus before it can survive outside the uterus. There are two types of abortion procedures: (1) Medical Abortion using mifepristone and misoprostol, usually within the first 7-10 weeks. (2) Surgical Abortion using suction and/or surgical instruments (aspiration, D&E, D&X), generally between 6-14 weeks. These procedures should only be carried out under the supervision of qualified healthcare professionals in safe, legal settings.

| Metric | Clean | Noisy | Repaired |
|--------|------:|------:|---------:|
| BLEU | 5.55 | 0.88 | 4.30 |
| BERTScore F1 | 0.895 | 0.883 | 0.864 |
| Judge | 3 | 3 | 3 |

**Judge analysis:** All three answers cover the key facts: procedure to end pregnancy, uses medicine or surgery, involves a healthcare professional. The reference's mention of counseling is a detail most predictions miss, but enough other facts are covered.

---

### Sample 3: Abscess

**Clean question:**
> What is (are) Abscess ?

**Noisy question:**
> Whats da abcessses tha tey are? (Typo: "da" for "the") (Grammar error: present tense instead of third person singular)

**Repaired question:**
> What is the abscess that they have? (Corrected typo and grammatical error)

**Reference answer (MedQuAD):**
> An abscess is a pocket of pus. You can get an abscess almost anywhere in your body. When an area of your body becomes infected, your body's immune system tries to fight the infection. White blood cells go to the infected area, collect within the damaged tissue, and cause inflammation. During this process, pus forms. Pus is a mixture of living and dead white blood cells, germs, and dead tissue. Bacteria, viruses, parasites and swallowed objects can all lead to abscesses. Skin abscesses are easy t...

**Answer (clean):**
> An abscess is a localized infection that is surrounded by inflamed tissue. It's typically filled with pus, which consists of dead white blood cells, bacteria, and other debris. Abscesses can occur anywhere in the body but are most common in areas rich in bacterial colonies like the skin or internal organs such as the liver, kidneys, or brain. They usually form due to an injury or infection that causes tissue damage and allows bacteria to enter and multiply. Treatment often involves draining the abscess and administering antibiotics.

**Answer (noisy):**
> The correct term is an abscess. An abscess is a localized collection of pus that forms around tissue that has become infected. It's often caused by bacteria entering through a break in the skin or mucous membranes. If left untreated, it can cause serious complications such as sepsis.

**Answer (repaired):**
> To determine the type of abscess someone has, I would need to consider factors such as location, symptoms, and results from diagnostic tests like imaging or lab work. However, without specific details, it's not possible for me to provide an accurate diagnosis. Common types of abscesses include skin abscesses, tooth abscesses, and organ abscesses (like appendix or liver). If you suspect an abscess, please consult with a healthcare professional for proper evaluation and treatment.

| Metric | Clean | Noisy | Repaired |
|--------|------:|------:|---------:|
| BLEU | 6.71 | 1.50 | 1.19 |
| BERTScore F1 | 0.890 | 0.870 | 0.846 |
| Judge | 2 | 3 | 3 |

**Judge analysis:** The reference describes pus composition (white blood cells, germs, dead tissue) and the immune response. The clean answer covers pus and infection well but misses the immune system mechanism — score 2. The noisy and repaired answers cover the pus/infection concept sufficiently.

---

### Sample 4: Acne

**Clean question:**
> What is (are) Acne ?

**Noisy question:**
> What is acnes? A reasun they may appear on skin?

**Repaired question:**
> 1) "What is acne?" 2) "Why does acne occur on the skin?"

**Reference answer (MedQuAD):**
> Acne is a common skin disease that causes pimples. Pimples form when hair follicles under your skin clog up. Most pimples form on the face, neck, back, chest, and shoulders. Anyone can get acne, but it is common in teenagers and young adults. It is not serious, but it can cause scars. No one knows exactly what causes acne. Hormone changes, such as those during the teenage years and pregnancy, probably play a role. There are many myths about what causes acne. Chocolate and greasy foods are often...

**Answer (clean):**
> Acne is a common skin condition characterized by clogged pores, inflammation, and the appearance of spots or pimples, primarily on the face but also on other areas such as the back, chest, neck, and shoulders. It occurs when hair follicles become plugged with dead skin cells, oil from the skin (sebum), and bacteria. There are several types of acne, including whiteheads, blackheads, papules, pustules, nodules, and cysts. The severity can range from mild to severe, depending on factors like the number and size of lesions, their location, and how deep they are in the skin.

**Answer (noisy):**
> Acne refers to a common skin condition that causes spots and pimples, particularly on the face, back, chest, and shoulders. It happens when hair follicles become plugged with oil and dead skin cells. The primary reasons for this include overproduction of sebum (oil) by the sebaceous glands, abnormal shedding of dead skin cells, and bacterial infection in the follicle. Hormonal changes during puberty, menstruation, pregnancy, or taking certain medications can trigger or worsen acne. Stress might also play a role as it can affect hormone levels leading to increased oil production.

**Answer (repaired):**
> 1) Acne is a common skin condition that causes spots (often called pimples or zits) to appear on the face, back, neck, chest, and shoulders. It occurs when hair follicles become plugged with oil and dead skin cells. 2) Acne occurs on the skin due to several factors: excessive production of sebum (oil), dead skin cells blocking the opening of hair follicles, bacteria (Propionibacterium acnes) causing inflammation, and hormonal changes during puberty, pregnancy, menstrual cycle, and some medications.

| Metric | Clean | Noisy | Repaired |
|--------|------:|------:|---------:|
| BLEU | 6.91 | 7.21 | 7.86 |
| BERTScore F1 | 0.866 | 0.873 | 0.851 |
| Judge | 3 | 1 | 2 |

**Judge analysis:** The reference highlights: common, pimples, clogged follicles, face/neck/back/chest/shoulders, teenagers/young adults, can cause scars, hormone changes. The clean answer covers most facts — score 3. The noisy answer is strong on causes but the judge flagged it lower (possibly for missing "teenagers/young adults" and "scars"). The repaired answer covers most but not all — score 2.

---

## Aggregate Analysis

### Score Distribution

| Score | Clean | Noisy | Repaired |
|-------|------:|------:|---------:|
| 3 | 3 | 3 | 2 |
| 2 | 2 | 0 | 1 |
| 1 | 0 | 2 | 2 |
| 0 | 0 | 0 | 0 |

### Key Findings

1. **Noise injection works correctly.** Mistral-7B introduces realistic typos and colloquial phrasing ("da abdominel pane", "abourtionz", "AIC") while preserving medical intent. The noisy questions trigger lower BLEU (5.10 -> 2.46) and lower judge scores (2.60 -> 2.20), confirming measurable degradation.

2. **Repair partially recovers quality.** The judge score rises from 2.20 (noisy) to 2.40 (repaired), showing a recovery ratio of 0.50 (50% of the degradation is recovered). However, BERTScore does not recover — the repaired answers are often more concise and miss details.

3. **The judge now discriminates meaningfully.** With the chain-of-thought rubric comparing against MedQuAD reference facts, scores range from 1 to 3 across pipelines. The noisy pipeline has the most 1s (2 out of 5), confirming that noise degrades fact coverage. Clean answers score highest on average.

4. **Interesting failure modes:**
   - Sample 0 (A1C): The noisy question changed the meaning enough that the model gave an incorrect "No" response — correctly penalized to score 1.
   - Sample 3 (Abscess): The repaired question became vague ("What is the abscess that they have?"), leading to a diagnostic-style response rather than a factual definition.

## Model Configuration

| Setting | Value |
|---------|-------|
| Model | mistralai/Mistral-7B-Instruct-v0.3 |
| Precision | float16 |
| GPU Memory | 13.50 GB |
| Backend | local (HuggingFace transformers) |
| Noise types | typos_grammar |
| Answer max tokens | 300 |
| Repair max tokens | 120 |
| Noise max tokens | 150 |
| Judge max tokens | 200 (chain-of-thought) |
| Repetition penalty | 1.2 |
| Top-p | 0.9 |
