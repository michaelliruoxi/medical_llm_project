# n=50 Deck — Light Polish Design

**Date:** 2026-04-27
**Source:** `MedQuAD_Robustness_Final_Presentation_n50_updated.pptx` (19 slides, 16:9, 13.33"×7.5")
**Output:** `MedQuAD_Robustness_Final_Presentation_n50_polished.pptx` (new file, original untouched)
**Audience / venue:** Practicum final, ~15–20 min, faculty + classmates, presenter-led.
**Scope:** Option 1 — light polish. Keep all 19 slides and the existing order. No content rewriting, no slide cuts.

## Goals

1. Commit to one visual identity instead of a hesitant 13-color palette.
2. Enforce a 14pt body / 12pt table-cell font floor (current low is 6.8pt).
3. Make slide 8 (the headline result) read as a hero, not a bullet slide.
4. Retheme charts (slides 8, 9, 10) so winners/losers read at a glance.

## Visual System

### Palette (committed)
- Primary teal: `#127C7A`
- Slate dark: `#25323D` (text + dark backgrounds)
- Gold accent: `#E0A13A` (highlights, callout numbers, hero stats — not body)
- Negative: `#C94C4C` (cherry — chart bars and metric-tension "got worse" only)
- Positive: `#2D9D78` (green — chart bars and metric-tension "improved" only)
- Light bg shades: `#FFFFFF`, `#F2F6F7` (table row tint), `#DCE9ED` (chart gridlines)
- Drop on sight: `#3867D6`, `#7A4B8F`, `#0E5F60`, `#52616B`-as-fill, `#EAF0FF`, `#FCEBEB`, `#FFF4DA`, `#EAF7F1`, `#F3F7F9`, `#F7FAFC`, `#E7F4F2` — replace with the committed shades.

### Motif
Filled teal circle, ~0.35" diameter, white section number ("01", "02", …) inside, placed left of every section title. Repeated on slide 16 (interpretation) numbered insights.

### Type scale
| Element | Size |
|---|---|
| Slide title | 36pt bold |
| Section header / subhead | 20pt bold |
| Body | 14pt floor |
| Table cells | 12pt floor |
| Footer | 10pt |
| Stat callouts (e.g., big "11", "50") | 48pt+ bold gold |

Font kept: Aptos.

### Sandwich structure
Slide 1 and slide 19 use dark `#25323D` background with white text + gold accent. Slides 2–18 use white background.

## Per-Slide Changes

| # | Slide | Changes |
|---|---|---|
| 1 | Title | Dark bg, white headline, gold accent line under "Robustness", restyle existing 4-step diagram with teal circles. |
| 2 | Problem | 11/50/5 stats → 48pt gold; section title gets motif circle. |
| 3 | Audience bridge | 5-step arrow chain stays; Noise/Repair/Robustness/Recovery tiles get motif + 14pt body. |
| 4 | Design | Pipeline A/B/C labels in teal pill; fixed_repair vs self_repair tiles get gold-accent left border. |
| 5 | Setup | Stat numbers → 48pt; both tables → 12pt cells; trim "What it simulates" descriptions ~5 chars to prevent wrap. |
| 6 | Metrics | All 11 rows kept; cells → 12pt; alternate row shading `#F2F6F7`; colored vertical bar (teal/gold/cherry) on left of each family group. |
| 7 | Read metrics | 5 family blocks get motif + 14pt body; "Key reading rule" → gold-bordered callout box. |
| 8 | Headline | Chart sized up ~1.5×; mode-average text condensed to single-line gold callout; headline → 32pt. |
| 9 | Per-model | Bars retheme: positive deltas teal, negative cherry; sort by self_repair delta (Qwen3 on top). |
| 10 | Per-noise | Same retheme + sort logic. |
| 11 | Metric tension | improved/worse/flat counts → colored badges (green/cherry/grey). |
| 12 | Evidence: incomplete (air pollution) | Column header chips (CLEAN teal / NOISY cherry / REPAIRED gold); 12pt body floor; bottom metrics → horizontal stat bar with arrows. |
| 13 | Evidence: incomplete (anabolic steroids) | Same pattern as slide 12. |
| 14 | Evidence: layperson (abortion) | Same pattern. |
| 15 | Evidence: typos (anal disorders) | Same pattern. |
| 16 | Interpretation | 3 numbered insights get teal-circle motif (01/02/03). |
| 17 | Limitations | Cells → 12pt; alternating row shade. |
| 18 | Next steps | 4 tiles get teal/gold alternating left borders; "Practical takeaway" → gold-bordered box. |
| 19 | Thank you | Mirror slide 1: dark bg, white headline, gold accent on key sentence, italic muted invitation line. |

## Charts

- Slide 8: vertical bars, ~9"×4.5", clean=teal / noisy=slate `#52616B` / repaired=gold; chart title removed (slide already has one); axis labels ≥ 11pt; gridlines `#DCE9ED`.
- Slides 9, 10: horizontal bars, sorted by self_repair delta descending; positive=teal, negative=cherry.

## Implementation

- Tool: `python-pptx` for in-place mutation (preserves embedded charts; XML-safe).
- File copy first; never write the original.
- Use a single Python script to apply all changes idempotently so I can rerun if the user requests tweaks.

## Verification

LibreOffice is not installed locally, so I cannot render the deck to images. Verification path:
1. `python -m markitdown` on output to confirm no text was lost.
2. `python-pptx` shape inspection to confirm font sizes ≥ floor and palette colors are present.
3. User opens in PowerPoint and we iterate.

## Out of Scope

- Slide reordering or removal
- Copy rewriting beyond ~5-char trims to prevent wrap
- New chart types or new data
- Custom fonts beyond Aptos
- Visual companion / browser mockups (output is a pptx; user reviews the artifact directly)
