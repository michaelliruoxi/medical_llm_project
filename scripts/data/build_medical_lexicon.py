"""Build a domain medical-term lexicon from the cleaned MedQuAD corpus.

The output is a JSON file used by ``src.metrics.compute_medical_term_coverage``
to extract medical-looking tokens and bigrams from model answers and compute
recall/precision/F1 against the reference answer.

The lexicon is deterministic and reproducible:

    unigrams = content tokens that appear in at least MIN_DOC_FREQ MedQuAD
               answer documents, after stopword/length filtering, plus any
               token matching a medical suffix/prefix regex regardless of
               document frequency.
    bigrams  = adjacent content-word bigrams appearing in at least
               MIN_BIGRAM_DOC_FREQ documents (tighter threshold so we do not
               flood the lexicon with generic phrases).

Run once:

    python scripts/data/build_medical_lexicon.py

Output: data/processed/medical_lexicon.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "medquad_cleaned.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "medical_lexicon.json"

# Tokens must be at least this long to be considered content words.
MIN_TOKEN_LEN = 3
# Unigrams must appear in at least this many answer documents to enter the
# lexicon on frequency evidence alone.
MIN_DOC_FREQ = 3
# Bigrams have a tighter threshold to avoid flooding with boilerplate phrases.
MIN_BIGRAM_DOC_FREQ = 4

# Medical morphological patterns. A token matching any of these is always
# included in the lexicon, even if its MedQuAD document frequency is low.
MEDICAL_SUFFIX_RE = re.compile(
    r"(?:"
    r"itis|oma|osis|opathy|pathy|emia|aemia|uria|algia|gram|graphy|"
    r"plasty|ectomy|ostomy|otomy|scopy|lysis|plegia|paresis|trophy|"
    r"phagia|phasia|rrhea|rrhoea|rrhage|genesis|cyte|blast|cidal|"
    r"pneic|glycaemia|glycemia|kinase|ase|oxin"
    r")$",
    re.IGNORECASE,
)
MEDICAL_PREFIX_RE = re.compile(
    r"^(?:"
    r"cardio|neuro|hepato|nephro|gastro|dermato|osteo|arthro|hemo|haemo|"
    r"thrombo|pulmo|broncho|onco|immuno|endo|myo|angi|encephalo|"
    r"rhino|laryngo|esophag|oesophag|colo|recto|procto|oto|ophthalmo|"
    r"lympho|leuko|melano|carcino|sarcoma|adeno|fibro|chondro|gluco|"
    r"hyper|hypo|dys|anti|pro|re|anti|sub|trans"
    r")",
    re.IGNORECASE,
)

# Generic stopwords. Kept small and hand-picked: this is not a full NLTK
# stopword list because we want to retain some "medical-flavor" common
# words (e.g. "pain", "blood", "cell"). Stopwords removed are purely
# syntactic / non-content.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "while",
    "for", "to", "of", "in", "on", "at", "by", "with", "as", "from",
    "into", "onto", "over", "under", "between", "through", "about",
    "is", "are", "was", "were", "be", "been", "being", "am", "have", "has",
    "had", "having", "do", "does", "did", "done", "doing", "can", "could",
    "may", "might", "must", "shall", "should", "will", "would", "this",
    "that", "these", "those", "it", "its", "itself", "he", "she", "they",
    "them", "his", "her", "their", "there", "here", "when", "where", "why",
    "how", "which", "who", "whom", "what", "also", "because", "such",
    "very", "more", "most", "some", "any", "all", "each", "every", "no",
    "not", "than", "too", "so", "just", "only", "other", "another", "same",
    "different", "many", "much", "few", "little", "your", "you", "yours",
    "yourself", "our", "ours", "us", "we", "i", "me", "my",
    # Non-content boilerplate often present in MedQuAD answers:
    "summary", "source", "include", "includes", "including",
    "called", "known", "see", "also", "example", "examples",
    # Uninformative medical-sounding connectors:
    "normal", "type", "types", "kind", "kinds", "usually",
    "often", "sometimes", "severe", "mild",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def is_candidate_token(tok: str) -> bool:
    if len(tok) < MIN_TOKEN_LEN:
        return False
    if tok in STOPWORDS:
        return False
    # Digits-heavy tokens are unhelpful; we keep hyphenated medical tokens.
    return True


def is_medical_shape(tok: str) -> bool:
    """Return True when the token looks morphologically medical."""
    if MEDICAL_SUFFIX_RE.search(tok):
        return True
    if MEDICAL_PREFIX_RE.match(tok) and len(tok) >= 6:
        return True
    return False


def build_lexicon(source_csv: Path, output_json: Path) -> dict:
    df = pd.read_csv(source_csv)
    if "answer" not in df.columns:
        raise KeyError(
            f"{source_csv} must contain an 'answer' column; got {list(df.columns)}"
        )

    answers = [str(a) for a in df["answer"].dropna().tolist()]
    n_docs = len(answers)

    unigram_doc_freq: Counter = Counter()
    bigram_doc_freq: Counter = Counter()
    morphology_hits: set[str] = set()

    for ans in answers:
        tokens = [t for t in tokenize(ans) if is_candidate_token(t)]
        unique_unigrams = set(tokens)
        unigram_doc_freq.update(unique_unigrams)

        for tok in unique_unigrams:
            if is_medical_shape(tok):
                morphology_hits.add(tok)

        bigrams = {
            f"{a} {b}"
            for a, b in zip(tokens, tokens[1:])
            if is_candidate_token(a) and is_candidate_token(b)
        }
        bigram_doc_freq.update(bigrams)

    kept_unigrams = {
        tok for tok, df_count in unigram_doc_freq.items()
        if df_count >= MIN_DOC_FREQ
    }
    kept_unigrams |= morphology_hits
    kept_bigrams = {
        bg for bg, df_count in bigram_doc_freq.items()
        if df_count >= MIN_BIGRAM_DOC_FREQ
    }

    lexicon = {
        "meta": {
            "source_csv": str(source_csv),
            "n_docs": n_docs,
            "min_doc_freq_unigram": MIN_DOC_FREQ,
            "min_doc_freq_bigram": MIN_BIGRAM_DOC_FREQ,
            "morphology_hits": len(morphology_hits),
            "stopword_count": len(STOPWORDS),
            "n_unigrams": len(kept_unigrams),
            "n_bigrams": len(kept_bigrams),
        },
        "unigrams": sorted(kept_unigrams),
        "bigrams": sorted(kept_bigrams),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(lexicon, fh, indent=2, ensure_ascii=False)
    return lexicon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lex = build_lexicon(args.source, args.output)
    print(
        f"Wrote lexicon to {args.output}\n"
        f"  n_docs = {lex['meta']['n_docs']}\n"
        f"  unigrams = {lex['meta']['n_unigrams']}\n"
        f"  bigrams  = {lex['meta']['n_bigrams']}\n"
        f"  morphology_hits = {lex['meta']['morphology_hits']}"
    )


if __name__ == "__main__":
    main()
