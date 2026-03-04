"""Step A: Load MedQuAD XML files, clean, deduplicate, sample, and save as parquet."""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.utils import PROJECT_ROOT, load_config, setup_logging


logger = setup_logging()


def parse_xml_file(filepath: Path) -> list[dict]:
    """Extract question-answer pairs from a single MedQuAD XML file."""
    records = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        logger.warning("Skipping malformed XML: %s", filepath)
        return records

    for qa_pair in root.iter("QAPair"):
        question_el = qa_pair.find("Question")
        answer_el = qa_pair.find("Answer")

        if question_el is None or answer_el is None:
            continue

        q_text = question_el.text or question_el.get("text", "") or ""
        a_text = answer_el.text or ""

        # Some answers are nested inside child elements
        if not a_text.strip():
            parts = []
            for child in answer_el:
                if child.text:
                    parts.append(child.text)
                if child.tail:
                    parts.append(child.tail)
            a_text = " ".join(parts)

        q_text = _clean_text(q_text)
        a_text = _clean_text(a_text)

        if q_text and a_text:
            records.append({
                "question": q_text,
                "answer": a_text,
                "source": filepath.parent.name,
            })

    return records


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML artefacts."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_all_xml(raw_dir: Path) -> pd.DataFrame:
    """Recursively load all XML files under raw_dir."""
    records = []
    xml_files = list(raw_dir.rglob("*.xml"))
    logger.info("Found %d XML files in %s", len(xml_files), raw_dir)

    for fp in xml_files:
        records.extend(parse_xml_file(fp))

    logger.info("Parsed %d QA pairs total", len(records))
    return pd.DataFrame(records)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate questions (case-insensitive)."""
    before = len(df)
    df = df.copy()
    df["_q_lower"] = df["question"].str.lower().str.strip()
    df = df.drop_duplicates(subset="_q_lower", keep="first").drop(columns="_q_lower")
    logger.info("Deduplicated: %d -> %d", before, len(df))
    return df.reset_index(drop=True)


def run(n_examples: int | None = None):
    cfg = load_config()
    n = n_examples or cfg["n_examples"]
    seed = cfg.get("random_seed", 42)

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data"]
    out_dir = PROJECT_ROOT / cfg["paths"]["processed_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_xml(raw_dir)
    if df.empty:
        logger.error("No QA pairs found. Place MedQuAD XML folders in %s", raw_dir)
        return

    df = deduplicate(df)

    if len(df) < n:
        logger.warning("Only %d unique QA pairs available (requested %d). Using all.", len(df), n)
        n = len(df)

    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))

    out_path = out_dir / "medquad.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d examples to %s", len(df), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MedQuAD data")
    parser.add_argument("--n", type=int, default=None, help="Override n_examples")
    args = parser.parse_args()
    run(n_examples=args.n)
