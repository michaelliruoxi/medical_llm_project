"""Write a markdown comparison note for fixed_repair vs self_repair."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_METRICS = (
    "med_coverage",
    "med_f1",
    "geval",
    "intent_preservation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs" / "benchmarks" / "fixed_repair",
    )
    parser.add_argument(
        "--self-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs" / "benchmarks" / "self_repair",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / f"fixed_vs_self_repair_{date.today().isoformat()}.md",
    )
    return parser.parse_args()


def parse_percent(value) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def load_completed_mode_table(mode_dir: Path, expected_mode: str) -> pd.DataFrame:
    path = mode_dir / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison file: {path}")

    df = pd.read_csv(path)
    df = df.loc[df["Mode"] == expected_mode].copy()
    df["status_norm"] = df["Status"].astype(str).str.upper()
    completed = df.loc[df["status_norm"].str.startswith("COMPLETED")].copy()
    if completed.empty:
        raise RuntimeError(f"No completed rows found in {path}")

    completed["Model"] = completed["Model"].astype(str)
    for metric in TARGET_METRICS:
        completed[f"{metric}_repaired"] = pd.to_numeric(
            completed[f"{metric}_repaired"], errors="coerce"
        )
        completed[f"{metric}_noisy"] = pd.to_numeric(
            completed[f"{metric}_noisy"], errors="coerce"
        )
        completed[f"{metric}_recovery_pct"] = completed[f"{metric}_recovery%"].apply(parse_percent)
        completed[f"{metric}_repaired_minus_noisy"] = (
            completed[f"{metric}_repaired"] - completed[f"{metric}_noisy"]
        )
    return completed


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                if pd.isna(value):
                    cells.append("NA")
                else:
                    cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(fixed_df: pd.DataFrame, self_df: pd.DataFrame) -> str:
    merged = fixed_df.merge(
        self_df,
        on="Model",
        suffixes=("_fixed", "_self"),
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No overlapping completed models between fixed_repair and self_repair.")

    repaired_rows = []
    recovery_rows = []
    delta_rows = []
    win_rows = []

    verdict_signals: dict[str, float] = {}

    for metric in TARGET_METRICS:
        fixed_mean = merged[f"{metric}_repaired_fixed"].mean()
        self_mean = merged[f"{metric}_repaired_self"].mean()
        delta = self_mean - fixed_mean
        verdict_signals[metric] = delta

        repaired_rows.append(
            {
                "metric": metric,
                "fixed_repair_mean": fixed_mean,
                "self_repair_mean": self_mean,
                "self_minus_fixed": delta,
            }
        )

        fixed_recovery = merged[f"{metric}_recovery_pct_fixed"].mean()
        self_recovery = merged[f"{metric}_recovery_pct_self"].mean()
        recovery_rows.append(
            {
                "metric": metric,
                "fixed_recovery_pct": fixed_recovery,
                "self_recovery_pct": self_recovery,
                "self_minus_fixed_pct": self_recovery - fixed_recovery,
            }
        )

        fixed_gain = merged[f"{metric}_repaired_minus_noisy_fixed"].mean()
        self_gain = merged[f"{metric}_repaired_minus_noisy_self"].mean()
        delta_rows.append(
            {
                "metric": metric,
                "fixed_repaired_minus_noisy": fixed_gain,
                "self_repaired_minus_noisy": self_gain,
                "self_minus_fixed": self_gain - fixed_gain,
            }
        )

        fixed_wins = int((merged[f"{metric}_repaired_fixed"] > merged[f"{metric}_repaired_self"]).sum())
        self_wins = int((merged[f"{metric}_repaired_self"] > merged[f"{metric}_repaired_fixed"]).sum())
        ties = int((merged[f"{metric}_repaired_self"] == merged[f"{metric}_repaired_fixed"]).sum())
        win_rows.append(
            {
                "metric": metric,
                "fixed_model_wins": fixed_wins,
                "self_model_wins": self_wins,
                "ties": ties,
            }
        )

    repaired_table = pd.DataFrame(repaired_rows)
    recovery_table = pd.DataFrame(recovery_rows)
    delta_table = pd.DataFrame(delta_rows)
    wins_table = pd.DataFrame(win_rows)

    if (
        verdict_signals["intent_preservation"] > 0
        and verdict_signals["med_coverage"] < 0
        and verdict_signals["med_f1"] < 0
    ):
        verdict = (
            "Across the completed models, self-repair looks more like paraphrase-preserving cleanup "
            "than stronger medical repair: it improves intent preservation on average, but loses ground "
            "on medical content coverage and medical F1 relative to fixed GPT-5.4 repair."
        )
    elif verdict_signals["med_coverage"] > 0 and verdict_signals["med_f1"] > 0:
        verdict = (
            "Across the completed models, model-native self-repair beats fixed GPT-5.4 repair on the core "
            "medical-content metrics as well as the average repaired answer quality."
        )
    else:
        verdict = (
            "The result is mixed across models: neither repair strategy dominates every metric, so the main "
            "story depends on whether we prioritize medical content retention or surface-level intent cleanup."
        )

    lines = [
        f"# fixed_repair vs self_repair - {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        f"- Compared completed rows shared across both modes: `{len(merged)}` model(s)",
        "- Focus metrics: `med_coverage`, `med_f1`, `geval`, `intent_preservation`, and repaired-vs-noisy recovery",
        "",
        "## Headline",
        "",
        verdict,
        "",
        "## Average repaired scores by model",
        "",
        markdown_table(repaired_table),
        "",
        "## Average repaired-vs-noisy gains by model",
        "",
        markdown_table(delta_table),
        "",
        "## Average recovery percentages by model",
        "",
        markdown_table(recovery_table),
        "",
        "## Model win counts on repaired scores",
        "",
        markdown_table(wins_table, digits=0),
        "",
        "## Notes",
        "",
        "- Recovery percentages come from the benchmark summary columns and reflect how much the repaired answer climbs back toward the clean-answer score from the noisy-answer score.",
        "- Repaired-vs-noisy deltas are raw score differences (`repaired - noisy`) and help separate true recovery from overall baseline model strength.",
        "- This note uses completed models only, so rerunning it later will automatically reflect any updated benchmark outputs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fixed_df = load_completed_mode_table(args.fixed_dir, "fixed_repair")
    self_df = load_completed_mode_table(args.self_dir, "self_repair")
    report = build_report(fixed_df, self_df)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
