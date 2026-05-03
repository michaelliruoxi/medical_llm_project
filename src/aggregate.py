"""Aggregate metrics, compute robustness measures, and run paired statistical tests.

This module is the significance layer for the benchmark pipeline. It reads the
per-model wide-format ``samples_<label>.csv`` files produced by
``scripts/benchmarks/run_comparison.py``, reshapes them to long form, and
writes per-model summary / robustness / Wilcoxon / bootstrap-CI tables.

Library functions (``summary_by_pipeline``, ``robustness_metrics``,
``paired_tests``, ``compute_bootstrap_cis``) operate on long-form DataFrames
with columns ``id``, ``pipeline`` (clean/noisy/repaired), ``noise_type`` and
any subset of ``METRIC_COLS``.

Usage:
    python -m src.aggregate --mode fixed_repair
    python -m src.aggregate --mode self_repair --output-root data/outputs/benchmarks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import compute_recovery_statistics
from src.utils import PROJECT_ROOT, setup_logging


logger = setup_logging()

PIPELINES = ("clean", "noisy", "repaired")

METRIC_COLS = [
    "bleu",
    "chrf",
    "rouge_l",
    "token_f1",
    "exact_match",
    "med_coverage",
    "med_precision",
    "med_f1",
    "bertscore",
    "intent_preservation",
    "geval",
]

BASE_LONG_COLS = ["id", "noise_type", "pipeline"]


# ---------------------------------------------------------------------------
# Wide (benchmark samples_*.csv) -> long adapter
# ---------------------------------------------------------------------------

def samples_wide_to_long(samples_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape one model's wide-format samples dataframe into long form.

    The benchmark writes each sample row with per-pipeline metric columns
    like ``bleu_clean``, ``bleu_noisy``, ``bleu_repaired``. We melt that into
    one long row per (question_id, noise_type, pipeline) with one column per
    metric, which is the shape the statistical helpers below expect.
    """
    if samples_df.empty:
        return pd.DataFrame(columns=BASE_LONG_COLS + METRIC_COLS)

    records: list[dict] = []
    for _, row in samples_df.iterrows():
        try:
            qid = int(row["question_id"])
        except (KeyError, ValueError, TypeError):
            continue
        noise_type = row.get("noise_type", "")
        for pipeline in PIPELINES:
            rec = {"id": qid, "noise_type": noise_type, "pipeline": pipeline}
            for metric in METRIC_COLS:
                col = f"{metric}_{pipeline}"
                if col in samples_df.columns:
                    val = row[col]
                    try:
                        rec[metric] = float(val) if pd.notna(val) else np.nan
                    except (TypeError, ValueError):
                        rec[metric] = np.nan
            records.append(rec)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_by_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, std for each metric grouped by pipeline."""
    rows = []
    for pipeline, grp in df.groupby("pipeline"):
        row = {"pipeline": pipeline, "n": int(len(grp))}
        for col in METRIC_COLS:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_median"] = grp[col].median()
                row[f"{col}_std"] = grp[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


def summary_by_pipeline_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean for each metric grouped by (pipeline, noise_type)."""
    if "noise_type" not in df.columns:
        return pd.DataFrame()

    rows = []
    for (pipeline, nt), grp in df.groupby(["pipeline", "noise_type"]):
        row = {"pipeline": pipeline, "noise_type": nt, "n": int(len(grp))}
        for col in METRIC_COLS:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness metrics (per noise-type grouping on shared IDs)
# ---------------------------------------------------------------------------

def robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Degradation, Recovery, Recovery Ratio per metric per noise type."""
    clean = df[df["pipeline"] == "clean"]
    noisy = df[df["pipeline"] == "noisy"]
    repaired = df[df["pipeline"] == "repaired"]

    if clean.empty or noisy.empty:
        logger.warning("Need both clean and noisy pipelines for robustness metrics.")
        return pd.DataFrame()

    noise_types = noisy["noise_type"].unique() if "noise_type" in noisy.columns else ["all"]
    rows = []

    for nt in noise_types:
        noisy_nt = noisy[noisy["noise_type"] == nt] if "noise_type" in noisy.columns else noisy

        ids = set(clean["id"]) & set(noisy_nt["id"])
        c = clean[clean["id"].isin(ids)].set_index("id").sort_index()
        n = noisy_nt[noisy_nt["id"].isin(ids)].set_index("id").sort_index()

        rep = None
        if not repaired.empty:
            rep_nt = repaired[repaired["noise_type"] == nt] if "noise_type" in repaired.columns else repaired
            rep_ids = ids & set(rep_nt["id"])
            if rep_ids:
                rep = rep_nt[rep_nt["id"].isin(rep_ids)].set_index("id").sort_index()

        for col in METRIC_COLS:
            if col not in c.columns or col not in n.columns:
                continue
            clean_mean = c[col].mean()
            noisy_mean = n[col].mean()
            repaired_mean = rep[col].mean() if rep is not None and col in rep.columns else np.nan
            recovery_stats = compute_recovery_statistics(
                clean_mean,
                noisy_mean,
                repaired_mean,
                metric_name=col,
            )

            rows.append({
                "noise_type": nt,
                "metric": col,
                "n_pairs": int(len(ids)),
                "clean_mean": clean_mean,
                "noisy_mean": noisy_mean,
                "repaired_mean": repaired_mean,
                "degradation": recovery_stats["degradation"],
                "recovery": recovery_stats["recovery"],
                "recovery_ratio": recovery_stats["recovery_ratio"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank tests: clean vs noisy, noisy vs repaired (paired by id)."""
    clean_all = df[df["pipeline"] == "clean"].set_index("id").sort_index()
    noisy_all = df[df["pipeline"] == "noisy"]
    repaired_all = df[df["pipeline"] == "repaired"]

    noise_types = noisy_all["noise_type"].unique() if "noise_type" in noisy_all.columns else ["all"]
    rows = []

    for nt in noise_types:
        noisy = noisy_all[noisy_all["noise_type"] == nt] if "noise_type" in noisy_all.columns else noisy_all
        noisy = noisy.set_index("id").sort_index()

        if not repaired_all.empty and "noise_type" in repaired_all.columns:
            repaired = repaired_all[repaired_all["noise_type"] == nt].set_index("id").sort_index()
        elif not repaired_all.empty:
            repaired = repaired_all.set_index("id").sort_index()
        else:
            repaired = pd.DataFrame()

        shared_cn = sorted(set(clean_all.index) & set(noisy.index))
        shared_nr = sorted(set(noisy.index) & set(repaired.index)) if not repaired.empty else []

        for col in METRIC_COLS:
            if col not in clean_all.columns or col not in noisy.columns:
                continue

            c_vals = clean_all.loc[shared_cn, col].dropna().values
            n_vals = noisy.loc[shared_cn, col].dropna().values
            # align after dropna: use intersection of non-na indices
            paired_cn = (
                pd.concat(
                    [clean_all.loc[shared_cn, col], noisy.loc[shared_cn, col]],
                    axis=1, keys=["c", "n"],
                )
                .dropna()
            )
            c_vals = paired_cn["c"].values
            n_vals = paired_cn["n"].values
            stat_cn, p_cn = np.nan, np.nan
            if len(c_vals) >= 1 and np.any(c_vals - n_vals):
                try:
                    stat_cn, p_cn = stats.wilcoxon(c_vals, n_vals, zero_method="wilcox")
                except ValueError:
                    pass
            d_cn = _cohens_d(c_vals, n_vals) if len(c_vals) else np.nan

            stat_nr, p_nr, d_nr = np.nan, np.nan, np.nan
            n_pairs_nr = 0
            if shared_nr and col in repaired.columns:
                paired_nr = (
                    pd.concat(
                        [noisy.loc[shared_nr, col], repaired.loc[shared_nr, col]],
                        axis=1, keys=["n", "r"],
                    )
                    .dropna()
                )
                n2 = paired_nr["n"].values
                r_vals = paired_nr["r"].values
                n_pairs_nr = int(len(n2))
                if len(n2) >= 1 and np.any(n2 - r_vals):
                    try:
                        stat_nr, p_nr = stats.wilcoxon(n2, r_vals, zero_method="wilcox")
                    except ValueError:
                        pass
                d_nr = _cohens_d(r_vals, n2) if len(n2) else np.nan

            rows.append({
                "noise_type": nt,
                "metric": col,
                "n_pairs_clean_noisy": int(len(c_vals)),
                "wilcoxon_stat_clean_noisy": stat_cn,
                "p_value_clean_noisy": p_cn,
                "cohens_d_clean_noisy": d_cn,
                "n_pairs_noisy_repaired": n_pairs_nr,
                "wilcoxon_stat_noisy_repaired": stat_nr,
                "p_value_noisy_repaired": p_nr,
                "cohens_d_noisy_repaired": d_nr,
            })

    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_bootstrap_cis(df: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    """Bootstrap 95% CIs for each (pipeline, metric) combination."""
    rows = []
    for pipeline, grp in df.groupby("pipeline"):
        for col in METRIC_COLS:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().values
            if len(vals) < 2:
                continue
            lo, hi = bootstrap_ci(vals, n_boot=n_boot)
            rows.append({
                "pipeline": pipeline,
                "metric": col,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "ci_lower": lo,
                "ci_upper": hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def aggregate_one_model(
    samples_csv: Path,
    stats_dir: Path,
    n_boot: int = 10000,
) -> dict[str, Path]:
    """Run the full stats suite on one model's samples_*.csv.

    Writes four CSVs under ``stats_dir``:
        summary_<label>.csv
        summary_noise_<label>.csv
        robustness_<label>.csv
        paired_tests_<label>.csv
        bootstrap_cis_<label>.csv
    """
    label = samples_csv.stem[len("samples_"):] if samples_csv.stem.startswith("samples_") else samples_csv.stem
    stats_dir.mkdir(parents=True, exist_ok=True)

    samples_df = pd.read_csv(samples_csv)
    long_df = samples_wide_to_long(samples_df)

    out: dict[str, Path] = {}
    if long_df.empty:
        logger.warning("%s: no rows after reshape; skipping stats", label)
        return out

    summary = summary_by_pipeline(long_df)
    if not summary.empty:
        path = stats_dir / f"summary_{label}.csv"
        summary.to_csv(path, index=False)
        out["summary"] = path

    summary_noise = summary_by_pipeline_noise(long_df)
    if not summary_noise.empty:
        path = stats_dir / f"summary_noise_{label}.csv"
        summary_noise.to_csv(path, index=False)
        out["summary_noise"] = path

    robust = robustness_metrics(long_df)
    if not robust.empty:
        path = stats_dir / f"robustness_{label}.csv"
        robust.to_csv(path, index=False)
        out["robustness"] = path

    tests = paired_tests(long_df)
    if not tests.empty:
        path = stats_dir / f"paired_tests_{label}.csv"
        tests.to_csv(path, index=False)
        out["paired_tests"] = path

    cis = compute_bootstrap_cis(long_df, n_boot=n_boot)
    if not cis.empty:
        path = stats_dir / f"bootstrap_cis_{label}.csv"
        cis.to_csv(path, index=False)
        out["bootstrap_cis"] = path

    logger.info(
        "Aggregated %s: %d long rows -> %d tables under %s",
        label, len(long_df), len(out), stats_dir,
    )
    return out


def run_for_mode(
    mode: str,
    output_root: Path,
    n_boot: int = 10000,
) -> list[str]:
    mode_dir = output_root / mode
    if not mode_dir.exists():
        logger.error("No output directory for mode=%s: %s", mode, mode_dir)
        return []

    samples = sorted(mode_dir.glob("samples_*.csv"))
    if not samples:
        logger.warning("No samples_*.csv in %s", mode_dir)
        return []

    stats_dir = mode_dir / "stats"
    labels: list[str] = []
    for sp in samples:
        label = sp.stem[len("samples_"):]
        aggregate_one_model(sp, stats_dir, n_boot=n_boot)
        labels.append(label)
    logger.info("Aggregated %d model(s) for mode=%s into %s", len(labels), mode, stats_dir)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("end_to_end", "fixed_repair", "self_repair"),
        default="fixed_repair",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs" / "benchmarks",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    run_for_mode(args.mode, args.output_root, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
