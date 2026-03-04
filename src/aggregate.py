"""Step E (part 3): Aggregate metrics, compute robustness measures, and run statistical tests."""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import PROJECT_ROOT, load_config, setup_logging


logger = setup_logging()

METRIC_COLS = ["bleu", "bertscore_f1", "judge_score"]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_by_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, std for each metric grouped by pipeline."""
    rows = []
    for pipeline, grp in df.groupby("pipeline"):
        row = {"pipeline": pipeline}
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
        row = {"pipeline": pipeline, "noise_type": nt}
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

        # Match rows by id
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
            degradation = c[col].mean() - n[col].mean()
            recovery = (rep[col].mean() - n[col].mean()) if rep is not None and col in rep.columns else np.nan
            recovery_ratio = (recovery / degradation) if degradation != 0 and not np.isnan(recovery) else np.nan

            rows.append({
                "noise_type": nt,
                "metric": col,
                "clean_mean": c[col].mean(),
                "noisy_mean": n[col].mean(),
                "repaired_mean": rep[col].mean() if rep is not None and col in rep.columns else np.nan,
                "degradation": degradation,
                "recovery": recovery,
                "recovery_ratio": recovery_ratio,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank tests: clean vs noisy, noisy vs repaired."""
    clean = df[df["pipeline"] == "clean"].set_index("id").sort_index()
    noisy_all = df[df["pipeline"] == "noisy"]
    repaired_all = df[df["pipeline"] == "repaired"]

    noise_types = noisy_all["noise_type"].unique() if "noise_type" in noisy_all.columns else ["all"]
    rows = []

    for nt in noise_types:
        noisy = noisy_all[noisy_all["noise_type"] == nt] if "noise_type" in noisy_all.columns else noisy_all
        noisy = noisy.set_index("id").sort_index()
        repaired = (
            repaired_all[repaired_all["noise_type"] == nt].set_index("id").sort_index()
            if not repaired_all.empty and "noise_type" in repaired_all.columns
            else repaired_all.set_index("id").sort_index() if not repaired_all.empty
            else pd.DataFrame()
        )

        shared_cn = sorted(set(clean.index) & set(noisy.index))
        shared_nr = sorted(set(noisy.index) & set(repaired.index)) if not repaired.empty else []

        for col in METRIC_COLS:
            if col not in clean.columns or col not in noisy.columns:
                continue

            # Clean vs Noisy
            c_vals = clean.loc[shared_cn, col].values
            n_vals = noisy.loc[shared_cn, col].values
            try:
                stat_cn, p_cn = stats.wilcoxon(c_vals, n_vals)
            except ValueError:
                stat_cn, p_cn = np.nan, np.nan
            d_cn = _cohens_d(c_vals, n_vals)

            # Noisy vs Repaired
            stat_nr, p_nr, d_nr = np.nan, np.nan, np.nan
            if shared_nr and col in repaired.columns:
                n2 = noisy.loc[shared_nr, col].values
                r_vals = repaired.loc[shared_nr, col].values
                try:
                    stat_nr, p_nr = stats.wilcoxon(n2, r_vals)
                except ValueError:
                    pass
                d_nr = _cohens_d(r_vals, n2)

            rows.append({
                "noise_type": nt,
                "metric": col,
                "wilcoxon_stat_clean_noisy": stat_cn,
                "p_value_clean_noisy": p_cn,
                "cohens_d_clean_noisy": d_cn,
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
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return diff.mean() / diff.std() if diff.std() > 0 else 0.0


def compute_bootstrap_cis(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap 95% CIs for each (pipeline, metric) combination."""
    rows = []
    for pipeline, grp in df.groupby("pipeline"):
        for col in METRIC_COLS:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().values
            if len(vals) < 2:
                continue
            lo, hi = bootstrap_ci(vals)
            rows.append({
                "pipeline": pipeline,
                "metric": col,
                "mean": vals.mean(),
                "ci_lower": lo,
                "ci_upper": hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]
    metrics_path = out_dir / "metrics.parquet"

    df = pd.read_parquet(metrics_path)
    logger.info("Loaded %d metric rows", len(df))

    # Summary tables
    summary = summary_by_pipeline(df)
    summary.to_csv(out_dir / "summary_by_pipeline.csv", index=False)
    logger.info("Summary by pipeline:\n%s", summary.to_string(index=False))

    summary_noise = summary_by_pipeline_noise(df)
    if not summary_noise.empty:
        summary_noise.to_csv(out_dir / "summary_by_pipeline_noise.csv", index=False)

    # Robustness
    robust = robustness_metrics(df)
    if not robust.empty:
        robust.to_csv(out_dir / "robustness_metrics.csv", index=False)
        logger.info("Robustness metrics:\n%s", robust.to_string(index=False))

    # Statistical tests
    tests = paired_tests(df)
    if not tests.empty:
        tests.to_csv(out_dir / "statistical_tests.csv", index=False)
        logger.info("Statistical tests:\n%s", tests.to_string(index=False))

    # Bootstrap CIs
    cis = compute_bootstrap_cis(df)
    if not cis.empty:
        cis.to_csv(out_dir / "bootstrap_cis.csv", index=False)
        logger.info("Bootstrap CIs:\n%s", cis.to_string(index=False))

    logger.info("All aggregate outputs saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results and compute statistics")
    parser.parse_args()
    run()
