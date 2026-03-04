"""Generate publication-ready tables (LaTeX + Markdown) and bar charts from aggregate outputs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import PROJECT_ROOT, load_config, setup_logging


logger = setup_logging()


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    ncols = len(df.columns)
    col_fmt = "l" + "r" * (ncols - 1)
    latex = df.to_latex(index=False, float_format="%.3f", column_format=col_fmt)
    return (
        f"\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n{latex}\\end{{table}}\n"
    )


def generate_tables(out_dir: Path):
    """Write LaTeX and markdown versions of all summary CSVs."""
    summary_path = out_dir / "summary_by_pipeline.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        latex = _df_to_latex(df, "Summary by Pipeline", "tab:summary_pipeline")
        (out_dir / "summary_by_pipeline.tex").write_text(latex, encoding="utf-8")
        md = df.to_markdown(index=False, floatfmt=".3f")
        (out_dir / "summary_by_pipeline.md").write_text(md, encoding="utf-8")
        logger.info("Wrote summary_by_pipeline tables")

    robust_path = out_dir / "robustness_metrics.csv"
    if robust_path.exists():
        df = pd.read_csv(robust_path)
        latex = _df_to_latex(df, "Robustness Metrics", "tab:robustness")
        (out_dir / "robustness_metrics.tex").write_text(latex, encoding="utf-8")
        md = df.to_markdown(index=False, floatfmt=".3f")
        (out_dir / "robustness_metrics.md").write_text(md, encoding="utf-8")
        logger.info("Wrote robustness_metrics tables")

    tests_path = out_dir / "statistical_tests.csv"
    if tests_path.exists():
        df = pd.read_csv(tests_path)
        latex = _df_to_latex(df, "Statistical Tests", "tab:tests")
        (out_dir / "statistical_tests.tex").write_text(latex, encoding="utf-8")
        md = df.to_markdown(index=False, floatfmt=".4f")
        (out_dir / "statistical_tests.md").write_text(md, encoding="utf-8")
        logger.info("Wrote statistical_tests tables")

    ci_path = out_dir / "bootstrap_cis.csv"
    if ci_path.exists():
        df = pd.read_csv(ci_path)
        latex = _df_to_latex(df, "Bootstrap 95\\% Confidence Intervals", "tab:cis")
        (out_dir / "bootstrap_cis.tex").write_text(latex, encoding="utf-8")
        md = df.to_markdown(index=False, floatfmt=".3f")
        (out_dir / "bootstrap_cis.md").write_text(md, encoding="utf-8")
        logger.info("Wrote bootstrap_cis tables")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(out_dir: Path):
    """Create bar charts for degradation/recovery by noise type and pipeline comparison."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline comparison grouped bar chart
    summary_path = out_dir / "summary_by_pipeline.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        metrics = [c.replace("_mean", "") for c in df.columns if c.endswith("_mean")]
        if metrics:
            plot_df = pd.melt(
                df, id_vars=["pipeline"],
                value_vars=[f"{m}_mean" for m in metrics],
                var_name="metric", value_name="score"
            )
            plot_df["metric"] = plot_df["metric"].str.replace("_mean", "")

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x="metric", y="score", hue="pipeline", ax=ax)
            ax.set_title("Pipeline Comparison")
            ax.set_ylabel("Score")
            ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(charts_dir / "pipeline_comparison.png", dpi=150)
            plt.close(fig)
            logger.info("Saved pipeline_comparison.png")

    # Degradation vs Recovery by noise type
    robust_path = out_dir / "robustness_metrics.csv"
    if robust_path.exists():
        df = pd.read_csv(robust_path)
        for metric in df["metric"].unique():
            mdf = df[df["metric"] == metric].copy()
            if mdf.empty or mdf["noise_type"].nunique() < 1:
                continue

            fig, ax = plt.subplots(figsize=(9, 5))
            x = range(len(mdf))
            width = 0.35
            ax.bar([i - width / 2 for i in x], mdf["degradation"], width, label="Degradation")
            ax.bar([i + width / 2 for i in x], mdf["recovery"], width, label="Recovery")
            ax.set_xticks(list(x))
            ax.set_xticklabels(mdf["noise_type"], rotation=30, ha="right")
            ax.set_title(f"Degradation vs Recovery — {metric}")
            ax.set_ylabel("Score Delta")
            ax.legend()
            fig.tight_layout()
            fig.savefig(charts_dir / f"deg_vs_rec_{metric}.png", dpi=150)
            plt.close(fig)
            logger.info("Saved deg_vs_rec_%s.png", metric)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["paths"]["outputs"]

    generate_tables(out_dir)
    generate_charts(out_dir)
    logger.info("Report generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate report tables and charts")
    parser.parse_args()
    run()
