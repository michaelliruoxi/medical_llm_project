"""Cross-platform orchestrator — runs the full pipeline sequentially."""

import argparse
import subprocess
import sys


STEPS = [
    ("Step 1: Ingest",             [sys.executable, "-m", "src.ingest"]),
    ("Step 2: Noise injection",    [sys.executable, "-m", "src.noise"]),
    ("Step 3a: Answer (clean)",    [sys.executable, "-m", "src.answer", "--mode", "clean"]),
    ("Step 3b: Answer (noisy)",    [sys.executable, "-m", "src.answer", "--mode", "noisy"]),
    ("Step 4: Repair",             [sys.executable, "-m", "src.repair"]),
    ("Step 5: Answer (repaired)",  [sys.executable, "-m", "src.answer", "--mode", "repaired"]),
    ("Step 6: Evaluate metrics",   [sys.executable, "-m", "src.evaluate_metrics"]),
    ("Step 7: LLM judge",          [sys.executable, "-m", "src.judge"]),
    ("Step 8: Aggregate",          [sys.executable, "-m", "src.aggregate"]),
    ("Step 9: Report tables",      [sys.executable, "-m", "src.report_tables"]),
]


def main():
    parser = argparse.ArgumentParser(description="Run the full MedQuAD robustness pipeline")
    parser.add_argument("--pilot", action="store_true",
                        help="Run with 20 samples instead of the full dataset")
    args = parser.parse_args()

    pilot_modules = {"src.ingest", "src.noise", "src.repair"}

    for label, cmd in STEPS:
        module_name = cmd[2] if len(cmd) > 2 else ""
        if args.pilot and module_name in pilot_modules:
            cmd = cmd + ["--n", "20"]

        print(f"\n{'='*60}")
        print(f"=== {label} ===")
        print(f"{'='*60}\n")

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\nERROR: {label} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("=== Pipeline complete ===")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
