#!/usr/bin/env bash
set -euo pipefail

PILOT_N=""
if [[ "${1:-}" == "--pilot" ]]; then
    PILOT_N="--n 20"
    echo "=== PILOT MODE: 20 samples ==="
fi

echo "=== Step 1: Ingest ==="
python -m src.ingest $PILOT_N

echo "=== Step 2: Noise injection ==="
python -m src.noise $PILOT_N

echo "=== Step 3a: Answer (clean) ==="
python -m src.answer --mode clean

echo "=== Step 3b: Answer (noisy) ==="
python -m src.answer --mode noisy

echo "=== Step 4: Repair ==="
python -m src.repair $PILOT_N

echo "=== Step 5: Answer (repaired) ==="
python -m src.answer --mode repaired

echo "=== Step 6: Evaluate metrics ==="
python -m src.evaluate_metrics

echo "=== Step 7: LLM judge ==="
python -m src.judge

echo "=== Step 8: Aggregate ==="
python -m src.aggregate

echo "=== Step 9: Report tables ==="
python -m src.report_tables

echo "=== Done ==="
