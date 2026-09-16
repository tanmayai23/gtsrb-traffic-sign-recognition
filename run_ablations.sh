#!/usr/bin/env bash
# Ablation study: retrain with one design decision removed at a time.
# Each run writes its own checkpoint and metrics so the baseline is untouched.
set -uo pipefail

run() {
  name="$1"; shift
  echo "=== ablation: ${name} ==="
  python -m src.cli train --epochs 15 --out "ablations/${name}" \
      --results "ablations/${name}" "$@" > "ablations/${name}_train.log" 2>&1
  python -m src.cli evaluate --checkpoint "ablations/${name}/best.pt" \
      --results "ablations/${name}" > "ablations/${name}_eval.log" 2>&1
  echo "  done: ${name}"
}

run no_clahe     --no-clahe
run no_balance   --balance none
run half_width   --width-mult 0.5

echo "ALL ABLATIONS COMPLETE"
