#!/usr/bin/env bash
# End-to-end pipeline: data -> training -> evaluation.
# Usage:  bash run_all.sh [epochs]
set -euo pipefail

EPOCHS="${1:-18}"

echo "=============================================="
echo " GTSRB traffic sign recognition - full run"
echo "=============================================="
python -m src.cli info

echo
echo "[1/4] preparing data (downloads ~365 MB the first time)"
python -m src.cli prepare

echo
echo "[2/4] training for ${EPOCHS} epochs"
python -m src.cli train --epochs "${EPOCHS}"

echo
echo "[3/4] evaluating on the official test set"
python -m src.cli evaluate

echo
echo "[4/4] running tests"
python -m unittest discover tests

echo
echo "Done. Metrics and figures are in results/"
