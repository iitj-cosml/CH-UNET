#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

for enf_mass in true false; do
    echo "=================================================================="
    echo "[run_pipeline] training  (enf_mass=${enf_mass})"
    echo "=================================================================="
    python train.py --enf_mass "${enf_mass}"

    echo "=================================================================="
    echo "[run_pipeline] predicting (enf_mass=${enf_mass})"
    echo "=================================================================="
    python pred.py --enf_mass "${enf_mass}"
done

echo "[run_pipeline] done. Results in test_results/"
