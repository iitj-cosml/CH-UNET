#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

for enf_cons in true false; do
    echo "=================================================================="
    echo "[run_pipeline] training  (enf_cons=${enf_cons})"
    echo "=================================================================="
    python train.py --enf_cons "${enf_cons}"

    echo "=================================================================="
    echo "[run_pipeline] predicting (enf_cons=${enf_cons})"
    echo "=================================================================="
    python pred.py --enf_cons "${enf_cons}"
done

echo "[run_pipeline] done. Results in results/"
