#!/usr/bin/env bash
# Re-run the three Protocol 2 baselines on splits_v2. Usage: bash run_all.sh <gpu_id>
# RF runs on CPU; EEGNet and the Transformer use the given GPU. Writes only into ./results.
set -euo pipefail
GPU="${1:?usage: bash run_all.sh <gpu_id>   (check nvidia-smi first)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"   # [release] was an absolute interpreter path
mkdir -p "$HERE/results/logs"

"$PY" -B "$HERE/run_baselines_voting_v2.py" --check           # no training; validates splits_v2

for m in rf eegnet transformer; do
  if [[ -f "$HERE/results/${m}_voting.json" ]]; then echo "$m: result exists, skipping"; continue; fi
  echo "=== $m  $(date '+%F %T')"
  CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" -B "$HERE/run_baselines_voting_v2.py" --method "$m" --cuda "$GPU" \
    2>&1 | tee "$HERE/results/logs/${m}.log"
done

"$PY" -B "$HERE/summarize_baselines_v2.py" | tee "$HERE/results/logs/summary.log"
