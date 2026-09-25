#!/usr/bin/env bash
# Protocol 2 head-only control (Reviewer 2, comment 2). Usage: bash run_head_only_p2.sh <gpu_id>
# Reads the five locked Protocol 2 checkpoints and splits_v2; writes only into ./results.
set -euo pipefail

GPU="${1:?usage: bash run_head_only_p2.sh <gpu_id>  (check nvidia-smi first)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"   # [release]
PY="${PYTHON:-python}"   # [release] was an absolute interpreter path
CKPT="${EEGLLAVA_P2_RUN:-$REPO/outputs/protocol2/R1a_seed1234}/ckpt"
SPLITS="$REPO/splits/protocol2_v2"
OUT="$HERE/results"
mkdir -p "$OUT/logs"

# GPU-free check that the aggregation reproduces the locked 80.1% / 0.849 headline.
"$PY" -B "$HERE/aggregate_head_only_p2.py" --selfcheck

for k in 0 1 2 3 4; do
  if [[ -f "$OUT/probe_p2_fold${k}.json" ]]; then
    echo "fold $k: output exists, skipping"
    continue
  fi
  "$PY" -B "$HERE/probe_heldout_p2.py" --cuda "$GPU" \
    --checkpoint "$CKPT/fold_${k}_best.pth" \
    --split "$SPLITS/fold_${k}.json" \
    --output "$OUT/probe_p2_fold${k}.json" 2>&1 | tee "$OUT/logs/fold_${k}.log"
done

"$PY" -B "$HERE/aggregate_head_only_p2.py" --results-dir "$OUT" \
  --output "$OUT/E16_head_only_protocol2.json"
