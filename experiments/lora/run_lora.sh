#!/usr/bin/env bash
# Protocol 1, 55-token EEG-LLaVA with LoRA in Stage 2: drive the short processes of train_p1_lora.py on one GPU.
# Usage: bash run_lora.sh <gpu_id> [seed]        (seed defaults to 42, the reported Protocol 1 checkpoint)
# Each phase is its own process (< 2 h, the cluster's limit for unregistered GPU processes) and saves its state,
# so rerunning this script after an interruption skips finished phases and resumes Stage 2 from its last chunk.
# ALLOW_BUSY=1 skips the idle-GPU check. Writes only into ./runs/seed<seed>.
set -euo pipefail
GPU="${1:?usage: bash run_lora.sh <gpu_id> [seed]   (check nvidia-smi first)}"
SEED="${2:-42}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"   # [release] was an absolute interpreter path
OUT="$HERE/runs/seed$SEED"
BUSY_MIB=1000
MAX_STAGE2_CALLS=10

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "bad GPU id: $GPU"; exit 1; }
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || { echo "no GPU $GPU"; exit 1; }
if (( used > BUSY_MIB )) && [[ "${ALLOW_BUSY:-0}" != 1 ]]; then
  echo "GPU $GPU already uses ${used} MiB; pick an idle GPU or set ALLOW_BUSY=1"; exit 1
fi
mkdir -p "$OUT/logs"

run() {   # run <phase> <log name>
  echo "=== $1 on GPU $GPU  $(date '+%F %T')  log: runs/seed$SEED/logs/$2.log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" -B "$HERE/train_p1_lora.py" --phase "$1" --seed "$SEED" --cuda 0 2>&1 | tee "$OUT/logs/$2.log"
  echo "--- $1 finished  $(date '+%F %T')"
}

if [[ ! -f "$OUT/ckpt/stage1.pt" ]]; then
  run check check          # ~2 min: model build, LoRA, one forward/generate; writes nothing
  run stage1 stage1
fi
n=1
while [[ ! -f "$OUT/ckpt/stage2.done" ]]; do
  (( n <= MAX_STAGE2_CALLS )) || { echo "Stage 2 did not finish after $MAX_STAGE2_CALLS calls"; exit 1; }
  run stage2 "stage2_call$(date '+%m%d_%H%M')"
  n=$((n + 1))
done
if [[ ! -f "$OUT/logs/fold_0_result.json" ]]; then
  run test test
fi
echo "done: $OUT/logs/fold_0_result.json"
