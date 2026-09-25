#!/usr/bin/env bash
# Re-run the seven Protocol 1 Table 3 baselines on the locked 1,404-segment split, then compare segment-level
# diagnosis with eye-level soft voting.
#
# Usage: bash run_all.sh <gpu_ids>      e.g.  bash run_all.sh 3   |   bash run_all.sh 3,5   |   bash run_all.sh 3,4,5
#   1-3 distinct GPU indices as nvidia-smi prints them (PCI bus order). Each deep baseline is its own process that
#   sees only its GPU; one model per GPU at a time, the remaining models queue for the next free GPU.
#   SVM/LDA run on the CPU alongside. Writes only into ./results. ALLOW_BUSY=1 skips the idle-GPU check.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"   # [release] was an absolute interpreter path
MAX_GPUS=3
BUSY_MIB=1000
QUEUE=(lstm cnnlstm transformer cnn1d eegnet)   # the five deep baselines, presumed slowest first

IFS=',' read -r -a GPUS <<< "${1:?usage: bash run_all.sh <gpu_ids>   e.g. 3,4,5 (check nvidia-smi first)}"
(( ${#GPUS[@]} >= 1 && ${#GPUS[@]} <= MAX_GPUS )) || { echo "give 1-$MAX_GPUS GPU ids, got ${#GPUS[@]}"; exit 1; }
[[ -z "$(printf '%s\n' "${GPUS[@]}" | sort | uniq -d)" ]] || { echo "duplicate GPU ids: $1"; exit 1; }
for g in "${GPUS[@]}"; do
  [[ "$g" =~ ^[0-9]+$ ]] || { echo "bad GPU id: $g"; exit 1; }
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits) || { echo "no GPU $g"; exit 1; }
  if (( used > BUSY_MIB )) && [[ "${ALLOW_BUSY:-0}" != 1 ]]; then
    echo "GPU $g already uses ${used} MiB; pick an idle GPU or set ALLOW_BUSY=1"; exit 1
  fi
done

mkdir -p "$HERE/results/logs" "$HERE/results/dl"
declare -A GPU_OF=() MODEL_OF=()   # pid -> GPU / model of a running deep baseline
FREE=("${GPUS[@]}")
FAILED=()
CLASSICAL_PID="" CLASSICAL_RC=""
trap 'echo "interrupted, stopping jobs"; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM


if [[ -f "$HERE/results/classical_p1.json" ]]; then echo "classical: result exists, skipping"; else
  echo "=== SVM / LDA on CPU  $(date '+%F %T')  log: results/logs/classical.log"
  "$PY" -B "$HERE/baseline_p1.py" > "$HERE/results/logs/classical.log" 2>&1 &
  CLASSICAL_PID=$!
fi

reap_one() {   # wait for any background job to end; free its GPU and record failures
  local pid="" rc=0
  wait -n -p pid || rc=$?
  if [[ "$pid" == "$CLASSICAL_PID" ]]; then
    CLASSICAL_RC=$rc
    echo "--- SVM / LDA finished, exit $rc  $(date '+%F %T')"
    (( rc == 0 )) || FAILED+=(classical)
    return
  fi
  local g=${GPU_OF[$pid]} m=${MODEL_OF[$pid]}
  unset "GPU_OF[$pid]" "MODEL_OF[$pid]"
  FREE+=("$g")
  echo "--- $m finished on GPU $g, exit $rc  $(date '+%F %T')"
  (( rc == 0 )) || FAILED+=("dl_$m")
}

for m in "${QUEUE[@]}"; do
  if [[ -f "$HERE/results/dl/$m.json" ]]; then echo "$m: result exists, skipping"; continue; fi
  while (( ${#FREE[@]} == 0 )); do reap_one; done
  g=${FREE[0]}; FREE=("${FREE[@]:1}")
  echo "=== $m on GPU $g  $(date '+%F %T')  log: results/logs/dl_$m.log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" \
    "$PY" -B "$HERE/dl_baselines_p1.py" --cuda 0 --model "$m" --epochs 50 --batch_size 64 \
    > "$HERE/results/logs/dl_$m.log" 2>&1 &
  GPU_OF[$!]=$g; MODEL_OF[$!]=$m
done
while (( ${#GPU_OF[@]} > 0 )) || [[ -n "$CLASSICAL_PID" && -z "$CLASSICAL_RC" ]]; do reap_one; done

if (( ${#FAILED[@]} > 0 )); then
  echo "FAILED: ${FAILED[*]}; comparison not run"
  for f in "${FAILED[@]}"; do echo "----- results/logs/$f.log (last 20 lines)"; tail -n 20 "$HERE/results/logs/$f.log"; done
  exit 1
fi

echo "done: results/classical_p1.json, results/dl/*.json  (RF row: python rf_full_comparison_p1.py)"
