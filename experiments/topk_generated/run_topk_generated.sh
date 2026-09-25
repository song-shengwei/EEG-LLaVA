#!/usr/bin/env bash
# Fig. 15(c): Top-K channel subsets with generated decisions on the Protocol 1 seed-42 checkpoint.
#
# Usage:  EEGLLAVA_P1_CKPT=<checkpoints/protocol1_seed42/fold_0_best.pth> bash run_topk_generated.sh <gpu_id>
#
# Step 1: intact input and the six single-channel masks; summarize.py rank orders the channels by
#         the accuracy drop on the test set (the rule of the submission version of this analysis).
# Step 2: Top-K keeps the K highest-ranked channels and zeroes the others (K = 1..4);
#         summarize.py final writes results/fig15c_generated_E11schema.json.
# The paper's run used an RTX 4090; on an A40 one near-tie decision differed in a 100-segment check.
# gen_masking.py appends one line per segment, so an interrupted condition resumes where it stopped.
set -euo pipefail
GPU="${1:?usage: bash run_topk_generated.sh <gpu_id>}"
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p logs
run() { "$PY" -B gen_masking.py --cuda "$GPU" --name "$1" --mask "$2" > "logs/$1.log" 2>&1; echo "$1 exit $?"; }

run baseline ""
for c in PO3 POz PO4 O1 Oz O2; do run "loo_$c" "$c"; done
"$PY" -B summarize.py rank

for k in 1 2 3 4; do
  m=$("$PY" -B -c "import json,sys;o=json.load(open('ranking.json'))['order'];k=int(sys.argv[1]);print(','.join(c for c in ['PO3','POz','PO4','O1','Oz','O2'] if c not in o[:k]))" "$k")
  run "top$k" "$m"
done
"$PY" -B summarize.py final
