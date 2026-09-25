#!/usr/bin/env bash
# Protocol 1 (eye-disjoint split, Table 3): train one seed end to end, then evaluate.
#
# Usage:  bash training/run_protocol1.sh <gpu_id> [seed=42]
#
# Steps (identical arguments to the runs reported in the paper):
#   1. auxiliary compact-Transformer branch  (training/prepare_component.py --component aux)
#   2. CBraMod adaptation, Phase 0            (training/prepare_component.py --component mainenc)
#   3. Stage 1 (projectors, LLM frozen) + Stage 2 (projectors + LLM), complete checkpoint
#   4. segment-level metrics, DeLong / McNemar / bootstrap against the depth-12 RF (Fig. 10)
# Components are fitted on the training eyes only and selected on the validation eyes; the
# held-out test eyes are scored once, after training.
set -euo pipefail
GPU="${1:?usage: bash training/run_protocol1.sh <gpu_id> [seed]}"
SEED="${2:-42}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
SPLITS="$REPO/splits/protocol1"
COMP="$REPO/outputs/protocol1/components/seed$SEED/ckpt"
OUT="$REPO/outputs/protocol1/seed$SEED"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
mkdir -p "$COMP" "$OUT"
cd "$REPO"

for component in aux mainenc; do
  if [[ -f "$COMP/fold_0_${component}.pth" ]]; then echo "$component: exists, skipping"; continue; fi
  "$PY" training/prepare_component.py --component "$component" --split-dir "$SPLITS" \
    --output-dir "$COMP" --seed "$SEED" --cuda 0
done

"$PY" src/H_dual_branch/train_fold_dual.py --fold 0 --cuda 0 --seed "$SEED" \
  --aux_epochs 50 --aux_lr 0.001 --stage1_epochs 50 --stage2_epochs 20 \
  --lr1 0.001 --lr2 0.00002 --batch_size 8 --max_length 128 --num_workers 2 \
  --use_spectral --class_weight subject --subject_balanced --phase0 c2_hi_lr \
  --splits_dir "$SPLITS" --split_unit eye --out_dir "$OUT" \
  --component_ckpt_dir "$COMP" --complete_checkpoint

"$PY" evaluation/evaluate_protocol1.py --result "$OUT/logs/fold_0_result.json" \
  --split "$SPLITS/fold_0.json" --output-dir "$REPO/outputs/protocol1/analysis_seed$SEED"
