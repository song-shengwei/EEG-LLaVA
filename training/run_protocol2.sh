#!/usr/bin/env bash
# Protocol 2 (subject-disjoint five-fold cross-validation, Table 4 / Fig. 11 / Fig. 12).
#
# Usage:  bash training/run_protocol2.sh <gpu_id> [folds="0 1 2 3 4"]
#
# For every fold the trainer fits the auxiliary branch (Phase A) and adapts CBraMod (Phase 0) on
# that fold's training participants, then runs Stage 1 and Stage 2. The R1a wrapper selects the
# Stage 2 epoch on the fold's validation participants: every second epoch the subject-level mean
# score is evaluated; epochs with validation ROC-AUC >= 0.85 are preferred and ranked by BAcc, then
# ROC-AUC; if no epoch reaches 0.85 they are ranked by ROC-AUC, then BAcc; ties go to the earliest
# epoch. Test participants are scored once, with the selected weights.
# After all five folds, the subject-level soft-voting summary (mean segment score per
# participant, threshold 0.5, 10,000 participant bootstrap resamples) and the Table 4 / operating
# point package are written.
set -euo pipefail
GPU="${1:?usage: bash training/run_protocol2.sh <gpu_id> [folds]}"
FOLDS="${2:-0 1 2 3 4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
SPLITS="$REPO/splits/protocol2_v2"
OUT="$REPO/outputs/protocol2/R1a_seed1234"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$OUT"
cd "$REPO"

for k in $FOLDS; do
  if [[ -f "$OUT/logs/fold_${k}_result.json" ]]; then echo "fold $k: result exists, skipping"; continue; fi
  "$PY" training/train_fold_dual_r1a.py --selection-auc-floor 0.85 --selection-eval-every 2 \
    --fold "$k" --cuda 0 --seed 1234 \
    --aux_epochs 50 --aux_lr 0.001 --stage1_epochs 50 --stage2_epochs 20 \
    --lr1 0.001 --lr2 0.00002 --batch_size 8 --max_length 128 --num_workers 2 \
    --use_spectral --class_weight subject --subject_balanced --phase0 c2_hi_lr \
    --splits_dir "$SPLITS" --split_unit subject --out_dir "$OUT" --complete_checkpoint
done

if [[ -f "$OUT/logs/fold_0_result.json" && -f "$OUT/logs/fold_4_result.json" ]]; then
  "$PY" evaluation/r0_subject_evaluator.py --results-dir "$OUT/logs" --splits-dir "$SPLITS" \
    --output-dir "$OUT/final_5fold_subject_summary"
  "$PY" evaluation/build_r1a_paper_metrics.py --run-dir "$OUT" \
    --output-dir "$REPO/outputs/protocol2/paper_metrics"
fi
