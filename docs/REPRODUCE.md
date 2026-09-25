# Reproducing the paper

## Conventions

Run every command from the repository root, with the data and weights placed as described in the README. `$CKPT1` denotes the Protocol 1 seed-42 checkpoint (`checkpoints/protocol1_seed42/fold_0_best.pth`) or a checkpoint you trained yourself (`outputs/protocol1/seed42/ckpt/fold_0_best.pth`).

Two metric definitions recur below:
- **Loss-based score:** for each segment, the softmax over the negative language-model losses of the two answer templates ("healthy" / "glaucoma"). A score above 0.5 favours glaucoma.
- **Generated decision:** the model generates an answer, and the segment is called glaucoma if the text contains "glaucoma".

GPU runs assume one visible GPU (`CUDA_VISIBLE_DEVICES=<id>`, `--cuda 0`).

## 1. Data

| Step | Command | Output |
|---|---|---|
| Raw recordings → LMDB | `EEGLLAVA_RAW_DIR=data/raw EEGLLAVA_LABEL_SHEET=data/metadata/participants.csv EEGLLAVA_LMDB=data/processed_lmdb python preprocessing/preprocess_glaucoma_openbci.py` | 8,861 segments of 6 × 5 × 200 and the eye-level `__keys__` split. The script deletes and rewrites `EEGLLAVA_LMDB`. |
| Protocol 1 split | `python splits/build_protocol1_split.py --data-dir data/processed_lmdb --output-dir <dir>` | `fold_0.json`, identical to `splits/protocol1/fold_0.json` (6,114 / 1,343 / 1,404 segments) |
| Protocol 2 source split | `python splits/build_subject_splits_e07.py` | `splits/e07_subject_cv/fold_{0..4}.json` |
| Protocol 2 split | `python splits/build_splits_v2.py` | `splits/protocol2_v2/fold_{0..4}.json`. The test blocks are the same as in the source split; the validation participants are rotated within each class. |

## 2. Protocol 1: eye-disjoint split (Table 3, Figs. 9, 10, 13a)

`bash training/run_protocol1.sh <gpu> 42` runs four steps:
1. The auxiliary branch.
2. The CBraMod adaptation.
3. Stage 1 and Stage 2, writing `outputs/protocol1/seed42/logs/fold_0_result.json` and `ckpt/fold_0_best.pth`.
4. `evaluation/evaluate_protocol1.py`.

Step 4 writes `protocol1_metrics_and_stats.json` and `protocol1_predictions.csv`:

| Paper item | Field |
|---|---|
| Table 3, EEG-LLaVA (loss-based): BAcc 76.6 %, ROC-AUC 0.829, PR-AUC 0.804 | `m_loss_based` |
| Table 3, EEG-LLaVA (discrete): BAcc 76.6 % | `m_generated_answer` |
| Fig. 10(a), DeLong vs. RF: ΔAUC +0.042, p < 0.001 | `paired_tests.delong_m_loss_vs_rf` |
| Fig. 10(b), McNemar | `paired_tests.mcnemar_m_generated_vs_rf` |

**RF comparator of the paired tests** (depth 12; RF ROC-AUC 0.786):
- Inputs: `baselines/protocol1/rf_depth12/{delong_rf_scores.json, mcnemar_rf_predictions.json}`.
- Regenerate them with `python baselines/protocol1/rf_depth12/rf_depth12_paired.py`. On CPU this takes about 10 s and reproduces the shipped files.

**Fig. 13(a), three seeds:** run `training/run_protocol1.sh <gpu> <seed>` for seeds 42, 1234 and 2027. The test BAcc shown is `m_generated_answer.balanced_accuracy`: 76.6, 71.1 and 73.3 %.

## 3. Table 3: baselines and LoRA variant

| Row | Command | Result file |
|---|---|---|
| SVM (RBF), LDA | `python baselines/protocol1/baseline_p1.py` (CPU, about 2 min) | `baselines/protocol1/results/classical_p1.json` |
| EEGNet, Bi-LSTM, CNN-LSTM, Transformer, CNN-1D | `bash baselines/protocol1/run_all.sh <gpu_ids>` (also runs SVM/LDA) | `baselines/protocol1/results/dl/<model>.json` |
| Random Forest | `python baselines/protocol1/rf_full_comparison_p1.py` (CPU, 1 min) | `baselines/protocol1/results/rf_full_comparison_p1.json`; test BAcc 71.9 %, ROC-AUC 0.789 |
| EEG-LLaVA (LoRA) | `bash experiments/lora/run_lora.sh <gpu> 42` | `experiments/lora/runs/seed42/logs/fold_0_result.json` |

The LoRA run reuses the seed-42 Protocol 1 components from `outputs/protocol1/components/seed42/ckpt`, so run `training/run_protocol1.sh` first. It is split into short resumable processes (`--phase check|stage1|stage2|test`).

## 4. Protocol 2: subject-disjoint five-fold CV (Table 4, Figs. 11, 12)

`bash training/run_protocol2.sh <gpu>` trains folds 0–4 into `outputs/protocol2/R1a_seed1234/`. It then runs two scripts.

**`evaluation/r0_subject_evaluator.py`** writes `final_5fold_subject_summary/`:
- Participant-level soft voting: the mean loss-based score of all segments of a participant, thresholded at 0.5.
- Results: BAcc 80.1 % [71.2, 88.2], ROC-AUC 0.849 [0.761, 0.924], from 10,000 participant bootstrap resamples with seed 20260725.

**`evaluation/build_r1a_paper_metrics.py`** writes `outputs/protocol2/paper_metrics/`:

| Paper item | Output |
|---|---|
| Table 4 | `table4_r1a.csv`; fold mean ± SD in `fold_mean_sample_std` |
| Fig. 11(a), per-fold segment-level values | `per_fold` |
| Fig. 11(c), segment level 66.1 % / 0.707 vs. participant level 80.1 % / 0.849 | `pooled_segment`, `headline` |
| Aggregation rules (mean probability, mean logit, hard voting 76.5 %) | `aggregation_robustness` |
| Fig. 12 operating points: default, Youden, high-sensitivity 0.91 / 0.54; PPV and NPV vs. prevalence | `operating_points` |

**Fig. 11(b) baselines:**
- Run `bash baselines/protocol2/run_all.sh <gpu>`: RF on CPU, EEGNet and Transformer on the GPU, on `splits/protocol2_v2`.
- Each `baselines/protocol2/results/<method>_voting.json` stores the per-fold segment-, eye- and participant-level values.
- The baseline bars in Fig. 11(b) are the five-fold means of the per-fold segment-level BAcc / ROC-AUC (`per_fold[*].segment`): RF 67.3 % / 71.0, Transformer 68.0 % / 73.7, EEGNet 60.3 % / 63.8.
- `summarize_baselines_v2.py` additionally reports the participant-level (soft-voted) values of the baselines.
- Passing `--baseline-dir baselines/protocol2/results` to `build_r1a_paper_metrics.py` adds the same participant-level rows to its package.

## 5. Ablations and analyses

| Paper item | Command | Reported value |
|---|---|---|
| Fig. 13(b), learning curve | see below | generated-decision accuracy |
| Fig. 14(a), no EEG–language alignment | Protocol 1 trainer command of `run_protocol1.sh` with `--stage1_epochs 0 --out_dir outputs/ablations/no_alignment_seed42` (seed 42, same components) | 74.7 % |
| Fig. 14(b), unfreeze last two encoder layers | same with `--unfreeze_enc_last_n 2 --out_dir outputs/ablations/unfreeze2_seed42` | 76.3 % |
| Fig. 15(a)(b), row and single-channel masking (loss-based BAcc) | `python experiments/evaluate_channel_masking.py --cuda 0 --checkpoint $CKPT1 --split splits/protocol1/fold_0.json --output outputs/fig15_channel_masking.json` | row: 48.3 / 60.0 %; channels: `leave_one_out` |
| Fig. 15(c), Top-K subsets (generated decisions) | `EEGLLAVA_P1_CKPT=$CKPT1 bash experiments/topk_generated/run_topk_generated.sh <gpu>` | Top-4 {PO4, O1, PO3, Oz}: BAcc 66.8 % |
| Fig. 16, held-out probe on the 55 mean-pooled EEG tokens | `python experiments/probe_heldout.py --cuda 0 --checkpoint $CKPT1 --split splits/protocol1/fold_0.json --output outputs/probe_heldout_seed42.json` | ROC-AUC 0.797, BAcc 72.7 % |
| Sec. 5.8, head-only control (Protocol 2) | `bash experiments/head_only_p2/run_head_only_p2.sh <gpu>`, then `python experiments/head_only_p2/paired_llm_vs_head.py` | 73.1 % / 0.809 and 71.2 % / 0.811; LLM readout +7.0 pp [+0.3, +14.0] |
| Fig. 20(a), Rich Report variant | `python experiments/train_rich_final.py --cuda 0 --seed 42 --split splits/protocol1/fold_0.json --component-dir outputs/protocol1/components/seed42/ckpt --output-dir outputs/rich_report_seed42 --stage1-epochs 50 --stage2-epochs 20 --batch-size 8 --max-length 192` | 75.6 % / 0.827 |
| Figs. 7–8, band-wise power differences, Cohen's d, Welch PSD | `python analysis/regen_spectral_descriptives.py --data-dir data/processed_lmdb --output-dir <empty dir>` | `spectral_descriptive_audit.json` and the figure panels |

**Fig. 14 metric.** The Fig. 14 values are the generated-decision accuracy (`test_acc`) of `logs/fold_0_result.json`. The generated-decision balanced accuracies of the same runs are identical to one decimal.

**Fig. 13(b), learning curve:**
- Only the Stage 1–2 training segments are subsampled. The auxiliary branch and the CBraMod adaptation are the fixed components of the same seed, trained on the full training split.
- Build the subsamples:
  ```bash
  python experiments/build_learning_splits.py --source-split splits/protocol1/fold_0.json --output-root outputs/learning_splits            # seeds 42, 1234 (and 3407)
  python experiments/build_learning_splits_seeds.py --source-split splits/protocol1/fold_0.json --output-root outputs/learning_splits --seeds 2027
  ```
- For each seed S and fraction F in {10, 25, 50, 75}, run the Protocol 1 trainer command of `run_protocol1.sh` with `--seed S --splits_dir outputs/learning_splits/seedS/fraction_F --out_dir outputs/learning_curve/fraction_F/seedS --component_ckpt_dir outputs/protocol1/components/seedS/ckpt`.
- The 100 % point is the full Protocol 1 run of each seed.
- The curve shows the mean ± SD over seeds 42, 1234 and 2027 of `accuracy_score(test_labels_discrete, test_preds_discrete)`: 68.6 ± 0.9 % at 10 %, 73.7 ± 2.8 % at 100 %.

## 6. System performance (Fig. 17)

**(a) RTX 4090, Protocol 2 fold-0 checkpoint.**

Command:
```bash
python deployment/profile_r1a.py --cuda 0 \
  --checkpoint $EEGLLAVA_CKPT_DIR/protocol2_R1a_seed1234/fold_0_best.pth \
  --output outputs/profile/r1a_profile_results.json
```
- The checkpoint needs its `.sha256` and `.manifest.json` sidecars.
- Results: single template forward pass 47.1 ms, loss-based score (two passes) 94.4 ms, peak memory 1,305 MiB.

**(b) Laptop (RTX 4070 Laptop GPU / CPU).**

Run the scripts in `deployment/edge/` in this order:
1. `check_env.py`
2. `profile_m1_m3.py --checkpoint {p1_seed42,p2_r1a_fold0} --device {cpu,cuda} --precision {fp32,bf16,int8}`
3. `participant_m4.py`
4. `agreement_m5.py`
5. `summarize.py`

The RTX 4090 reference scores they compare against are in `deployment/edge/reference/`. Each script documents its options with `--help`.

## Not included

- Plotting scripts for the figure layouts. The numbers behind every panel come from the outputs listed above.
- The demographic summary (Table 2). It is computed from `participants.csv` of the dataset release.
- The Section 3 group statistics of SSVEP-band power (pooled reduction and participant-level Mann–Whitney test).
