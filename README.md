# EEG-LLaVA

Code for **"EEG-LLaVA: A Low-Cost Wearable SSVEP-EEG System with LLM-Driven Interpretable Glaucoma Screening"**.

EEG-LLaVA screens for glaucoma from six-channel occipital SSVEP-EEG recorded with a dry-electrode headband. Each 5-second segment (6 channels × 5 one-second windows × 200 samples) is encoded into 55 EEG tokens by three parallel paths:
- an adapted CBraMod encoder (30 tokens),
- a compact task-trained Transformer (21 tokens),
- a time/band descriptor path (4 tokens).

The tokens are mapped into the embedding space of Qwen3-0.6B, which produces the screening decision as text.

The model is evaluated under two protocols:
- **Protocol 1** uses an eye-disjoint split (Table 3).
- **Protocol 2** uses subject-disjoint five-fold cross-validation with participant-level soft voting (Table 4, Figs. 11–12).

> The repository is anonymised for peer review.

## Repository layout

| Path | Contents |
|---|---|
| `src/llamaG/` | Base EEG-LLaVA model: CBraMod encoder (`model/`), projector and Qwen3 wrapper (`model/glaucoma_llava.py`), encoder-adaptation head (`backbone/`), prompt/answer templates (`data/`) |
| `src/H_dual_branch/` | 55-token model (`dual_branch.py`) and the per-fold trainer (`train_fold_dual.py`) |
| `src/C_5fold_clean_encoder/` | Shared training utilities imported by the trainer: data cache, Phase-0 encoder adaptation, loss-based and generated-answer evaluation |
| `training/` | Protocol 1 component training, the Protocol 2 selection wrapper, and one-command scripts for both protocols |
| `evaluation/` | Protocol 1 metrics and paired tests, Protocol 2 participant-level evaluation, Table 4 / operating-point package, strict checkpoint re-scoring |
| `splits/` | Split builders and the exact split files used in the paper |
| `preprocessing/` | Raw OpenBCI text files → LMDB of 5-second segments |
| `baselines/` | Protocol 1 baselines (Table 3, Fig. 9), the Random Forest comparator of the paired tests (Fig. 10), Protocol 2 baselines (Fig. 11b) |
| `experiments/` | Ablations and analyses: channel masking (Fig. 15), held-out probe (Fig. 16), head-only control (Sec. 5.8), LoRA variant (Table 3), Rich Report variant (Fig. 20), learning-curve subsamples (Fig. 13b) |
| `analysis/` | Spectral descriptive statistics (Figs. 7–8) |
| `deployment/` | RTX 4090 latency/memory profile (Fig. 17a) and the laptop evaluation (Fig. 17b) |
| `docs/REPRODUCE.md` | Paper result → script → command |

## Installation

Tested with Python 3.12.13, PyTorch 2.6.0 (CUDA 12.4) and transformers 5.5.0 on NVIDIA RTX 4090 and A40 GPUs. Training peaks at about 14 GB of GPU memory. Inference needs about 1.3 GB.

```bash
conda env create -f environment.yml
conda activate eegllava
# or: pip install -r requirements.txt
```

## Data and pretrained weights

Nothing below is stored in this repository.

| What | Where | Default location (override with) |
|---|---|---|
| Preprocessed dataset, LMDB of 8,861 segments from 95 participants | Zenodo record of the paper (see the paper's data availability statement) | `data/processed_lmdb/` (`EEGLLAVA_LMDB`) |
| Raw OpenBCI recordings and `participants.csv`; needed only to rerun preprocessing | same Zenodo record | `data/raw/`, `data/metadata/participants.csv` (`EEGLLAVA_RAW_DIR`, `EEGLLAVA_LABEL_SHEET`) |
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `pretrained/Qwen3-0.6B/` (`EEGLLAVA_LLM`) |
| CBraMod pretrained weights | [CBraMod](https://github.com/wjq-learning/CBraMod), file [`pretrained_weights.pth`](https://huggingface.co/weighting666/CBraMod/blob/main/pretrained_weights.pth) | `pretrained/cbramod/pretrained_weights.pth` (`EEGLLAVA_CBRAMOD`) |
| Trained checkpoints (Protocol 1 seed 42; Protocol 2 folds 0–4), each with its `.sha256` and `.manifest.json` sidecar and its training result JSON | same Zenodo record | `checkpoints/protocol1_seed42/`, `checkpoints/protocol2_R1a_seed1234/` (`EEGLLAVA_P1_CKPT`, `EEGLLAVA_CKPT_DIR`) |

The files used for the paper have these SHA-256 digests:
- Qwen3-0.6B `model.safetensors`: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`
- CBraMod `pretrained_weights.pth`: `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`

The trainers and loaders work fully offline once these files are in place. Set `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` to make sure nothing is downloaded.

Run outputs go to `outputs/`. Several analysis scripts find the Protocol 2 run through `EEGLLAVA_P2_RUN` (default `outputs/protocol2/R1a_seed1234`). They find its metrics package through `EEGLLAVA_P2_METRICS` (default `outputs/protocol2/paper_metrics/r1a_paper_metrics.json`).

## Quick start

Re-score the held-out test segments with a released checkpoint. The check loads the checkpoint strictly, scores every Protocol 1 test segment with the loss-based readout, and compares the scores with those stored in the training result file:

```bash
python evaluation/r0_verify_checkpoint.py \
  --checkpoint checkpoints/protocol1_seed42/fold_0_best.pth \
  --result-json checkpoints/protocol1_seed42/fold_0_result.json \
  --cuda 0 --output-json outputs/verify_protocol1_seed42.json
# expected: "status": "pass", "segment_auc": 0.8287..., "max_abs_score_error": 0.0 on an RTX 4090
```

Then compute the Protocol 1 metrics and paired tests:

```bash
python evaluation/evaluate_protocol1.py \
  --result checkpoints/protocol1_seed42/fold_0_result.json \
  --split splits/protocol1/fold_0.json --output-dir outputs/protocol1/analysis
# BAcc 76.6 %, ROC-AUC 0.829; DeLong vs. RF ΔAUC +0.042
```

## Training from scratch

```bash
bash training/run_protocol1.sh <gpu_id> 42     # Protocol 1, one seed
bash training/run_protocol2.sh <gpu_id>        # Protocol 2, five folds
```

On an RTX 4090, Stage 1 + Stage 2 took about 4 h for Protocol 1 and about 3 h per Protocol 2 fold. The auxiliary branch and the CBraMod adaptation come on top of that. The exact commands behind every table and figure are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md). GPU training is not bit-for-bit deterministic across hardware and driver versions. Retrained models therefore reproduce the reported numbers only approximately. Re-scoring the released checkpoints reproduces the reported scores; this was checked on an RTX 4090.

## How this code relates to the runs reported in the paper

- **Configurable paths.** File locations of the original compute environment were replaced by the defaults and environment variables above. Helper code that only managed the internal GPU job queue was removed: a queue kill switch in `training/prepare_component.py` and a bookkeeping subprocess in `experiments/evaluate_channel_masking.py`. Model, training and evaluation code are otherwise unchanged.
- **Base weights at load time.** The final checkpoints record the location of Qwen3-0.6B and of the CBraMod initialisation from the original environment. The loaders now take these from `EEGLLAVA_LLM` and `EEGLLAVA_CBRAMOD`. Every parameter is then overwritten by a strict load of the checkpoint.
- **LLM autocast.** `src/llamaG/model/glaucoma_llava.py` makes the autocast context around the LLM configurable, which the laptop evaluation needs. The default is identical to the original `autocast("cuda", bfloat16)`.
- **Trainer edits after the runs.**
  - `src/H_dual_branch/train_fold_dual.py` gained the optional arguments `--select_metric` and `--stage2_eval_every` after the final runs. Their defaults (`acc`, `2`) reproduce the earlier behaviour.
  - `training/train_fold_dual_r1a.py` accepts and ignores those keywords.
  - The byte-exact versions used for the runs were not preserved.
- **Protocol 2 components.** The paper's Protocol 2 run reused Phase A/Phase 0 components that the same trainer had produced earlier with the same seed and folds. `training/run_protocol2.sh` trains them inside the same process.
- **Preprocessing.** `preprocessing/preprocess_glaucoma_openbci.py` reads the label sheet as `.csv` or `.xlsx` and accepts `Yes`/`yes`. Run on the released raw recordings and `participants.csv`, it rebuilds the released LMDB byte for byte.

## Verification of this release

The following were re-run from this repository on the released data and compared with the results reported in the paper:

| Check | Result |
|---|---|
| Preprocessing, raw recordings → LMDB | all 8,862 entries byte-identical |
| Split builders (Protocol 1, Protocol 2 with its source split, learning-curve subsamples) | all split files byte-identical |
| Protocol 1 checkpoint re-scoring (RTX 4090) | max absolute score difference 0; ROC-AUC 0.8287 |
| Protocol 1 metrics, DeLong, McNemar and bootstrap tests | identical |
| Protocol 2 participant-level evaluation (BAcc 80.1 %, ROC-AUC 0.849, bootstrap CIs) | identical |
| Table 4, fold mean ± SD, aggregation rules, operating points | identical |
| Table 3 SVM / LDA / RF rows; RF comparator of the paired tests | identical |
| Protocol 2 RF baseline | identical, up to 4e-16 in scores |
| Head-only control; Top-K summary from generated decisions; spectral statistics | identical |
| All GPU scripts | import and argument parsing checked; not retrained |

## License

Code: MIT (see `LICENSE`). The dataset and checkpoints are distributed under the licenses stated in their Zenodo record. Qwen3-0.6B is licensed under Apache-2.0 by its authors. For CBraMod, see its repository.

## Citation

Citation information will be added after review.
