"""
E02: 完整对比分析脚本
三个维度的对比：

1. 所有指标对比（Acc / PR-AUC / ROC-AUC / Sensitivity / Specificity）
   - RF 和 EEG-LLaVA 的完整指标
2. Val 集上的对比（体现 EEG-LLaVA 的泛化优势）
   - RF val acc vs EEG-LLaVA val acc (80%+)
3. EEG-LLaVA 的 loss-based 连续分数 → 真实的 PR-AUC / ROC-AUC
   - 参考 test.py 的 evaluate_by_loss 逻辑（移植到 glaucoma 版本）

用法：
  CUDA_DEVICE_ORDER=PCI_BUS_ID python full_comparison.py --cuda 7
"""

import argparse
import pickle
import numpy as np
import lmdb
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)

import os, sys  # [release]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'llamaG'))  # [release]
from model.glaucoma_llava import EEGLlavaModel
from data.glaucoma_llava_dataset import get_data_loaders, STAGE2_TEMPLATE, STAGE2_ANSWER

DATA_DIR  = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[2] / 'data' / 'processed_lmdb'))  # [release]
CKPT      = None  # [release] Part 2 (older EEG-LLaVA checkpoint) is not part of the release
LLM_PATH  = os.environ.get('EEGLLAVA_LLM', str(Path(__file__).resolve().parents[2] / 'pretrained' / 'Qwen3-0.6B'))
EEG_W     = None  # [release] see CKPT
OUT_DIR   = Path(__file__).parent


# ── 特征提取（与 baseline.py 一致）────────────────────────
def extract_features(eeg_np):
    feats = []
    signal = eeg_np.reshape(6, -1)  # (6, 1000)
    for ch in range(6):
        s = signal[ch]
        feats += [np.mean(s), np.std(s), np.max(s)-np.min(s),
                  np.mean(np.abs(s)), np.sqrt(np.mean(s**2)), np.sum(s**2)]
        fft   = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), d=1.0/200)
        for lo, hi in [(0.5,4),(4,8),(8,13),(13,30),(30,50),(6,20)]:
            mask = (freqs>=lo)&(freqs<hi)
            feats.append(fft[mask].mean() if mask.any() else 0)
        psd = fft**2; psd_n = psd/(psd.sum()+1e-8)
        feats.append(-np.sum(psd_n*np.log(psd_n+1e-8)))
    return np.array(feats, dtype=np.float32)


def load_split_features(data_dir, split):
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys = pickle.loads(txn.get('__keys__'.encode()))[split]
    X, y = [], []
    with db.begin(write=False) as txn:
        for key in keys:
            pair = pickle.loads(txn.get(key.encode()))
            X.append(extract_features(pair['sample']/100.0))
            y.append(int(pair['label']))
    db.close()
    return np.array(X), np.array(y)


def compute_metrics(truths, preds, scores=None, name=''):
    tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
    acc  = accuracy_score(truths, preds)
    bacc = balanced_accuracy_score(truths, preds)
    sens = tp / (tp+fn+1e-8)
    spec = tn / (tn+fp+1e-8)
    if scores is not None:
        roc = roc_auc_score(truths, scores)
        prec, rec, _ = precision_recall_curve(truths, scores, pos_label=1)
        pr  = auc(rec, prec)
    else:
        roc, pr = float('nan'), float('nan')
    return dict(name=name, acc=acc, bacc=bacc, sens=sens, spec=spec,
                roc_auc=roc, pr_auc=pr,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def print_metrics(m):
    print(f"\n{'='*55}")
    print(f"[{m['name']}]")
    print(f"  Acc (balanced):  {m['bacc']:.4f}  (raw: {m['acc']:.4f})")
    print(f"  ROC-AUC:         {m['roc_auc']:.4f}")
    print(f"  PR-AUC:          {m['pr_auc']:.4f}")
    print(f"  Sensitivity:     {m['sens']:.4f}  (glaucoma recall)")
    print(f"  Specificity:     {m['spec']:.4f}  (healthy recall)")
    print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")


# ── Part 1: RF on train/val/test ──────────────────────────
def run_rf_all_splits(data_dir):
    print("\n" + "="*55)
    print("Part 1: Random Forest — Train/Val/Test 全集对比")
    print("="*55)

    X_tr, y_tr = load_split_features(data_dir, 'train')
    X_va, y_va = load_split_features(data_dir, 'val')
    X_te, y_te = load_split_features(data_dir, 'test')

    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=3407))
    ])
    rf.fit(X_tr, y_tr)

    results = {}
    for split, X, y in [('Train', X_tr, y_tr), ('Val', X_va, y_va), ('Test', X_te, y_te)]:
        preds  = rf.predict(X)
        scores = rf.predict_proba(X)[:, 1]
        m = compute_metrics(y, preds, scores, f'RF — {split}')
        print_metrics(m)
        results[split] = m

    return results


# ── Part 2: EEG-LLaVA loss-based eval（连续分数）─────────
@torch.no_grad()
def evaluate_llava_loss_based(model, data_loader, device, split_name):
    """
    用 loss 作为连续置信度分数，计算真实的 PR-AUC / ROC-AUC
    参考 test.py 的 evaluate_by_loss，适配 glaucoma 版本
    """
    model.eval()
    tokenizer = model.tokenizer

    answer_healthy  = STAGE2_ANSWER[0]   # "The EEG signal indicates a healthy visual system."
    answer_glaucoma = STAGE2_ANSWER[1]   # "The EEG signal indicates glaucoma."

    all_truths, all_scores, all_preds = [], [], []

    for batch in tqdm(data_loader, desc=f"LLaVA loss-eval [{split_name}]", mininterval=10):
        eeg = batch['eeg'].to(device)
        labels_tensor = batch['labels']

        for i in range(eeg.shape[0]):
            # Ground truth
            valid_ids = labels_tensor[i][labels_tensor[i] != -100]
            gt_text   = tokenizer.decode(valid_ids, skip_special_tokens=True).strip().lower()
            gt_label  = 1 if 'glaucoma' in gt_text else 0
            all_truths.append(gt_label)

            eeg_single = eeg[i:i+1]
            losses = []
            for answer in [answer_healthy, answer_glaucoma]:
                full_text  = STAGE2_TEMPLATE + "\n" + answer + tokenizer.eos_token
                prompt_text = STAGE2_TEMPLATE + "\n"
                full_tok   = tokenizer(full_text, return_tensors='pt',
                                       padding=False, truncation=True,
                                       max_length=128).to(device)
                prompt_tok = tokenizer(prompt_text, return_tensors='pt',
                                       padding=False, truncation=True,
                                       max_length=128)
                lbl = full_tok.input_ids.clone()
                lbl[:, :prompt_tok.input_ids.shape[1]] = -100
                out = model(eeg_single, full_tok.input_ids, full_tok.attention_mask, lbl)
                losses.append(out.loss.item())

            # softmax(-loss) → P(glaucoma)
            logits = torch.tensor([-losses[0], -losses[1]])
            probs  = torch.softmax(logits, dim=0)
            score  = probs[1].item()
            all_scores.append(score)
            all_preds.append(1 if score > 0.5 else 0)

    truths = np.array(all_truths)
    preds  = np.array(all_preds)
    scores = np.array(all_scores)
    return compute_metrics(truths, preds, scores, f'EEG-LLaVA (loss-based) — {split_name}')


def run_llava_all_splits(cuda):
    print("\n" + "="*55)
    print("Part 2: EEG-LLaVA — Loss-based 连续分数评估")
    print("(Acc / PR-AUC / ROC-AUC / Sensitivity / Specificity)")
    print("="*55)

    device = torch.device(f'cuda:{cuda}')
    torch.cuda.set_device(cuda)

    model = EEGLlavaModel(
        llm_path=LLM_PATH, eeg_encoder_weights=EEG_W,
        freeze_eeg_encoder=True, freeze_llm=False,
        eeg_dim=200, num_channels=6, num_patches=5,
    )
    ckpt = torch.load(CKPT, map_location='cpu')
    model.projector.load_state_dict(ckpt['projector'])
    if 'llm' in ckpt:
        model.llm.load_state_dict(ckpt['llm'])
    model = model.to(device)
    model.eval()
    print(f"Loaded ckpt: epoch={ckpt.get('epoch')} acc={ckpt.get('acc',0):.4f}")

    loaders = get_data_loaders(
        data_dir=DATA_DIR, tokenizer=model.tokenizer,
        batch_size=8, num_workers=4, max_length=128, stage=2,
    )

    results = {}
    for split in ['val', 'test']:
        m = evaluate_llava_loss_based(model, loaders[split], device, split)
        print_metrics(m)
        results[split] = m

    return results


# ── Part 3: 综合对比表 ────────────────────────────────────
def print_comparison_table(rf_results, llava_results):
    print("\n" + "="*75)
    print("Part 3: 综合对比表（Paper Table 1 素材）")
    print("="*75)
    print(f"{'Method':<35} {'Split':<6} {'Acc':>7} {'ROC-AUC':>9} {'PR-AUC':>8} {'Sens':>7} {'Spec':>7}")
    print("-"*75)

    for split in ['Val', 'Test']:
        rf = rf_results.get(split, {})
        if rf:
            print(f"{'Random Forest + hand-crafted':<35} {split:<6} "
                  f"{rf['acc']:>7.4f} {rf['roc_auc']:>9.4f} {rf['pr_auc']:>8.4f} "
                  f"{rf['sens']:>7.4f} {rf['spec']:>7.4f}")

    for split in ['val', 'test']:
        ll = llava_results.get(split, {})
        if ll:
            print(f"{'EEG-LLaVA (Ours, loss-based)':<35} {split.capitalize():<6} "
                  f"{ll['acc']:>7.4f} {ll['roc_auc']:>9.4f} {ll['pr_auc']:>8.4f} "
                  f"{ll['sens']:>7.4f} {ll['spec']:>7.4f}")

    print("="*75)
    print("\n关键对比点：")
    if 'Val' in rf_results and 'val' in llava_results:
        rf_val  = rf_results['Val']['acc']
        ll_val  = llava_results['val']['acc']
        print(f"  Val Acc gap:  RF={rf_val:.4f} vs LLaVA={ll_val:.4f}  Δ={ll_val-rf_val:+.4f}")
    if 'Test' in rf_results and 'test' in llava_results:
        rf_te  = rf_results['Test']['roc_auc']
        ll_te  = llava_results['test']['roc_auc']
        print(f"  Test ROC-AUC: RF={rf_te:.4f} vs LLaVA={ll_te:.4f}  Δ={ll_te-rf_te:+.4f}")
        rf_pr  = rf_results['Test']['pr_auc']
        ll_pr  = llava_results['test']['pr_auc']
        print(f"  Test PR-AUC:  RF={rf_pr:.4f} vs LLaVA={ll_pr:.4f}  Δ={ll_pr-rf_pr:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=7)
    args = parser.parse_args()

    # Part 1
    rf_results = run_rf_all_splits(DATA_DIR)

    # Part 2
    llava_results = run_llava_all_splits(args.cuda)

    # Part 3
    print_comparison_table(rf_results, llava_results)


if __name__ == '__main__':
    main()
