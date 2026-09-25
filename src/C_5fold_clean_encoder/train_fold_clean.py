"""
C: Subject-Level 5-fold CV with a PER-FOLD ENCODER — leakage-free re-run of A_5fold_95subj.

Why this exists
---------------
A_5fold_95subj (and A_5fold_bs32) load ONE frozen encoder for all 5 folds:

    glaucoma_backbone_weights_new/epoch30_acc_0.72034_pr_0.80560_roc_0.80099.pth

That checkpoint was produced by backbone/finetune_glaucoma.py with GLAUCOMA LABELS
(BCEWithLogitsLoss, --frozen defaults to False so all 209 backbone tensors got gradients)
on the LMDB's builtin __keys__['train'] = 6114 segments / 88 subjects. Because a 5-fold CV
puts every sample in test exactly once, the union of the 5 test sets IS the whole 8861-segment
database, so those 6114 segments reappear as test data byte-for-byte (69.0% of the pooled test
set; 88/95 = 92.6% of test subjects were label-exposed). freeze_eeg_encoder=True does not undo
this — it guarantees the memorised weights are applied to test unchanged.
See notes/编码器微调数据泄露_证据文档.md for the full audit.

What this script changes
------------------------
  Phase 0 (NEW): fine-tune the encoder from the self-supervised foundation checkpoint using
                 ONLY fold_i['train']; select the checkpoint on fold_i['val']; never touch
                 fold_i['test'] or the LMDB builtin __keys__ split.
  Phase 1/2/3  : byte-identical recipe to A_5fold_95subj/train_fold_dump.py (seed 3407,
                 s1=50ep/lr1e-3, s2=20ep/lr2e-5, bs=8, max_length=128), except the Stage-2
                 class weight is now computed per fold instead of the hardcoded 3281/2833
                 (those two numbers are the class counts of the ENCODER's training split, i.e.
                 a whole-database statistic). Pass --class_weight_mode legacy to restore the
                 constant if you want the encoder to be the single changed variable.

So the only substantive difference from A is: the encoder no longer knows the test subjects.

Usage
  CUDA_DEVICE_ORDER=PCI_BUS_ID python train_fold_clean.py --fold 0 --cuda 0

Output
  ckpt/fold_{N}_encoder.pth   per-fold encoder (backbone.* + classifier.*, same format the
                              downstream EEGLlavaModel expects)
  ckpt/fold_{N}_best.pth      Stage-2 projector + LLM
  logs/fold_{N}_result.json   per-segment loss-based scores for subject-level voting
"""

import argparse
import copy
import json
import pickle
import random
from pathlib import Path
from timeit import default_timer as timer

import lmdb
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve, auc as sk_auc)
from functools import partial

import os, sys  # [release] os is used by the path overrides below
LLAMAG_ROOT = str(Path(__file__).resolve().parents[1] / 'llamaG')  # [release] was an absolute server path
sys.path.insert(0, LLAMAG_ROOT)
from model.glaucoma_llava import EEGLlavaModel
from backbone.model_for_glaucoma import Model as BackboneModel
from data.glaucoma_llava_dataset import STAGE2_TEMPLATE, STAGE1_TEMPLATE, STAGE1_ANSWER, STAGE2_ANSWER

# ── Constants ─────────────────────────────────────────────
DATA_DIR   = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[2] / 'data' / 'processed_lmdb'))  # [release]
LLM_PATH   = os.environ.get('EEGLLAVA_LLM', str(Path(__file__).resolve().parents[2] / 'pretrained' / 'Qwen3-0.6B'))  # [release]
# Self-supervised foundation checkpoint (CBraMod-style EEGEncoder init, label-free).
# This is the SAME starting point backbone/finetune_glaucoma.py uses (--foundation_dir default);
# what we drop is its label-supervised whole-database fine-tune.
FOUNDATION_WEIGHTS = os.environ.get('EEGLLAVA_CBRAMOD', str(Path(__file__).resolve().parents[2] / 'pretrained' / 'cbramod' / 'pretrained_weights.pth'))  # [release]
# Fold definitions are the ORIGINAL E07 splits, identical to A_5fold_95subj, so segment-level
# numbers stay comparable to Table 4. Subject-disjoint; both eyes of a subject in the same fold.
SPLITS_DIR = Path(__file__).resolve().parents[2] / 'splits' / 'e07_subject_cv'  # [release]
LOG_DIR    = Path(__file__).parent / 'logs'
SAVE_DIR   = Path(__file__).parent / 'ckpt'
LOG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ── Global lmdb cache: open once per process ──────────────
_LMDB_CACHE: dict = {}   # data_dir -> {key -> pair}

def load_lmdb_cache(data_dir, keys):
    global _LMDB_CACHE
    if data_dir not in _LMDB_CACHE:
        _LMDB_CACHE[data_dir] = {}
    cache = _LMDB_CACHE[data_dir]
    missing = [k for k in keys if k not in cache]
    if missing:
        db = lmdb.open(data_dir, readonly=True, lock=False,
                       readahead=False, meminit=False)
        with db.begin(write=False) as txn:
            for k in missing:
                cache[k] = pickle.loads(txn.get(k.encode()))
        db.close()
    return cache


# ── Phase 0: encoder datasets ─────────────────────────────
class RawEEGDataset(Dataset):
    """(sample/100, label) — same normalisation as backbone/glaucoma_dataset.py."""

    def __init__(self, data_dir, keys):
        self.cache = load_lmdb_cache(data_dir, keys)
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        pair = self.cache[self.keys[idx]]
        return pair['sample'] / 100.0, float(pair['label'])

    @staticmethod
    def collate(batch):
        x = torch.from_numpy(np.array([b[0] for b in batch])).float()
        y = torch.from_numpy(np.array([b[1] for b in batch])).float()
        return x, y


class _BackboneParams:
    """Shell object BackboneModel(param) expects."""

    def __init__(self, cuda, dropout=0.1, classifier='all_patch_reps'):
        self.use_pretrained_weights = True
        self.foundation_dir = FOUNDATION_WEIGHTS
        self.cuda = cuda
        self.classifier = classifier
        self.dropout = dropout


@torch.no_grad()
def encoder_scores(model, loader, device):
    model.eval()
    truths, scores = [], []
    for x, y in tqdm(loader, desc='EncEval', mininterval=10):
        pred = model(x.to(device))
        scores += torch.sigmoid(pred).float().cpu().numpy().tolist()
        truths += y.long().cpu().numpy().tolist()
    return np.asarray(truths), np.asarray(scores)


def encoder_metrics(truths, scores, thr=0.5):
    preds = (scores > thr).astype(int)
    pr, rc, _ = precision_recall_curve(truths, scores, pos_label=1)
    return {
        'n': int(len(truths)),
        'bacc': float(balanced_accuracy_score(truths, preds)),
        'roc_auc': float(roc_auc_score(truths, scores)),
        'pr_auc': float(sk_auc(rc, pr)),
    }


def train_encoder(fold_data, args, device):
    """Fine-tune the encoder on THIS fold's train subjects only; select on this fold's val.

    Mirrors backbone/finetune_glaucoma.py (30ep, bs=64, lr=1e-4, wd=5e-2, multi_lr, clip=1,
    cosine to 1e-6, pos_weight from train class counts, eval every 5 epochs, keep best val
    ROC-AUC) — the ONLY change is the data it sees.
    """
    tr_keys = fold_data['keys']['train']
    va_keys = fold_data['keys']['val']
    te_keys = fold_data['keys']['test']

    tr_ds = RawEEGDataset(DATA_DIR, tr_keys)
    va_ds = RawEEGDataset(DATA_DIR, va_keys)
    te_ds = RawEEGDataset(DATA_DIR, te_keys)
    tr_ld = DataLoader(tr_ds, batch_size=args.enc_bs, shuffle=True,
                       num_workers=0, collate_fn=RawEEGDataset.collate)
    va_ld = DataLoader(va_ds, batch_size=args.enc_bs, shuffle=False,
                       num_workers=0, collate_fn=RawEEGDataset.collate)
    te_ld = DataLoader(te_ds, batch_size=args.enc_bs, shuffle=False,
                       num_workers=0, collate_fn=RawEEGDataset.collate)

    model = BackboneModel(_BackboneParams(args.cuda)).to(device)

    tr_labels = np.array([int(tr_ds.cache[k]['label']) for k in tr_keys])
    n_pos = int(tr_labels.sum())
    n_neg = int(len(tr_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"[Phase 0] fold train class balance: healthy={n_neg}, glaucoma={n_pos}, "
          f"pos_weight={pos_weight.item():.3f}")
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        (backbone_params if 'backbone' in name else other_params).append(param)
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.enc_lr},
        {'params': other_params, 'lr': 0.001 * (args.enc_bs / 256) ** 0.5},
    ], weight_decay=args.enc_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.enc_epochs * len(tr_ld), eta_min=1e-6)

    best_auc, best_epoch, best_states = 0.0, 0, None
    val_curve = []
    for epoch in range(args.enc_epochs):
        model.train()
        losses = []
        start = timer()
        for x, y in tqdm(tr_ld, desc=f'Enc Epoch {epoch+1}', mininterval=10):
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.enc_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        print(f"Enc Epoch {epoch+1}: loss={np.mean(losses):.5f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}, time={(timer()-start)/60:.2f}min")

        if (epoch + 1) % 5 == 0 or epoch == args.enc_epochs - 1:
            m = encoder_metrics(*encoder_scores(model, va_ld, device))
            val_curve.append({'epoch': epoch + 1, **m})
            print(f"  Enc Val: bacc={m['bacc']:.5f}, pr_auc={m['pr_auc']:.5f}, "
                  f"roc_auc={m['roc_auc']:.5f}")
            if m['roc_auc'] > best_auc:
                best_auc, best_epoch = m['roc_auc'], epoch + 1
                best_states = copy.deepcopy(model.state_dict())
                print(f"  New best encoder (val roc_auc={best_auc:.5f} @ epoch {best_epoch})")

    if best_states is None:                       # degenerate val (single class) fallback
        best_states = copy.deepcopy(model.state_dict())
        best_epoch = args.enc_epochs
    model.load_state_dict(best_states)

    # Held-out encoder metrics are optional. New protocol-consistency runs skip them so the
    # final test labels remain untouched until the complete model and scoring rule are frozen.
    te_m = None
    if not getattr(args, 'skip_encoder_test', False):
        te_m = encoder_metrics(*encoder_scores(model, te_ld, device))
        print(f"[Phase 0] encoder held-out test (segment level): bacc={te_m['bacc']:.5f}, "
              f"roc_auc={te_m['roc_auc']:.5f}, pr_auc={te_m['pr_auc']:.5f}")
    else:
        print("[Phase 0] held-out test scoring skipped by protocol lock")

    enc_path = SAVE_DIR / f'fold_{args.fold}_encoder.pth'
    torch.save(best_states, enc_path)             # keeps 'backbone.'/'classifier.' prefixes
    print(f"[Phase 0] encoder saved: {enc_path}")

    del model, tr_ld, va_ld, te_ld, tr_ds, va_ds, te_ds
    import gc; gc.collect(); torch.cuda.empty_cache()

    return str(enc_path), {
        'best_epoch': best_epoch,
        'best_val_roc_auc': best_auc,
        'val_curve': val_curve,
        'test_segment': te_m,
        'train_class_counts': {'healthy': n_neg, 'glaucoma': n_pos},
        'pos_weight': float(pos_weight.item()),
    }


# ── Stage 1/2 dataset (identical to A_5fold_95subj) ───────
class FoldDataset(Dataset):
    def __init__(self, data_dir, keys, tokenizer, max_length=128, stage=2):
        self.cache = load_lmdb_cache(data_dir, keys)
        self.keys = keys
        self.tokenizer = tokenizer
        self.max_length = max_length

        if stage == 1:
            self.prompt = STAGE1_TEMPLATE
            self.answer_map = STAGE1_ANSWER
        else:
            self.prompt = STAGE2_TEMPLATE
            self.answer_map = STAGE2_ANSWER

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        pair = self.cache[key]
        eeg = torch.tensor(pair['sample'] / 100.0, dtype=torch.float32)
        label = int(pair['label'])
        answer = self.answer_map[label]
        full_text = self.prompt + "\n" + answer + self.tokenizer.eos_token
        return {
            'eeg': eeg,
            'text': full_text,
            'prompt': self.prompt + "\n",
            'label': label,
        }


def collate_fn(batch, tokenizer, max_length):
    eeg_batch = torch.stack([b['eeg'] for b in batch])
    full_texts = [b['text'] for b in batch]
    prompt_texts = [b['prompt'] for b in batch]

    full_tokens = tokenizer(full_texts, padding='max_length', truncation=True,
                            max_length=max_length, return_tensors='pt')
    prompt_tokens = tokenizer(prompt_texts, padding=False, truncation=True,
                              max_length=max_length)

    labels = full_tokens.input_ids.clone()
    for i, prompt_ids in enumerate(prompt_tokens.input_ids):
        prompt_len = len(prompt_ids)
        labels[i, :prompt_len] = -100
    labels[full_tokens.attention_mask == 0] = -100

    return {
        'eeg': eeg_batch,
        'input_ids': full_tokens.input_ids,
        'attention_mask': full_tokens.attention_mask,
        'labels': labels,
        'label': torch.tensor([b['label'] for b in batch]),
    }


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, loader, device, return_outputs=False):
    """Discrete generate-mode evaluation (for val early stopping)."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc='Eval', mininterval=10):
        eeg = batch['eeg'].to(device)
        labels_tensor = batch['labels']
        for i in range(eeg.shape[0]):
            valid_ids = labels_tensor[i][labels_tensor[i] != -100]
            gt_text = model.tokenizer.decode(valid_ids, skip_special_tokens=True).strip().lower()
            gt_label = 1 if 'glaucoma' in gt_text else 0
            pred_text = model.generate(eeg[i:i+1], STAGE2_TEMPLATE + '\n', max_new_tokens=32)
            pred_label = 1 if 'glaucoma' in pred_text.lower() else 0
            all_preds.append(pred_label)
            all_labels.append(gt_label)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)
    if return_outputs:
        return acc, auc, cm, all_preds, all_labels
    return acc, auc, cm


@torch.no_grad()
def evaluate_loss_based(model, loader, device):
    """Loss-based scoring: compute CE loss for both answers, softmax → AUC."""
    import torch.nn.functional as F
    model.eval()
    prompt = STAGE2_TEMPLATE + '\n'
    ans_h = STAGE2_ANSWER[0]
    ans_g = STAGE2_ANSWER[1]
    all_scores, all_labels = [], []
    for batch in tqdm(loader, desc='LossEval', mininterval=10):
        eeg = batch['eeg'].to(device)
        labels_tensor = batch['labels']
        for i in range(eeg.shape[0]):
            valid_ids = labels_tensor[i][labels_tensor[i] != -100]
            gt_text = model.tokenizer.decode(valid_ids, skip_special_tokens=True).strip().lower()
            gt = 1 if 'glaucoma' in gt_text else 0
            eeg_i = eeg[i:i+1]
            enc_p = model.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            enc_h = model.tokenizer(prompt + ans_h, return_tensors='pt', add_special_tokens=False)
            enc_g = model.tokenizer(prompt + ans_g, return_tensors='pt', add_special_tokens=False)
            plen = enc_p['input_ids'].shape[1]
            ids_h = enc_h['input_ids'].to(device)
            lab_h = ids_h.clone(); lab_h[:, :plen] = -100
            loss_h = model(eeg_i, ids_h, enc_h['attention_mask'].to(device), lab_h).loss.item()
            ids_g = enc_g['input_ids'].to(device)
            lab_g = ids_g.clone(); lab_g[:, :plen] = -100
            loss_g = model(eeg_i, ids_g, enc_g['attention_mask'].to(device), lab_g).loss.item()
            probs = F.softmax(torch.tensor([-loss_h, -loss_g]), dim=0)
            all_scores.append(probs[1].item())
            all_labels.append(gt)
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    preds = (scores > 0.5).astype(int)
    bacc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(labels, preds)
    return bacc, auc, cm, scores.tolist(), labels.tolist()


def train_stage(model, loader, optimizer, scheduler, device, epochs, healthy_weight,
                clip_value=1.0, run_eval_fn=None, eval_every=2):
    best_acc, best_epoch, best_states = 0, 0, None
    trainable = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(epochs):
        model.train()
        losses = []
        start = timer()
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}', mininterval=10):
            eeg = batch['eeg'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            raw_labels = batch['label']
            sample_weights = torch.where(raw_labels == 0,
                                         torch.tensor(healthy_weight),
                                         torch.tensor(1.0))
            optimizer.zero_grad()
            outputs = model(eeg, input_ids, attention_mask, labels, sample_weights)
            outputs.loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip_value)
            optimizer.step()
            scheduler.step()
            losses.append(outputs.loss.item())

        elapsed = (timer() - start) / 60
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}, time={elapsed:.1f}min")

        if run_eval_fn and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            acc, auc, cm = run_eval_fn()
            print(f"Val acc={acc:.5f}, auc={auc:.5f}\n{cm}")
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                best_states = copy.deepcopy(model.state_dict())
                print(f"New best! acc={acc:.5f} at epoch {best_epoch}")

    return best_acc, best_epoch, best_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',       type=int, default=0)
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--seed',       type=int, default=3407)
    # Stage 1/2 — identical defaults to A_5fold_95subj
    parser.add_argument('--stage1_epochs', type=int, default=50)
    parser.add_argument('--stage2_epochs', type=int, default=20)
    parser.add_argument('--lr1',        type=float, default=1e-3)
    parser.add_argument('--lr2',        type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--class_weight_mode', choices=['perfold', 'legacy'], default='perfold',
                        help="perfold: healthy weight = n_glaucoma/n_healthy of THIS fold's "
                             "train split. legacy: the 3281/2833 constant hardcoded in A "
                             "(class counts of the encoder's whole-database train split).")
    # Phase 0 — encoder, mirrors backbone/finetune_glaucoma.py defaults
    parser.add_argument('--enc_epochs', type=int, default=30)
    parser.add_argument('--enc_bs',     type=int, default=64)
    parser.add_argument('--enc_lr',     type=float, default=1e-4)
    parser.add_argument('--enc_wd',     type=float, default=5e-2)
    parser.add_argument('--enc_clip',   type=float, default=1.0)
    parser.add_argument('--reuse_encoder', action='store_true',
                        help='Skip Phase 0 if ckpt/fold_N_encoder.pth already exists (resume).')
    parser.add_argument('--encoder_path', type=str, default=None,
                        help='Skip Phase 0 entirely and freeze THIS encoder instead. Pass the '
                             'label-free CBraMod foundation checkpoint to drop encoder domain '
                             'adaptation altogether — the probe (probe_encoders.py) found the '
                             'unadapted foundation encoder scores at least as well as the '
                             'per-fold supervised one, and it removes the leakage pathway '
                             'entirely because no glaucoma data ever reaches the encoder.')
    parser.add_argument('--unfreeze_enc_last_n', type=int, default=0,
                        help='Unfreeze the top N encoder layers during Stage 2 (0 = fully '
                             'frozen, the default and what C_5fold used).')
    parser.add_argument('--splits_dir', type=str, default=None,
                        help='Fold definitions to use. Default = the original E07 splits. Pass '
                             'splits_v2/ to use the corrected train/val partition (test sets are '
                             'byte-identical either way, so results stay comparable).')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where logs/ and ckpt/ go. Default = this script\'s directory.')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)

    global LOG_DIR, SAVE_DIR
    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS_DIR
    if args.out_dir:
        LOG_DIR = Path(args.out_dir) / 'logs'; SAVE_DIR = Path(args.out_dir) / 'ckpt'
        LOG_DIR.mkdir(parents=True, exist_ok=True); SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"splits: {splits_dir}\noutputs: {LOG_DIR.parent}")

    split_file = splits_dir / f'fold_{args.fold}.json'
    assert split_file.exists(), f"Missing split file: {split_file}"
    with open(split_file) as f:
        fold_data = json.load(f)

    # Guard: the whole point of this re-run. Abort loudly if the splits ever stop being disjoint.
    sub = fold_data['subjects']
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not (set(sub[a]) & set(sub[b])), f"subject overlap between {a} and {b}!"
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not (set(fold_data['keys'][a]) & set(fold_data['keys'][b])), \
            f"segment-key overlap between {a} and {b}!"

    print(f"{'='*60}")
    print(f"C: clean-encoder subject-level CV | Fold {args.fold} | cuda:{args.cuda}")
    print(f"  Train: {fold_data['stats']['train']['subjects']} subjects, "
          f"{fold_data['stats']['train']['samples']} samples")
    print(f"  Val:   {fold_data['stats']['val']['subjects']} subjects, "
          f"{fold_data['stats']['val']['samples']} samples")
    print(f"  Test:  {fold_data['stats']['test']['subjects']} subjects, "
          f"{fold_data['stats']['test']['samples']} samples")
    print(f"{'='*60}")

    # ── Phase 0: per-fold encoder ─────────────────────────
    enc_path = SAVE_DIR / f'fold_{args.fold}_encoder.pth'
    if args.encoder_path:
        assert Path(args.encoder_path).exists(), f"missing {args.encoder_path}"
        print(f"\n[Phase 0] SKIPPED — freezing the given encoder instead:\n  {args.encoder_path}")
        eeg_weights = args.encoder_path
        enc_info = {'phase0': 'skipped', 'encoder_path': args.encoder_path}
    elif args.reuse_encoder and enc_path.exists():
        print(f"\n[Phase 0] reusing existing encoder: {enc_path}")
        eeg_weights, enc_info = str(enc_path), {'reused': True}
    else:
        print("\n[Phase 0] Fine-tuning the EEG encoder on THIS fold's train subjects only...")
        eeg_weights, enc_info = train_encoder(fold_data, args, device)

    # ── Stage 1 ───────────────────────────────────────────
    print("\n[Stage 1] Training projector...")
    model = EEGLlavaModel(
        llm_path=LLM_PATH, eeg_encoder_weights=eeg_weights,
        freeze_eeg_encoder=True, freeze_llm=True,
        eeg_dim=200, num_channels=6, num_patches=5,
    ).to(device)

    col_fn = partial(collate_fn, tokenizer=model.tokenizer, max_length=args.max_length)

    train_ds = FoldDataset(DATA_DIR, fold_data['keys']['train'], model.tokenizer,
                           args.max_length, stage=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=col_fn)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(trainable, lr=args.lr1, weight_decay=0.01)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=len(train_loader) * args.stage1_epochs, eta_min=1e-6)

    for epoch in range(args.stage1_epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'S1 Epoch {epoch+1}', mininterval=10):
            eeg = batch['eeg'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            opt1.zero_grad()
            outputs = model(eeg, input_ids, attention_mask, labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt1.step()
            sch1.step()
            losses.append(outputs.loss.item())
        print(f"S1 Epoch {epoch+1}: loss={np.mean(losses):.4f}")

    stage1_projector = copy.deepcopy(model.projector.state_dict())
    print("Stage 1 done.")

    del train_loader, train_ds, model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 2 ───────────────────────────────────────────
    print("\n[Stage 2] Finetuning projector + LLM...")
    model2 = EEGLlavaModel(
        llm_path=LLM_PATH, eeg_encoder_weights=eeg_weights,
        freeze_eeg_encoder=True, freeze_llm=False,
        eeg_dim=200, num_channels=6, num_patches=5,
    ).to(device)
    model2.projector.load_state_dict(stage1_projector)
    if args.unfreeze_enc_last_n > 0:
        # Gradients reach the top N encoder layers during Stage 2. Still fold-train data only,
        # so this does not create leakage; it just gives the encoder task-specific capacity that
        # a generic foundation checkpoint lacks.
        model2.unfreeze_encoder_last_n(args.unfreeze_enc_last_n)

    col_fn2 = partial(collate_fn, tokenizer=model2.tokenizer, max_length=args.max_length)
    train_ds2 = FoldDataset(DATA_DIR, fold_data['keys']['train'], model2.tokenizer,
                            args.max_length, stage=2)
    val_ds2   = FoldDataset(DATA_DIR, fold_data['keys']['val'],   model2.tokenizer,
                            args.max_length, stage=2)
    test_ds2  = FoldDataset(DATA_DIR, fold_data['keys']['test'],  model2.tokenizer,
                            args.max_length, stage=2)

    train_loader2 = DataLoader(train_ds2, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=col_fn2)
    val_loader2   = DataLoader(val_ds2,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=col_fn2)
    test_loader2  = DataLoader(test_ds2,  batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=col_fn2)

    if args.class_weight_mode == 'legacy':
        healthy_weight = 3281 / 2833
    else:
        tr_labels = np.array([int(train_ds2.cache[k]['label'])
                              for k in fold_data['keys']['train']])
        n_g = int(tr_labels.sum()); n_h = int(len(tr_labels) - n_g)
        healthy_weight = n_g / n_h
    print(f"[Stage 2] healthy sample weight = {healthy_weight:.5f} "
          f"(mode={args.class_weight_mode})")

    trainable2 = [p for p in model2.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(trainable2, lr=args.lr2, weight_decay=0.01)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=len(train_loader2) * args.stage2_epochs, eta_min=1e-6)

    best_acc, best_epoch, best_states = train_stage(
        model2, train_loader2, opt2, sch2, device,
        epochs=args.stage2_epochs,
        healthy_weight=healthy_weight,
        run_eval_fn=lambda: evaluate(model2, val_loader2, device),
        eval_every=2,
    )

    if best_states:
        model2.load_state_dict(best_states)

    # ── Test ──────────────────────────────────────────────
    print(f"\n{'='*60}\nTest Evaluation (Fold {args.fold})\n{'='*60}")

    test_acc, test_auc, test_cm = evaluate(model2, test_loader2, device)
    print(f"Test acc={test_acc:.5f}, auc(discrete)={test_auc:.5f}")
    print(test_cm)

    print("Running loss-based evaluation...")
    lb_acc, lb_auc, lb_cm, lb_scores, lb_labels = evaluate_loss_based(model2, test_loader2, device)
    print(f"Test acc(loss)={lb_acc:.5f}, auc(loss)={lb_auc:.5f}")
    print(lb_cm)

    # val 的 loss-based 分数:集成规则与阈值都要在 val 上选,没有它明天还得再跑一遍
    print("Scoring validation split (for val-based selection)...")
    v_bacc, v_auc, _, v_scores, v_labels = evaluate_loss_based(model2, val_loader2, device)
    print(f"Val   acc(loss)={v_bacc:.5f}, auc(loss)={v_auc:.5f}")

    ckpt_path = SAVE_DIR / f'fold_{args.fold}_best.pth'
    torch.save({
        'projector': model2.projector.state_dict(),
        'llm': model2.llm.state_dict(),
        'fold': args.fold,
        'best_val_epoch': best_epoch,
        'encoder_ckpt': str(eeg_weights),
    }, ckpt_path)
    print(f"Model saved: {ckpt_path}")

    result = {
        'fold': args.fold,
        'best_val_acc': best_acc,
        'best_val_epoch': best_epoch,
        'test_acc': test_acc,
        'test_auc_discrete': test_auc,
        'test_auc_loss_based': lb_auc,
        'test_bacc_loss_based': lb_acc,
        'test_cm': test_cm.tolist(),
        'test_cm_loss_based': lb_cm.tolist(),
        'args': vars(args),
        'encoder': {'ckpt': str(eeg_weights), 'healthy_weight_stage2': healthy_weight,
                    **enc_info},
        # ── per-segment dump for subject-level majority voting ──
        # lb_scores[i] corresponds to fold_data['keys']['test'][i]
        # (test_loader2 uses shuffle=False, so loader order == split key order).
        'test_keys': fold_data['keys']['test'],
           'val_keys': fold_data['keys']['val'], 'val_scores_lb': v_scores, 'val_labels_lb': v_labels,
        'test_scores_lb': lb_scores,
        'test_labels_lb': lb_labels,
    }
    assert len(lb_scores) == len(fold_data['keys']['test']), \
        f"score/key length mismatch: {len(lb_scores)} vs {len(fold_data['keys']['test'])}"
    result_file = LOG_DIR / f'fold_{args.fold}_result.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {result_file}")


if __name__ == '__main__':
    main()
