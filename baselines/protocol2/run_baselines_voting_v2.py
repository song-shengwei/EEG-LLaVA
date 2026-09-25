#!/usr/bin/env python3
"""
Subject-level baselines — the number nobody has computed yet.

E12_subject_cv_baselines reported RF / EEGNet / Transformer only at the SEGMENT level
(RF 0.7235, Transformer 0.7233, EEGNet 0.6418). EEG-LLaVA's case rests on aggregating segment
scores to the participant, so the honest comparison has to aggregate the baselines the same way.
Without it we cannot say whether EEG-LLaVA leads, ties, or trails at the unit a screening
decision is actually made.

Models, hyperparameters and feature extraction are copied verbatim from
the earlier E12 cross-validation baseline script (not released) so the segment-level
numbers reproduce; the only addition is that per-segment probabilities are dumped and then
aggregated (soft vote = mean probability) to eye and participant level, with a
participant-clustered bootstrap CI — exactly the protocol aggregate.py applies to EEG-LLaVA.

The 5 fold test sets partition the database, so pooling gives one out-of-fold prediction for
every one of the 95 participants.

Usage:
  python run_baselines_voting.py --method rf                 # CPU only
  CUDA_DEVICE_ORDER=PCI_BUS_ID python run_baselines_voting.py --method eegnet --cuda 4
  CUDA_DEVICE_ORDER=PCI_BUS_ID python run_baselines_voting.py --method transformer --cuda 4
"""
import argparse, hashlib, json, os, pickle, random, sys  # [release] os added
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

import lmdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

DATA_DIR   = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[2] / 'data' / 'processed_lmdb'))  # [release]
# v2: the ONLY functional change from the first-split baselines/run_baselines_voting.py
# is the fold definition. The original read the first five-fold split, in which folds 1-4 share
# one validation group; EEG-LLaVA (R1a) was trained on splits_v2. Test blocks are identical.
REPO_ROOT  = Path(__file__).resolve().parents[2]  # [release]
SPLITS_DIR = REPO_ROOT / 'splits' / 'protocol2_v2'
OLD_SPLITS = REPO_ROOT / 'splits' / 'e07_subject_cv'
OUT_DIR    = Path(__file__).resolve().parent / 'results'
N_FOLDS, SEED = 5, 3407
random.seed(20260803); np.random.seed(20260803)


def load_samples(keys):
    db = lmdb.open(DATA_DIR, readonly=True, lock=False, readahead=False, meminit=False)
    out = []
    with db.begin(write=False) as txn:
        for k in keys:
            p = pickle.loads(txn.get(k.encode()))
            out.append((p['sample'].astype(np.float32) / 100.0, int(p['label'])))
    db.close()
    return out


def extract_features(eeg_np):
    """Verbatim from E12 run_cv_baselines.py."""
    feats = []
    signal = eeg_np.reshape(6, -1)
    for ch in range(6):
        s = signal[ch]
        feats += [np.mean(s), np.std(s), np.max(s) - np.min(s),
                  np.mean(np.abs(s)), np.sqrt(np.mean(s ** 2)), np.sum(s ** 2)]
        fft = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), d=1.0 / 200)
        for lo, hi in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50), (8, 11.8)]:
            m = fft[(freqs >= lo) & (freqs < hi)]
            feats.append(m.mean() if len(m) else 0.0)
        psd = fft ** 2; psd_n = psd / (psd.sum() + 1e-8)
        feats.append(-np.sum(psd_n * np.log(psd_n + 1e-8)))
    return np.array(feats, dtype=np.float32)


class FoldDatasetDL(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        eeg, lbl = self.samples[i]
        return (torch.tensor(eeg.reshape(6, -1), dtype=torch.float32),
                torch.tensor(lbl, dtype=torch.float32))


class EEGNet(nn.Module):
    def __init__(self, n_ch=6, seq_len=1000, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
                                      nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                                       nn.BatchNorm2d(F1 * D), nn.ELU(),
                                       nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
                                       nn.BatchNorm2d(F2), nn.ELU(),
                                       nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.classifier = nn.Linear(F2 * (seq_len // 32), 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.separable(self.depthwise(self.temporal(x)))
        return self.classifier(x.flatten(1))


class TransformerClassifier(nn.Module):
    def __init__(self, n_ch=6, seq_len=1000, d_model=64, nhead=4, num_layers=3,
                 dropout=0.1, patch_size=50):
        super().__init__()
        self.patch_size = patch_size
        n_patches = seq_len // patch_size
        self.patch_proj = nn.Linear(n_ch * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B, C, T // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, T // self.patch_size, -1)
        x = self.patch_proj(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        return self.classifier(self.norm(self.encoder(x))[:, 0])


def train_dl(model, train_s, val_s, device, epochs=50, lr=1e-3, bs=64):
    tr = DataLoader(FoldDatasetDL(train_s), batch_size=bs, shuffle=True, num_workers=4)
    vl = DataLoader(FoldDatasetDL(val_s), batch_size=bs, shuffle=False, num_workers=4)
    labels = [s[1] for s in train_s]
    pos_w = torch.tensor([labels.count(0) / max(labels.count(1), 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr), eta_min=1e-5)
    best, best_state = 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            loss = crit(model(xb.to(device)).squeeze(1), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, yb in vl:
                pr = torch.sigmoid(model(xb.to(device)).squeeze(1)).cpu().numpy()
                vp.extend((pr > 0.5).astype(int)); vt.extend(yb.int().tolist())
        v = balanced_accuracy_score(vt, vp)
        if v > best:
            best = v
            best_state = {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score_dl(model, samples, device, bs=64):
    ld = DataLoader(FoldDatasetDL(samples), batch_size=bs, shuffle=False, num_workers=4)
    model.eval(); s, y = [], []
    for xb, yb in ld:
        s += torch.sigmoid(model(xb.to(device)).squeeze(1)).cpu().numpy().tolist()
        y += yb.int().tolist()
    return np.array(s), np.array(y)


def subj_of(k): return '_'.join(k.split('_')[:3])
def eye_of(k):  return '_'.join(k.split('_')[:4])


def agg(keys, s, y, fn):
    g, gy = defaultdict(list), {}
    for k, si, yi in zip(keys, s, y):
        u = fn(k); g[u].append(si); gy[u] = yi
    us = list(g)
    return np.array([np.mean(g[u]) for u in us]), np.array([gy[u] for u in us])


def metrics(s, y):
    return {'n': int(len(y)), 'auc': float(roc_auc_score(y, s)),
            'bacc': float(balanced_accuracy_score(y, (s > 0.5).astype(int)))}


def boot_ci(S, Y, f, n=5000):
    v, m = [], len(Y)
    for _ in range(n):
        j = np.random.randint(0, m, m); yy = Y[j]
        if 0 < yy.sum() < m: v.append(f(yy, S[j]))
    v = np.sort(np.array(v))
    return float(v[int(.025 * len(v))]), float(v[int(.975 * len(v))])


def check_splits():
    """Fold sanity: disjoint train/val/test per fold, 95 participants each tested once, and test
    blocks byte-identical to the first split so results stay comparable with Table 4."""
    tested = []
    for fold in range(N_FOLDS):
        new = json.load(open(SPLITS_DIR / f'fold_{fold}.json'))['keys']
        old = json.load(open(OLD_SPLITS / f'fold_{fold}.json'))['keys']
        subj = {k: {subj_of(x) for x in new[k]} for k in ('train', 'val', 'test')}
        assert not (subj['train'] & subj['val'] or subj['train'] & subj['test'] or subj['val'] & subj['test']), \
            f'fold {fold}: participant overlap between train/val/test'
        assert set(new['test']) == set(old['test']), f'fold {fold}: test block differs from the first split'
        tested += sorted(subj['test'])
        print(f"  fold{fold}: train {len(subj['train'])} / val {len(subj['val'])} / test {len(subj['test'])} "
              f"participants; segments {len(new['train'])}/{len(new['val'])}/{len(new['test'])}; "
              f"train differs from first split by {len(subj['train'] ^ {subj_of(x) for x in old['train']})} participants")
    assert len(tested) == len(set(tested)) == 95, f'expected 95 participants tested once, got {len(set(tested))}'
    print('  splits_v2 OK: 95 participants, each tested exactly once; test blocks identical to the first split')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['rf', 'eegnet', 'transformer'])
    ap.add_argument('--check', action='store_true', help='validate splits_v2 and exit (no training)')
    ap.add_argument('--cuda', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    args = ap.parse_args()
    if args.check:
        check_splits()
        return
    if not args.method:
        ap.error('--method is required unless --check is given')
    out = OUT_DIR / f'{args.method}_voting.json'
    if out.exists():
        ap.error(f'refusing to overwrite {out}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check_splits()
    dev = torch.device(f'cuda:{args.cuda}') if args.method != 'rf' else None
    if dev is not None:
        torch.cuda.set_device(args.cuda)

    pooled_k, pooled_s, pooled_y, per_fold = [], [], [], []
    for fold in range(N_FOLDS):
        split = json.load(open(SPLITS_DIR / f'fold_{fold}.json'))
        tr_k, va_k, te_k = (split['keys'][x] for x in ('train', 'val', 'test'))
        random.seed(SEED + fold); np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold); torch.cuda.manual_seed_all(SEED + fold)

        tr_s = load_samples(tr_k); te_s = load_samples(te_k)
        if args.method == 'rf':
            X = np.stack([extract_features(e) for e, _ in tr_s])
            yv = np.array([l for _, l in tr_s])
            Xt = np.stack([extract_features(e) for e, _ in te_s])
            yt = np.array([l for _, l in te_s])
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('rf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                                           class_weight='balanced', n_jobs=-1,
                                                           random_state=SEED))])
            pipe.fit(X, yv)
            s = pipe.predict_proba(Xt)[:, 1]
        else:
            va_s = load_samples(va_k)
            model = (EEGNet() if args.method == 'eegnet' else TransformerClassifier()).to(dev)
            model = train_dl(model, tr_s, va_s, dev, epochs=args.epochs)
            s, yt = score_dl(model, te_s, dev)
            del model; torch.cuda.empty_cache()
            yt = np.array([l for _, l in te_s])

        yt = np.array([l for _, l in te_s])
        seg = metrics(s, yt)
        eS, eY = agg(te_k, s, yt, eye_of); sS, sY = agg(te_k, s, yt, subj_of)
        per_fold.append({'fold': fold, 'segment': seg,
                         'eye': metrics(eS, eY), 'subject': metrics(sS, sY)})
        print(f"  fold{fold}: seg AUC {seg['auc']:.4f} | eye {metrics(eS, eY)['auc']:.4f} | "
              f"subj {metrics(sS, sY)['auc']:.4f} (n={len(sY)})")
        pooled_k += te_k; pooled_s += s.tolist(); pooled_y += yt.tolist()

    S, Y = np.array(pooled_s), np.array(pooled_y)
    res = {'method': args.method, 'splits': str(SPLITS_DIR), 'seed': SEED, 'epochs': args.epochs,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'per_fold': per_fold, 'pooled': {},
           'test_keys': pooled_k, 'test_scores': S.tolist(), 'test_labels': Y.tolist()}
    print(f"\npooled ({len(Y)} segments):")
    for name, fn in (('segment', None), ('eye', eye_of), ('subject', subj_of)):
        gS, gY = (S, Y) if fn is None else agg(pooled_k, S, Y, fn)
        m = metrics(gS, gY)
        if fn is not None:
            m['auc_ci'] = boot_ci(gS, gY, lambda y, x: roc_auc_score(y, x))
            m['bacc_ci'] = boot_ci(gS, gY,
                                   lambda y, x: balanced_accuracy_score(y, (x > 0.5).astype(int)))
            print(f"  {name:<8} n={m['n']:<4} AUC={m['auc']:.4f} "
                  f"[{m['auc_ci'][0]:.3f},{m['auc_ci'][1]:.3f}]  BAcc={m['bacc']:.4f} "
                  f"[{m['bacc_ci'][0]:.3f},{m['bacc_ci'][1]:.3f}]")
        else:
            print(f"  {name:<8} n={m['n']:<4} AUC={m['auc']:.4f}  BAcc={m['bacc']:.4f}")
        res['pooled'][name] = m

    json.dump(res, open(out, 'w'))
    print(f"\nsaved -> {out}")


if __name__ == '__main__':
    main()
