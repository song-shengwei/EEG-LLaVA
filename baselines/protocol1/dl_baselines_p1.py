"""
E02: 深度学习 Baseline
方法: EEGNet, CNN-1D, Transformer, LSTM, CNN-LSTM
端到端训练，直接从原始 EEG 输入，输出二分类概率

所有模型统一设置：
  - 输入: (B, 6, 1000)  — 6通道，5patch×200拼接
  - 输出: binary classification (healthy / glaucoma)
  - 训练: 50 epochs, lr=1e-3, AdamW, class-weighted loss
  - 评估: Acc / PR-AUC / ROC-AUC / Sensitivity / Specificity

用法:
  python dl_baselines.py --cuda 0
  python dl_baselines.py --cuda 0 --model eegnet

p1 副本（baselines/protocol1）：只加两件事，启动时核对 LMDB __keys__ 与锁定的 Protocol 1
划分逐键相同；每个模型训练完立即把指标和逐段测试分数存进 results/dl/<model>.json（一模型一文件，
可以一卡一模型并行跑）。模型，超参数，pos_weight（737/1523，原样保留）与选模规则均未改。
--model <单个> 时原代码仍在 setup_seed(3407) 之后按原顺序建出全部五个模型，故初始权重与 --model all
相同；训练阶段的随机数从种子处开始，与串行跑时前面模型消耗过的状态不同（EEGNet 排第一，两种跑法相同）。
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import numpy as np
import lmdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)

sys.dont_write_bytecode = True

DATA_DIR = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[2] / 'data' / 'processed_lmdb'))  # [release]
OUT_DIR  = Path(__file__).resolve().parent / 'results'
DL_DIR   = OUT_DIR / 'dl'
MODEL_ARGS = {'EEGNet': 'eegnet', 'CNN-1D': 'cnn1d', 'Transformer': 'transformer', 'LSTM': 'lstm', 'CNN-LSTM': 'cnnlstm'}
REPO_ROOT    = Path(__file__).resolve().parents[2]  # [release]
LOCKED_SPLIT = REPO_ROOT / 'splits' / 'protocol1' / 'fold_0.json'
EXPECTED = {'train': 6114, 'val': 1343, 'test': 1404}
SEQ_LEN  = 1000   # 5 patches × 200
N_CH     = 6
N_CLASS  = 1      # binary, BCE loss


def setup_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Dataset ───────────────────────────────────────────────
class EEGRawDataset(Dataset):
    def __init__(self, data_dir, split):
        db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
        with db.begin(write=False) as txn:
            keys = pickle.loads(txn.get('__keys__'.encode()))[split]
        self.keys = list(keys)
        self.samples = []
        with db.begin(write=False) as txn:
            for key in keys:
                pair = pickle.loads(txn.get(key.encode()))
                eeg = pair['sample'] / 100.0          # (6, 5, 200)
                eeg = eeg.reshape(N_CH, SEQ_LEN)      # (6, 1000)
                self.samples.append((eeg, int(pair['label'])))
        db.close()

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        eeg, label = self.samples[idx]
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ── 模型定义 ──────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG (Lawhern et al., 2018)
    输入: (B, 1, C, T) — 1通道，C电极，T时间点
    """
    def __init__(self, n_ch=6, seq_len=1000, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        feat_len = seq_len // 32
        self.classifier = nn.Linear(F2 * feat_len, 1)

    def forward(self, x):
        # x: (B, 6, 1000) → (B, 1, 6, 1000)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(1)
        return self.classifier(x)


class CNN1D(nn.Module):
    """
    普通 1D-CNN：多尺度卷积 + 全连接分类
    """
    def __init__(self, n_ch=6, seq_len=1000, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            # 多尺度卷积
            nn.Conv1d(n_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(4),                       # (B, 128, 250)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(4),                       # (B, 256, 62)
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),               # (B, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))


class TransformerClassifier(nn.Module):
    """
    普通 Transformer Encoder 分类器
    将每个时间步作为 token，用 CLS token 分类
    """
    def __init__(self, n_ch=6, seq_len=1000, d_model=64, nhead=4,
                 num_layers=3, dropout=0.1, patch_size=50):
        super().__init__()
        self.patch_size = patch_size
        n_patches = seq_len // patch_size        # 20 patches
        self.patch_proj = nn.Linear(n_ch * patch_size, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, 6, 1000)
        B, C, T = x.shape
        # 分 patch
        x = x.reshape(B, C, T // self.patch_size, self.patch_size)  # (B, C, n_p, ps)
        x = x.permute(0, 2, 1, 3).reshape(B, T // self.patch_size, -1)  # (B, n_p, C*ps)
        x = self.patch_proj(x)                  # (B, n_p, d_model)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, n_p+1, d_model)
        x = x + self.pos_embed
        x = self.norm(self.encoder(x))
        return self.classifier(x[:, 0])         # CLS token


class LSTMClassifier(nn.Module):
    """
    双向 LSTM：逐时间步处理，取最后隐状态分类
    """
    def __init__(self, n_ch=6, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_ch, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, 6, 1000) → (B, 1000, 6)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])      # 最后时间步


class CNNLSTM(nn.Module):
    """
    CNN-LSTM：CNN 提取局部特征，LSTM 建模时序依赖
    """
    def __init__(self, n_ch=6, seq_len=1000, hidden=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(4),                    # (B, 64, 250)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(5),                    # (B, 128, 50)
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x):
        x = self.cnn(x)                         # (B, 128, 50)
        x = x.permute(0, 2, 1)                  # (B, 50, 128)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])


# ── 训练和评估 ────────────────────────────────────────────
def check_locked_split(data_dir):
    """p1: the LMDB's embedded split must equal the locked Protocol 1 split key for key, in order."""
    locked = json.load(open(LOCKED_SPLIT))['keys']
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        embedded = pickle.loads(txn.get('__keys__'.encode()))
    db.close()
    for split, n in EXPECTED.items():
        assert len(embedded[split]) == n, f'{split}: {len(embedded[split])} segments, expected {n}'
        assert list(embedded[split]) == locked[split], f'{split}: LMDB __keys__ differs from {LOCKED_SPLIT}'
    print(f"split OK: LMDB __keys__ == locked Protocol 1 split "
          f"({EXPECTED['train']}/{EXPECTED['val']}/{EXPECTED['test']})")


def compute_metrics(truths, preds, scores, name):
    tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
    bacc = balanced_accuracy_score(truths, preds)
    roc  = roc_auc_score(truths, scores)
    prec, rec, _ = precision_recall_curve(truths, scores, pos_label=1)
    pr   = auc(rec, prec)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return dict(name=name, bacc=bacc, roc_auc=roc, pr_auc=pr,
                sens=sens, spec=spec, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truths, preds, scores = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logit = model(x).squeeze(1)
        score = torch.sigmoid(logit)
        pred  = (score > 0.5).long()
        truths.extend(y.long().cpu().tolist())
        preds.extend(pred.cpu().tolist())
        scores.extend(score.cpu().tolist())
    return np.array(truths), np.array(preds), np.array(scores)


def train_model(model, name, train_loader, val_loader, test_loader,
                device, epochs=50, lr=1e-3):
    # 类别权重（3281 glaucoma, 2833 healthy）
    pos_weight = torch.tensor([737/1523]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auc, best_state = 0, None

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            tr, pr, sc = evaluate(model, val_loader, device)
            val_auc = roc_auc_score(tr, sc)
            print(f"  [{name}] Epoch {epoch+1}: loss={np.mean(losses):.4f}, val_auc={val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def run_model(name, model, train_ds, val_ds, test_ds, device,
              epochs=50, lr=1e-3, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    model = model.to(device)
    print(f"\n{'='*55}\nTraining [{name}] ...")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    model = train_model(model, name, train_loader, val_loader, test_loader,
                        device, epochs, lr)

    # Val
    tr, pr, sc = evaluate(model, val_loader, device)
    val_m = compute_metrics(tr, pr, sc, f'{name} — Val')

    # Test
    tr, pr, sc = evaluate(model, test_loader, device)
    test_m = compute_metrics(tr, pr, sc, f'{name} — Test')
    test_out = {'test_labels': tr.tolist(), 'test_preds': pr.tolist(), 'test_scores': sc.tolist()}

    for m in [val_m, test_m]:
        print(f"\n  [{m['name']}]")
        print(f"    Acc(bal)={m['bacc']:.4f} ROC={m['roc_auc']:.4f} "
              f"PR={m['pr_auc']:.4f} Sens={m['sens']:.4f} Spec={m['spec']:.4f}")

    return val_m, test_m, test_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model',      type=str, default='all',
                        choices=['all', 'eegnet', 'cnn1d', 'transformer', 'lstm', 'cnnlstm'])
    args = parser.parse_args()
    wanted = list(MODEL_ARGS.values()) if args.model == 'all' else [args.model]
    for a in wanted:
        if (DL_DIR / f'{a}.json').exists():
            parser.error(f'refusing to overwrite {DL_DIR / f"{a}.json"}')
    check_locked_split(DATA_DIR)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    setup_seed(3407)
    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"GPU: {torch.cuda.get_device_name(args.cuda)}  (CUDA_VISIBLE_DEVICES={visible}, --cuda {args.cuda})")

    print("Loading datasets...")
    train_ds = EEGRawDataset(DATA_DIR, 'train')
    val_ds   = EEGRawDataset(DATA_DIR, 'val')
    test_ds  = EEGRawDataset(DATA_DIR, 'test')
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    model_configs = {
        'EEGNet':      EEGNet(n_ch=N_CH, seq_len=SEQ_LEN),
        'CNN-1D':      CNN1D(n_ch=N_CH, seq_len=SEQ_LEN),
        'Transformer': TransformerClassifier(n_ch=N_CH, seq_len=SEQ_LEN),
        'LSTM':        LSTMClassifier(n_ch=N_CH),
        'CNN-LSTM':    CNNLSTM(n_ch=N_CH, seq_len=SEQ_LEN),
    }

    if args.model != 'all':
        key = {'eegnet':'EEGNet','cnn1d':'CNN-1D',
               'transformer':'Transformer','lstm':'LSTM','cnnlstm':'CNN-LSTM'}[args.model]
        model_configs = {key: model_configs[key]}

    all_results = {}
    DL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in model_configs.items():
        val_m, test_m, test_out = run_model(
            name, model, train_ds, val_ds, test_ds,
            device, epochs=args.epochs, batch_size=args.batch_size
        )
        all_results[name] = {'val': val_m, 'test': test_m}
        # p1: one file per model, written as soon as it finishes; the test loader is unshuffled,
        # so the per-segment outputs follow test_ds.keys
        rec = {'model': name, 'model_arg': MODEL_ARGS[name], 'models_run_in_process': args.model,
               'split': str(LOCKED_SPLIT), 'seed': 3407, 'epochs': args.epochs, 'batch_size': args.batch_size,
               'cuda_visible_devices': visible, 'gpu_name': torch.cuda.get_device_name(args.cuda),
               'script_sha256': script_sha,
               'metrics': {s: {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}
                           for s, m in all_results[name].items()},
               'test_keys': test_ds.keys, **test_out}
        out_file = DL_DIR / f'{MODEL_ARGS[name]}.json'
        with open(out_file, 'x') as f:
            json.dump(rec, f)
        print(f"  saved -> {out_file}")

    # 汇总表
    print("\n" + "="*75)
    print("Summary Table (Paper Table 1 素材)")
    print("="*75)
    print(f"{'Method':<20} {'Split':<6} {'Acc(bal)':>9} {'ROC-AUC':>9} {'PR-AUC':>8} {'Sens':>7} {'Spec':>7}")
    print("-"*75)
    for name, res in all_results.items():
        for split in ['val', 'test']:
            m = res[split]
            print(f"{name:<20} {split.capitalize():<6} "
                  f"{m['bacc']:>9.4f} {m['roc_auc']:>9.4f} {m['pr_auc']:>8.4f} "
                  f"{m['sens']:>7.4f} {m['spec']:>7.4f}")
    print("="*75)
    # p1: results are saved per model inside the training loop (results/dl/<model>.json)


if __name__ == '__main__':
    main()
