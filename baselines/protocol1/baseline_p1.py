"""
E02: 传统机器学习 Baseline
方法: SVM, Random Forest, LDA
特征: 从 lmdb 读取 EEG，提取时域+频域特征，subject-level split

用法:
  python baseline.py --data_dir <lmdb_path>

p1 副本（baselines/protocol1）：划分实际是 LMDB __keys__ 里的 Protocol 1 按眼划分，
不是上面写的 subject-level。副本只加四件事：启动时核对 __keys__ 与锁定划分逐键相同；固定 NumPy
全局种子（只影响 SVM 的概率校准）；补打 BAcc；把 train-only 一组的逐段测试分数存进
results/classical_p1.json。模型与特征均未改。
"""

import argparse
import hashlib
import os  # [release] used by the LMDB default below
import json
import pickle
import sys
import numpy as np
import lmdb
from pathlib import Path

sys.dont_write_bytecode = True

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

REPO_ROOT    = Path(__file__).resolve().parents[2]  # [release]
LOCKED_SPLIT = REPO_ROOT / 'splits' / 'protocol1' / 'fold_0.json'
OUT_FILE     = Path(__file__).resolve().parent / 'results' / 'classical_p1.json'
EXPECTED     = {'train': 6114, 'val': 1343, 'test': 1404}


# ── 特征提取 ──────────────────────────────────────────────
def extract_features(eeg_np):
    """
    eeg_np: (6, 5, 200) — 6 channels, 5 patches, 200 samples
    返回: 1D feature vector
    """
    feats = []
    signal = eeg_np.reshape(6, -1)  # (6, 1000)

    for ch in range(6):
        s = signal[ch]
        # 时域特征
        feats.append(np.mean(s))
        feats.append(np.std(s))
        feats.append(np.max(s) - np.min(s))       # peak-to-peak
        feats.append(np.mean(np.abs(s)))           # MAV
        feats.append(np.sqrt(np.mean(s ** 2)))     # RMS
        feats.append(np.sum(s ** 2))               # energy

        # 频域特征 (FFT)
        fft = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), d=1.0/200)  # 假设 200Hz 采样
        # 各频段能量
        delta = fft[(freqs >= 0.5) & (freqs < 4)].mean() if any((freqs >= 0.5) & (freqs < 4)) else 0
        theta = fft[(freqs >= 4) & (freqs < 8)].mean() if any((freqs >= 4) & (freqs < 8)) else 0
        alpha = fft[(freqs >= 8) & (freqs < 13)].mean() if any((freqs >= 8) & (freqs < 13)) else 0
        beta  = fft[(freqs >= 13) & (freqs < 30)].mean() if any((freqs >= 13) & (freqs < 30)) else 0
        gamma = fft[(freqs >= 30) & (freqs < 50)].mean() if any((freqs >= 30) & (freqs < 50)) else 0
        # SSVEP 目标频率附近的峰值能量（若刺激频率已知可精确设置）
        ssvep_band = fft[(freqs >= 6) & (freqs < 20)].max() if any((freqs >= 6) & (freqs < 20)) else 0

        feats.extend([delta, theta, alpha, beta, gamma, ssvep_band])

        # 频谱熵
        psd = fft ** 2
        psd_norm = psd / (psd.sum() + 1e-8)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
        feats.append(spectral_entropy)

    return np.array(feats, dtype=np.float32)


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


def load_split(data_dir, split):
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys = pickle.loads(txn.get('__keys__'.encode()))[split]

    X, y = [], []
    with db.begin(write=False) as txn:
        for key in keys:
            pair = pickle.loads(txn.get(key.encode()))
            eeg = pair['sample'] / 100.0  # (6, 5, 200)
            label = int(pair['label'])
            feats = extract_features(eeg)
            X.append(feats)
            y.append(label)

    db.close()
    return np.array(X), np.array(y), list(keys)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"\n{'='*50}")
    print(f"[{name}]")
    print(f"  Acc:         {acc:.5f}")
    print(f"  BAcc:        {bacc:.5f}")
    print(f"  AUC-ROC:     {auc:.5f}")
    print(f"  Sensitivity: {sensitivity:.5f}  (青光眼检出率)")
    print(f"  Specificity: {specificity:.5f}  (健康正确率)")
    print(f"  Confusion Matrix:\n{cm}")
    return acc, auc, bacc, np.asarray(y_prob, dtype=float), np.asarray(y_pred, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('EEGLLAVA_LMDB', str(REPO_ROOT / 'data' / 'processed_lmdb')))  # [release]
    args = parser.parse_args()
    if OUT_FILE.exists():
        parser.error(f'refusing to overwrite {OUT_FILE}')
    check_locked_split(args.data_dir)
    # p1: SVC(probability=True) draws its Platt-scaling CV seed from NumPy's global RNG. predict() and the
    # rank-based AUC do not depend on it, but mean probabilities per eye do, so fix it for reproducibility.
    np.random.seed(3407)

    print("Loading data...")
    X_train, y_train, _ = load_split(args.data_dir, 'train')
    X_val,   y_val,   _ = load_split(args.data_dir, 'val')
    X_test,  y_test,  test_keys = load_split(args.data_dir, 'test')

    # 合并 train+val 用于最终测试（模拟实际场景）
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Feature dim: {X_train.shape[1]}")

    models = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced'))
        ]),
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=3407))
        ]),
        'LDA': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())
        ]),
    }

    print("\n=== 在 Train 集训练，Test 集评估 ===")
    results = {}
    saved = {}
    for name, clf in models.items():
        acc, auc, bacc, prob, pred = evaluate_model(name, clf, X_train, y_train, X_test, y_test)
        results[name] = (acc, auc, bacc)
        saved[name] = {'test_scores': prob.tolist(), 'test_preds': pred.tolist()}

    print("\n=== 在 Train+Val 集训练，Test 集评估 ===")
    for name, clf in models.items():
        evaluate_model(f"{name} (train+val)", clf, X_trainval, y_trainval, X_test, y_test)

    print("\n=== Summary ===")
    for name, (acc, auc, bacc) in results.items():
        print(f"  {name}: acc={acc:.4f}, bacc={bacc:.4f}, auc={auc:.4f}")

    # p1: train-only runs are the ones Table 3 reports; keep their per-segment test outputs
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = {'split': str(LOCKED_SPLIT), 'fit_on': 'train only',
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'test_keys': test_keys, 'test_labels': y_test.tolist(), 'models': saved}
    with open(OUT_FILE, 'x') as f:
        json.dump(out, f)
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == '__main__':
    main()
