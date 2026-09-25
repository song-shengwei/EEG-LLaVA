#!/usr/bin/env python3
"""Random-forest comparator for the Protocol 1 paired tests (DeLong, McNemar; Fig. 10).

The paired tests compare EEG-LLaVA with a Random Forest trained on 78 hand-crafted features per
segment (per channel: mean, SD, range, mean |x|, RMS, energy, mean FFT magnitude in 0.5-4, 4-8,
8-13, 13-30, 30-50 and 8-11.8 Hz, spectral entropy), StandardScaler, then
RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=3407).
It is fitted on the Protocol 1 training segments and scored once on the 1,404 test segments.
(The Table 3 RF row is a different, depth-unlimited forest: see ../full_comparison.py.)

The feature function and the pipeline are copied verbatim from the scripts that produced the
archived comparator (E13 McNemar / E15 DeLong). This script writes the two input files that
evaluation/evaluate_protocol1.py reads:
  delong_rf_scores.json        {"rf_scores": [...], "trues": [...]}
  mcnemar_rf_predictions.json  {"rf": {"preds": [...]}, "trues": [...]}
Segment order is the Protocol 1 test order (splits/protocol1/fold_0.json, identical to the LMDB
__keys__['test'] list).

Usage:  python rf_depth12_paired.py [--output-dir DIR] [--check-against DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = os.environ.get("EEGLLAVA_LMDB", str(REPO_ROOT / "data" / "processed_lmdb"))
SPLIT = REPO_ROOT / "splits" / "protocol1" / "fold_0.json"
SEED = 3407


def extract_features(eeg_np):
    feats = []
    signal = eeg_np.reshape(6, -1)
    for ch in range(6):
        s = signal[ch]
        feats += [np.mean(s), np.std(s), np.max(s)-np.min(s),
                  np.mean(np.abs(s)), np.sqrt(np.mean(s**2)), np.sum(s**2)]
        fft   = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), d=1.0/200)
        for lo, hi in [(0.5,4),(4,8),(8,13),(13,30),(30,50),(8,11.8)]:
            m = fft[(freqs>=lo)&(freqs<hi)]
            feats.append(m.mean() if len(m) else 0.0)
        psd = fft**2; psd_n = psd/(psd.sum()+1e-8)
        feats.append(-np.sum(psd_n*np.log(psd_n+1e-8)))
    return np.array(feats, dtype=np.float32)


def load(keys):
    db = lmdb.open(DATA_DIR, readonly=True, lock=False, readahead=False, meminit=False)
    out = []
    with db.begin(write=False) as txn:
        for k in keys:
            pair = pickle.loads(txn.get(k.encode()))
            out.append((pair['sample'].astype(np.float32) / 100.0, int(pair['label'])))
    db.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--check-against", type=Path, default=None,
                    help="directory with previously written files; report whether they are reproduced")
    args = ap.parse_args()

    split = json.loads(SPLIT.read_text())["keys"]
    train, test = load(split["train"]), load(split["test"])
    X_tr = np.stack([extract_features(e) for e, _ in train]); y_tr = np.array([l for _, l in train])
    X_te = np.stack([extract_features(e) for e, _ in test]); y_te = np.array([l for _, l in test])
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                                   class_weight='balanced', n_jobs=-1,
                                                   random_state=SEED))])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te).tolist()
    scores = pipe.predict_proba(X_te)[:, 1].tolist()
    trues = y_te.tolist()
    print(f"RF depth 12: AUC {roc_auc_score(trues, scores):.6f}  BAcc {balanced_accuracy_score(trues, preds):.6f}  "
          f"n_test {len(trues)}")

    delong = {"rf_scores": scores, "trues": trues}
    mcnemar = {"rf": {"preds": preds}, "trues": trues}
    if args.check_against is not None:
        old_d = json.loads((args.check_against / "delong_rf_scores.json").read_text())
        old_m = json.loads((args.check_against / "mcnemar_rf_predictions.json").read_text())
        same_pred = old_m["rf"]["preds"] == preds and old_m["trues"] == trues
        max_diff = float(np.max(np.abs(np.array(old_d["rf_scores"]) - np.array(scores))))
        print(f"reproduces shipped files: predictions identical {same_pred}; max |score diff| {max_diff:.3g}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "delong_rf_scores.json").write_text(json.dumps(delong) + "\n")
    (args.output_dir / "mcnemar_rf_predictions.json").write_text(json.dumps(mcnemar) + "\n")
    print(f"written to {args.output_dir}")


if __name__ == "__main__":
    main()
