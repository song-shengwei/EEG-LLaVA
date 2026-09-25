#!/usr/bin/env python3
"""Held-out probe of the pre-LLM 55-token fused representation.

`extract_fused_probe.py` reproduces the major E05 definition, which cross-validates the
probe *inside the test split* and fits PCA and the scaler on all selected samples before
splitting. Its AUC is therefore not comparable with the pipeline's held-out test AUC, and
the major's claim that the probe exceeds the pipeline (0.827 vs. 0.811) rests on that
comparison.

This script measures the quantity that is actually comparable: the probe is fitted only on
the training split, every preprocessing step is fitted on training data alone, and it is
evaluated once on the complete test split with the same threshold the pipeline uses. The
result answers one question — how much of the pipeline's gap to a strong baseline is a
readout limitation rather than a representation limitation.

No model is trained, no checkpoint is selected and no test label informs any fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from final_model_io import T, load_complete_model


def encode(model, cache: dict, keys: list[str], device, label: str) -> tuple:
    features, labels = [], []
    for start in range(0, len(keys), 16):
        batch = keys[start:start + 16]
        eeg = torch.tensor(np.asarray([cache[key]["sample"] / 100.0 for key in batch]),
                           dtype=torch.float32, device=device)
        with torch.no_grad():
            tokens, _, _ = model.encode_eeg(eeg)
        if tokens.shape[1] != 55:
            raise RuntimeError(f"expected 55 EEG tokens, observed {tokens.shape}")
        features.append(tokens.float().mean(dim=1).cpu().numpy())
        labels.extend(int(cache[key]["label"]) for key in batch)
        if start % 800 == 0:
            print(f"{label}: {min(start + 16, len(keys))}/{len(keys)}", flush=True)
    return np.concatenate(features, axis=0), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    T.setup_seed(42)
    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(args.cuda)
    model, checkpoint = load_complete_model(args.checkpoint, device)

    fold = json.loads(args.split.read_text())
    train_keys, test_keys = fold["keys"]["train"], fold["keys"]["test"]
    if set(train_keys) & set(test_keys):
        raise RuntimeError("train and test keys overlap")
    cache = T.load_lmdb_cache(T.DATA_DIR, train_keys + test_keys)

    x_train, y_train = encode(model, cache, train_keys, device, "train")
    x_test, y_test = encode(model, cache, test_keys, device, "test")

    # Every transform is fitted on training data only.
    probe = Pipeline([
        ("pca", PCA(n_components=50, random_state=42)),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    probe.fit(x_train, y_train)
    probability = probe.predict_proba(x_test)[:, 1]
    pred = (probability > 0.5).astype(int)

    train_probability = probe.predict_proba(x_train)[:, 1]
    result = {
        "experiment": "held-out probe of the final 55-token pre-LLM fused representation",
        "seed": checkpoint["args"]["seed"],
        "checkpoint": str(args.checkpoint.resolve()),
        "split": str(args.split.resolve()),
        "representation": "mean of 30 CBraMod + 21 compact-Transformer + 4 spectral tokens",
        "probe": "PCA(50) + StandardScaler + LogisticRegression(C=1), fitted on train only",
        "protocol": "fit on the training split; evaluate once on the complete test split",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "test_auc": float(roc_auc_score(y_test, probability)),
        "test_bacc": float(balanced_accuracy_score(y_test, pred)),
        "test_sensitivity": float((pred[y_test == 1] == 1).mean()),
        "test_specificity": float((pred[y_test == 0] == 0).mean()),
        "train_auc_in_sample": float(roc_auc_score(y_train, train_probability)),
        "threshold": 0.5,
        "test_labels": y_test.tolist(),
        "test_probability_glaucoma": probability.tolist(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "n_train", "n_test", "test_auc", "test_bacc", "test_sensitivity",
        "test_specificity", "train_auc_in_sample")}, indent=2))


if __name__ == "__main__":
    main()
