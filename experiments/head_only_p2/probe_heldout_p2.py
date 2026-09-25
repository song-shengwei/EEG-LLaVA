#!/usr/bin/env python3
"""Protocol 2 head-only control: a separate two-class softmax head replaces the LLM readout.

Copy of `experiments/probe_heldout.py` (the script behind
the Fig. 16 held-out probe); the original is not modified. For one Protocol 2 fold it fits a
separate classification head on the training participants only and scores every held-out
segment once. The language model is loaded with the checkpoint but never executed: no decoder
forward pass, no template loss and no text generation take part in the score.

Two representations are read out with the same head, so that the two things Reviewer 2 asks
about (the LLM, and the joint mapping-decoder tuning) are removed one at a time:

  post_mapping  the 55 EEG tokens at the input of the language model, mean-pooled (1024-d).
                These tokens have passed through the mappings that were tuned jointly with
                the decoder in Stages 2-3. Identical to the Fig. 16 probe definition.
                Removes: the LLM readout.  Keeps: the jointly tuned mappings.

  pre_mapping   the outputs of the frozen EEG components before any mapping: mean-pooled
                CBraMod tokens (200-d), mean-pooled compact-Transformer tokens (64-d) and the
                78 raw time/band descriptors. Nothing in this vector has been updated by
                alignment or task tuning.
                Removes: the LLM readout and the joint mapping-decoder tuning.

Head: linear layer + two-class softmax, fitted by maximum likelihood (logistic regression,
C=1) after PCA(50) and standardisation, all fitted on the training split only. The
pre_mapping vector concatenates blocks with different units, so it is standardised once
before PCA; this is fixed here in advance and is the only difference between the two recipes.

No network is trained, no checkpoint is selected and no test label informs any fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Imports below come from archived directories; leave no bytecode there.
sys.dont_write_bytecode = True

# final_model_io locates the Response directory from its own path, so it is imported
# in place rather than copied.
REPO_ROOT = Path(__file__).resolve().parents[2]  # [release]
ORIGINAL_SCRIPTS = REPO_ROOT / "experiments"
sys.path.insert(0, str(ORIGINAL_SCRIPTS))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPRESENTATIONS = {
    "post_mapping": "mean of the 55 EEG tokens at the LLM input (30 CBraMod + 21 "
                    "compact-Transformer + 4 descriptor tokens), after the jointly tuned mappings",
    "pre_mapping": "frozen EEG components before any mapping: mean CBraMod tokens (200) + mean "
                   "compact-Transformer tokens (64) + 78 raw time/band descriptors",
}
HEAD_DESCRIPTION = {
    "post_mapping": "PCA(50) + StandardScaler + linear head with two-class softmax "
                    "(LogisticRegression, C=1), fitted on train only",
    "pre_mapping": "StandardScaler + PCA(50) + StandardScaler + linear head with two-class "
                   "softmax (LogisticRegression, C=1), fitted on train only",
}


def build_head(representation: str) -> Pipeline:
    steps = [
        ("pca", PCA(n_components=50, random_state=42)),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]
    if representation == "pre_mapping":
        steps.insert(0, ("block_scaler", StandardScaler()))
    return Pipeline(steps)


def two_class_softmax(head: Pipeline, x: np.ndarray) -> np.ndarray:
    """Softmax over the two class logits of the linear head -> (n, 2) = [P(healthy), P(glaucoma)].

    The fitted head has one logit difference z = logit(glaucoma) - logit(healthy); the two
    class logits are written symmetrically as (-z/2, +z/2), which leaves the softmax unchanged.
    """
    z = head.decision_function(x)
    logits = np.stack([-z / 2.0, z / 2.0], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    reference = head.predict_proba(x)
    if not np.allclose(softmax, reference, atol=1e-10):
        raise RuntimeError("two-class softmax disagrees with the fitted head's probabilities")
    return softmax


def fit_and_score(representation: str, x_train, y_train, x_test, y_test) -> dict:
    head = build_head(representation)
    head.fit(x_train, y_train)
    softmax = two_class_softmax(head, x_test)
    probability = softmax[:, 1]
    pred = softmax.argmax(axis=1)
    train_probability = two_class_softmax(head, x_train)[:, 1]
    return {
        "representation": REPRESENTATIONS[representation],
        "head": HEAD_DESCRIPTION[representation],
        "feature_dim": int(x_train.shape[1]),
        "test_auc": float(roc_auc_score(y_test, probability)),
        "test_bacc": float(balanced_accuracy_score(y_test, pred)),
        "test_sensitivity": float((pred[y_test == 1] == 1).mean()),
        "test_specificity": float((pred[y_test == 0] == 0).mean()),
        "train_auc_in_sample": float(roc_auc_score(y_train, train_probability)),
        "test_softmax_glaucoma": probability.tolist(),
    }


def encode(model, cache: dict, keys: list[str], device, label: str) -> tuple:
    import torch

    post, pre, labels = [], [], []
    for start in range(0, len(keys), 16):
        batch = keys[start:start + 16]
        eeg = torch.tensor(np.asarray([cache[key]["sample"] / 100.0 for key in batch]),
                           dtype=torch.float32, device=device)
        with torch.no_grad():
            tokens, _, _ = model.encode_eeg(eeg)
            cbramod = model.eeg_encoder(eeg).reshape(eeg.shape[0], -1, model.eeg_dim)
            compact = model.aux_encoder.features(eeg)
            descriptors = model.spectral.descriptors(eeg)
        if tokens.shape[1] != 55:
            raise RuntimeError(f"expected 55 EEG tokens, observed {tokens.shape}")
        if cbramod.shape[1:] != (30, 200) or compact.shape[1:] != (21, 64) \
                or descriptors.shape[1] != 78:
            raise RuntimeError("unexpected pre-mapping shapes: "
                               f"{cbramod.shape}, {compact.shape}, {descriptors.shape}")
        post.append(tokens.float().mean(dim=1).cpu().numpy())
        pre.append(torch.cat([cbramod.float().mean(dim=1), compact.float().mean(dim=1),
                              descriptors.float()], dim=1).cpu().numpy())
        labels.extend(int(cache[key]["label"]) for key in batch)
        if start % 800 == 0:
            print(f"{label}: {min(start + 16, len(keys))}/{len(keys)}", flush=True)
    features = {"post_mapping": np.concatenate(post, axis=0),
                "pre_mapping": np.concatenate(pre, axis=0)}
    return features, np.asarray(labels)


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

    import torch
    from final_model_io import T, load_complete_model

    T.setup_seed(42)
    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(args.cuda)
    model, checkpoint = load_complete_model(args.checkpoint, device)
    if any(p.requires_grad for p in model.eeg_encoder.parameters()) or model.aux_trainable:
        raise RuntimeError("EEG encoders are expected to be frozen in the locked checkpoint")

    fold = json.loads(args.split.read_text())
    train_keys, test_keys = fold["keys"]["train"], fold["keys"]["test"]
    if set(train_keys) & set(test_keys):
        raise RuntimeError("train and test keys overlap")
    cache = T.load_lmdb_cache(T.DATA_DIR, train_keys + test_keys)

    x_train, y_train = encode(model, cache, train_keys, device, "train")
    x_test, y_test = encode(model, cache, test_keys, device, "test")

    # Every transform and the head are fitted on training data only.
    result = {
        "experiment": "Protocol 2 head-only control: separate two-class softmax head in place "
                      "of the LLM readout",
        "seed": checkpoint["args"]["seed"],
        "checkpoint": str(args.checkpoint.resolve()),
        "split": str(args.split.resolve()),
        "protocol": "fit on the training split; evaluate once on the complete test split",
        "llm_forward_passes": 0,
        "fold": fold.get("fold"),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "threshold": 0.5,
        "test_keys": list(test_keys),
        "test_labels": y_test.tolist(),
        "heads": {name: fit_and_score(name, x_train[name], y_train, x_test[name], y_test)
                  for name in REPRESENTATIONS},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {key: head[key] for key in (
        "feature_dim", "test_auc", "test_bacc", "test_sensitivity", "test_specificity",
        "train_auc_in_sample")} for name, head in result["heads"].items()}, indent=2))


if __name__ == "__main__":
    main()
