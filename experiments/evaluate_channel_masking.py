#!/usr/bin/env python3
"""Recompute Fig. 15 masking with the final seed-42 55-token checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from final_model_io import T, load_complete_model


CHANNELS = ["PO3", "POz", "PO4", "O1", "Oz", "O2"]
GROUPS = {
    "Upper (PO3+POz+PO4)": [0, 1, 2],
    "Lower (O1+Oz+O2)": [3, 4, 5],
}


# [release] launch_cpu_finalizer_once() removed (internal bookkeeping only).


@torch.no_grad()
def evaluate(model, cache: dict, keys: list[str], device: torch.device,
             masked: list[int]) -> dict:
    model.eval()
    prompt = T.STAGE2_TEMPLATE + "\n"
    encoded_prompt = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    plen = encoded_prompt.input_ids.shape[1]
    candidates = []
    for answer in (T.STAGE2_ANSWER[0], T.STAGE2_ANSWER[1]):
        encoded = model.tokenizer(prompt + answer, return_tensors="pt",
                                  add_special_tokens=False)
        labels = encoded.input_ids.clone()
        labels[:, :plen] = -100
        candidates.append((encoded.input_ids.to(device),
                           encoded.attention_mask.to(device), labels.to(device)))
    labels, scores = [], []
    for index, key in enumerate(keys):
        pair = cache[key]
        eeg = torch.tensor(pair["sample"] / 100.0, dtype=torch.float32,
                           device=device).unsqueeze(0)
        if masked:
            eeg[:, masked, :, :] = 0.0
        losses = [model(eeg, ids, attention, target).loss.item()
                  for ids, attention, target in candidates]
        scores.append(float(F.softmax(torch.tensor([-losses[0], -losses[1]]), dim=0)[1]))
        labels.append(int(pair["label"]))
        if (index + 1) % 100 == 0:
            print(f"  scored {index + 1}/{len(keys)}", flush=True)
    y = np.asarray(labels)
    s = np.asarray(scores)
    pred = (s > 0.5).astype(int)
    return {
        "n": len(keys),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, s)),
        "cm": confusion_matrix(y, pred).tolist(),
        "masked_indices": masked,
        "masked_channels": [CHANNELS[index] for index in masked],
        "scores": scores,
        "labels": labels,
    }


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
    keys = fold["keys"]["test"]
    cache = T.load_lmdb_cache(T.DATA_DIR, keys)

    result = {
        "experiment": "Fig. 15 final-method inference-time channel masking",
        "seed": checkpoint["args"]["seed"],
        "checkpoint": str(args.checkpoint.resolve()),
        "split": str(args.split.resolve()),
        "metric": "segment-level loss-based BAcc/AUC; threshold 0.5",
        "retraining": False,
        "baseline": evaluate(model, cache, keys, device, []),
        "leave_one_out": {}, "group_ablation": {}, "topk": {},
    }
    base = result["baseline"]["bacc"]
    for index, name in enumerate(CHANNELS):
        print(f"[LOO] masking {name}", flush=True)
        value = evaluate(model, cache, keys, device, [index])
        value["delta_bacc"] = value["bacc"] - base
        result["leave_one_out"][name] = value
    for name, indices in GROUPS.items():
        print(f"[Group] masking {name}", flush=True)
        value = evaluate(model, cache, keys, device, indices)
        value["delta_bacc"] = value["bacc"] - base
        result["group_ablation"][name] = value
    importance = sorted(result["leave_one_out"].items(),
                        key=lambda item: item[1]["delta_bacc"])
    result["importance_order_from_loo"] = [item[0] for item in importance]
    for k in (1, 2, 3, 4):
        retained = [item[0] for item in importance[:k]]
        masked = [index for index, name in enumerate(CHANNELS) if name not in retained]
        print(f"[Top-{k}] retaining {retained}", flush=True)
        value = evaluate(model, cache, keys, device, masked)
        value["retained_channels"] = retained
        result["topk"][f"top{k}"] = value
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
