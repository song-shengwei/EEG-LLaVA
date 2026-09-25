#!/usr/bin/env python3
"""R1a wrapper: select the unchanged M model by validation-subject metrics.

The underlying model, losses, optimizers, data splits, and test evaluator remain
those of H_dual_branch/train_fold_dual.py.  Only Stage-2 checkpoint selection is
replaced.  This file intentionally has the same basename as the underlying
trainer so the existing scoped task-renewal allow-list accepts it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # [release] was the internal project root
BASE_PATH = REPO_ROOT / "src" / "H_dual_branch" / "train_fold_dual.py"

spec = importlib.util.spec_from_file_location("_r1a_base_train_fold_dual", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import base trainer: {BASE_PATH}")
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)

ORIGINAL_EVALUATE = BASE.T.evaluate
SELECTION_HISTORY: list[dict] = []
SELECTION_AUC_FLOOR = 0.85
SELECTION_EVAL_EVERY = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_of(key: str) -> str:
    return "_".join(key.split("_")[:3])


def subject_metrics(keys, scores, labels) -> tuple[float, float, np.ndarray, int]:
    grouped: dict[str, list] = defaultdict(lambda: [[], None])
    for key, score, label in zip(keys, scores, labels):
        subject = subject_of(key)
        label = int(label)
        if grouped[subject][1] is not None and grouped[subject][1] != label:
            raise AssertionError(f"inconsistent label within subject {subject}")
        grouped[subject][0].append(float(score))
        grouped[subject][1] = label
    subject_scores = np.asarray(
        [np.mean(grouped[subject][0]) for subject in grouped], dtype=np.float64
    )
    subject_labels = np.asarray(
        [grouped[subject][1] for subject in grouped], dtype=np.int64
    )
    predictions = (subject_scores >= 0.5).astype(np.int64)
    bacc = float(balanced_accuracy_score(subject_labels, predictions))
    auc = float(roc_auc_score(subject_labels, subject_scores))
    matrix = confusion_matrix(subject_labels, predictions, labels=[0, 1])
    return bacc, auc, matrix, len(grouped)


def r1a_evaluate(model, loader, device, return_outputs=False):
    # Preserve the original generated-answer evaluator for the final test report.
    if return_outputs:
        return ORIGINAL_EVALUATE(model, loader, device, return_outputs=True)
    _, _, _, scores, labels = BASE.T.evaluate_loss_based(model, loader, device)
    keys = list(loader.dataset.keys)
    if not (len(keys) == len(scores) == len(labels)):
        raise AssertionError("validation key/score/label length mismatch")
    bacc, auc, matrix, n_subjects = subject_metrics(keys, scores, labels)
    print(
        f"R1a validation-subject metric: n={n_subjects}, "
        f"BAcc={bacc:.5f}, AUC={auc:.5f}\n{matrix}",
        flush=True,
    )
    return bacc, auc, matrix


def r1a_train_stage(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    epochs,
    healthy_weight,
    clip_value=1.0,
    run_eval_fn=None,
    eval_every=2,
    select_metric=None,
    history=None,
):
    """Train unchanged Stage 2; select only with validation-subject BAcc/AUC.

    `select_metric` and `history` exist only to absorb the keywords the base
    `train_stage_w` gained on 2026-08-08. R1a ignores both: its selection rule is the
    validation-subject BAcc/AUC rule fixed in PRE_REGISTRATION.json, and it records its own
    per-epoch trajectory in the `.r1a_selection.json` sidecar.
    """
    del eval_every, select_metric, history
    best_rank = None
    best_bacc, best_epoch, best_states = 0.0, 0, None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(epochs):
        model.train()
        losses = []
        start = timer()
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", mininterval=30):
            raw = batch["label"]
            weights = torch.where(
                raw == 0,
                torch.tensor(float(healthy_weight)),
                torch.tensor(1.0),
            )
            weights = weights * batch.get("sw", torch.ones_like(weights))
            optimizer.zero_grad()
            output = model(
                batch["eeg"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
                weights,
            )
            output.loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip_value)
            optimizer.step()
            scheduler.step()
            losses.append(output.loss.item())
        print(
            f"Epoch {epoch + 1}: loss={np.mean(losses):.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}, "
            f"time={(timer() - start) / 60:.1f}min"
        )
        if run_eval_fn and (
            (epoch + 1) % SELECTION_EVAL_EVERY == 0 or epoch == epochs - 1
        ):
            bacc, auc, matrix = run_eval_fn()
            eligible = bool(auc >= SELECTION_AUC_FLOOR)
            # Once any epoch reaches the AUC floor, maximize BAcc, then AUC.
            # If none reaches it, fall back to max AUC, then BAcc, and record this.
            rank = (
                int(eligible),
                bacc if eligible else auc,
                auc if eligible else bacc,
                -(epoch + 1),
            )
            row = {
                "epoch": epoch + 1,
                "subject_bacc": float(bacc),
                "subject_auc": float(auc),
                "auc_floor_eligible": eligible,
                "confusion_matrix": matrix.tolist(),
            }
            SELECTION_HISTORY.append(row)
            print(
                f"R1a selection candidate: epoch={epoch + 1}, "
                f"eligible={eligible}, BAcc={bacc:.5f}, AUC={auc:.5f}"
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_bacc, best_epoch = float(bacc), epoch + 1
                best_states = copy.deepcopy(model.state_dict())
                print(f"R1a new selected epoch: {best_epoch}", flush=True)
    return best_bacc, best_epoch, best_states


def option_value(arguments: list[str], name: str, default: str) -> str:
    if name not in arguments:
        return default
    position = arguments.index(name)
    if position + 1 >= len(arguments):
        raise ValueError(f"missing value after {name}")
    return arguments[position + 1]


def append_selection_audit(arguments: list[str]) -> None:
    out_dir = Path(option_value(arguments, "--out_dir", str(HERE)))
    fold = int(option_value(arguments, "--fold", "0"))
    result_path = out_dir / "logs" / f"fold_{fold}_result.json"
    checkpoint = out_dir / "ckpt" / f"fold_{fold}_best.pth"
    if not result_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("base trainer did not produce result/checkpoint")
    selected_epoch = int(json.loads(result_path.read_text())["best_val_epoch"])
    selected = next(
        row for row in SELECTION_HISTORY if row["epoch"] == selected_epoch
    )
    audit = {
        "candidate": "R1a_subject_aligned_checkpoint_selection",
        "model_architecture_changed": False,
        "training_loss_changed": False,
        "validation_unit": "subject",
        "subject_score": "mean loss-based probability over that subject's segments",
        "threshold": 0.5,
        "auc_floor": SELECTION_AUC_FLOOR,
        "selection_rule": (
            "among epochs with validation-subject AUC >= floor, maximize BAcc, "
            "then AUC, then choose earliest epoch; if none is eligible, maximize "
            "AUC, then BAcc, then choose earliest epoch"
        ),
        "fallback_used": not bool(selected["auc_floor_eligible"]),
        "evaluation_every_epochs": SELECTION_EVAL_EVERY,
        "selected": selected,
        "history": SELECTION_HISTORY,
        "wrapper_sha256": sha256(Path(__file__)),
        "base_trainer": str(BASE_PATH.resolve()),
        "base_trainer_sha256": sha256(BASE_PATH),
    }
    result = json.loads(result_path.read_text())
    result["r1a_selection"] = audit
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(result_path)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".r1a_selection.json")
    sidecar.write_text(json.dumps(audit, indent=2) + "\n")
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_protocol"] = "r1a_validation_subject_bacc_auc_floor"
    manifest["selection_sidecar"] = sidecar.name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    global SELECTION_AUC_FLOOR, SELECTION_EVAL_EVERY
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection-auc-floor", type=float, default=0.85)
    parser.add_argument("--selection-eval-every", type=int, default=2)
    options, remaining = parser.parse_known_args()
    if not 0.5 <= options.selection_auc_floor <= 1.0:
        parser.error("--selection-auc-floor must be in [0.5, 1.0]")
    if options.selection_eval_every < 1:
        parser.error("--selection-eval-every must be positive")
    SELECTION_AUC_FLOOR = options.selection_auc_floor
    SELECTION_EVAL_EVERY = options.selection_eval_every

    BASE.T.evaluate = r1a_evaluate
    BASE.train_stage_w = r1a_train_stage
    sys.argv = [sys.argv[0], *remaining]
    BASE.main()
    append_selection_audit(remaining)


if __name__ == "__main__":
    main()
