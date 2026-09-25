#!/usr/bin/env python3
"""Reload one complete R0 checkpoint and compare its segment scores with saved output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # [release] was the internal project root
H_DIR = REPO_ROOT / "src" / "H_dual_branch"
C_DIR = REPO_ROOT / "src" / "C_5fold_clean_encoder"
sys.path.insert(0, str(H_DIR))
sys.path.insert(0, str(C_DIR))

import train_fold_clean as T  # noqa: E402
from dual_branch import AuxTransformer, DualBranchEEGLlava  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_and_load(checkpoint: dict, device: torch.device) -> DualBranchEEGLlava:
    if checkpoint.get("checkpoint_format") != "dual_branch_complete_v1":
        raise ValueError(
            f"unsupported checkpoint format: {checkpoint.get('checkpoint_format')!r}"
        )
    spec = checkpoint["model_spec"]
    aux = AuxTransformer()
    model = DualBranchEEGLlava(
        llm_path=T.LLM_PATH,  # [release] was spec["llm_path"] (a server path)
        eeg_encoder_weights=T.FOUNDATION_WEIGHTS,  # [release] was spec["eeg_encoder_init"]
        freeze_eeg_encoder=True,
        freeze_llm=False,
        eeg_dim=int(spec["eeg_dim"]),
        num_channels=int(spec["num_channels"]),
        num_patches=int(spec["num_patches"]),
        aux_encoder=aux,
        aux_trainable=bool(spec["aux_trainable"]),
        use_spectral=bool(spec["use_spectral"]),
        n_spectral_tokens=int(spec["n_spectral_tokens"]),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(
            f"state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-atol", type=float, default=1e-6)
    args = parser.parse_args()

    if args.cuda not in {0, 1, 2, 3, 4}:
        parser.error("R0 policy permits only physical GPUs 0..4")
    if args.output_json.exists():
        parser.error(f"refusing to overwrite: {args.output_json}")

    observed_sha256 = sha256_file(args.checkpoint)
    manifest_path = args.checkpoint.with_suffix(args.checkpoint.suffix + ".manifest.json")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if observed_sha256 != manifest["sha256"]:
        raise AssertionError(
            f"checkpoint SHA-256 mismatch: {observed_sha256} != {manifest['sha256']}"
        )
    for required_flag in (
        "contains_spectral_state",
        "contains_aux_encoder_state",
        "contains_eeg_encoder_state",
    ):
        if not manifest.get(required_flag):
            raise AssertionError(f"checkpoint manifest flag is false: {required_flag}")

    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(args.cuda)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_and_load(checkpoint, device)

    with args.result_json.open() as handle:
        expected = json.load(handle)
    if int(expected["fold"]) != int(checkpoint["fold"]):
        raise AssertionError("checkpoint/result fold mismatch")

    max_length = int(checkpoint["args"]["max_length"])
    dataset = T.FoldDataset(
        T.DATA_DIR, expected["test_keys"], model.tokenizer, max_length, stage=2
    )
    collate = partial(T.collate_fn, tokenizer=model.tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    bacc_named_legacy, auc, cm, scores, labels = T.evaluate_loss_based(model, loader, device)

    expected_scores = np.asarray(expected["test_scores_lb"], dtype=np.float64)
    observed_scores = np.asarray(scores, dtype=np.float64)
    if observed_scores.shape != expected_scores.shape:
        raise AssertionError(
            f"score shape mismatch: {observed_scores.shape} != {expected_scores.shape}"
        )
    max_abs_score_error = float(np.max(np.abs(observed_scores - expected_scores)))
    if max_abs_score_error > args.score_atol:
        raise AssertionError(
            f"score reload mismatch: max abs error {max_abs_score_error} > {args.score_atol}"
        )
    if [int(label) for label in labels] != [int(label) for label in expected["test_labels_lb"]]:
        raise AssertionError("label order mismatch after checkpoint reload")

    report = {
        "status": "pass",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": observed_sha256,
        "checkpoint_format": checkpoint["checkpoint_format"],
        "fold": int(checkpoint["fold"]),
        "strict_state_dict_load": True,
        "max_abs_score_error": max_abs_score_error,
        "score_atol": args.score_atol,
        "n_test_segments": len(scores),
        "segment_accuracy_legacy_name": float(bacc_named_legacy),
        "segment_auc": float(auc),
        "segment_confusion_matrix": cm.tolist(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

