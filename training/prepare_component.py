#!/usr/bin/env python3
"""Train one Protocol-1-only M component without reading held-out test scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # [release] was the internal project root
H_DIR = REPO_ROOT / "src" / "H_dual_branch"
C_DIR = REPO_ROOT / "src" / "C_5fold_clean_encoder"
sys.path.insert(0, str(H_DIR))
sys.path.insert(0, str(C_DIR))

import train_fold_clean as T  # noqa: E402
from dual_branch import train_aux  # noqa: E402


# [release] stop_superseded_phase2_if_armed() removed: it only stopped an obsolete job of the internal
# GPU queue and did not touch training.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["aux", "mainenc"], required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"fold_0_{args.component}.pth"
    result_path = args.output_dir / f"fold_0_{args.component}_result.json"
    if output_path.exists() or result_path.exists():
        parser.error(f"refusing to overwrite component output: {output_path}")

    with (args.split_dir / "fold_0.json").open() as handle:
        fold = json.load(handle)
    if fold.get("split_unit") != "eye":
        raise AssertionError("expected a Protocol 1 eye-level split")

    T.setup_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(args.cuda)

    if args.component == "aux":
        cache = T.load_lmdb_cache(
            T.DATA_DIR, fold["keys"]["train"] + fold["keys"]["val"]
        )
        model, info = train_aux(
            cache,
            fold["keys"]["train"],
            fold["keys"]["val"],
            device,
            epochs=50,
            lr=1e-3,
            seed=args.seed,
        )
        atomic_torch_save(model.state_dict(), output_path)
        del model
        config = {"epochs": 50, "lr": 1e-3, "batch_size": 64}
    else:
        old_save_dir = T.SAVE_DIR
        T.SAVE_DIR = args.output_dir
        train_args = SimpleNamespace(
            fold=0,
            cuda=args.cuda,
            enc_epochs=30,
            enc_bs=64,
            enc_lr=5e-4,
            enc_wd=5e-2,
            enc_clip=1.0,
            skip_encoder_test=True,
        )
        produced_path, info = T.train_encoder(fold, train_args, device)
        T.SAVE_DIR = old_save_dir
        produced = Path(produced_path)
        if produced != output_path:
            produced.replace(output_path)
        config = {
            "epochs": 30,
            "lr": 5e-4,
            "batch_size": 64,
            "weight_decay": 5e-2,
            "held_out_test_scored": False,
        }

    report = {
        "status": "complete",
        "protocol": "Protocol 1 (Eye-Level Split)",
        "component": args.component,
        "seed": args.seed,
        "split_file": str((args.split_dir / "fold_0.json").resolve()),
        "train_segments": len(fold["keys"]["train"]),
        "validation_segments": len(fold["keys"]["val"]),
        "test_used_for_selection": False,
        "config": config,
        "training_info": info,
        "checkpoint": str(output_path.resolve()),
        "checkpoint_sha256": sha256_file(output_path),
    }
    with result_path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(args.output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    # The queue and trainer are separate host processes on this mounted volume.  A short grace
    # avoids a second metadata-visibility race at the exact child-exit boundary.
    time.sleep(2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
