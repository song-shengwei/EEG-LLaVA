#!/usr/bin/env python3
"""Reproducible batch-1 inference profile for the locked R1a model.

This script is read-only with respect to the R1a checkpoint.  It loads fold 0
only because all five folds share the same unified architecture; no performance
metric or checkpoint is selected here.  The output is used solely to synchronize
the paper's system-profile figure for the final 55-token model used in both
protocols.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import numpy as np
import psutil
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # [release] was the internal project root
DUAL_DIR = REPO_ROOT / "src" / "H_dual_branch"
R1A_RUN = Path(os.environ.get("EEGLLAVA_P2_RUN", REPO_ROOT / "outputs" / "protocol2" / "R1a_seed1234"))
DEFAULT_CHECKPOINT = R1A_RUN / "ckpt" / "fold_0_best.pth"
DEFAULT_RESULT = REPO_ROOT / "outputs" / "profile" / "r1a_profile_results.json"
DATA_DIR = Path(os.environ.get("EEGLLAVA_LMDB", REPO_ROOT / "data" / "processed_lmdb"))
LLAMAG_ROOT = REPO_ROOT / "src" / "llamaG"
LLM_PATH = os.environ.get("EEGLLAVA_LLM", str(REPO_ROOT / "pretrained" / "Qwen3-0.6B"))
CBRAMOD_INIT = os.environ.get("EEGLLAVA_CBRAMOD", str(REPO_ROOT / "pretrained" / "cbramod" / "pretrained_weights.pth"))

sys.path.insert(0, str(DUAL_DIR))
sys.path.insert(0, str(LLAMAG_ROOT))
from dual_branch import AuxTransformer, DualBranchEEGLlava  # noqa: E402
from data.glaucoma_llava_dataset import STAGE2_ANSWER, STAGE2_TEMPLATE  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_mem_mb(device: torch.device) -> float:
    return float(torch.cuda.memory_allocated(device) / 1024**2)


def gpu_reserved_mb(device: torch.device) -> float:
    return float(torch.cuda.memory_reserved(device) / 1024**2)


def cpu_mem_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


def timed(function, repeats: int = 20, warmup: int = 5) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
    torch.cuda.synchronize()
    values = []
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            function()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "repeats": int(repeats),
        "warmup": int(warmup),
    }


def load_real_sample(key: str, device: torch.device) -> torch.Tensor:
    database = lmdb.open(
        str(DATA_DIR), readonly=True, lock=False, readahead=False, meminit=False
    )
    with database.begin(write=False) as transaction:
        raw = transaction.get(key.encode())
        if raw is None:
            raise KeyError(f"sample key missing from LMDB: {key}")
        pair = pickle.loads(raw)
    database.close()
    sample = np.asarray(pair["sample"], dtype=np.float32) / 100.0
    tensor = torch.from_numpy(sample).reshape(1, 6, 5, 200)
    return tensor.to(device)


def tokenized_prompt(model, device: torch.device):
    prompt = STAGE2_TEMPLATE + "\n"
    encoded = model.tokenizer(
        [prompt], return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    return prompt, ids, mask, ids.clone()


def candidate(model, prompt: str, answer: str, device: torch.device):
    prompt_ids = model.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    encoded = model.tokenizer(
        prompt + answer, return_tensors="pt", add_special_tokens=False
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return ids, mask, labels


def validate_checkpoint(checkpoint_path: Path) -> tuple[dict, str]:
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    manifest_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".manifest.json"
    )
    if not checkpoint_path.is_file() or not sidecar.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("checkpoint, SHA sidecar, or manifest is missing")
    actual = sha256(checkpoint_path)
    expected = sidecar.read_text().split()[0]
    manifest = json.loads(manifest_path.read_text())
    if actual != expected or manifest.get("sha256") != actual:
        raise RuntimeError("R1a checkpoint SHA-256 validation failed")
    if manifest.get("checkpoint_format") != "dual_branch_complete_v1":
        raise RuntimeError(f"unexpected checkpoint format: {manifest}")
    return manifest, actual


def load_model(checkpoint_path: Path, device: torch.device):
    manifest, checkpoint_sha = validate_checkpoint(checkpoint_path)
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    specification = checkpoint["model_spec"]
    if not specification.get("use_spectral"):
        raise RuntimeError("locked R1a checkpoint does not contain the spectral branch")
    auxiliary = AuxTransformer()
    model = DualBranchEEGLlava(
        llm_path=LLM_PATH,  # [release] was specification["llm_path"] (a server path)
        eeg_encoder_weights=CBRAMOD_INIT,  # [release] was specification["eeg_encoder_init"]
        freeze_eeg_encoder=True,
        freeze_llm=False,
        eeg_dim=specification["eeg_dim"],
        num_channels=specification["num_channels"],
        num_patches=specification["num_patches"],
        aux_encoder=auxiliary,
        aux_trainable=False,
        use_spectral=True,
        n_spectral_tokens=specification["n_spectral_tokens"],
    )
    incompatibility = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatibility}")
    del checkpoint
    model = model.to(device).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    return model, manifest, checkpoint_sha, load_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.repeats < 5 or args.warmup < 2:
        parser.error("profiling requires at least 5 repeats and 2 warm-up runs")

    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(args.cuda)
    model, manifest, checkpoint_sha, load_seconds = load_model(args.checkpoint, device)

    fold_result = json.loads((R1A_RUN / "logs" / "fold_0_result.json").read_text())
    sample_key = fold_result["test_keys"][0]
    eeg = load_real_sample(sample_key, device)
    prompt, _, _, _ = tokenized_prompt(model, device)
    healthy = candidate(model, prompt, STAGE2_ANSWER[0], device)
    glaucoma = candidate(model, prompt, STAGE2_ANSWER[1], device)

    with torch.inference_mode():
        main_features = model.eeg_encoder(eeg).reshape(1, 30, 200)
        aux_features = model.aux_encoder.features(eeg)

    def measure(function, repeats=None, warmup=None):
        return timed(
            function,
            repeats=args.repeats if repeats is None else repeats,
            warmup=args.warmup if warmup is None else warmup,
        )

    components = {
        "cbramod_encoder": measure(lambda: model.eeg_encoder(eeg)),
        "cbramod_projector": measure(lambda: model.projector(main_features)),
        "aux_transformer": measure(lambda: model.aux_encoder.features(eeg)),
        "aux_projector": measure(lambda: model.aux_proj(aux_features)),
        "spectral_branch": measure(lambda: model.spectral(eeg)),
        "combined_encode_eeg": measure(lambda: model.encode_eeg(eeg)),
        "single_template_forward": measure(
            lambda: model(eeg, *healthy),
            repeats=max(5, args.repeats // 2),
            warmup=max(2, args.warmup // 2),
        ),
    }

    def loss_based_score():
        for ids, mask, target in (healthy, glaucoma):
            model(eeg, ids, mask, target)

    components["loss_based_score_two_passes"] = measure(
        loss_based_score,
        repeats=max(5, args.repeats // 2),
        warmup=max(2, args.warmup // 2),
    )
    components["generate_20_tokens"] = measure(
        lambda: model.generate(eeg, prompt, max_new_tokens=20),
        repeats=5,
        warmup=2,
    )

    memory = {
        "model_loaded_mb": gpu_mem_mb(device),
        "model_reserved_mb": gpu_reserved_mb(device),
        "cpu_process_mb": cpu_mem_mb(),
    }
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(eeg, *healthy)
    memory["single_template_forward_peak_mb"] = float(
        torch.cuda.max_memory_allocated(device) / 1024**2
    )
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        loss_based_score()
    memory["loss_based_score_peak_mb"] = float(
        torch.cuda.max_memory_allocated(device) / 1024**2
    )
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model.generate(eeg, prompt, max_new_tokens=20)
    memory["generate_20_tokens_peak_mb"] = float(
        torch.cuda.max_memory_allocated(device) / 1024**2
    )

    loss_ms = components["loss_based_score_two_passes"]["mean_ms"]
    peak_mb = max(
        memory["single_template_forward_peak_mb"],
        memory["loss_based_score_peak_mb"],
        memory["generate_20_tokens_peak_mb"],
    )
    result = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "paper system-profile synchronization only",
        "selection_or_training_performed": False,
        "architecture": {
            "core_cbramod_tokens": 30,
            "aux_transformer_tokens": 21,
            "spectral_tokens": 4,
            "total_eeg_tokens": 55,
            "cross_model_vote_or_ensemble": False,
        },
        "device": {
            "name": torch.cuda.get_device_name(args.cuda),
            "visible_cuda_index": args.cuda,
            "capability": list(torch.cuda.get_device_capability(args.cuda)),
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "manifest": manifest,
            "fold_used_for_architecture_profile": 0,
        },
        "sample_key": sample_key,
        "model_load_time_s": float(load_seconds),
        "memory": memory,
        "component_latency_bs1": components,
        "derived": {
            "loss_based_samples_per_second": float(1000.0 / loss_ms),
            "maximum_profiled_peak_mb": float(peak_mb),
            "fits_12gb_consumer_gpu": bool(peak_mb < 12 * 1024),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
