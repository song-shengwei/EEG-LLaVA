#!/usr/bin/env python3
"""Strict loader for the locked seed-42 complete 55-token checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]  # [release] was the internal project root
H_DIR = REPO_ROOT / "src" / "H_dual_branch"
C_DIR = REPO_ROOT / "src" / "C_5fold_clean_encoder"
sys.path.insert(0, str(H_DIR))
sys.path.insert(0, str(C_DIR))
from dual_branch import AuxTransformer, DualBranchEEGLlava  # noqa: E402
import train_fold_clean as T  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete_model(checkpoint_path: Path, device: torch.device,
                        verify_sha: bool = True):
    if verify_sha:
        sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        expected = sidecar.read_text().split()[0]
        observed = sha256_file(checkpoint_path)
        if observed != expected:
            raise RuntimeError(f"checkpoint SHA mismatch: {observed} != {expected}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_format") != "dual_branch_complete_v1":
        raise RuntimeError("only dual_branch_complete_v1 checkpoints are accepted")
    spec = checkpoint["model_spec"]
    component_sources = checkpoint["component_sources"]
    aux = AuxTransformer()
    # The complete state will overwrite this, but loading the component here independently
    # verifies that the archived source recorded by the training run is still usable.
    if Path(component_sources["aux"]).is_file():  # [release] training-time file, usually absent
        aux.load_state_dict(torch.load(component_sources["aux"], map_location="cpu"))
    model = DualBranchEEGLlava(
        llm_path=T.LLM_PATH,  # [release] was spec["llm_path"] (a server path)
        eeg_encoder_weights=T.FOUNDATION_WEIGHTS,  # [release] was component_sources["main_encoder"]
        freeze_eeg_encoder=True,
        freeze_llm=False,
        eeg_dim=spec["eeg_dim"],
        num_channels=spec["num_channels"],
        num_patches=spec["num_patches"],
        aux_encoder=aux,
        aux_trainable=False,
        use_spectral=spec["use_spectral"],
        n_spectral_tokens=spec.get("n_spectral_tokens", 4),
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch; missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    return model, checkpoint

