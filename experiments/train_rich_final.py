#!/usr/bin/env python3
"""Train the Fig. 20 Rich Report variant of the final 55-token method."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from final_model_io import AuxTransformer, DualBranchEEGLlava, T


CHANNELS = ["PO3", "POz", "PO4", "O1", "Oz", "O2"]
HEALTHY_REFERENCE_POWER = 5.84
ANSWER_H = (
    "The occipital EEG shows robust SSVEP responses, with peak activation at Oz "
    "(reference), within the normal reference range. The strong visual evoked potentials "
    "across all occipital channels reflect intact primary visual cortex function and healthy "
    "retinal ganglion cell activity. No signs of glaucomatous visual pathway damage are detected."
)
ANSWER_G = (
    "The occipital EEG shows markedly attenuated SSVEP activity, with peak residual power at "
    "Oz (reference), approximately 88% below the healthy reference level. The diminished visual "
    "evoked response across all occipital channels is consistent with retinal ganglion cell loss "
    "and optic nerve degeneration characteristic of glaucoma. This subject is recommended for "
    "further ophthalmological examination including visual field testing and optical coherence "
    "tomography."
)


def ssvep_features(eeg: np.ndarray) -> tuple[str, float, float]:
    signal = eeg.reshape(6, -1) * 100.0
    powers = []
    for channel in range(6):
        spectrum = np.abs(np.fft.rfft(signal[channel]))
        frequencies = np.fft.rfftfreq(signal.shape[1], d=1.0 / 200.0)
        selected = (frequencies >= 8.0) & (frequencies <= 11.8)
        psd = spectrum ** 2 / (200.0 * signal.shape[1])
        powers.append(float(psd[selected].mean()))
    index = int(np.argmax(powers))
    reduction = max(0.0, (HEALTHY_REFERENCE_POWER - powers[index]) /
                    HEALTHY_REFERENCE_POWER * 100.0)
    return CHANNELS[index], powers[index], reduction


def rich_answer(eeg: np.ndarray, label: int) -> str:
    channel, power, reduction = ssvep_features(eeg)
    if label == 1:
        return (
            f"The occipital EEG shows markedly attenuated SSVEP activity, with peak residual "
            f"power at {channel} ({power:.2f} μV²/Hz in the 8–12 Hz band), approximately "
            f"{reduction:.0f}% below the healthy reference level. The diminished visual evoked "
            f"response across all occipital channels is consistent with retinal ganglion cell "
            f"loss and optic nerve degeneration characteristic of glaucoma. This subject is "
            f"recommended for further ophthalmological examination including visual field "
            f"testing and optical coherence tomography."
        )
    excess = max(0.0, (power - HEALTHY_REFERENCE_POWER) /
                 HEALTHY_REFERENCE_POWER * 100.0)
    level = f"{excess:.0f}% above" if excess > 5 else "within"
    return (
        f"The occipital EEG shows robust SSVEP responses, with peak activation at {channel} "
        f"({power:.2f} μV²/Hz in the 8–12 Hz band), {level} the normal reference range. "
        f"The strong visual evoked potentials across all occipital channels reflect intact "
        f"primary visual cortex function and healthy retinal ganglion cell activity. No signs "
        f"of glaucomatous visual pathway damage are detected."
    )


class RichDataset(Dataset):
    def __init__(self, keys: list[str], tokenizer, max_length: int, segment_weights=None):
        self.keys = keys
        self.cache = T.load_lmdb_cache(T.DATA_DIR, keys)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segment_weights = segment_weights or {}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        pair = self.cache[key]
        eeg = pair["sample"].astype(np.float32) / 100.0
        label = int(pair["label"])
        prompt = T.STAGE2_TEMPLATE + "\n"
        return {
            "eeg": torch.tensor(eeg, dtype=torch.float32),
            "text": prompt + rich_answer(eeg, label) + self.tokenizer.eos_token,
            "prompt": prompt,
            "label": label,
            "sw": self.segment_weights.get(key, 1.0),
        }


def collate_rich(batch: list[dict], tokenizer, max_length: int) -> dict:
    full = tokenizer([item["text"] for item in batch], padding="max_length",
                     truncation=True, max_length=max_length, return_tensors="pt")
    prompts = tokenizer([item["prompt"] for item in batch], padding=False,
                        truncation=True, max_length=max_length)
    labels = full.input_ids.clone()
    for index, prompt_ids in enumerate(prompts.input_ids):
        labels[index, :len(prompt_ids)] = -100
    labels[full.attention_mask == 0] = -100
    return {
        "eeg": torch.stack([item["eeg"] for item in batch]),
        "input_ids": full.input_ids,
        "attention_mask": full.attention_mask,
        "labels": labels,
        "label": torch.tensor([item["label"] for item in batch]),
        "sw": torch.tensor([item["sw"] for item in batch], dtype=torch.float32),
    }


@torch.no_grad()
def loss_evaluate(model, keys: list[str], device: torch.device,
                  max_length: int) -> dict:
    model.eval()
    cache = T.load_lmdb_cache(T.DATA_DIR, keys)
    prompt = T.STAGE2_TEMPLATE + "\n"
    prompt_length = model.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
    candidates = []
    for answer in (ANSWER_H, ANSWER_G):
        tokenized = model.tokenizer(
            prompt + answer + model.tokenizer.eos_token, return_tensors="pt",
            truncation=True, max_length=max_length, add_special_tokens=False)
        labels = tokenized.input_ids.clone()
        labels[:, :prompt_length] = -100
        candidates.append((tokenized.input_ids.to(device),
                           tokenized.attention_mask.to(device), labels.to(device)))
    y, scores = [], []
    for index, key in enumerate(keys):
        pair = cache[key]
        eeg = torch.tensor(pair["sample"] / 100.0, dtype=torch.float32,
                           device=device).unsqueeze(0)
        losses = [model(eeg, ids, attention, labels).loss.item()
                  for ids, attention, labels in candidates]
        scores.append(float(F.softmax(torch.tensor([-losses[0], -losses[1]]), dim=0)[1]))
        y.append(int(pair["label"]))
        if (index + 1) % 200 == 0:
            print(f"  RichLossEval {index + 1}/{len(keys)}", flush=True)
    y_array = np.asarray(y)
    score_array = np.asarray(scores)
    pred = (score_array > 0.5).astype(int)
    return {
        "bacc": float(balanced_accuracy_score(y_array, pred)),
        "auc": float(roc_auc_score(y_array, score_array)),
        "cm": confusion_matrix(y_array, pred).tolist(),
        "scores": scores, "labels": y,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    result_path = args.output_dir / "logs" / "rich_result.json"
    checkpoint_path = args.output_dir / "ckpt" / "rich_best.pth"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite {args.output_dir}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    T.setup_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}")
    torch.cuda.set_device(args.cuda)
    fold = json.loads(args.split.read_text())
    if fold.get("split_unit") != "eye":
        raise AssertionError("expected Protocol 1 eye split")
    aux_path = args.component_dir / "fold_0_aux.pth"
    main_path = args.component_dir / "fold_0_mainenc.pth"
    aux = AuxTransformer()
    aux.load_state_dict(torch.load(aux_path, map_location="cpu"))

    def build(freeze_llm: bool):
        return DualBranchEEGLlava(
            llm_path=T.LLM_PATH, eeg_encoder_weights=str(main_path),
            freeze_eeg_encoder=True, freeze_llm=freeze_llm,
            eeg_dim=200, num_channels=6, num_patches=5,
            aux_encoder=aux, aux_trainable=False, use_spectral=True,
        ).to(device)

    # Stage 1 is byte-equivalent in definition to the final simple-report run.
    print("[Stage 1] final 55-token alignment", flush=True)
    model = build(True)
    collate_stage1 = partial(T.collate_fn, tokenizer=model.tokenizer, max_length=128)
    stage1 = DataLoader(
        T.FoldDataset(T.DATA_DIR, fold["keys"]["train"], model.tokenizer, 128, stage=1),
        batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_stage1)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(stage1) * args.stage1_epochs, eta_min=1e-6)
    for epoch in range(args.stage1_epochs):
        model.train()
        losses = []
        for batch in tqdm(stage1, desc=f"Rich S1 {epoch + 1}", mininterval=30):
            optimizer.zero_grad()
            output = model(batch["eeg"].to(device), batch["input_ids"].to(device),
                           batch["attention_mask"].to(device), batch["labels"].to(device))
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(output.loss.item())
        print(f"Rich S1 {epoch + 1}: loss={np.mean(losses):.5f}", flush=True)
    projector_state = copy.deepcopy(model.projector.state_dict())
    auxiliary_projector_state = copy.deepcopy(model.aux_proj.state_dict())
    del model, stage1
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("[Stage 2] rich report task tuning", flush=True)
    model = build(False)
    model.projector.load_state_dict(projector_state)
    model.aux_proj.load_state_dict(auxiliary_projector_state)
    train_keys = fold["keys"]["train"]
    cache = T.load_lmdb_cache(T.DATA_DIR, train_keys)
    subject_of = lambda key: "_".join(key.split("_")[:3])
    subject_labels = {}
    for key in train_keys:
        subject_labels.setdefault(subject_of(key), int(cache[key]["label"]))
    glaucoma = sum(subject_labels.values())
    healthy = len(subject_labels) - glaucoma
    healthy_weight = glaucoma / healthy
    counts = Counter(subject_of(key) for key in train_keys)
    segment_weights = {key: 1.0 / counts[subject_of(key)] for key in train_keys}
    mean_weight = float(np.mean(list(segment_weights.values())))
    segment_weights = {key: value / mean_weight for key, value in segment_weights.items()}
    collate_stage2 = partial(collate_rich, tokenizer=model.tokenizer,
                             max_length=args.max_length)
    stage2 = DataLoader(
        RichDataset(train_keys, model.tokenizer, args.max_length, segment_weights),
        batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_stage2)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(stage2) * args.stage2_epochs, eta_min=1e-6)
    best_bacc, best_epoch, best_state = -1.0, 0, None
    validation_curve = []
    for epoch in range(args.stage2_epochs):
        model.train()
        losses = []
        for batch in tqdm(stage2, desc=f"Rich S2 {epoch + 1}", mininterval=30):
            raw = batch["label"]
            weights = torch.where(raw == 0, torch.tensor(float(healthy_weight)),
                                  torch.tensor(1.0)) * batch["sw"]
            optimizer.zero_grad()
            output = model(batch["eeg"].to(device), batch["input_ids"].to(device),
                           batch["attention_mask"].to(device), batch["labels"].to(device),
                           weights)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(output.loss.item())
        print(f"Rich S2 {epoch + 1}: loss={np.mean(losses):.5f}", flush=True)
        if (epoch + 1) % 2 == 0 or epoch == args.stage2_epochs - 1:
            validation = loss_evaluate(
                model, fold["keys"]["val"], device, args.max_length)
            validation_curve.append({"epoch": epoch + 1, "bacc": validation["bacc"],
                                     "auc": validation["auc"], "cm": validation["cm"]})
            print(f"Rich val: {validation_curve[-1]}", flush=True)
            if validation["bacc"] > best_bacc:
                best_bacc, best_epoch = validation["bacc"], epoch + 1
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no rich checkpoint selected on validation")
    model.load_state_dict(best_state)
    test = loss_evaluate(model, fold["keys"]["test"], device, args.max_length)
    checkpoint = {
        "checkpoint_format": "dual_branch_rich_complete_v1",
        "model_state_dict": model.state_dict(),
        "seed": args.seed, "best_val_epoch": best_epoch,
        "split": str(args.split.resolve()),
        "components": {"aux": str(aux_path.resolve()), "main_encoder": str(main_path.resolve())},
        "tokens": 55, "max_length": args.max_length,
    }
    temporary = checkpoint_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    digest = sha256_file(checkpoint_path)
    checkpoint_path.with_suffix(".pth.sha256").write_text(
        f"{digest}  {checkpoint_path.name}\n")
    result = {
        "experiment": "Fig. 20 final 55-token Rich Report",
        "seed": args.seed, "best_val_bacc": best_bacc,
        "best_val_epoch": best_epoch, "validation_curve": validation_curve,
        "test_bacc": test["bacc"], "test_auc": test["auc"], "test_cm": test["cm"],
        "test_scores": test["scores"], "test_labels": test["labels"],
        "simple_report_comparator": "locked final seed-42 Protocol-1 run",
        "selection": "validation BAcc only; held-out test scored once after selection",
        "class_weight": {"mode": "subject", "healthy_weight": healthy_weight},
        "subject_balanced_segments": True,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": digest,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "best_val_bacc", "best_val_epoch", "test_bacc", "test_auc")}, indent=2))
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()

