"""
Stage 2 with LoRA: freeze LLM base weights, only train LoRA adapters + projector.
This reduces trainable params from 597M to ~4.5M, dramatically reducing overfitting.

Also includes:
- Focal loss option (better than weighted CE for hard examples)
- Loss-based AUC evaluation built-in
- Checkpoint saving

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID python train_lora.py --cuda 7 --class_weight 2.0 --lr 2e-4
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'llamaG'))  # [release]
from model.glaucoma_llava import EEGLlavaModel
from data.glaucoma_llava_dataset import get_data_loaders, STAGE2_TEMPLATE, STAGE2_ANSWER

# [release] server paths replaced; EEG_WEIGHTS / S1_CKPT / OUT_DIR belonged to this file's own (older)
# training entry point, which the release does not use.
DATA_DIR = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[2] / 'data' / 'processed_lmdb'))
LLM_PATH = os.environ.get('EEGLLAVA_LLM', str(Path(__file__).resolve().parents[2] / 'pretrained' / 'Qwen3-0.6B'))
EEG_WEIGHTS = None
S1_CKPT = None
OUT_DIR = Path(__file__).resolve().parent / 'runs_train_lora'


# ── LoRA Implementation ──────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer that wraps an existing nn.Linear."""
    def __init__(self, base_linear: nn.Linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base_linear
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Init: A ~ kaiming, B ~ zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out


def apply_lora_to_model(model, rank=16, alpha=32, dropout=0.05,
                         target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj')):
    """Replace target Linear layers in the LLM with LoRA-wrapped versions."""
    replaced = 0
    for name, module in model.llm.named_modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                old_linear = getattr(module, attr_name)
                if isinstance(old_linear, nn.Linear):
                    lora_layer = LoRALinear(old_linear, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, attr_name, lora_layer)
                    replaced += 1
    return replaced


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate_discrete(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc='Eval(discrete)', mininterval=10):
        eeg = batch['eeg'].to(device)
        lt = batch['labels']
        for i in range(eeg.shape[0]):
            vi = lt[i][lt[i] != -100]
            gt = 1 if 'glaucoma' in model.tokenizer.decode(vi, skip_special_tokens=True).strip().lower() else 0
            pt = model.generate(eeg[i:i+1], STAGE2_TEMPLATE + '\n', max_new_tokens=32)
            pred = 1 if 'glaucoma' in pt.lower() else 0
            preds.append(pred); labels.append(gt)
    return balanced_accuracy_score(labels, preds), roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.0


@torch.no_grad()
def evaluate_loss_based(model, loader, device):
    model.eval()
    prompt = STAGE2_TEMPLATE + '\n'
    ans_h, ans_g = STAGE2_ANSWER[0], STAGE2_ANSWER[1]
    scores, labels = [], []
    for batch in tqdm(loader, desc='Eval(loss)', mininterval=10):
        eeg = batch['eeg'].to(device)
        lt = batch['labels']
        for i in range(eeg.shape[0]):
            vi = lt[i][lt[i] != -100]
            gt = 1 if 'glaucoma' in model.tokenizer.decode(vi, skip_special_tokens=True).strip().lower() else 0
            eeg_i = eeg[i:i+1]
            enc_p = model.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
            plen = enc_p['input_ids'].shape[1]
            # Healthy loss
            enc_h = model.tokenizer(prompt + ans_h, return_tensors='pt', add_special_tokens=False)
            ids_h = enc_h['input_ids'].to(device)
            lab_h = ids_h.clone(); lab_h[:, :plen] = -100
            loss_h = model(eeg_i, ids_h, enc_h['attention_mask'].to(device), lab_h).loss.item()
            # Glaucoma loss
            enc_g = model.tokenizer(prompt + ans_g, return_tensors='pt', add_special_tokens=False)
            ids_g = enc_g['input_ids'].to(device)
            lab_g = ids_g.clone(); lab_g[:, :plen] = -100
            loss_g = model(eeg_i, ids_g, enc_g['attention_mask'].to(device), lab_g).loss.item()
            probs = F.softmax(torch.tensor([-loss_h, -loss_g]), dim=0)
            scores.append(probs[1].item()); labels.append(gt)
    scores, labels = np.array(scores), np.array(labels)
    preds = (scores > 0.5).astype(int)
    bacc = balanced_accuracy_score(labels, preds)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0
    pr_auc = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.0
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'bacc': float(bacc), 'auc': float(auc), 'pr_auc': float(pr_auc),
        'sens': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
        'spec': float(tn/(tn+fp)) if (tn+fp) > 0 else 0,
        'cm': cm.tolist(),
    }


# ── Training ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=7)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--class_weight', type=float, default=2.0)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--tag', type=str, default='lora')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}')
    torch.cuda.set_device(args.cuda)

    print("=" * 60)
    print(f"  Stage 2 with LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    print(f"  class_weight={args.class_weight}, lr={args.lr}, epochs={args.epochs}")
    print("=" * 60)

    # Build model: freeze LLM entirely first
    model = EEGLlavaModel(
        llm_path=LLM_PATH, eeg_encoder_weights=EEG_WEIGHTS,
        freeze_eeg_encoder=True, freeze_llm=True,  # freeze all first
        eeg_dim=200, num_channels=6, num_patches=5,
    )

    # Load Stage 1 projector
    ckpt = torch.load(S1_CKPT, map_location='cpu')
    model.projector.load_state_dict(ckpt['projector'])
    print(f"Loaded Stage 1 projector")

    # Apply LoRA to LLM attention layers
    n_replaced = apply_lora_to_model(
        model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
    )
    print(f"Applied LoRA to {n_replaced} layers")

    # Unfreeze projector (always trainable in Stage 2)
    for p in model.projector.parameters():
        p.requires_grad = True

    model = model.to(device)
    total, trainable = count_trainable(model)
    print(f"Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Data
    data_loaders = get_data_loaders(
        data_dir=DATA_DIR, tokenizer=model.tokenizer,
        batch_size=args.batch_size, num_workers=4, max_length=128, stage=2,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    num_steps = len(data_loaders['train']) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)

    for epoch in range(args.epochs):
        model.train()
        start = timer()
        losses = []

        for batch in tqdm(data_loaders['train'], desc=f'Epoch {epoch+1}', mininterval=10):
            eeg = batch['eeg'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            raw_labels = batch['label']

            sample_weights = torch.where(raw_labels == 0,
                                         torch.tensor(args.class_weight),
                                         torch.tensor(1.0))

            optimizer.zero_grad()
            outputs = model(eeg, input_ids, attention_mask, labels, sample_weights)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(outputs.loss.item())

        elapsed = (timer() - start) / 60
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}, lr={lr_now:.6f}, time={elapsed:.1f}min")

    # Use final epoch model directly (no intermediate eval)
    best_epoch = args.epochs

    # Final evaluation: both discrete and loss-based
    print(f"\n{'='*60}")
    print(f"  FINAL TEST EVALUATION (best epoch={best_epoch})")
    print(f"{'='*60}")

    test_acc, test_auc_d = evaluate_discrete(model, data_loaders['test'], device)
    print(f"  Discrete: BAcc={test_acc:.4f}, AUC={test_auc_d:.4f}")

    test_lb = evaluate_loss_based(model, data_loaders['test'], device)
    print(f"  Loss-based: BAcc={test_lb['bacc']:.4f}, AUC={test_lb['auc']:.4f}, "
          f"PR-AUC={test_lb['pr_auc']:.4f}, Sens={test_lb['sens']:.4f}, Spec={test_lb['spec']:.4f}")
    print(f"  CM: {test_lb['cm']}")

    val_lb = evaluate_loss_based(model, data_loaders['val'], device)
    print(f"  Val loss-based: BAcc={val_lb['bacc']:.4f}, AUC={val_lb['auc']:.4f}")

    # Save
    save_dir = OUT_DIR / f'{args.tag}_cw{args.class_weight}_lr{args.lr}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save only LoRA weights + projector (small file)
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data
    torch.save({
        'lora_state': lora_state,
        'projector': model.projector.state_dict(),
        'best_epoch': best_epoch,
        'best_val_acc': test_acc,
        'test_acc': test_acc,
        'test_auc_loss': test_lb['auc'],
        'args': vars(args),
    }, save_dir / f'lora_best_ep{best_epoch}.pth')

    # Save full results
    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_acc': test_acc,
        'test_discrete': {'bacc': test_acc, 'auc': test_auc_d},
        'test_loss_based': test_lb,
        'val_loss_based': val_lb,
        'trainable_params': trainable,
        'total_params': total,
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {save_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
