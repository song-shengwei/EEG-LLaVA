#!/usr/bin/env python3
"""
Dual-branch EEG encoding: the frozen foundation representation plus a task-trained branch,
both projected into the LLM's embedding space and concatenated as EEG tokens.

Why: under one identical subject-disjoint protocol, a 3-layer 171K-parameter Transformer trained
end-to-end from raw signal reaches participant-level AUC 0.831, while the frozen CBraMod
representation feeding EEG-LLaVA reaches 0.757. Score-level fusion of the two recovers 0.804.
Fusing at the *token* level instead lets the language model see both representations directly,
which score averaging cannot do.

Nothing about the existing architecture is removed:
  - the CBraMod branch, its 30 tokens and the projector are untouched, so the "30 EEG tokens ≈
    CLIP image patches" framing survives;
  - the external Transformer baseline in Table 1 stays a baseline — this branch is trained
    separately, per fold, and is a component of our model rather than a comparison method;
  - the concatenation reuses the exact pattern already used for region/VFI tokens
    (glaucoma_llava.py:440,446), and forward() reads the token count dynamically
    (`n_eeg = eeg_emb.shape[0]`, marked "勿写死"), so no change is needed there.

Leakage discipline: the auxiliary branch is trained on that fold's TRAIN participants only, with
checkpoint selection on that fold's VAL participants. It never sees test.
"""
import copy
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import os; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'llamaG')))  # [release] was an absolute server path
from model.glaucoma_llava import EEGLlavaModel


class AuxTransformer(nn.Module):
    """The E12 baseline Transformer, exposed so its token sequence can be reused.

    Identical definition to Exp/E12_subject_cv_baselines/run_cv_baselines.py — 3 layers,
    d_model 64, 50-sample patches — so its standalone behaviour matches the reported baseline.
    `features()` returns the full token sequence (CLS + 20 patches) rather than only the CLS,
    giving the projector something with temporal structure to work with.
    """

    def __init__(self, n_ch=6, seq_len=1000, d_model=64, nhead=4, num_layers=3,
                 dropout=0.1, patch_size=50):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        n_patches = seq_len // patch_size
        self.n_tokens = n_patches + 1
        self.patch_proj = nn.Linear(n_ch * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def features(self, x):
        """x: (B, 6, 1000) or (B, 6, 5, 200) -> (B, n_tokens, d_model)"""
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        B, C, T = x.shape
        h = x.reshape(B, C, T // self.patch_size, self.patch_size)
        h = h.permute(0, 2, 1, 3).reshape(B, T // self.patch_size, -1)
        h = self.patch_proj(h)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1) + self.pos_embed
        return self.norm(self.encoder(h))

    def forward(self, x):
        return self.classifier(self.features(x)[:, 0])


class SpectralBranch(nn.Module):
    """A third EEG token stream built from the SSVEP spectral descriptors.

    Motivation: on identical folds, the Random-Forest baseline — which sees ONLY handcrafted
    spectral features — reaches participant-level AUC 0.818, within noise of the 171K Transformer's
    0.831. So most of the discriminative signal is spectral. The paper already computes these
    descriptors (peak power, relative reduction, inter-channel CV, asymmetry) — but only to fill
    the human-readable report, never as model input. This branch feeds them to the language model
    as tokens, which also gives a direct answer to the 1AE's question about what the LLM actually
    consumes.

    Features per channel (13), matching Exp/E12's extract_features so the representation is the
    one the RF baseline demonstrated: mean, std, peak-to-peak, mean|x|, RMS, energy, six band
    means (delta/theta/alpha/beta/gamma/stimulus 8-11.8 Hz), and spectral entropy. 6 channels
    -> 78 dims -> MLP -> n_tokens tokens in the LLM embedding space.

    Computed inside the model with torch.fft, so nothing has to be precomputed or cached and the
    branch stays self-contained.
    """

    BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50), (8, 11.8))
    FS = 200.0

    def __init__(self, llm_dim=1024, n_tokens=4, hidden=256, n_ch=6):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = n_ch * (6 + len(self.BANDS) + 1)
        self.norm = nn.LayerNorm(self.dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_tokens * llm_dim),
        )
        self.llm_dim = llm_dim

    @torch.no_grad()
    def descriptors(self, x):
        """x: (B, 6, 5, 200) or (B, 6, 1000) -> (B, 78)"""
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        B, C, T = x.shape
        feats = [x.mean(-1), x.std(-1), x.amax(-1) - x.amin(-1),
                 x.abs().mean(-1), x.pow(2).mean(-1).sqrt(), x.pow(2).sum(-1)]
        mag = torch.fft.rfft(x.float(), dim=-1).abs()                 # (B, C, T//2+1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.FS).to(x.device)
        for lo, hi in self.BANDS:
            m = (freqs >= lo) & (freqs < hi)
            feats.append(mag[..., m].mean(-1) if m.any()
                         else torch.zeros(B, C, device=x.device))
        psd = mag.pow(2)
        psd = psd / (psd.sum(-1, keepdim=True) + 1e-8)
        feats.append(-(psd * (psd + 1e-8).log()).sum(-1))             # spectral entropy
        return torch.stack(feats, dim=-1).reshape(B, -1)              # (B, C*13)

    def forward(self, x):
        d = self.descriptors(x).to(self.norm.weight.dtype)
        return self.net(self.norm(d)).view(x.shape[0], self.n_tokens, self.llm_dim)


class _RawDS(Dataset):
    def __init__(self, cache, keys):
        self.cache, self.keys = cache, keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        p = self.cache[self.keys[i]]
        return (torch.tensor(p['sample'].reshape(6, -1) / 100.0, dtype=torch.float32),
                torch.tensor(float(p['label'])))


def train_aux(cache, tr_keys, va_keys, device, epochs=50, lr=1e-3, bs=64, seed=3407):
    """Train the auxiliary branch on THIS fold's train split; select on THIS fold's val split.

    Same recipe as the E12 baseline (50 ep, lr 1e-3, bs 64, AdamW wd 1e-4, cosine, pos_weight,
    best val balanced accuracy) so the branch reaches the quality the baseline demonstrated.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(_RawDS(cache, tr_keys), batch_size=bs, shuffle=True, num_workers=0)
    va = DataLoader(_RawDS(cache, va_keys), batch_size=bs, shuffle=False, num_workers=0)
    m = AuxTransformer().to(device)
    lab = [int(cache[k]['label']) for k in tr_keys]
    pw = torch.tensor([lab.count(0) / max(lab.count(1), 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr), eta_min=1e-5)
    best, best_state, best_auc = -1, None, 0.0
    for _ in range(epochs):
        m.train()
        for x, y in tr:
            opt.zero_grad()
            crit(m(x.to(device)).squeeze(1), y.to(device)).backward()
            opt.step(); sch.step()
        m.eval(); vp, vs, vt = [], [], []
        with torch.no_grad():
            for x, y in va:
                pr = torch.sigmoid(m(x.to(device)).squeeze(1)).cpu().numpy()
                vp += (pr > 0.5).astype(int).tolist(); vs += pr.tolist()
                vt += y.int().tolist()
        v = balanced_accuracy_score(vt, vp)
        if v > best:
            best = v
            best_auc = float(roc_auc_score(vt, vs))
            best_state = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, {'val_bacc': float(best), 'val_auc': best_auc}


class DualBranchEEGLlava(EEGLlavaModel):
    """EEGLlavaModel + a second EEG token stream from the task-trained branch.

    encode_eeg() is the only override: it appends the auxiliary tokens after the CBraMod tokens,
    the same way region/VFI tokens are appended upstream. forward() needs no change because it
    reads the token count from the tensor.
    """

    def __init__(self, *args, aux_encoder=None, aux_trainable=False,
                 use_spectral=False, n_spectral_tokens=4, **kwargs):
        super().__init__(*args, **kwargs)
        assert aux_encoder is not None, "aux_encoder required"
        self.aux_encoder = aux_encoder
        self.aux_trainable = aux_trainable
        for p in self.aux_encoder.parameters():
            p.requires_grad = aux_trainable
        llm_dim = self.projector.proj[0].out_features if hasattr(self.projector, 'proj') \
            else self.llm.get_input_embeddings().weight.shape[1]
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_encoder.d_model, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.spectral = SpectralBranch(llm_dim=llm_dim, n_tokens=n_spectral_tokens) \
            if use_spectral else None
        n_tok = 30 + aux_encoder.n_tokens + (n_spectral_tokens if use_spectral else 0)
        print(f"[DualBranch] tokens = 30 (CBraMod) + {aux_encoder.n_tokens} (aux)"
              + (f" + {n_spectral_tokens} (spectral)" if use_spectral else "")
              + f" = {n_tok}; aux_trainable={aux_trainable}; "
              f"extra trainable params="
              f"{sum(p.numel() for p in self.aux_proj.parameters()) + (sum(p.numel() for p in self.spectral.parameters()) if use_spectral else 0):,}")

    def encode_eeg(self, eeg_signal):
        eeg_embeds, ring_scores, vfi_pred = super().encode_eeg(eeg_signal)
        if self.aux_trainable:
            feats = self.aux_encoder.features(eeg_signal)
        else:
            with torch.no_grad():
                feats = self.aux_encoder.features(eeg_signal)
        parts = [eeg_embeds, self.aux_proj(feats.to(eeg_embeds.dtype))]
        if self.spectral is not None:
            parts.append(self.spectral(eeg_signal).to(eeg_embeds.dtype))
        return torch.cat(parts, dim=1), ring_scores, vfi_pred
