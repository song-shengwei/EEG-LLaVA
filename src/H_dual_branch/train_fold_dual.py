"""
H: EEG-LLaVA with dual-branch EEG encoding — one fold.

Pipeline per fold, all on that fold's TRAIN participants, selected on that fold's VAL:
  Phase A  train the auxiliary Transformer branch (E12 baseline recipe: 50 ep, lr 1e-3, bs 64)
  Stage 1  train projector + aux_proj, LLM frozen        (50 ep, lr 1e-3, bs 8)
  Stage 2  train projector + aux_proj + LLM              (20 ep, lr 2e-5, bs 8)
  Test     discrete + loss-based scoring, per-segment scores dumped for voting

The CBraMod branch is the label-free public foundation checkpoint, frozen, unchanged. The
auxiliary branch supplies 21 extra EEG tokens. Everything else matches C_5fold_clean_encoder so
the only variable versus that run is the second branch.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID python train_fold_dual.py --fold 0 --cuda 3
"""
import argparse, copy, hashlib, json, os, sys, time
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent / 'C_5fold_clean_encoder'
sys.path.insert(0, str(C_DIR)); sys.path.insert(0, str(HERE))
import train_fold_clean as T
from dual_branch import AuxTransformer, train_aux, DualBranchEEGLlava

CBRAMOD = os.environ.get('EEGLLAVA_CBRAMOD', str(HERE.parents[1] / 'pretrained' / 'cbramod' / 'pretrained_weights.pth'))  # [release] was an absolute server path
LOG_DIR = HERE / 'logs'; LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = HERE / 'ckpt'; CKPT_DIR.mkdir(parents=True, exist_ok=True)


class WeightedFoldDataset(T.FoldDataset):
    """FoldDataset that also carries a per-segment loss weight (for subject balancing)."""

    def __init__(self, *a, seg_w=None, **kw):
        super().__init__(*a, **kw)
        self.seg_w = seg_w

    def __getitem__(self, i):
        d = super().__getitem__(i)
        d['sw'] = self.seg_w.get(self.keys[i], 1.0) if self.seg_w else 1.0
        return d


def collate_w(batch, tokenizer, max_length):
    out = T.collate_fn(batch, tokenizer=tokenizer, max_length=max_length)
    out['sw'] = torch.tensor([b.get('sw', 1.0) for b in batch], dtype=torch.float32)
    return out


def train_stage_w(model, loader, optimizer, scheduler, device, epochs, healthy_weight,
                  clip_value=1.0, run_eval_fn=None, eval_every=2, select_metric='acc',
                  history=None):
    """T.train_stage, but the class weight is multiplied by the per-segment weight.

    `select_metric` chooses which validation quantity drives checkpoint selection. The
    default 'acc' reproduces the original behaviour exactly: selection on validation
    accuracy at the fixed 0.5 threshold, with the validation AUC computed in the same call
    and then discarded. 'auc' selects on validation AUC instead, which uses the same
    predictions but keeps their ranking information. Either way the returned accuracy is
    the accuracy at the selected epoch, so `best_val_acc` keeps its meaning downstream.

    `history` optionally collects one record per evaluation, so the selection trajectory
    can be audited without re-running training. Validation only; test is never touched.
    """
    if select_metric not in ('acc', 'auc'):
        raise ValueError(f"select_metric must be 'acc' or 'auc', got {select_metric!r}")
    best_acc, best_epoch, best_states = 0, 0, None
    best_score = -1.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(epochs):
        model.train(); losses = []; start = timer()
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}', mininterval=30):
            raw = batch['label']
            w = torch.where(raw == 0, torch.tensor(float(healthy_weight)), torch.tensor(1.0))
            w = w * batch.get('sw', torch.ones_like(w))
            optimizer.zero_grad()
            out = model(batch['eeg'].to(device), batch['input_ids'].to(device),
                        batch['attention_mask'].to(device), batch['labels'].to(device), w)
            out.loss.backward()
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip_value)
            optimizer.step(); scheduler.step(); losses.append(out.loss.item())
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.6f}, time={(timer()-start)/60:.1f}min")
        if run_eval_fn and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            acc, auc, cm = run_eval_fn()
            print(f"Val acc={acc:.5f}, auc={auc:.5f}\n{cm}")
            if history is not None:
                history.append({'epoch': epoch + 1, 'val_acc': float(acc),
                                'val_auc': float(auc)})
            score = acc if select_metric == 'acc' else auc
            if score > best_score:
                best_score, best_acc, best_epoch = score, acc, epoch + 1
                best_states = copy.deepcopy(model.state_dict())
                print(f"New best! {select_metric}={score:.5f} "
                      f"(acc={acc:.5f}, auc={auc:.5f}) at epoch {best_epoch}")
    return best_acc, best_epoch, best_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--cuda', type=int, default=3)
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--aux_epochs', type=int, default=50)
    ap.add_argument('--aux_lr', type=float, default=1e-3)
    ap.add_argument('--stage1_epochs', type=int, default=50)
    ap.add_argument('--stage2_epochs', type=int, default=20)
    ap.add_argument('--lr1', type=float, default=1e-3)
    ap.add_argument('--lr2', type=float, default=2e-5)
    ap.add_argument('--select_metric', choices=['acc', 'auc'], default='acc',
                    help='Stage-2/3 checkpoint selection criterion on the validation '
                         'split. Default "acc" reproduces the locked runs exactly. "auc" '
                         'selects on validation AUC, which is computed either way but is '
                         'discarded under "acc". Validation only; test is never used.')
    ap.add_argument('--stage2_eval_every', type=int, default=2,
                    help='Validate every N epochs during Stage 2/3. Default 2 reproduces '
                         'the locked runs (10 candidate checkpoints over 20 epochs).')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_length', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--aux_trainable', action='store_true',
                    help='let Stage 2 gradients reach the auxiliary branch too')
    ap.add_argument('--unfreeze_enc_last_n', type=int, default=0,
                    help='unfreeze the last N main EEG-encoder layers during Stage 2')
    ap.add_argument('--use_spectral', action='store_true',
                    help='Add the SSVEP spectral-descriptor branch. Motivated by the RF baseline '
                         'reaching participant AUC 0.818 from handcrafted spectral features '
                         'alone — most of the signal is spectral, and the paper already computes '
                         'these descriptors for the report without feeding them to the model.')
    ap.add_argument('--class_weight', default='segment',
                    choices=['segment', 'subject', 'paper'],
                    help="Weight applied to healthy samples in Stage 2. 'segment' = this fold's "
                         "glaucoma/healthy SEGMENT ratio (~1.14, what C_5fold used); 'subject' = "
                         "the PARTICIPANT ratio (~1.57), which matches the unit we evaluate on; "
                         "'paper' = 2.0, the value the submitted manuscript states in Section 4 "
                         "(the code has always used ~1.16, so paper and code disagree today).")
    ap.add_argument('--subject_balanced', action='store_true',
                    help='Also weight each segment by 1/(number of segments that participant '
                         'contributes), renormalised to mean 1. Segment counts range 28-152 per '
                         'participant (5.4x), so without this the heavily-recorded participants '
                         'dominate a loss whose evaluation unit is the participant.')
    ap.add_argument('--phase0', choices=['none', 'c2_hi_lr', 'c4_scratch'], default='none',
                    help="Main-branch encoder. 'none' freezes the public CBraMod checkpoint "
                         "as-is (what H_dual_branch ran). D_5fold_cbramod showed that an "
                         "unadapted encoder does poorly through the projector+LLM path "
                         "(val acc 0.40-0.55 vs 0.53-0.65 for an adapted one) even though a "
                         "plain MLP head does fine on the same features — so the LLaVA path "
                         "needs the encoder adapted. 'c2_hi_lr' = 30 ep / lr 5e-4 from the "
                         "foundation checkpoint (keeps the CBraMod-transfer framing); "
                         "'c4_scratch' = 100 ep / lr 5e-4 random init (won on val, but drops "
                         "the foundation and would require rewriting the intro's rationale). "
                         "Both are trained on THIS fold's train split only.")
    ap.add_argument('--splits_dir', type=str, default=None,
                    help='Fold definitions. Default = original E07 splits; pass splits_v2/ for '
                         'the corrected train/val partition (test sets identical either way).')
    ap.add_argument('--split_unit', choices=['subject', 'eye'], default='subject',
                    help="Disjoint unit enforced by the split audit. Protocol 2 uses 'subject'; "
                         "Protocol 1 uses 'eye' and explicitly permits the other eye of the "
                         "same participant to occur in another split.")
    ap.add_argument('--out_dir', type=str, default=None,
                    help="Where logs/ and ckpt/ go. Default = this script's directory.")
    ap.add_argument('--component_ckpt_dir', type=str, default=None,
                    help='Optional read-only directory containing fold_N_aux.pth and '
                         'fold_N_mainenc.pth. This lets a reproducibility rebuild reuse the '
                         'exact locked phase-A/phase-0 components while writing all new outputs '
                         'to --out_dir.')
    ap.add_argument('--complete_checkpoint', action='store_true',
                    help='Save a versioned full model state (including the EEG encoders and '
                         'spectral branch), configuration, and a SHA-256 sidecar. Required for '
                         'new R0/R1 experiments; legacy partial checkpoint behaviour is kept '
                         'only for backwards compatibility.')
    args = ap.parse_args()
    if not 0 <= args.unfreeze_enc_last_n <= 12:
        ap.error('--unfreeze_enc_last_n must be between 0 and 12')

    T.setup_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}'); torch.cuda.set_device(args.cuda)

    global LOG_DIR, CKPT_DIR
    splits_dir = Path(args.splits_dir) if args.splits_dir else T.SPLITS_DIR
    if args.out_dir:
        LOG_DIR = Path(args.out_dir) / 'logs'; CKPT_DIR = Path(args.out_dir) / 'ckpt'
        LOG_DIR.mkdir(parents=True, exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"splits: {splits_dir}\noutputs: {LOG_DIR.parent}")

    result_path = LOG_DIR / f'fold_{args.fold}_result.json'
    best_path = CKPT_DIR / f'fold_{args.fold}_best.pth'
    if result_path.exists() or best_path.exists():
        ap.error(f'refusing to overwrite completed fold output: {result_path} / {best_path}')

    component_dir = Path(args.component_ckpt_dir) if args.component_ckpt_dir else None
    if component_dir is not None and not component_dir.is_dir():
        ap.error(f'--component_ckpt_dir is not a directory: {component_dir}')

    fd = json.load(open(splits_dir / f'fold_{args.fold}.json'))
    sub = fd['subjects']
    subj_of_key = lambda k: '_'.join(k.split('_')[:3])
    eye_of_key = lambda k: '_'.join(k.split('_')[:4])
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not (set(fd['keys'][a]) & set(fd['keys'][b])), f"key overlap {a}/{b}"
        if args.split_unit == 'subject':
            assert not (set(sub[a]) & set(sub[b])), f"subject overlap {a}/{b}"
        else:
            eyes_a = {eye_of_key(key) for key in fd['keys'][a]}
            eyes_b = {eye_of_key(key) for key in fd['keys'][b]}
            assert not (eyes_a & eyes_b), f"eye overlap {a}/{b}"
            subject_overlap = {
                subj_of_key(key) for key in fd['keys'][a]
            } & {
                subj_of_key(key) for key in fd['keys'][b]
            }
            print(f"[Protocol 1 audit] {a}/{b}: eye overlap=0; "
                  f"permitted subject overlap={len(subject_overlap)}")

    print(f"{'='*60}\nH dual-branch | Fold {args.fold} | cuda:{args.cuda}")
    for s in ('train', 'val', 'test'):
        st = fd['stats'][s]
        print(f"  {s:5s}: {st['subjects']} subjects, {st['samples']} samples")
    print('=' * 60)

    # ── Phase A: auxiliary branch ─────────────────────────
    aux_path = CKPT_DIR / f'fold_{args.fold}_aux.pth'
    aux_load_path = aux_path
    if not aux_load_path.exists() and component_dir is not None:
        candidate = component_dir / f'fold_{args.fold}_aux.pth'
        if candidate.exists():
            aux_load_path = candidate
    cache = T.load_lmdb_cache(T.DATA_DIR, fd['keys']['train'] + fd['keys']['val'])
    if aux_load_path.exists():
        aux = AuxTransformer().to(device)
        aux.load_state_dict(torch.load(aux_load_path, map_location='cpu'))
        aux_info = {'reused': True, 'source': str(aux_load_path)}
        print(f"[Phase A] reusing {aux_load_path}")
    else:
        t0 = timer()
        print("[Phase A] training auxiliary Transformer branch (fold-train only)...")
        aux, aux_info = train_aux(cache, fd['keys']['train'], fd['keys']['val'], device,
                                  epochs=args.aux_epochs, lr=args.aux_lr, seed=args.seed)
        torch.save(aux.state_dict(), aux_path)
        print(f"[Phase A] done in {(timer()-t0)/60:.1f} min, val {aux_info}")

    # ── Phase 0 (optional): adapt the main-branch encoder on this fold's train split ──
    main_enc = CBRAMOD
    enc_info = {'phase0': 'none'}
    if args.phase0 != 'none':
        enc_path = CKPT_DIR / f'fold_{args.fold}_mainenc.pth'
        enc_load_path = enc_path
        if not enc_load_path.exists() and component_dir is not None:
            candidate = component_dir / f'fold_{args.fold}_mainenc.pth'
            if candidate.exists():
                enc_load_path = candidate
        if enc_load_path.exists():
            main_enc = str(enc_load_path)
            enc_info = {'phase0': args.phase0, 'reused': True, 'source': str(enc_load_path)}
            print(f"[Phase 0] reusing {enc_load_path}")
        else:
            import types
            recipe = {'c2_hi_lr': (30, 5e-4), 'c4_scratch': (100, 5e-4)}[args.phase0]
            a = types.SimpleNamespace(fold=args.fold, cuda=args.cuda,
                                      enc_epochs=recipe[0], enc_bs=64, enc_lr=recipe[1],
                                      enc_wd=5e-2, enc_clip=1.0)
            if args.phase0 == 'c4_scratch':
                _orig = T._BackboneParams.__init__
                def _p(self, cuda, dropout=0.1, classifier='all_patch_reps'):
                    _orig(self, cuda, dropout, classifier); self.use_pretrained_weights = False
                T._BackboneParams.__init__ = _p
            print(f"[Phase 0] adapting main encoder, recipe={args.phase0} "
                  f"({recipe[0]} ep, lr {recipe[1]})")
            saved_dir = T.SAVE_DIR
            T.SAVE_DIR = CKPT_DIR
            main_enc, enc_info = T.train_encoder(fd, a, device)
            T.SAVE_DIR = saved_dir
            Path(main_enc).rename(enc_path)
            main_enc = str(enc_path)
            enc_info['phase0'] = args.phase0
            print(f"[Phase 0] main encoder -> {enc_path}")

    def build(freeze_llm):
        m = DualBranchEEGLlava(
            llm_path=T.LLM_PATH, eeg_encoder_weights=main_enc,
            freeze_eeg_encoder=True, freeze_llm=freeze_llm,
            eeg_dim=200, num_channels=6, num_patches=5,
            aux_encoder=aux, aux_trainable=(args.aux_trainable and not freeze_llm),
            use_spectral=args.use_spectral,
        ).to(device)
        return m

    # ── Stage 1: projector + aux_proj ─────────────────────
    print("\n[Stage 1] projector + aux_proj ...")
    model = build(freeze_llm=True)
    col = partial(T.collate_fn, tokenizer=model.tokenizer, max_length=args.max_length)
    ds1 = T.FoldDataset(T.DATA_DIR, fd['keys']['train'], model.tokenizer, args.max_length, stage=1)
    ld1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers, collate_fn=col)
    tr1 = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(tr1, lr=args.lr1, weight_decay=0.01)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=len(ld1) * args.stage1_epochs, eta_min=1e-6)
    for ep in range(args.stage1_epochs):
        model.train(); losses = []
        for b in tqdm(ld1, desc=f'S1 Epoch {ep+1}', mininterval=30):
            opt1.zero_grad()
            out = model(b['eeg'].to(device), b['input_ids'].to(device),
                        b['attention_mask'].to(device), b['labels'].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(tr1, 1.0)
            opt1.step(); sch1.step(); losses.append(out.loss.item())
        print(f"S1 Epoch {ep+1}: loss={np.mean(losses):.4f}")
    s1_proj = copy.deepcopy(model.projector.state_dict())
    s1_aux = copy.deepcopy(model.aux_proj.state_dict())
    del ld1, ds1, model
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ── Stage 2: + LLM ────────────────────────────────────
    print("\n[Stage 2] projector + aux_proj + LLM ...")
    model2 = build(freeze_llm=False)
    if args.unfreeze_enc_last_n:
        model2.unfreeze_encoder_last_n(args.unfreeze_enc_last_n)
    model2.projector.load_state_dict(s1_proj)
    model2.aux_proj.load_state_dict(s1_aux)
    col2 = partial(collate_w, tokenizer=model2.tokenizer, max_length=args.max_length)
    mk = lambda keys, sh, w=None: DataLoader(
        WeightedFoldDataset(T.DATA_DIR, keys, model2.tokenizer, args.max_length, stage=2, seg_w=w),
        batch_size=args.batch_size, shuffle=sh, num_workers=args.num_workers, collate_fn=col2)
    ld_tr = None  # built after weights are computed
    ld_va, ld_te = mk(fd['keys']['val'], False), mk(fd['keys']['test'], False)

    tr_keys = fd['keys']['train']
    cache_tr = T.load_lmdb_cache(T.DATA_DIR, tr_keys)
    lab = np.array([int(cache_tr[k]['label']) for k in tr_keys])
    n_g = int(lab.sum()); n_h = int(len(lab) - n_g)
    subj_of = lambda k: '_'.join(k.split('_')[:3])
    subj_lab = {}
    for k in tr_keys:
        subj_lab.setdefault(subj_of(k), int(cache_tr[k]['label']))
    s_g = sum(subj_lab.values()); s_h = len(subj_lab) - s_g
    healthy_w = {'segment': n_g / n_h, 'subject': s_g / s_h, 'paper': 2.0}[args.class_weight]
    print(f"[Stage 2] class_weight={args.class_weight} -> healthy weight = {healthy_w:.5f} "
          f"(segments {n_g}/{n_h} = {n_g/n_h:.3f}; participants {s_g}/{s_h} = {s_g/s_h:.3f})")

    seg_w = None
    if args.subject_balanced:
        from collections import Counter
        cnt = Counter(subj_of(k) for k in tr_keys)
        w = {k: 1.0 / cnt[subj_of(k)] for k in tr_keys}
        m = float(np.mean(list(w.values())))
        seg_w = {k: v / m for k, v in w.items()}       # mean 1, so the LR scale is unchanged
        vv = np.array(list(seg_w.values()))
        print(f"[Stage 2] subject-balanced segment weights: {len(cnt)} participants, "
              f"segments/participant {min(cnt.values())}-{max(cnt.values())}, "
              f"weight range {vv.min():.3f}-{vv.max():.3f}")

    ld_tr = mk(tr_keys, True, seg_w)          # built here: needs seg_w from above

    tr2 = [p for p in model2.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(tr2, lr=args.lr2, weight_decay=0.01)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=len(ld_tr) * args.stage2_epochs, eta_min=1e-6)
    val_history = []
    best_acc, best_ep, best_states = train_stage_w(
        model2, ld_tr, opt2, sch2, device, epochs=args.stage2_epochs,
        healthy_weight=healthy_w,
        run_eval_fn=lambda: T.evaluate(model2, ld_va, device),
        eval_every=args.stage2_eval_every, select_metric=args.select_metric,
        history=val_history)
    if best_states:
        model2.load_state_dict(best_states)

    # ── Test ──────────────────────────────────────────────
    print(f"\n{'='*60}\nTest (Fold {args.fold})\n{'='*60}")
    acc, auc, cm, discrete_preds, discrete_labels = T.evaluate(
        model2, ld_te, device, return_outputs=True)
    print(f"Test acc={acc:.5f}, auc(discrete)={auc:.5f}\n{cm}")
    lb_acc, lb_auc, lb_cm, lb_s, lb_y = T.evaluate_loss_based(model2, ld_te, device)
    print(f"Test acc(loss)={lb_acc:.5f}, auc(loss)={lb_auc:.5f}\n{lb_cm}")

    print("Scoring validation split (for val-based selection)...")
    v_bacc, v_auc, _, v_scores, v_labels = T.evaluate_loss_based(model2, ld_va, device)
    print(f"Val   acc(loss)={v_bacc:.5f}, auc(loss)={v_auc:.5f}")

    if args.complete_checkpoint:
        checkpoint = {
            'checkpoint_format': 'dual_branch_complete_v1',
            'model_state_dict': model2.state_dict(),
            'model_spec': {
                'llm_path': T.LLM_PATH,
                'eeg_encoder_init': CBRAMOD,
                'eeg_dim': 200,
                'num_channels': 6,
                'num_patches': 5,
                'aux_model': 'AuxTransformer_default',
                'aux_trainable': bool(args.aux_trainable),
                'use_spectral': bool(args.use_spectral),
                'n_spectral_tokens': 4,
            },
            'fold': args.fold,
            'best_val_epoch': best_ep,
            'args': vars(args),
            'splits_dir': str(splits_dir.resolve()),
            'component_sources': {
                'aux': str(aux_load_path.resolve()),
                'main_encoder': str(Path(main_enc).resolve()),
            },
        }
        tmp_path = best_path.with_suffix(best_path.suffix + '.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(best_path)
        sha256 = hashlib.sha256()
        with open(best_path, 'rb') as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b''):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        manifest = {
            'checkpoint': best_path.name,
            'sha256': digest,
            'checkpoint_format': checkpoint['checkpoint_format'],
            'fold': args.fold,
            'state_dict_keys': len(checkpoint['model_state_dict']),
            'contains_spectral_state': any(
                key.startswith('spectral.') for key in checkpoint['model_state_dict']),
            'contains_aux_encoder_state': any(
                key.startswith('aux_encoder.') for key in checkpoint['model_state_dict']),
            'contains_eeg_encoder_state': any(
                key.startswith('eeg_encoder.') for key in checkpoint['model_state_dict']),
        }
        with open(best_path.with_suffix(best_path.suffix + '.manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        with open(best_path.with_suffix(best_path.suffix + '.sha256'), 'w') as f:
            f.write(f'{digest}  {best_path.name}\n')
        print(f"Complete checkpoint saved: {best_path} ({digest})")
    else:
        torch.save({'projector': model2.projector.state_dict(),
                    'aux_proj': model2.aux_proj.state_dict(),
                    'llm': model2.llm.state_dict(), 'fold': args.fold,
                    'best_val_epoch': best_ep, 'aux_ckpt': str(aux_load_path)},
                   best_path)

    res = {'fold': args.fold, 'best_val_acc': best_acc, 'best_val_epoch': best_ep,
           'select_metric': args.select_metric, 'val_history': val_history,
           'test_acc': acc, 'test_auc_discrete': auc,
           'test_auc_loss_based': lb_auc, 'test_bacc_loss_based': lb_acc,
           'test_cm': cm.tolist(), 'test_cm_loss_based': lb_cm.tolist(),
           'aux': aux_info, 'main_encoder': enc_info, 'args': vars(args),
           'test_keys': fd['keys']['test'],
           'test_preds_discrete': discrete_preds,
           'test_labels_discrete': discrete_labels,
           'val_keys': fd['keys']['val'], 'val_scores_lb': v_scores, 'val_labels_lb': v_labels, 'test_scores_lb': lb_s, 'test_labels_lb': lb_y}
    assert len(lb_s) == len(fd['keys']['test'])
    with open(result_path, 'w') as f:
        json.dump(res, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    for directory in (LOG_DIR, CKPT_DIR):
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    time.sleep(2)  # separate queue process must observe every completion artifact before exit
    print(f"Result saved: {result_path}")


if __name__ == '__main__':
    main()
