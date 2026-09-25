#!/usr/bin/env python3
"""
Protocol 1, 55-token EEG-LLaVA with LoRA in Stage 2 instead of full LLM fine-tuning.

Everything except Stage 2's LLM update follows the reported Protocol 1 final run (seed 42,
outputs/protocol1/seed42): the locked eye-level split, the seed's P1 components
(auxiliary Transformer, adapted CBraMod; reused, not retrained), Stage 1 (projector + aux_proj, LLM frozen,
50 epochs, lr 1e-3), Stage 2 class weighting ('subject', subject-balanced), selection on validation accuracy
of generated answers every 2 epochs, and the same test evaluators. The code for all of that is
src/H_dual_branch/train_fold_dual.py and its imports, loaded read-only; the Stage 1 and Stage 2
loops below restate its main() line for line.

What changes, only in Stage 2:
  - the LLM stays frozen; LoRA adapters (rank 16, alpha 32, dropout 0.05) wrap q/k/v/o_proj in all 28 layers,
    using the implementation of train_lora.py (the 30-token LoRA reference of Table 3);
  - learning rate 2e-4 (that reference's LoRA rate; full fine-tuning used 2e-5).

The cluster kills unregistered GPU processes after 2 h and a full run takes about 4 h, so the run is split into
short processes, each saving its state atomically (run_lora.sh drives them):
  --phase check    build the model, apply LoRA, print trainable parameters, one forward pass; writes nothing
  --phase stage1   Stage 1 (about 70 min)                                 -> ckpt/stage1.pt
  --phase stage2   Stage 2, --chunk_epochs epochs per call (about 70 min)  -> ckpt/stage2_state.pt, ckpt/stage2.done
  --phase test     score test and validation with the selected epoch      -> logs/fold_0_result.json, ckpt/fold_0_best.pth
RNG states are carried between the processes, so the chunked run follows the random stream a single process
would; Stage 1 follows the original run's stream exactly (compare its losses with the original log).
"""
import argparse, copy, gc, hashlib, importlib.util, json, os, random, sys
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

sys.dont_write_bytecode = True

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # [release]
H_TRAINER = REPO_ROOT / 'src' / 'H_dual_branch' / 'train_fold_dual.py'
LORA_REF  = HERE / 'train_lora.py'
P1_RUN    = REPO_ROOT / 'outputs' / 'protocol1'  # [release] components are read from P1_RUN/components/seed<S>/ckpt
SPLITS_DIR = REPO_ROOT / 'splits' / 'protocol1'
EXPECTED  = {'train': 6114, 'val': 1343, 'test': 1404}
FOLD = 0
N_LORA_LAYERS, N_LORA_PARAMS = 112, 4_587_520      # 28 Qwen3 layers x q/k/v/o at rank 16
RUNTIME_ARGS = ('phase', 'cuda', 'chunk_epochs')    # may differ between the processes of one run


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_module('_p1lora_h_trainer', H_TRAINER)   # also imports T (train_fold_clean) and dual_branch
L = load_module('_p1lora_lora_ref', LORA_REF)
T = H.T


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def get_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def frozen_names(model):
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def partial_state(model):
    """Trainable parameters and all buffers; the frozen parameters never change, so this is the full state."""
    frozen = frozen_names(model)
    return {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items() if k not in frozen}


def load_partial(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f'unexpected keys: {unexpected[:5]}'
    assert set(missing) == frozen_names(model), 'partial state does not cover every trainable parameter / buffer'


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['check', 'stage1', 'stage2', 'test'])
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str, default=None, help='default: ./runs/seed<seed>')
    ap.add_argument('--component_ckpt_dir', type=str, default=None,
                    help='default: the P1 components of this seed (fold_0_aux.pth, fold_0_mainenc.pth)')
    ap.add_argument('--stage1_epochs', type=int, default=50)
    ap.add_argument('--lr1', type=float, default=1e-3)
    ap.add_argument('--stage2_epochs', type=int, default=20)
    ap.add_argument('--lr2', type=float, default=2e-4, help='LoRA rate of the 30-token LoRA reference')
    ap.add_argument('--stage2_eval_every', type=int, default=2)
    ap.add_argument('--chunk_epochs', type=int, default=10, help='Stage 2 epochs per process (keep each < 2 h)')
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_length', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--class_weight', default='subject', choices=['segment', 'subject', 'paper'])
    ap.add_argument('--no_subject_balanced', action='store_true')
    args = ap.parse_args()
    if args.chunk_epochs % args.stage2_eval_every:
        ap.error('--chunk_epochs must be a multiple of --stage2_eval_every')
    args.out_dir = Path(args.out_dir) if args.out_dir else HERE / 'runs' / f'seed{args.seed}'
    args.component_ckpt_dir = (Path(args.component_ckpt_dir) if args.component_ckpt_dir
                               else P1_RUN / 'components' / f'seed{args.seed}' / 'ckpt')
    return args


def config_of(args):
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in RUNTIME_ARGS}
    cfg['sources_sha256'] = {str(p): sha256_file(p) for p in (
        Path(__file__).resolve(), H_TRAINER, H_TRAINER.parent / 'dual_branch.py', Path(T.__file__), LORA_REF,
        SPLITS_DIR / f'fold_{FOLD}.json')}
    return cfg


def check_config(args, ckpt_dir):
    """All processes of one run must use identical settings and unchanged source files."""
    path = ckpt_dir / 'config.json'
    cfg = config_of(args)
    if path.exists():
        old = json.load(open(path))
        diff = {k: (old.get(k), cfg[k]) for k in cfg if old.get(k) != cfg[k]}
        assert not diff, f"settings or source files differ from this run's config.json: {diff}"
    else:
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
    return cfg


def load_split():
    fd = json.load(open(SPLITS_DIR / f'fold_{FOLD}.json'))
    eye = lambda k: '_'.join(k.split('_')[:4])
    for s, n in EXPECTED.items():
        assert len(fd['keys'][s]) == n, f'{s}: {len(fd["keys"][s])} segments, expected {n}'
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not (set(fd['keys'][a]) & set(fd['keys'][b])), f'key overlap {a}/{b}'
        assert not ({eye(k) for k in fd['keys'][a]} & {eye(k) for k in fd['keys'][b]}), f'eye overlap {a}/{b}'
    print(f"split: {SPLITS_DIR / f'fold_{FOLD}.json'}  "
          f"({EXPECTED['train']}/{EXPECTED['val']}/{EXPECTED['test']} segments, eye-disjoint)")
    return fd


class Run:
    """The part of H_dual_branch/train_fold_dual.py main() every phase repeats, in the same order."""

    def __init__(self, args):
        self.args = args
        T.setup_seed(args.seed)
        self.device = torch.device(f'cuda:{args.cuda}'); torch.cuda.set_device(args.cuda)
        print(f"GPU: {torch.cuda.get_device_name(args.cuda)}  (CUDA_VISIBLE_DEVICES="
              f"{os.environ.get('CUDA_VISIBLE_DEVICES')}, --cuda {args.cuda})")
        self.fd = load_split()
        comp = args.component_ckpt_dir
        self.aux_path, self.enc_path = comp / f'fold_{FOLD}_aux.pth', comp / f'fold_{FOLD}_mainenc.pth'
        for p in (self.aux_path, self.enc_path):
            assert p.exists(), f'missing component {p}'
        T.load_lmdb_cache(T.DATA_DIR, self.fd['keys']['train'] + self.fd['keys']['val'])
        self.aux = H.AuxTransformer().to(self.device)
        self.aux.load_state_dict(torch.load(self.aux_path, map_location='cpu'))
        print(f"[Phase A] reusing {self.aux_path}\n[Phase 0] reusing {self.enc_path}")

    def build(self, freeze_llm):
        return H.DualBranchEEGLlava(
            llm_path=T.LLM_PATH, eeg_encoder_weights=str(self.enc_path),
            freeze_eeg_encoder=True, freeze_llm=freeze_llm,
            eeg_dim=200, num_channels=6, num_patches=5,
            aux_encoder=self.aux, aux_trainable=False, use_spectral=True,
        ).to(self.device)

    def build_lora(self, s1):
        """Stage 2 model: LLM frozen, LoRA on q/k/v/o, projector / aux_proj from Stage 1."""
        a = self.args
        model = self.build(freeze_llm=True)
        n = L.apply_lora_to_model(model, rank=a.lora_rank, alpha=a.lora_alpha, dropout=a.lora_dropout,
                                  target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'))
        model.to(self.device)
        model.projector.load_state_dict(s1['projector'])
        model.aux_proj.load_state_dict(s1['aux_proj'])
        groups = {'lora': 0, 'projector': 0, 'aux_proj': 0, 'spectral': 0, 'other': 0}
        for name, p in model.named_parameters():
            if p.requires_grad:
                g = 'lora' if '.lora_' in name else name.split('.')[0]
                groups[g if g in groups else 'other'] += p.numel()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(groups.values())
        print(f"[LoRA] wrapped {n} layers; trainable {trainable:,} of {total:,} ({100 * trainable / total:.2f}%): "
              + ', '.join(f'{k} {v:,}' for k, v in groups.items()))
        assert n == N_LORA_LAYERS, f'LoRA wrapped {n} layers, expected {N_LORA_LAYERS}'
        if (a.lora_rank, a.lora_alpha) == (16, 32):
            assert groups['lora'] == N_LORA_PARAMS, f"LoRA parameters {groups['lora']:,} != {N_LORA_PARAMS:,}"
        assert groups['other'] == 0, 'parameters outside LoRA / projector / aux_proj / spectral are trainable'
        self.lora_info = {'layers': n, 'rank': a.lora_rank, 'alpha': a.lora_alpha, 'dropout': a.lora_dropout,
                          'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                          'trainable_params': trainable, 'total_params': total, 'trainable_by_group': groups}
        return model

    def stage2_loaders(self, model):
        """Stage 2 data and weights, as in H main()."""
        a, fd = self.args, self.fd
        col2 = partial(H.collate_w, tokenizer=model.tokenizer, max_length=a.max_length)
        mk = lambda keys, sh, w=None: DataLoader(
            H.WeightedFoldDataset(T.DATA_DIR, keys, model.tokenizer, a.max_length, stage=2, seg_w=w),
            batch_size=a.batch_size, shuffle=sh, num_workers=a.num_workers, collate_fn=col2)
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
        healthy_w = {'segment': n_g / n_h, 'subject': s_g / s_h, 'paper': 2.0}[a.class_weight]
        print(f"[Stage 2] class_weight={a.class_weight} -> healthy weight = {healthy_w:.5f} "
              f"(segments {n_g}/{n_h} = {n_g/n_h:.3f}; participants {s_g}/{s_h} = {s_g/s_h:.3f})")
        seg_w = None
        if not a.no_subject_balanced:
            from collections import Counter
            cnt = Counter(subj_of(k) for k in tr_keys)
            w = {k: 1.0 / cnt[subj_of(k)] for k in tr_keys}
            m = float(np.mean(list(w.values())))
            seg_w = {k: v / m for k, v in w.items()}
            vv = np.array(list(seg_w.values()))
            print(f"[Stage 2] subject-balanced segment weights: {len(cnt)} participants, "
                  f"segments/participant {min(cnt.values())}-{max(cnt.values())}, "
                  f"weight range {vv.min():.3f}-{vv.max():.3f}")
        ld_tr = mk(tr_keys, True, seg_w)
        return ld_tr, ld_va, ld_te, healthy_w


def phase_stage1(args, run, ckpt_dir):
    """Stage 1 exactly as in H main(): projector + aux_proj (+ spectral), LLM frozen."""
    out = ckpt_dir / 'stage1.pt'
    assert not out.exists(), f'{out} exists; Stage 1 is done'
    device = run.device
    print("\n[Stage 1] projector + aux_proj ...")
    model = run.build(freeze_llm=True)
    col = partial(T.collate_fn, tokenizer=model.tokenizer, max_length=args.max_length)
    ds1 = T.FoldDataset(T.DATA_DIR, run.fd['keys']['train'], model.tokenizer, args.max_length, stage=1)
    ld1 = DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=col)
    tr1 = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(tr1, lr=args.lr1, weight_decay=0.01)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=len(ld1) * args.stage1_epochs, eta_min=1e-6)
    history = []
    for ep in range(args.stage1_epochs):
        model.train(); losses = []; t0 = timer()
        for b in tqdm(ld1, desc=f'S1 Epoch {ep+1}', mininterval=30):
            opt1.zero_grad()
            o = model(b['eeg'].to(device), b['input_ids'].to(device),
                      b['attention_mask'].to(device), b['labels'].to(device))
            o.loss.backward()
            torch.nn.utils.clip_grad_norm_(tr1, 1.0)
            opt1.step(); sch1.step(); losses.append(o.loss.item())
        history.append(float(np.mean(losses)))
        print(f"S1 Epoch {ep+1}: loss={np.mean(losses):.4f}  ({(timer() - t0) / 60:.1f} min)")
    s1 = {'projector': copy.deepcopy(model.projector.state_dict()),
          'aux_proj': copy.deepcopy(model.aux_proj.state_dict())}
    del ld1, ds1, model
    gc.collect(); torch.cuda.empty_cache()
    s1 = {k: {n: t.cpu() for n, t in v.items()} for k, v in s1.items()}
    atomic_save({**s1, 'loss_history': history, 'rng_after_stage1': get_rng()}, out)
    print(f"Stage 1 saved: {out}")


def phase_stage2(args, run, ckpt_dir):
    """One chunk of Stage 2 (train_stage_w with select_metric='acc'), resumable."""
    done_flag, state_path = ckpt_dir / 'stage2.done', ckpt_dir / 'stage2_state.pt'
    assert not done_flag.exists(), f'{done_flag} exists; Stage 2 is done'
    s1 = torch.load(ckpt_dir / 'stage1.pt', map_location='cpu', weights_only=False)
    state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else None
    if state is None:
        set_rng(s1['rng_after_stage1'])        # continue the stream exactly where Stage 1 left it
    print("\n[Stage 2] projector + aux_proj + LoRA adapters (LLM frozen) ...")
    model = run.build_lora(s1)
    ld_tr, ld_va, _, healthy_w = run.stage2_loaders(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr2, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(ld_tr) * args.stage2_epochs, eta_min=1e-6)
    if state is None:
        start, best, history = 0, {'score': -1.0, 'acc': 0, 'epoch': 0, 'state': None}, []
    else:
        load_partial(model, state['model'])
        opt.load_state_dict(state['optimizer']); sch.load_state_dict(state['scheduler'])
        start, best, history = state['epochs_done'], state['best'], state['val_history']
        set_rng(state['rng'])
        print(f"[Stage 2] resumed after epoch {start}; best so far acc={best['acc']:.5f} at epoch {best['epoch']}")
    end = min(start + args.chunk_epochs, args.stage2_epochs)
    device = run.device
    for epoch in range(start, end):
        model.train(); losses = []; t0 = timer()
        for batch in tqdm(ld_tr, desc=f'Epoch {epoch+1}', mininterval=30):
            raw = batch['label']
            w = torch.where(raw == 0, torch.tensor(float(healthy_w)), torch.tensor(1.0))
            w = w * batch.get('sw', torch.ones_like(w))
            opt.zero_grad()
            o = model(batch['eeg'].to(device), batch['input_ids'].to(device),
                      batch['attention_mask'].to(device), batch['labels'].to(device), w)
            o.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sch.step(); losses.append(o.loss.item())
        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f}, "
              f"lr={opt.param_groups[0]['lr']:.6f}, time={(timer() - t0) / 60:.1f}min")
        if (epoch + 1) % args.stage2_eval_every == 0 or epoch == args.stage2_epochs - 1:
            acc, auc, cm = T.evaluate(model, ld_va, device)
            print(f"Val acc={acc:.5f}, auc={auc:.5f}\n{cm}")
            history.append({'epoch': epoch + 1, 'val_acc': float(acc), 'val_auc': float(auc)})
            if acc > best['score']:
                best = {'score': acc, 'acc': acc, 'epoch': epoch + 1, 'state': partial_state(model)}
                print(f"New best! acc={acc:.5f} (auc={auc:.5f}) at epoch {epoch + 1}")
    atomic_save({'model': partial_state(model), 'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
                 'epochs_done': end, 'best': best, 'val_history': history, 'rng': get_rng(),
                 'lora': run.lora_info, 'healthy_weight': healthy_w}, state_path)
    print(f"Stage 2 state saved after epoch {end}: {state_path}")
    if end == args.stage2_epochs:
        done_flag.write_text(f"best epoch {best['epoch']}, val acc {best['acc']:.5f}\n")
        print(f"Stage 2 done; best epoch {best['epoch']} (val acc {best['acc']:.5f})")


def phase_test(args, run, ckpt_dir, log_dir, cfg):
    result_path, best_path = log_dir / f'fold_{FOLD}_result.json', ckpt_dir / f'fold_{FOLD}_best.pth'
    assert (ckpt_dir / 'stage2.done').exists(), 'Stage 2 is not finished'
    assert not (result_path.exists() or best_path.exists()), f'refusing to overwrite {result_path} / {best_path}'
    s1 = torch.load(ckpt_dir / 'stage1.pt', map_location='cpu', weights_only=False)
    st = torch.load(ckpt_dir / 'stage2_state.pt', map_location='cpu', weights_only=False)
    best = st['best']
    model = run.build_lora(s1)
    if best['state'] is not None:
        load_partial(model, best['state'])
    _, ld_va, ld_te, _ = run.stage2_loaders(model)
    device, fd = run.device, run.fd

    print(f"\n{'='*60}\nTest (Fold {FOLD}), selected epoch {best['epoch']}\n{'='*60}")
    acc, auc, cm, discrete_preds, discrete_labels = T.evaluate(model, ld_te, device, return_outputs=True)
    print(f"Test acc={acc:.5f}, auc(discrete)={auc:.5f}\n{cm}")
    lb_acc, lb_auc, lb_cm, lb_s, lb_y = T.evaluate_loss_based(model, ld_te, device)
    print(f"Test acc(loss)={lb_acc:.5f}, auc(loss)={lb_auc:.5f}\n{lb_cm}")
    print("Scoring validation split ...")
    v_bacc, v_auc, _, v_scores, v_labels = T.evaluate_loss_based(model, ld_va, device)
    print(f"Val   acc(loss)={v_bacc:.5f}, auc(loss)={v_auc:.5f}")
    bacc_d = float(balanced_accuracy_score(discrete_labels, discrete_preds))
    bacc_lb = float(balanced_accuracy_score(lb_y, (np.asarray(lb_s) > 0.5).astype(int)))
    print(f"Test BAcc discrete={bacc_d:.5f}, BAcc loss-based={bacc_lb:.5f}")

    checkpoint = {
        'checkpoint_format': 'dual_branch_lora_complete_v1',
        'model_state_dict': model.state_dict(),
        'model_spec': {'llm_path': T.LLM_PATH, 'eeg_encoder_init': H.CBRAMOD, 'eeg_dim': 200, 'num_channels': 6,
                       'num_patches': 5, 'aux_model': 'AuxTransformer_default', 'aux_trainable': False,
                       'use_spectral': True, 'n_spectral_tokens': 4, 'lora': run.lora_info},
        'fold': FOLD, 'best_val_epoch': best['epoch'], 'config': cfg,
        'component_sources': {'aux': str(run.aux_path), 'main_encoder': str(run.enc_path)},
    }
    atomic_save(checkpoint, best_path)
    digest = sha256_file(best_path)
    with open(best_path.with_suffix(best_path.suffix + '.sha256'), 'w') as f:
        f.write(f'{digest}  {best_path.name}\n')
    print(f"Complete checkpoint saved: {best_path} ({digest})")

    res = {'fold': FOLD, 'variant': '55-token EEG-LLaVA, Stage 2 LoRA', 'lora': run.lora_info,
           'best_val_acc': best['acc'], 'best_val_epoch': best['epoch'], 'select_metric': 'acc',
           'val_history': st['val_history'], 'stage1_loss_history': s1['loss_history'],
           'healthy_weight': st['healthy_weight'],
           # field names as in H_dual_branch results (test_acc is accuracy, test_auc_discrete equals discrete BAcc)
           'test_acc': acc, 'test_auc_discrete': auc, 'test_auc_loss_based': lb_auc, 'test_bacc_loss_based': lb_acc,
           # explicit balanced accuracies, to avoid the field-name trap above
           'test_balanced_accuracy_discrete': bacc_d, 'test_balanced_accuracy_loss_based': bacc_lb,
           'test_cm': cm.tolist(), 'test_cm_loss_based': lb_cm.tolist(), 'config': cfg,
           'aux': {'reused': True, 'source': str(run.aux_path)},
           'main_encoder': {'phase0': 'c2_hi_lr', 'reused': True, 'source': str(run.enc_path)},
           'test_keys': fd['keys']['test'], 'test_preds_discrete': discrete_preds,
           'test_labels_discrete': discrete_labels, 'val_keys': fd['keys']['val'], 'val_scores_lb': v_scores,
           'val_labels_lb': v_labels, 'test_scores_lb': lb_s, 'test_labels_lb': lb_y,
           'checkpoint_sha256': digest}
    assert len(lb_s) == len(fd['keys']['test'])
    tmp = result_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(result_path)
    print(f"Result saved: {result_path}")


def phase_check(args, run):
    """Everything a real run needs, without training or writing: model build, LoRA, one forward, one generate."""
    model = run.build(freeze_llm=True)
    s1 = {'projector': model.projector.state_dict(), 'aux_proj': model.aux_proj.state_dict()}
    del model; gc.collect(); torch.cuda.empty_cache()
    model = run.build_lora(s1)
    ld_tr, _, _, healthy_w = run.stage2_loaders(model)
    b = next(iter(ld_tr))
    w = torch.where(b['label'] == 0, torch.tensor(float(healthy_w)), torch.tensor(1.0)) * b['sw']
    model.train()
    o = model(b['eeg'].to(run.device), b['input_ids'].to(run.device),
              b['attention_mask'].to(run.device), b['labels'].to(run.device), w)
    o.loss.backward()
    grads = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[check] forward loss {o.loss.item():.4f} (finite: {bool(torch.isfinite(o.loss))}); "
          f"{grads}/{n_tr} trainable tensors received gradients")
    model.eval()
    with torch.no_grad():
        text = model.generate(b['eeg'][:1].to(run.device), T.STAGE2_TEMPLATE + '\n', max_new_tokens=32)
    print(f"[check] generate() ok: {text!r}")
    print("[check] OK; nothing was written")


def main():
    args = parse()
    ckpt_dir, log_dir = args.out_dir / 'ckpt', args.out_dir / 'logs'
    print(f"phase: {args.phase}   outputs: {args.out_dir}")
    cfg = None
    if args.phase != 'check':
        ckpt_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
        cfg = check_config(args, ckpt_dir)
    run = Run(args)
    if args.phase == 'check':
        phase_check(args, run)
    elif args.phase == 'stage1':
        phase_stage1(args, run, ckpt_dir)
    elif args.phase == 'stage2':
        phase_stage2(args, run, ckpt_dir)
    else:
        phase_test(args, run, ckpt_dir, log_dir, cfg)


if __name__ == '__main__':
    main()
