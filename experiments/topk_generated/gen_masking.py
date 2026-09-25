"""Fig. 15(c) with the SUBMITTED method on the final model.

Submitted method (the submission-version E11 channel-ablation script (not released)): for every test segment, zero the masked channels,
call model.generate(eeg, STAGE2_TEMPLATE + '\\n', max_new_tokens=32), predict glaucoma iff 'glaucoma' appears in the
lower-cased text; accuracy and roc_auc_score are computed on these 0/1 decisions; channels are ranked by the accuracy
drop of single-channel masking on the test set; Top-K keeps the K highest-ranked channels and masks the rest.

This file only produces the decisions for ONE masking condition, resumably (one JSON line per segment), so that a
reaper kill or a restart loses at most one line. Model loading and data reading are imported read-only from the
locked final pipeline (experiments/final_model_io.py), exactly as the locked
loss-based masking script did. Inference only; writes only under this directory.
"""
import argparse, json, os, sys, time  # [release] os added
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # [release]
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
import torch  # noqa: E402
from final_model_io import T, load_complete_model  # noqa: E402

CHANNELS = ['PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
CKPT = Path(os.environ.get('EEGLLAVA_P1_CKPT', REPO_ROOT / 'checkpoints' / 'protocol1_seed42' / 'fold_0_best.pth'))  # [release]
SPLIT = REPO_ROOT / 'splits' / 'protocol1' / 'fold_0.json'  # [release]

ap = argparse.ArgumentParser()
ap.add_argument('--cuda', type=int, required=True)
ap.add_argument('--name', required=True)                 # e.g. baseline, loo_PO4, top2
ap.add_argument('--mask', default='')                    # comma-separated channel names to zero
ap.add_argument('--limit', type=int, default=0)          # first N test segments only (reproduction check / timing)
args = ap.parse_args()
masked = [CHANNELS.index(c) for c in args.mask.split(',') if c]
out = HERE / 'decisions' / f'{args.name}.jsonl'
out.parent.mkdir(exist_ok=True)
done = set()
if out.exists():
    done = {json.loads(l)['key'] for l in out.read_text().splitlines() if l.strip()}

T.setup_seed(42)
device = torch.device(f'cuda:{args.cuda}'); torch.cuda.set_device(args.cuda)
model, _ = load_complete_model(CKPT, device)
model.eval()
keys = json.loads(SPLIT.read_text())['keys']['test']
if args.limit: keys = keys[:args.limit]
todo = [k for k in keys if k not in done]
cache = T.load_lmdb_cache(T.DATA_DIR, todo) if todo else {}
print(f'[{args.name}] mask={[CHANNELS[i] for i in masked]} todo={len(todo)} done={len(done)}', flush=True)
t0 = time.time()
with open(out, 'a') as fh, torch.no_grad():
    for n, key in enumerate(todo, 1):
        pair = cache[key]
        eeg = torch.tensor(pair['sample'] / 100.0, dtype=torch.float32, device=device).unsqueeze(0)
        if masked:
            eeg[:, masked, :, :] = 0.0
        text = model.generate(eeg, T.STAGE2_TEMPLATE + '\n', max_new_tokens=32)
        pred = 1 if 'glaucoma' in text.lower() else 0
        fh.write(json.dumps({'key': key, 'label': int(pair['label']), 'pred': pred, 'text': text}) + '\n'); fh.flush()
        if n % 50 == 0:
            print(f'[{args.name}] {n}/{len(todo)}  {(time.time()-t0)/n:.2f} s/segment', flush=True)
print(f'[{args.name}] finished {len(todo)} in {time.time()-t0:.0f} s', flush=True)
