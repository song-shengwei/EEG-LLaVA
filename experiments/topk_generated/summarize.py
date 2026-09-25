"""Collect the generated decisions into the SAME schema and with the SAME metrics as the submitted E11 script
(the submission-version E11 channel-ablation script (not released) -> logs/channel_ablation_results.json):
acc = accuracy_score, auc = roc_auc_score on the 0/1 decisions, cm = confusion_matrix,
LOO delta_acc = acc_masked - acc_baseline, ranking = sorted by delta_acc (largest drop first), Top-K = first K names.
Usage: python summarize.py rank   -> prints the ranking and writes ranking.json (after stage 1)
       python summarize.py final  -> writes ../results/fig15c_generated_E11schema.json (after stage 2)
"""
import json, sys
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
HERE = Path(__file__).resolve().parent
CH = ['PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
SPLIT = HERE.parents[1] / 'splits' / 'protocol1' / 'fold_0.json'  # [release]
KEYS = json.loads(SPLIT.read_text())['keys']['test']

def load(name):
    rows = {r['key']: r for r in (json.loads(l) for l in (HERE / 'decisions' / f'{name}.jsonl').read_text().splitlines() if l.strip())}
    assert set(rows) == set(KEYS), f'{name}: {len(rows)} of {len(KEYS)} segments'
    y = [rows[k]['label'] for k in KEYS]; p = [rows[k]['pred'] for k in KEYS]
    try: auc = roc_auc_score(y, p)
    except Exception: auc = 0.0
    return {'acc': accuracy_score(y, p), 'auc': auc, 'cm': confusion_matrix(y, p).tolist()}

mode = sys.argv[1]
base = load('baseline')
loo = {c: load(f'loo_{c}') for c in CH}
for c in CH: loo[c]['delta_acc'] = loo[c]['acc'] - base['acc']
importance = sorted(loo.items(), key=lambda x: x[1]['delta_acc'])     # identical to the submitted line 157
order = [c for c, _ in importance]
if mode == 'rank':
    (HERE / 'ranking.json').write_text(json.dumps({'order': order, 'baseline_acc': base['acc'],
        'delta_acc': {c: loo[c]['delta_acc'] for c in CH}}, indent=2) + '\n')
    print('baseline acc', round(base['acc'] * 100, 2)); print('ranking', order)
    for c in order: print(f'  {c}: acc {loo[c]["acc"]*100:.2f}  delta {loo[c]["delta_acc"]*100:+.2f}')
else:
    topk = {}
    for k in (1, 2, 3, 4):
        m = load(f'top{k}'); m['channels'] = order[:k]; topk[f'top{k}'] = m
    res = {'baseline': {**base, 'masked': []}, 'leave_one_out': loo, 'topk': topk,
           'note': 'final seed-42 55-token checkpoint; submitted E11 method (generated decisions, test-set ranking); RTX 4090'}
    out = HERE / 'results' / 'fig15c_generated_E11schema.json'  # [release]
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + '\n'); print('written', out)
    for k, v in topk.items(): print(f'  {k} {v["channels"]}: acc {v["acc"]*100:.2f} auc {v["auc"]*100:.2f}')
