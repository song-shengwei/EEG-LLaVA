#!/usr/bin/env python3
"""
Rebuild the fold definitions with a working train/val partition — test sets byte-identical.

The bug in the original build_subject_splits.py:

    other_g = [s for i, f in enumerate(g_folds) if i != fold_idx for s in f]  # chunks in order
    val_g   = other_g[:max(1, len(other_g) // 5)]                             # ALWAYS the front

`other_g` is the non-test chunks concatenated in chunk order, and val always takes the front of
it. For fold_idx in {1,2,3,4} that front is chunk 0, so those four folds get the SAME validation
participants. Measured on the shipped splits:

    fold0 val <- chunk 1 (14)      fold1..4 val <- chunk 0 (14/15/15/15, identical people)
    union of the five val sets = 29 unique participants, not 73
    14 participants (15%) never appear in ANY fold's training set — they are val in four folds
    and test in one

Consequences: (a) any val-based selection is fitting the same 15 people, which is why threshold /
aggregation-rule / encoder-recipe selection all behaved erratically; (b) 15% of the cohort never
contributes training signal.

What is NOT wrong: every participant is in exactly one test set, and train∩test = val∩test = 0 in
every fold. **The reported test metrics are valid.** Only the train/val split is defective.

The fix: keep the five test chunks exactly as they are — so every result stays comparable to
Table 4 — and rotate which slice of the remaining participants becomes validation.

    val = other[fold_idx * k : (fold_idx + 1) * k]      # a different slice per fold

Output: splits_v2/fold_{0..4}.json, same schema as the originals.

Usage: python build_splits_v2.py [--check-only]
"""
import argparse, json, os, pickle  # [release] os added
from collections import Counter, defaultdict
from pathlib import Path

import lmdb

DATA_DIR = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[1] / 'data' / 'processed_lmdb'))  # [release]
ORIG = Path(__file__).resolve().parent / 'e07_subject_cv'  # [release]
OUT = Path(__file__).resolve().parent / 'protocol2_v2'  # [release]
N_FOLDS = 5


def subject_of(key):
    p = key.split('_')
    return f"{p[0]}_{p[1]}_{p[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    # subject -> keys, from the same LMDB the originals were built from
    db = lmdb.open(DATA_DIR, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        ks = pickle.loads(txn.get('__keys__'.encode()))
        all_keys = ks['train'] + ks['val'] + ks['test']
        subject_keys = defaultdict(list)
        subject_label = {}
        for k in all_keys:
            s = subject_of(k)
            subject_keys[s].append(k)
            if s not in subject_label:
                subject_label[s] = int(pickle.loads(txn.get(k.encode()))['label'])
    db.close()

    # Recover the five test chunks from the shipped splits — we keep them byte-identical.
    chunks = []
    for i in range(N_FOLDS):
        d = json.load(open(ORIG / f'fold_{i}.json'))
        chunks.append(list(d['subjects']['test']))
    flat = [s for c in chunks for s in c]
    assert len(flat) == len(set(flat)) == len(subject_keys), "test chunks are not a partition"
    print(f"test chunks (kept as-is): {[len(c) for c in chunks]}, total {len(flat)} subjects")

    stats = []
    for f in range(N_FOLDS):
        test = chunks[f]
        other = [s for i in range(N_FOLDS) if i != f for s in chunks[i]]
        # Rotate the validation slice WITHIN each class, so every fold gets a different set of
        # participants and the class ratio stays close to the cohort's (58 glaucoma / 37 healthy).
        # The original code sliced the front of a class-ordered list, which is what collapsed the
        # five validation sets onto the same people.
        val = []
        for lab in (1, 0):
            pool = [s for s in other if subject_label[s] == lab]
            k = max(1, round(len(pool) / N_FOLDS))
            start = (f * k) % len(pool)
            sl = pool[start:start + k]
            if len(sl) < k:
                sl += pool[:k - len(sl)]
            val += sl
        val_set = set(val)
        train = [s for s in other if s not in val_set]

        # stratification sanity: both classes present in val
        vlab = Counter(subject_label[s] for s in val)
        assert len(vlab) == 2, f"fold {f}: val has one class only: {dict(vlab)}"

        keys = {n: [kk for s in grp for kk in subject_keys[s]]
                for n, grp in (('train', train), ('val', val), ('test', test))}
        fold = {'fold': f,
                'subjects': {'train': train, 'val': val, 'test': test},
                'keys': keys,
                'stats': {n: {'subjects': len(grp), 'samples': len(keys[n])}
                          for n, grp in (('train', train), ('val', val), ('test', test))}}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            assert not (set(fold['subjects'][a]) & set(fold['subjects'][b]))
            assert not (set(keys[a]) & set(keys[b]))
        if not args.check_only:
            json.dump(fold, open(OUT / f'fold_{f}.json', 'w'), indent=2)
        stats.append(fold['stats'])
        print(f"  fold{f}: train {len(train)} / val {len(val)} (glaucoma {vlab[1]}, healthy {vlab[0]})"
              f" / test {len(test)}   samples {[fold['stats'][n]['samples'] for n in ('train','val','test')]}")

    # the property that was broken
    vals = [set(json.load(open(OUT / f'fold_{f}.json'))['subjects']['val'])
            if not args.check_only else None for f in range(N_FOLDS)]
    if not args.check_only:
        union = set().union(*vals)
        print(f"\nval union: {len(union)} unique subjects (was 29)")
        print("val pairwise overlap:",
              {f"{a}&{b}": len(vals[a] & vals[b]) for a in range(5) for b in range(a + 1, 5)})
        never_train = [s for s in subject_keys
                       if all(s not in json.load(open(OUT / f'fold_{f}.json'))['subjects']['train']
                              for f in range(N_FOLDS))]
        print(f"never in any training set: {len(never_train)} subjects (was 14)")
        # test identity check against the originals
        same = all(json.load(open(OUT / f'fold_{f}.json'))['subjects']['test']
                   == json.load(open(ORIG / f'fold_{f}.json'))['subjects']['test']
                   for f in range(N_FOLDS))
        print(f"test splits identical to the originals: {same}   <- results stay comparable")
        print(f"\nwritten to {OUT}")


if __name__ == '__main__':
    main()
