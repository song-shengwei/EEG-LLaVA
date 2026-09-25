"""
E07: 构建 Subject-Level 5折 Cross-Validation 划分
与现有 lmdb 完全不交叉：不修改原始 lmdb，只生成 split 索引文件

输出：
  splits/fold_{0..4}.json  每折的 train/val/test subject 列表和对应 sample keys

用法：
  python build_subject_splits.py
"""

import os  # [release]
import pickle
import json
import random
import lmdb
from pathlib import Path
from collections import defaultdict

DATA_DIR = os.environ.get('EEGLLAVA_LMDB', str(Path(__file__).resolve().parents[1] / 'data' / 'processed_lmdb'))  # [release]
OUT_DIR = Path(__file__).resolve().parent / 'e07_subject_cv'  # [release] was ./splits
OUT_DIR.mkdir(exist_ok=True)

SEED = 3407
N_FOLDS = 5


def get_subject_id(key):
    """
    key 格式: P_MMDD_P{num}_{eye}_{session}_{idx}
    subject = P_MMDD_P{num}  (同一个人的左右眼算同一 subject)
    """
    parts = key.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"  # e.g. P_0601_P3


def main():
    # 加载所有 keys（合并 train/val/test）
    db = lmdb.open(DATA_DIR, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        all_splits = pickle.loads(txn.get('__keys__'.encode()))

    all_keys = (all_splits['train'] + all_splits['val'] + all_splits['test'])

    # 读取每个 key 的 label
    key_to_label = {}
    with db.begin(write=False) as txn:
        for key in all_keys:
            pair = pickle.loads(txn.get(key.encode()))
            key_to_label[key] = int(pair['label'])
    db.close()

    # 按 subject 聚合 keys
    subject_to_keys = defaultdict(list)
    for key in all_keys:
        subj = get_subject_id(key)
        subject_to_keys[subj].append(key)

    # 每个 subject 的 label（取第一个 key 的 label，同一 subject 应同类）
    subject_to_label = {}
    for subj, keys in subject_to_keys.items():
        labels = [key_to_label[k] for k in keys]
        subject_to_label[subj] = max(set(labels), key=labels.count)

    # 按 label 分组，做分层划分
    glaucoma_subjects = sorted([s for s, l in subject_to_label.items() if l == 1])
    healthy_subjects  = sorted([s for s, l in subject_to_label.items() if l == 0])

    rng = random.Random(SEED)
    rng.shuffle(glaucoma_subjects)
    rng.shuffle(healthy_subjects)

    print(f"Total subjects: {len(subject_to_keys)}")
    print(f"  Glaucoma: {len(glaucoma_subjects)}")
    print(f"  Healthy:  {len(healthy_subjects)}")

    # 5折划分（分层）
    def kfold_split(subjects, n_folds):
        folds = [[] for _ in range(n_folds)]
        for i, s in enumerate(subjects):
            folds[i % n_folds].append(s)
        return folds

    g_folds = kfold_split(glaucoma_subjects, N_FOLDS)
    h_folds = kfold_split(healthy_subjects, N_FOLDS)

    for fold_idx in range(N_FOLDS):
        test_subjects  = g_folds[fold_idx] + h_folds[fold_idx]
        other_g = [s for i, f in enumerate(g_folds) if i != fold_idx for s in f]
        other_h = [s for i, f in enumerate(h_folds) if i != fold_idx for s in f]

        # 从剩余中取 20% 作 val
        val_g = other_g[:max(1, len(other_g) // 5)]
        val_h = other_h[:max(1, len(other_h) // 5)]
        train_g = other_g[len(val_g):]
        train_h = other_h[len(val_h):]

        train_subjects = train_g + train_h
        val_subjects   = val_g + val_h

        # 展开成 sample keys
        def subjects_to_keys(subjs):
            keys = []
            for s in subjs:
                keys.extend(subject_to_keys[s])
            return keys

        fold_data = {
            'fold': fold_idx,
            'subjects': {
                'train': train_subjects,
                'val':   val_subjects,
                'test':  test_subjects,
            },
            'keys': {
                'train': subjects_to_keys(train_subjects),
                'val':   subjects_to_keys(val_subjects),
                'test':  subjects_to_keys(test_subjects),
            },
            'stats': {
                'train': {'subjects': len(train_subjects), 'samples': len(subjects_to_keys(train_subjects))},
                'val':   {'subjects': len(val_subjects),   'samples': len(subjects_to_keys(val_subjects))},
                'test':  {'subjects': len(test_subjects),  'samples': len(subjects_to_keys(test_subjects))},
            }
        }

        out_file = OUT_DIR / f'fold_{fold_idx}.json'
        with open(out_file, 'w') as f:
            json.dump(fold_data, f, indent=2)

        print(f"\nFold {fold_idx}:")
        for split in ['train', 'val', 'test']:
            s = fold_data['stats'][split]
            print(f"  {split:5s}: {s['subjects']:2d} subjects, {s['samples']:4d} samples")

    print(f"\nSplit files saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
