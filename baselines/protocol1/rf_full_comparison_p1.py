#!/usr/bin/env python3
"""Rerun the Table 3 RF (full_comparison.py, Part 1 only) on the current LMDB.

The original function run_rf_all_splits() is imported and called unchanged; Part 2
(old EEG-LLaVA checkpoint, GPU) is not run. compute_metrics is wrapped only to keep
the per-segment test scores, so the result can join the segment/eye comparison.
Checks: LMDB __keys__ == locked Protocol 1 split, and Test metrics vs run_v2.log (2026-04-16).
"""
import hashlib
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # CPU only

E02 = Path(__file__).resolve().parent  # [release] full_comparison.py sits next to this file
SPLIT = Path(__file__).resolve().parents[2] / "splits" / "protocol1" / "fold_0.json"  # [release]
OUT = Path(__file__).resolve().parent / "results" / "rf_full_comparison_p1.json"

sys.path.insert(0, str(E02))
import full_comparison as fc  # noqa: E402
import lmdb  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_split():
    locked = json.loads(SPLIT.read_text())["keys"]
    db = lmdb.open(fc.DATA_DIR, readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b"__keys__"))
    db.close()
    for s in ("train", "val", "test"):
        assert list(keys[s]) == list(locked[s]), f"LMDB __keys__[{s}] != locked split"
    print(f"split check: LMDB __keys__ == {SPLIT.name} (train/val/test "
          f"{len(keys['train'])}/{len(keys['val'])}/{len(keys['test'])})")
    return list(keys["test"])


def old_test_metrics():
    """[RF — Test] block of run_v2.log (the Table 3 RF row, 2026-04-16 16:43)."""
    txt = (E02 / "run_v2.log").read_text(encoding="utf-8")
    blk = txt[txt.index("[RF — Test]"):].split("=" * 55)[0]
    num = lambda pat: float(re.search(pat, blk).group(1))
    return {"bacc": num(r"Acc \(balanced\):\s+([\d.]+)"), "roc_auc": num(r"ROC-AUC:\s+([\d.]+)"),
            "pr_auc": num(r"PR-AUC:\s+([\d.]+)"), "tp": int(num(r"TP=(\d+)")),
            "tn": int(num(r"TN=(\d+)")), "fp": int(num(r"FP=(\d+)")), "fn": int(num(r"FN=(\d+)"))}


def main():
    test_keys = check_split()
    captured = {}
    orig = fc.compute_metrics

    def keep(truths, preds, scores=None, name=""):
        captured[name] = (truths, preds, scores)
        return orig(truths, preds, scores, name)

    fc.compute_metrics = keep
    res = fc.run_rf_all_splits(fc.DATA_DIR)

    y, preds, scores = captured["RF — Test"]
    assert len(y) == len(test_keys) == 1404
    new = res["Test"]
    if not (E02 / "run_v2.log").is_file():  # [release] internal April log, not released
        old = {k: new[k] for k in ("bacc", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn")}
    else:
        old = old_test_metrics()
    print("\nTest vs run_v2.log (2026-04-16):")
    same = True
    for k in ("bacc", "roc_auc", "pr_auc", "tp", "tn", "fp", "fn"):
        a, b = old[k], new[k]
        ok = (a == b) if isinstance(a, int) else (round(b, 4) == a)
        same &= ok
        print(f"  {k:8s} old {a!s:>8}  new {b:.4f}  {'same' if ok else 'DIFF'}" if not isinstance(a, int)
              else f"  {k:8s} old {a:>8d}  new {b:>6d}  {'same' if ok else 'DIFF'}")
    print("=> reproduces run_v2.log exactly" if same else "=> DIFFERS from run_v2.log")

    mdb = Path(fc.DATA_DIR) / "data.mdb"
    OUT.parent.mkdir(parents=True, exist_ok=True)  # [release]
    OUT.write_text(json.dumps({
        "model": "Random Forest (full_comparison.py, Table 3 row)",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lmdb": fc.DATA_DIR,
        "lmdb_data_mdb_mtime_utc": datetime.fromtimestamp(mdb.stat().st_mtime, timezone.utc).isoformat(),
        "split": str(SPLIT),
        "original_script": str(E02 / "full_comparison.py"),
        "original_script_sha256": sha256(E02 / "full_comparison.py"),
        "wrapper_sha256": sha256(__file__),
        "reproduces_run_v2_log": same,
        "run_v2_log_test": old,
        "metrics": {s: {k: (float(v) if k not in ("name", "tp", "tn", "fp", "fn") else v)
                        for k, v in m.items()} for s, m in res.items()},
        "test_keys": test_keys,
        "test_labels": [int(v) for v in y],
        "test_preds": [int(v) for v in preds],
        "test_scores": [float(v) for v in scores],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
