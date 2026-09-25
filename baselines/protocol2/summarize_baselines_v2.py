#!/usr/bin/env python3
"""Participant-level summary of the Protocol 2 baselines re-run on splits_v2.

Reads results/{rf,eegnet,transformer}_voting.json written by run_baselines_voting_v2.py and
recomputes participant-level metrics with the SAME functions that produce the locked EEG-LLaVA
headline (subject parser, ">= 0.5" decision, 10,000 participant bootstrap resamples, fixed seed),
imported read-only from evaluation/r0_subject_evaluator.py.

Writes, inside this directory only:
  results/E12_participant_level_baselines_splits_v2.json   (same schema as the paper's E12 baseline file, plus CIs)
  results/protocol2_subject_baselines_splits_v2.csv        (same columns as the paper's baseline table)
and prints old-split vs splits_v2 side by side, with the archived matched Transformer as a cross-check.

`--selfcheck` runs the identical aggregation on the archived first-split baseline outputs and must
reproduce the values currently in the manuscript; it needs no new results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os, sys  # [release]
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # [release]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from r0_subject_evaluator import bootstrap_ci, classification_metrics, subject_of  # noqa: E402

OLD_DIR = HERE / "first_split_results"  # [release] internal, not released (comparison skipped if absent)
LOCKED = Path(os.environ.get("EEGLLAVA_P2_METRICS", REPO_ROOT / "outputs" / "protocol2" / "paper_metrics"
                             / "r1a_paper_metrics.json"))
MATCHED_TRANSFORMER = HERE / "matched_transformer_results.json"  # [release] internal, skipped if absent
METHODS = (("eegnet", "EEGNet"), ("transformer", "Transformer"), ("rf", "Random Forest"))
RESAMPLES, SEED = 10000, 20260725


def participant_level(path: Path) -> dict:
    data = json.loads(path.read_text())
    keys, scores, labels = data["test_keys"], data["test_scores"], data["test_labels"]
    if not (len(keys) == len(scores) == len(labels) == 8861) or len(set(keys)) != 8861:
        raise RuntimeError(f"{path.name}: expected 8,861 unique out-of-fold segments")
    per = defaultdict(lambda: {"s": [], "y": set()})
    for k, s, y in zip(keys, scores, labels):
        per[subject_of(k)]["s"].append(float(s))
        per[subject_of(k)]["y"].add(int(y))
    if len(per) != 95 or any(len(v["y"]) != 1 for v in per.values()):
        raise RuntimeError(f"{path.name}: expected 95 participants with one label each")
    subjects = sorted(per)
    s = [sum(per[u]["s"]) / len(per[u]["s"]) for u in subjects]
    y = [next(iter(per[u]["y"])) for u in subjects]
    m = classification_metrics(s, y)
    m.pop("predictions")
    seg = classification_metrics([float(x) for x in scores], [int(x) for x in labels])
    seg.pop("predictions")
    return {"n": 95, **m, "bootstrap_95_ci": bootstrap_ci(s, y, RESAMPLES, SEED),
            "pooled_segment": {"n": len(scores), **seg}, "splits": data.get("splits"),
            "script_sha256": data.get("script_sha256")}


def fmt(m: dict) -> str:
    ci = m["bootstrap_95_ci"]
    return (f"BAcc {100 * m['balanced_accuracy']:5.1f}% [{100 * ci['balanced_accuracy'][0]:.1f}, "
            f"{100 * ci['balanced_accuracy'][1]:.1f}]  AUC {m['auc']:.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]  "
            f"sens {100 * m['sensitivity']:.1f}  spec {100 * m['specificity']:.1f}")


def selfcheck() -> None:
    locked = {b["method"]: b for b in json.loads(LOCKED.read_text())["protocol2_subject_level_baselines"]}
    for stem, name in METHODS:
        m = participant_level(OLD_DIR / f"{stem}_voting.json")
        ok = abs(m["auc"] - locked[name]["auc"]) < 1e-9 and \
            abs(m["balanced_accuracy"] - locked[name]["balanced_accuracy"]) < 1e-9
        print(f"{'OK  ' if ok else 'FAIL'} {name:<14} recomputed {100 * m['balanced_accuracy']:.1f}% / {m['auc']:.3f}"
              f" | manuscript {100 * locked[name]['balanced_accuracy']:.1f}% / {locked[name]['auc']:.3f}")
        if not ok:
            raise SystemExit("selfcheck failed")
    print("selfcheck passed: this aggregation reproduces the baseline values now in the manuscript")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    res = HERE / "results"
    out_json = res / "E12_participant_level_baselines_splits_v2.json"
    out_csv = res / "protocol2_subject_baselines_splits_v2.csv"
    if out_json.exists() or out_csv.exists():
        ap.error("refusing to overwrite existing summary files in results/")

    locked = json.loads(LOCKED.read_text())
    ours = locked["headline"]
    new, old = {}, {}
    for stem, name in METHODS:
        new[name] = participant_level(res / f"{stem}_voting.json")
        if (OLD_DIR / f"{stem}_voting.json").is_file():  # [release]
            old[name] = participant_level(OLD_DIR / f"{stem}_voting.json")
        if not str(new[name]["splits"]).endswith(("splits_v2", "protocol2_v2")):  # [release] repo name added
            raise RuntimeError(f"{name}: result was not produced on splits_v2")

    print("participant level, n = 95, threshold 0.5, 10,000 participant bootstrap resamples\n")
    for _, name in METHODS:
        if name in old:  # [release]
            print(f"{name:<14} first split : {fmt(old[name])}")
        print(f"{'':<14} splits_v2   : {fmt(new[name])}\n")
    oc = locked["headline_bootstrap_95_ci"]
    print(f"{'EEG-LLaVA':<14} splits_v2   : BAcc {100 * ours['balanced_accuracy']:5.1f}% "
          f"[{100 * oc['balanced_accuracy'][0]:.1f}, {100 * oc['balanced_accuracy'][1]:.1f}]  "
          f"AUC {ours['auc']:.3f} [{oc['auc'][0]:.3f}, {oc['auc'][1]:.3f}]")
    if MATCHED_TRANSFORMER.is_file():
        p0 = json.loads(MATCHED_TRANSFORMER.read_text())["pooled_test_subject"]
        print(f"\ncross-check, archived Transformer branch evaluated on splits_v2 (different seed): "
              f"BAcc {100 * p0['bacc']:.1f}%  AUC {p0['auc']:.3f}")

    out_json.write_text(json.dumps({
        "_schema": "participant-level Protocol 2 baselines, same evaluation unit and functions as the EEG-LLaVA headline",
        "_source": str(res),
        "_note": "re-run of the first-split baselines on splits_v2 (the split EEG-LLaVA R1a uses); "
                 "models, hyperparameters, features and seeds unchanged; test blocks identical to the first split",
        "baselines": {stem: {"acc": new[name]["balanced_accuracy"], "auc": new[name]["auc"],
                             "sensitivity": new[name]["sensitivity"], "specificity": new[name]["specificity"],
                             "confusion_matrix": new[name]["confusion_matrix"],
                             "bootstrap_95_ci": new[name]["bootstrap_95_ci"],
                             "pooled_segment": new[name]["pooled_segment"]} for stem, name in METHODS},
        "ours": {"acc": ours["balanced_accuracy"], "auc": ours["auc"]},
        "first_split_values_superseded": {stem: {"acc": old[name]["balanced_accuracy"], "auc": old[name]["auc"]}
                                          for stem, name in METHODS if name in old},  # [release]
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    with out_csv.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["method", "n", "auc", "balanced_accuracy", "sensitivity", "specificity", "confusion_matrix", "threshold"])
        for _, name in METHODS:
            m = new[name]
            w.writerow([name, 95, m["auc"], m["balanced_accuracy"], m["sensitivity"], m["specificity"],
                        json.dumps(m["confusion_matrix"]), 0.5])
        w.writerow(["EEG-LLaVA (R1a)", 95, ours["auc"], ours["balanced_accuracy"], ours["sensitivity"],
                    ours["specificity"], json.dumps(ours["confusion_matrix"]), 0.5])
    print(f"\nwritten: {out_json.name}, {out_csv.name}")


if __name__ == "__main__":
    main()
