#!/usr/bin/env python3
"""Participant-level aggregation of the Protocol 2 head-only control.

Reads the five per-fold outputs of `probe_heldout_p2.py`, averages the per-segment two-class
softmax scores P(glaucoma) within each held-out participant (soft voting), thresholds the
average at 0.5 (the score that selects the report template), and reports the same
participant-level metrics and bootstrap confidence intervals as the locked language-model
readout (80.1% BAcc / 0.849 AUC). This is done for both heads written by the probe script:
`post_mapping` (removes the LLM readout) and `pre_mapping` (also removes the joint
mapping-decoder tuning).

The subject parser, metrics and bootstrap are imported from the locked evaluator
`evaluation/r0_subject_evaluator.py`, which is not modified.

`--selfcheck` runs the identical aggregation on the archived language-model fold results and
must reproduce the locked headline; it needs no GPU and no probe output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os  # [release]
import sys
from collections import defaultdict
from pathlib import Path

# Imports below come from archived directories; leave no bytecode there.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]  # [release]
EVALUATOR_DIR = REPO_ROOT / "evaluation"
sys.path.insert(0, str(EVALUATOR_DIR))
from r0_subject_evaluator import (  # noqa: E402
    bootstrap_ci, classification_metrics, subject_of)

HERE = Path(__file__).resolve().parent
LLM_RESULTS = Path(os.environ.get("EEGLLAVA_P2_RUN", REPO_ROOT / "outputs" / "protocol2" / "R1a_seed1234")) / "logs"
LOCKED = Path(os.environ.get("EEGLLAVA_P2_METRICS", REPO_ROOT / "outputs" / "protocol2" / "paper_metrics"
                             / "r1a_paper_metrics.json"))  # [release]
N_FOLDS, N_SUBJECTS, N_SEGMENTS = 5, 95, 8861
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20260725
HEADS = {
    "post_mapping": "removes the LLM readout; keeps the jointly tuned mappings",
    "pre_mapping": "removes the LLM readout and the joint mapping-decoder tuning",
}


def load_probe_folds(results_dir: Path, head: str) -> list[dict]:
    folds = []
    for k in range(N_FOLDS):
        data = json.loads((results_dir / f"probe_p2_fold{k}.json").read_text())
        if "test_keys" not in data or head not in data.get("heads", {}):
            raise RuntimeError(f"fold {k}: no {head} head; rerun with probe_heldout_p2.py")
        if data.get("llm_forward_passes") != 0:
            raise RuntimeError(f"fold {k}: result does not certify an LLM-free score")
        if data.get("fold") not in (None, k):
            raise RuntimeError(f"fold {k}: file records fold {data['fold']}")
        checkpoint = Path(data["checkpoint"])
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        folds.append({
            "fold": k,
            "keys": data["test_keys"],
            "scores": data["heads"][head]["test_softmax_glaucoma"],
            "labels": data["test_labels"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sidecar.read_text().split()[0] if sidecar.is_file() else None,
            "split": data["split"],
            "head": data["heads"][head]["head"],
            "representation": data["heads"][head]["representation"],
            "feature_dim": data["heads"][head]["feature_dim"],
            "n_train": data["n_train"],
        })
    return folds


def load_llm_folds() -> list[dict]:
    folds = []
    for k in range(N_FOLDS):
        data = json.loads((LLM_RESULTS / f"fold_{k}_result.json").read_text())
        folds.append({"fold": k, "keys": data["test_keys"],
                      "scores": data["test_scores_lb"], "labels": data["test_labels_lb"]})
    return folds


def aggregate(folds: list[dict]) -> dict:
    per_subject: dict[str, dict] = defaultdict(lambda: {"scores": [], "labels": set(),
                                                        "folds": set()})
    seg_scores, seg_labels = [], []
    for fold in folds:
        if not (len(fold["keys"]) == len(fold["scores"]) == len(fold["labels"])):
            raise RuntimeError(f"fold {fold['fold']}: keys/scores/labels lengths differ")
        for key, score, label in zip(fold["keys"], fold["scores"], fold["labels"]):
            entry = per_subject[subject_of(key)]
            entry["scores"].append(float(score))
            entry["labels"].add(int(label))
            entry["folds"].add(fold["fold"])
            seg_scores.append(float(score))
            seg_labels.append(int(label))

    for subject, entry in per_subject.items():
        if len(entry["labels"]) != 1:
            raise RuntimeError(f"{subject}: inconsistent labels {entry['labels']}")
        if len(entry["folds"]) != 1:
            raise RuntimeError(f"{subject}: appears in folds {sorted(entry['folds'])}")
    if len(per_subject) != N_SUBJECTS:
        raise RuntimeError(f"expected {N_SUBJECTS} participants, found {len(per_subject)}")
    if len(seg_scores) != N_SEGMENTS:
        raise RuntimeError(f"expected {N_SEGMENTS} segments, found {len(seg_scores)}")

    subjects = sorted(per_subject)
    scores = [sum(per_subject[s]["scores"]) / len(per_subject[s]["scores"]) for s in subjects]
    labels = [next(iter(per_subject[s]["labels"])) for s in subjects]

    headline = classification_metrics(scores, labels)
    predictions = headline.pop("predictions")
    pooled = classification_metrics(seg_scores, seg_labels)
    pooled.pop("predictions")
    return {
        "headline": {"n": len(subjects), **headline},
        "headline_bootstrap_95_ci": bootstrap_ci(scores, labels, BOOTSTRAP_RESAMPLES,
                                                 BOOTSTRAP_SEED),
        "pooled_segment": {"n": len(seg_scores), **pooled},
        "subjects": [{"subject_id": s, "label": l, "mean_score": sc, "prediction": p,
                      "template": "glaucoma" if p == 1 else "healthy",
                      "n_segments": len(per_subject[s]["scores"]),
                      "fold": next(iter(per_subject[s]["folds"]))}
                     for s, l, sc, p in zip(subjects, labels, scores, predictions)],
    }


def selfcheck() -> None:
    result = aggregate(load_llm_folds())
    locked = json.loads(LOCKED.read_text())["headline"]
    for name in ("auc", "balanced_accuracy", "sensitivity", "specificity"):
        observed, expected = result["headline"][name], locked[name]
        status = "OK " if abs(observed - expected) < 1e-9 else "FAIL"
        print(f"{status} {name}: aggregated {observed:.6f} | locked {expected:.6f}")
        if status == "FAIL":
            raise SystemExit("selfcheck failed: aggregation does not reproduce the headline")
    print("confusion matrix:", result["headline"]["confusion_matrix"],
          "| locked:", locked["confusion_matrix"])
    print("selfcheck passed: this aggregation reproduces the locked 80.1% / 0.849 headline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path,
                        default=HERE / "results" / "E16_head_only_protocol2.json")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    locked = json.loads(LOCKED.read_text())
    heads = {}
    for head, removes in HEADS.items():
        folds = load_probe_folds(args.results_dir, head)
        heads[head] = {"removes": removes, "head": folds[0]["head"],
                       "representation": folds[0]["representation"],
                       "feature_dim": folds[0]["feature_dim"], **aggregate(folds)}
    output = {
        "_schema": "Protocol 2 head-only control: separate head -> two-class softmax per "
                   "segment -> participant mean (soft voting) -> 0.5 threshold -> template "
                   "selection; no LLM forward pass contributes to any score",
        "fit_data": "keys.train of each fold only; keys.val is not used",
        "aggregation": "mean softmax P(glaucoma) over all held-out segments of a participant",
        "threshold": 0.5,
        "bootstrap": {"unit": "participant", "resamples": BOOTSTRAP_RESAMPLES,
                      "seed": BOOTSTRAP_SEED},
        "heads": heads,
        "llm_readout_reference": {"headline": locked["headline"],
                                  "bootstrap_95_ci": locked["headline_bootstrap_95_ci"],
                                  "pooled_segment": locked["pooled_segment"]},
        "folds": [{k: f[k] for k in ("fold", "checkpoint", "checkpoint_sha256", "split",
                                     "n_train")} | {"n_test": len(f["keys"])}
                  for f in folds],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for name, entry in heads.items():
        head, ci = entry["headline"], entry["headline_bootstrap_95_ci"]
        print(f"{name:<13} BAcc {head['balanced_accuracy']:.4f} {ci['balanced_accuracy']}  "
              f"AUC {head['auc']:.4f} {ci['auc']}  sens {head['sensitivity']:.4f}  "
              f"spec {head['specificity']:.4f}")
    ref = locked["headline"]
    print(f"LLM readout   BAcc {ref['balanced_accuracy']:.4f}  AUC {ref['auc']:.4f}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
