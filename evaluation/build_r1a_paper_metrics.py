#!/usr/bin/env python3
"""Build the manuscript-facing metric package for the locked R1a five-fold run.

This script performs no model selection and no threshold tuning on the outer test
folds.  It only derives tables and operating-point summaries from the already
locked out-of-fold predictions.  The default headline threshold remains 0.5.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


import argparse  # [release]
import os  # [release]

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
# [release] internal locations replaced by options: --run-dir (Protocol 2 R1a run with logs/ and
# final_5fold_subject_summary/), optional --baseline-dir ({eegnet,transformer,rf}_voting.json), --output-dir.
_AP = argparse.ArgumentParser()
_AP.add_argument("--run-dir", type=Path, default=Path(os.environ.get(
    "EEGLLAVA_P2_RUN", REPO_ROOT / "outputs" / "protocol2" / "R1a_seed1234")))
_AP.add_argument("--baseline-dir", type=Path, default=None)
_AP.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "protocol2" / "paper_metrics")
_ARGS = _AP.parse_args()
R1_RUN = _ARGS.run_dir
R0_AUDIT = None  # [release] internal pre-R1a audit, not released; its comparison is skipped
BASELINES = _ARGS.baseline_dir
OUT = _ARGS.output_dir


def subject_of(key: str) -> str:
    return "_".join(key.split("_")[:3])


def auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return math.nan
    wins = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def metrics(scores: list[float], labels: list[int], threshold: float = 0.5) -> dict:
    predictions = [int(score >= threshold) for score in scores]
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "n": len(labels),
        "auc": auc(scores, labels),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": threshold,
    }


def group_subjects(
    keys: list[str], scores: list[float], labels: list[int], mode: str = "mean_probability"
) -> tuple[list[str], list[float], list[int]]:
    groups: dict[str, dict] = defaultdict(lambda: {"scores": [], "labels": set()})
    for key, score, label in zip(keys, scores, labels):
        subject = subject_of(key)
        groups[subject]["scores"].append(float(score))
        groups[subject]["labels"].add(int(label))

    subjects = sorted(groups)
    subject_scores: list[float] = []
    subject_labels: list[int] = []
    for subject in subjects:
        row = groups[subject]
        if len(row["labels"]) != 1:
            raise ValueError(f"inconsistent labels for {subject}: {row['labels']}")
        values = row["scores"]
        if mode == "mean_probability":
            score = statistics.fmean(values)
        elif mode == "mean_logit":
            eps = 1e-12
            logits = [
                math.log(min(1 - eps, max(eps, value)) / (1 - min(1 - eps, max(eps, value))))
                for value in values
            ]
            mean_logit = statistics.fmean(logits)
            score = 1 / (1 + math.exp(-mean_logit))
        elif mode == "hard_aggregation":
            score = statistics.fmean(int(value >= 0.5) for value in values)
        else:
            raise ValueError(mode)
        subject_scores.append(score)
        subject_labels.append(next(iter(row["labels"])))
    return subjects, subject_scores, subject_labels


def operating_points(scores: list[float], labels: list[int]) -> dict:
    ordered = sorted(set(scores))
    thresholds = [
        (left + right) / 2 for left, right in zip(ordered[:-1], ordered[1:])
    ]
    thresholds.extend([0.5, min(ordered) - 1e-12, max(ordered) + 1e-12])
    thresholds = sorted(set(thresholds))
    rows = [metrics(scores, labels, threshold) for threshold in thresholds]

    default = metrics(scores, labels, 0.5)
    youden = max(
        rows,
        key=lambda row: (
            row["sensitivity"] + row["specificity"],
            row["sensitivity"],
            row["threshold"],
        ),
    )
    eligible = [row for row in rows if row["sensitivity"] >= 0.90]
    high_sensitivity = max(
        eligible,
        key=lambda row: (
            row["specificity"],
            row["balanced_accuracy"],
            row["threshold"],
        ),
    )

    prevalence_rows = []
    sensitivity = high_sensitivity["sensitivity"]
    specificity = high_sensitivity["specificity"]
    for prevalence in (0.01, 0.035, 0.10, 0.50):
        ppv = sensitivity * prevalence / (
            sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
        )
        npv = specificity * (1 - prevalence) / (
            specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
        )
        prevalence_rows.append({"prevalence": prevalence, "ppv": ppv, "npv": npv})

    return {
        "default_fixed_0_5": default,
        "youden_descriptive_only": youden,
        "high_sensitivity_descriptive_only": high_sensitivity,
        "predictive_values_at_high_sensitivity_point": prevalence_rows,
        "note": (
            "Youden and high-sensitivity thresholds describe the locked OOF ROC curve; "
            "the headline BAcc remains fixed at threshold 0.5."
        ),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    all_keys: list[str] = []
    all_scores: list[float] = []
    all_labels: list[int] = []
    per_fold: list[dict] = []
    seen_keys: set[str] = set()
    seen_subjects: set[str] = set()

    for fold in range(5):
        path = R1_RUN / "logs" / f"fold_{fold}_result.json"
        with path.open() as handle:
            result = json.load(handle)
        keys = result["test_keys"]
        scores = [float(value) for value in result["test_scores_lb"]]
        labels = [int(value) for value in result["test_labels_lb"]]
        if not (len(keys) == len(scores) == len(labels)):
            raise ValueError(f"array length mismatch in fold {fold}")
        if seen_keys.intersection(keys):
            raise ValueError(f"repeated test key in fold {fold}")
        fold_subjects = {subject_of(key) for key in keys}
        if seen_subjects.intersection(fold_subjects):
            raise ValueError(f"repeated test subject in fold {fold}")
        seen_keys.update(keys)
        seen_subjects.update(fold_subjects)

        subjects, subject_scores, subject_labels = group_subjects(keys, scores, labels)
        per_fold.append(
            {
                "fold": fold,
                "subjects": len(subjects),
                "segment": metrics(scores, labels),
                "subject": metrics(subject_scores, subject_labels),
                "selected_epoch": result["r1a_selection"]["selected"]["epoch"],
                "selection_fallback_used": result["r1a_selection"]["fallback_used"],
            }
        )
        all_keys.extend(keys)
        all_scores.extend(scores)
        all_labels.extend(labels)

    subjects, subject_scores, subject_labels = group_subjects(
        all_keys, all_scores, all_labels
    )
    if len(subjects) != 95 or len(all_keys) != 8861:
        raise ValueError(
            f"expected 95 subjects/8861 segments, got {len(subjects)}/{len(all_keys)}"
        )

    locked_summary_path = R1_RUN / "final_5fold_subject_summary" / "evaluation_summary.json"
    with locked_summary_path.open() as handle:
        locked_summary = json.load(handle)
    pooled_subject = metrics(subject_scores, subject_labels)
    for name, observed in (
        ("auc", pooled_subject["auc"]),
        ("balanced_accuracy", pooled_subject["balanced_accuracy"]),
    ):
        expected = locked_summary["metrics"][name]
        if abs(observed - expected) > 1e-12:
            raise ValueError(f"locked summary mismatch for {name}: {observed} vs {expected}")

    fold_fields = ("balanced_accuracy", "auc")
    fold_mean_std = {}
    for unit in ("segment", "subject"):
        fold_mean_std[unit] = {}
        for field in fold_fields:
            values = [row[unit][field] for row in per_fold]
            fold_mean_std[unit][field] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }

    aggregation = {}
    for mode in ("mean_probability", "mean_logit", "hard_aggregation"):
        _, scores, labels = group_subjects(all_keys, all_scores, all_labels, mode)
        aggregation[mode] = metrics(scores, labels)

    baseline_rows = []
    for filename, display_name in (() if BASELINES is None else (  # [release] optional
        ("eegnet_voting.json", "EEGNet"),
        ("transformer_voting.json", "Transformer"),
        ("rf_voting.json", "Random Forest"),
    )):
        with (BASELINES / filename).open() as handle:
            data = json.load(handle)
        keys = data["test_keys"]
        scores = [float(value) for value in data["test_scores"]]
        labels = [int(value) for value in data["test_labels"]]
        _, grouped_scores, grouped_labels = group_subjects(keys, scores, labels)
        row_metrics = metrics(grouped_scores, grouped_labels)
        archived = data["pooled"]["subject"]
        if abs(row_metrics["auc"] - archived["auc"]) > 1e-12:
            raise ValueError(f"baseline AUC mismatch for {display_name}")
        if abs(row_metrics["balanced_accuracy"] - archived["bacc"]) > 1e-12:
            raise ValueError(f"baseline BAcc mismatch for {display_name}")
        baseline_rows.append({"method": display_name, **row_metrics})
    baseline_rows.append({"method": "EEG-LLaVA (R1a)", **pooled_subject})

    r0_rows = ({} if R0_AUDIT is None else  # [release]
               {row["subject_id"]: row for row in read_csv(R0_AUDIT / "subject_predictions.csv")})
    r1_rows = {
        row["subject_id"]: row
        for row in read_csv(R1_RUN / "final_5fold_subject_summary" / "subject_predictions.csv")
    }
    if r0_rows and set(r0_rows) != set(r1_rows):  # [release]
        raise ValueError("R0/R1a subject sets differ")
    changed = []
    for subject in sorted(r0_rows):
        old = r0_rows[subject]
        new = r1_rows[subject]
        if old["prediction"] != new["prediction"]:
            changed.append(
                {
                    "subject_id": subject,
                    "label": int(new["label"]),
                    "fold": int(new["fold"]),
                    "r0_score": float(old["score"]),
                    "r0_prediction": int(old["prediction"]),
                    "r1a_score": float(new["score"]),
                    "r1a_prediction": int(new["prediction"]),
                }
            )

    package = {
        "status": "pass",
        "scope": "locked R1a outer-fold predictions; no post-test model or threshold selection",
        "headline": pooled_subject,
        "headline_bootstrap_95_ci": locked_summary["bootstrap_95_ci"],
        "pooled_segment": metrics(all_scores, all_labels),
        "per_fold": per_fold,
        "fold_mean_sample_std": fold_mean_std,
        "aggregation_robustness": aggregation,
        "operating_points": operating_points(subject_scores, subject_labels),
        "protocol2_subject_level_baselines": baseline_rows,
        "r0_to_r1a_changed_binary_predictions": changed,
        "integrity": {
            "n_folds": 5,
            "n_segments": len(all_keys),
            "n_subjects": len(subjects),
            "test_keys_unique_across_folds": True,
            "test_subjects_unique_across_folds": True,
            "locked_summary_exact_match": True,
        },
        "inputs": {
            "r1a_run": str(R1_RUN),
            "locked_summary": str(locked_summary_path),
            "r0_subject_predictions": None,  # [release]
            "baseline_directory": str(BASELINES),
        },
    }

    with (OUT / "r1a_paper_metrics.json").open("w") as handle:
        json.dump(package, handle, indent=2)

    table4_rows = []
    for row in per_fold:
        table4_rows.append(
            {
                "fold": row["fold"],
                "subjects": row["subjects"],
                "segment_bacc": row["segment"]["balanced_accuracy"],
                "segment_auc": row["segment"]["auc"],
                "subject_bacc": row["subject"]["balanced_accuracy"],
                "subject_auc": row["subject"]["auc"],
                "selected_epoch": row["selected_epoch"],
                "selection_fallback_used": row["selection_fallback_used"],
            }
        )
    write_csv(
        OUT / "table4_r1a.csv",
        table4_rows,
        [
            "fold",
            "subjects",
            "segment_bacc",
            "segment_auc",
            "subject_bacc",
            "subject_auc",
            "selected_epoch",
            "selection_fallback_used",
        ],
    )

    write_csv(
        OUT / "protocol2_subject_baselines.csv",
        baseline_rows,
        [
            "method",
            "n",
            "auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "confusion_matrix",
            "threshold",
        ],
    )
    write_csv(
        OUT / "r0_r1a_changed_predictions.csv",
        changed,
        [
            "subject_id",
            "label",
            "fold",
            "r0_score",
            "r0_prediction",
            "r1a_score",
            "r1a_prediction",
        ],
    )

    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
