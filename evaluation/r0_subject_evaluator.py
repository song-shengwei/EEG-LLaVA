#!/usr/bin/env python3
"""Leakage-aware subject-level evaluator for R0 and later locked candidates.

The script never trains or tunes a threshold. It checks the five result files against
the declared split files, pools out-of-fold segment scores, emits exactly one mean score
per subject, and evaluates the fixed threshold 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


FOLD_RE = re.compile(r"fold_(\d+)_result\.json$")


def subject_of(key: str) -> str:
    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot derive subject from key: {key}")
    return "_".join(parts[:3])


def auc_pairwise(scores: list[float], labels: list[int]) -> float:
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


def classification_metrics(
    scores: list[float], labels: list[int], threshold: float = 0.5
) -> dict[str, object]:
    predictions = [int(score >= threshold) for score in scores]
    tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    bacc = (sensitivity + specificity) / 2
    return {
        "auc": auc_pairwise(scores, labels),
        "balanced_accuracy": bacc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": threshold,
        "predictions": predictions,
    }


def percentile_interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered))]]


def bootstrap_ci(
    scores: list[float], labels: list[int], n_resamples: int, seed: int
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    auc_values: list[float] = []
    bacc_values: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(len(scores)) for _ in scores]
        sampled_scores = [scores[index] for index in indices]
        sampled_labels = [labels[index] for index in indices]
        if 0 < sum(sampled_labels) < len(sampled_labels):
            metrics = classification_metrics(sampled_scores, sampled_labels)
            auc_values.append(float(metrics["auc"]))
            bacc_values.append(float(metrics["balanced_accuracy"]))
    if not auc_values or not bacc_values:
        raise ValueError("bootstrap produced no two-class samples")
    return {
        "auc": percentile_interval(auc_values),
        "balanced_accuracy": percentile_interval(bacc_values),
    }


def load_split(split_path: Path) -> dict:
    with split_path.open() as handle:
        split = json.load(handle)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_subjects = set(split["subjects"][left])
        right_subjects = set(split["subjects"][right])
        overlap_subjects = sorted(left_subjects & right_subjects)
        if overlap_subjects:
            raise ValueError(
                f"subject leakage in {split_path.name} {left}/{right}: "
                f"{overlap_subjects[:5]}"
            )
        left_keys = set(split["keys"][left])
        right_keys = set(split["keys"][right])
        overlap_keys = sorted(left_keys & right_keys)
        if overlap_keys:
            raise ValueError(
                f"segment leakage in {split_path.name} {left}/{right}: {overlap_keys[:5]}"
            )
    return split


def load_and_validate_folds(results_dir: Path, splits_dir: Path) -> list[dict]:
    result_paths = sorted(results_dir.glob("fold_*_result.json"))
    if not result_paths:
        raise FileNotFoundError(f"no fold result files in {results_dir}")

    folds: list[dict] = []
    seen_fold_ids: set[int] = set()
    seen_test_keys: set[str] = set()
    seen_test_subject_fold: dict[str, int] = {}

    for result_path in result_paths:
        match = FOLD_RE.search(result_path.name)
        if match is None:
            continue
        fold = int(match.group(1))
        if fold in seen_fold_ids:
            raise ValueError(f"duplicate fold id {fold}")
        seen_fold_ids.add(fold)

        with result_path.open() as handle:
            result = json.load(handle)
        if int(result.get("fold", result.get("args", {}).get("fold", -1))) != fold:
            raise ValueError(f"fold id mismatch in {result_path}")

        keys = result.get("test_keys")
        scores = result.get("test_scores_lb")
        labels = result.get("test_labels_lb")
        if not isinstance(keys, list) or not isinstance(scores, list) or not isinstance(labels, list):
            raise ValueError(f"missing test key/score/label arrays in {result_path}")
        if not (len(keys) == len(scores) == len(labels)):
            raise ValueError(f"test array length mismatch in {result_path}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate test segment key within {result_path}")

        split_path = splits_dir / f"fold_{fold}.json"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        split = load_split(split_path)
        if set(keys) != set(split["keys"]["test"]):
            missing = sorted(set(split["keys"]["test"]) - set(keys))
            extra = sorted(set(keys) - set(split["keys"]["test"]))
            raise ValueError(
                f"result/split test-key mismatch fold {fold}; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

        repeated_keys = sorted(seen_test_keys & set(keys))
        if repeated_keys:
            raise ValueError(f"test segments repeated across folds: {repeated_keys[:5]}")
        seen_test_keys.update(keys)

        for key in keys:
            subject = subject_of(key)
            prior_fold = seen_test_subject_fold.setdefault(subject, fold)
            if prior_fold != fold:
                raise ValueError(
                    f"subject {subject} appears in test folds {prior_fold} and {fold}"
                )

        folds.append(
            {
                "fold": fold,
                "path": str(result_path.resolve()),
                "keys": keys,
                "scores": [float(score) for score in scores],
                "labels": [int(label) for label in labels],
            }
        )

    expected_fold_ids = set(range(5))
    if seen_fold_ids != expected_fold_ids:
        raise ValueError(f"expected folds 0..4, found {sorted(seen_fold_ids)}")
    return sorted(folds, key=lambda item: item["fold"])


def aggregate_subjects(folds: list[dict]) -> list[dict]:
    groups: dict[str, dict] = defaultdict(
        lambda: {"scores": [], "labels": set(), "folds": set(), "n_segments": 0}
    )
    for fold_data in folds:
        for key, score, label in zip(
            fold_data["keys"], fold_data["scores"], fold_data["labels"]
        ):
            subject = subject_of(key)
            group = groups[subject]
            group["scores"].append(score)
            group["labels"].add(label)
            group["folds"].add(fold_data["fold"])
            group["n_segments"] += 1

    rows: list[dict] = []
    for subject, group in groups.items():
        if len(group["labels"]) != 1:
            raise ValueError(f"inconsistent labels within subject {subject}: {group['labels']}")
        if len(group["folds"]) != 1:
            raise ValueError(f"subject {subject} spans folds: {group['folds']}")
        score = sum(group["scores"]) / len(group["scores"])
        label = next(iter(group["labels"]))
        rows.append(
            {
                "subject_id": subject,
                "label": label,
                "score": score,
                "prediction": int(score >= 0.5),
                "fold": next(iter(group["folds"])),
                "n_segments": group["n_segments"],
            }
        )
    return sorted(rows, key=lambda row: (row["fold"], row["subject_id"]))


def assert_close(name: str, observed: float, expected: float | None, tolerance: float) -> None:
    if expected is not None and abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{name} mismatch: observed={observed:.12f}, expected={expected:.12f}, "
            f"tolerance={tolerance}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, default=95)
    parser.add_argument("--expect-auc", type=float)
    parser.add_argument("--expect-bacc", type=float)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds = load_and_validate_folds(args.results_dir, args.splits_dir)
    rows = aggregate_subjects(folds)
    if len(rows) != args.expected_subjects:
        raise AssertionError(
            f"subject count mismatch: observed={len(rows)}, expected={args.expected_subjects}"
        )

    scores = [row["score"] for row in rows]
    labels = [row["label"] for row in rows]
    metrics = classification_metrics(scores, labels)
    assert_close("AUC", float(metrics["auc"]), args.expect_auc, args.tolerance)
    assert_close(
        "balanced_accuracy",
        float(metrics["balanced_accuracy"]),
        args.expect_bacc,
        args.tolerance,
    )
    confidence_intervals = bootstrap_ci(
        scores, labels, args.bootstrap_resamples, args.bootstrap_seed
    )

    csv_path = args.output_dir / "subject_predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "label", "score", "prediction", "fold", "n_segments"],
        )
        writer.writeheader()
        writer.writerows(rows)

    serializable_metrics = {key: value for key, value in metrics.items() if key != "predictions"}
    summary = {
        "status": "pass",
        "aggregation": "arithmetic mean of segment glaucoma probabilities within subject",
        "threshold_rule": "fixed score >= 0.5",
        "n_folds": len(folds),
        "n_subjects": len(rows),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "metrics": serializable_metrics,
        "bootstrap_95_ci": confidence_intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "leakage_checks": {
            "within_fold_subject_disjoint": True,
            "within_fold_segment_disjoint": True,
            "result_keys_equal_declared_test_keys": True,
            "test_segments_unique_across_folds": True,
            "test_subjects_unique_across_folds": True,
        },
        "inputs": {
            "results_dir": str(args.results_dir.resolve()),
            "splits_dir": str(args.splits_dir.resolve()),
            "fold_result_files": [fold["path"] for fold in folds],
        },
    }
    summary_path = args.output_dir / "evaluation_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"saved: {csv_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
