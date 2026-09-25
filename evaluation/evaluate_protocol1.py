#!/usr/bin/env python3
"""Audit a completed Protocol 1 M run against the locked RF predictions.

This script never trains or selects a model.  It consumes the single held-out
test result emitted by ``train_fold_dual.py``, verifies ordering/labels, and
writes the manuscript metrics, paired tests, and per-segment audit table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


DEFAULT_DELONG = (Path(__file__).resolve().parents[1] / "baselines" / "protocol1" / "rf_depth12"
                  / "delong_rf_scores.json")  # [release]
DEFAULT_MCNEMAR = (Path(__file__).resolve().parents[1] / "baselines" / "protocol1" / "rf_depth12"
                   / "mcnemar_rf_predictions.json")  # [release]
BOOTSTRAP_SEED = 3407


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(labels, predictions, scores=None) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": matrix.tolist(),
    }
    if scores is not None:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        result["roc_auc_discrete"] = float(roc_auc_score(labels, predictions))
    return result


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values))
    left = 0
    while left < len(values):
        right = left
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[left:right] = 0.5 * (left + right - 1)
        left = right
    restored = np.empty(len(values))
    restored[order] = ranks + 1
    return restored


def fast_delong(predictions: np.ndarray, positive_count: int):
    positive_count = int(positive_count)
    classifiers, total = predictions.shape
    negative_count = total - positive_count
    tx = np.empty((classifiers, positive_count))
    ty = np.empty((classifiers, negative_count))
    tz = np.empty((classifiers, total))
    for row in range(classifiers):
        tx[row] = compute_midrank(predictions[row, :positive_count])
        ty[row] = compute_midrank(predictions[row, positive_count:])
        tz[row] = compute_midrank(predictions[row])
    aucs = (
        tz[:, :positive_count].sum(axis=1)
        / positive_count
        / negative_count
        - (positive_count + 1.0) / 2.0 / negative_count
    )
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = np.cov(v01) / positive_count + np.cov(v10) / negative_count
    return aucs, np.atleast_2d(covariance)


def delong_test(labels, scores_m, scores_rf) -> dict:
    labels = np.asarray(labels, dtype=int)
    order = (-labels).argsort()
    positive_count = int((labels[order] == 1).sum())
    predictions = np.vstack(
        [np.asarray(scores_m)[order], np.asarray(scores_rf)[order]]
    )
    aucs, covariance = fast_delong(predictions, positive_count)
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    standard_error = float(np.sqrt(max(float(variance), 0.0)))
    difference = float(aucs[0] - aucs[1])
    z_score = difference / (standard_error + 1e-12)
    p_value = float(2 * (1 - norm.cdf(abs(z_score))))
    return {
        "m_auc": float(aucs[0]),
        "rf_auc": float(aucs[1]),
        "delta_auc": difference,
        "standard_error": standard_error,
        "z": float(z_score),
        "p_value_two_sided": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def mcnemar_test(labels, predictions_m, predictions_rf) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions_m = np.asarray(predictions_m, dtype=int)
    predictions_rf = np.asarray(predictions_rf, dtype=int)
    m_correct = predictions_m == labels
    rf_correct = predictions_rf == labels
    a = int(np.sum(m_correct & rf_correct))
    b = int(np.sum(m_correct & ~rf_correct))
    c = int(np.sum(~m_correct & rf_correct))
    d = int(np.sum(~m_correct & ~rf_correct))
    if b + c == 0:
        statistic, p_value = 0.0, 1.0
        correction = "identical predictions"
    elif b + c < 25:
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1 - chi2.cdf(statistic, df=1))
        correction = "Yates continuity correction (b+c < 25)"
    else:
        statistic = float((b - c) ** 2 / (b + c))
        p_value = float(1 - chi2.cdf(statistic, df=1))
        correction = "uncorrected chi-square (matches archived E13 code)"
    return {
        "contingency": {"both_correct": a, "m_only_correct": b,
                        "rf_only_correct": c, "both_wrong": d},
        "correction": correction,
        "chi_square": statistic,
        "p_value_two_sided": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def bootstrap_differences(labels, values_m, values_rf, metric, n_boot: int) -> dict:
    labels = np.asarray(labels, dtype=int)
    values_m = np.asarray(values_m)
    values_rf = np.asarray(values_rf)
    generator = np.random.RandomState(BOOTSTRAP_SEED)
    differences = []
    for _ in range(n_boot):
        indices = generator.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        if metric == "auc":
            left = roc_auc_score(labels[indices], values_m[indices])
            right = roc_auc_score(labels[indices], values_rf[indices])
        elif metric == "balanced_accuracy":
            left = balanced_accuracy_score(labels[indices], values_m[indices])
            right = balanced_accuracy_score(labels[indices], values_rf[indices])
        else:
            raise ValueError(metric)
        differences.append(left - right)
    differences = np.asarray(differences)
    return {
        "metric": metric,
        "seed": BOOTSTRAP_SEED,
        "requested_replicates": n_boot,
        "valid_replicates": int(len(differences)),
        "delta_95ci": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "two_sided_sign_probability": float(
            min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--delong-baseline", type=Path, default=DEFAULT_DELONG)
    parser.add_argument("--mcnemar-baseline", type=Path, default=DEFAULT_MCNEMAR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    output_dir = args.output_dir or args.result.parent / "analysis"
    output_json = output_dir / "protocol1_metrics_and_stats.json"
    output_csv = output_dir / "protocol1_predictions.csv"
    if output_json.exists() or output_csv.exists():
        parser.error(f"refusing to overwrite audit outputs in {output_dir}")

    run = json.loads(args.result.read_text())
    split = json.loads(args.split.read_text())
    archived_delong = json.loads(args.delong_baseline.read_text())
    archived_mcnemar = json.loads(args.mcnemar_baseline.read_text())

    keys = run["test_keys"]
    scores_m = np.asarray(run["test_scores_lb"], dtype=float)
    labels_loss = np.asarray(run["test_labels_lb"], dtype=int)
    predictions_loss = (scores_m > 0.5).astype(int)
    predictions_discrete = np.asarray(run["test_preds_discrete"], dtype=int)
    labels_discrete = np.asarray(run["test_labels_discrete"], dtype=int)
    labels_delong = np.asarray(archived_delong["trues"], dtype=int)
    labels_mcnemar = np.asarray(archived_mcnemar["trues"], dtype=int)
    scores_rf = np.asarray(archived_delong["rf_scores"], dtype=float)
    predictions_rf = np.asarray(archived_mcnemar["rf"]["preds"], dtype=int)

    expected_n = len(split["keys"]["test"])
    arrays = {
        "keys": keys,
        "scores_m": scores_m,
        "labels_loss": labels_loss,
        "predictions_discrete": predictions_discrete,
        "labels_discrete": labels_discrete,
        "labels_delong": labels_delong,
        "labels_mcnemar": labels_mcnemar,
        "scores_rf": scores_rf,
        "predictions_rf": predictions_rf,
    }
    lengths = {name: len(values) for name, values in arrays.items()}
    if set(lengths.values()) != {expected_n}:
        raise AssertionError(f"paired length mismatch: expected {expected_n}, got {lengths}")
    if keys != split["keys"]["test"]:
        raise AssertionError("run test key order differs from locked Protocol 1 split")
    if not (
        np.array_equal(labels_loss, labels_discrete)
        and np.array_equal(labels_loss, labels_delong)
        and np.array_equal(labels_loss, labels_mcnemar)
    ):
        raise AssertionError("new/archived held-out labels are not in identical order")
    if not np.array_equal((scores_rf > 0.5).astype(int), predictions_rf):
        raise AssertionError("archived RF continuous and discrete predictions do not align")

    loss_metrics = classification_metrics(labels_loss, predictions_loss, scores_m)
    generated_metrics = classification_metrics(
        labels_loss, predictions_discrete, scores=None
    )
    rf_metrics = classification_metrics(labels_loss, predictions_rf, scores_rf)
    delong = delong_test(labels_loss, scores_m, scores_rf)
    mcnemar = mcnemar_test(labels_loss, predictions_discrete, predictions_rf)
    auc_bootstrap = bootstrap_differences(
        labels_loss, scores_m, scores_rf, "auc", args.bootstrap_replicates
    )
    bacc_bootstrap = bootstrap_differences(
        labels_loss,
        predictions_discrete,
        predictions_rf,
        "balanced_accuracy",
        args.bootstrap_replicates,
    )

    conclusion_audit = {
        "m_loss_auc_greater_than_rf": bool(
            loss_metrics["roc_auc"] > rf_metrics["roc_auc"]
        ),
        "m_loss_bacc_greater_than_rf": bool(
            loss_metrics["balanced_accuracy"] > rf_metrics["balanced_accuracy"]
        ),
        "delong_positive_and_significant": bool(
            delong["delta_auc"] > 0 and delong["p_value_two_sided"] < 0.05
        ),
        "mcnemar_remains_non_significant": bool(
            mcnemar["p_value_two_sided"] >= 0.05
        ),
    }
    conclusion_audit["all_major_direction_checks_pass"] = bool(
        all(conclusion_audit.values())
    )

    report = {
        "status": "complete",
        "selection_role": "held-out audit only; never used for model or seed selection",
        "protocol": split["protocol"],
        "split_unit": split["split_unit"],
        "metric_unit": split["metric_unit"],
        "n_test": expected_n,
        "threshold_rule": "glaucoma iff loss-based probability > 0.5",
        "m_loss_based": loss_metrics,
        "m_generated_answer": generated_metrics,
        "rf_locked_baseline": rf_metrics,
        "paired_tests": {
            "delong_m_loss_vs_rf": delong,
            "mcnemar_m_generated_vs_rf": mcnemar,
            "bootstrap_auc_difference": auc_bootstrap,
            "bootstrap_generated_bacc_difference": bacc_bootstrap,
        },
        "conclusion_audit": conclusion_audit,
        "alignment_audit": {
            "all_array_lengths": lengths,
            "test_keys_equal_locked_split": True,
            "new_and_archived_labels_identical": True,
            "rf_threshold_predictions_equal_archived_discrete": True,
        },
        "source_sha256": {
            "run_result": sha256(args.result),
            "locked_split": sha256(args.split),
            "archived_delong": sha256(args.delong_baseline),
            "archived_mcnemar": sha256(args.mcnemar_baseline),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    with output_csv.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["test_index", "key", "label", "m_loss_score", "m_loss_prediction",
             "m_generated_prediction", "rf_score", "rf_prediction"]
        )
        for index in range(expected_n):
            writer.writerow(
                [index, keys[index], int(labels_loss[index]), float(scores_m[index]),
                 int(predictions_loss[index]), int(predictions_discrete[index]),
                 float(scores_rf[index]), int(predictions_rf[index])]
            )
    print(json.dumps(report, indent=2))
    print(f"saved: {output_json}")
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
