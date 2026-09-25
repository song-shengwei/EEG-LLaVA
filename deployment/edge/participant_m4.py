#!/usr/bin/env python3
"""M4: end-to-end per-participant screening computation on this machine.

Uses the Protocol 2 checkpoint (R1a seed 1234, fold 0) and one participant from fold 0's
held-out test set, so every segment is out-of-fold for this model, exactly as in the primary
subject-disjoint evaluation.

Participant (pre-declared, deterministic, no look at any latency or score): among the fold-0
test participants, the one whose segment count is closest to the median segment count of all
95 participants (from p2_r1a_subject_predictions_rtx4090.csv); ties broken by the
lexicographically smallest participant id.  With the locked data this is P_0603_P8 (92 segments;
cohort median 92).

Timed region (per repeat), starting when all segments of the recording are in memory:
    for each segment: loss-based score (two template passes, evaluate_loss_based arithmetic)
    -> participant-level mean score -> fixed threshold (score >= 0.5, r0_subject_evaluator rule)
    -> deterministic SSVEP report fields -> fixed report template filled.
Model loading and raw-signal preprocessing are outside the timed region (see README_METRICS.md).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import common as C
import numpy as np
import torch

CHECKPOINT = "p2_r1a_fold0"


def choose_participant(fold_keys: list[str]) -> tuple[str, dict]:
    with C.SUBJECT_PREDICTIONS.open() as handle:
        rows = list(csv.DictReader(handle))
    cohort_median = statistics.median(int(row["n_segments"]) for row in rows)
    counts: dict[str, int] = {}
    for key in fold_keys:
        counts[C.subject_of(key)] = counts.get(C.subject_of(key), 0) + 1
    chosen = min(counts, key=lambda subject: (abs(counts[subject] - cohort_median), subject))
    reference_row = next(row for row in rows if row["subject_id"] == chosen)
    if int(reference_row["fold"]) != 0 or int(reference_row["n_segments"]) != counts[chosen]:
        raise RuntimeError(f"participant {chosen} is not a complete fold-0 test participant")
    return chosen, {"rule": "fold-0 test participant with segment count closest to the median "
                            "of all 95 participants; lexicographic tie-break",
                    "cohort_median_segments": cohort_median,
                    "fold0_candidates": dict(sorted(counts.items())),
                    "reference_row": reference_row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "int8"], default=None,
                        help="default: fp32 on cpu, bf16 on cuda")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--participant", default=None,
                        help="override the pre-declared choice (not recommended for the paper)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-segments", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=None,
                        help="smoke tests only: truncate the participant's recording")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    precision = args.precision or ("fp32" if args.device == "cpu" else "bf16")
    output = args.output or C.default_output("m4", CHECKPOINT, args.device, precision, args.tag)
    if output.exists() and not args.overwrite:
        parser.error(f"output exists: {output} (use --tag or --overwrite)")

    threads = C.setup_threads(args.threads)
    device = C.resolve_device(args.device)
    reference = C.load_reference(CHECKPOINT)
    fold_keys = reference["keys"]
    if args.participant:
        participant = args.participant
        selection = {"rule": "manual override via --participant"}
    else:
        participant, selection = choose_participant(fold_keys)
    keys = [key for key in fold_keys if C.subject_of(key) == participant]
    if not keys:
        parser.error(f"{participant} is not in the fold-0 test set")
    if args.max_segments:
        keys = keys[: args.max_segments]
    C.log(f"M4 | participant={participant} ({len(keys)} segments) device={device} "
          f"precision={precision} threads={threads}")

    model, checkpoint_info = C.load_model(CHECKPOINT, device, precision,
                                          verify_sha256=not args.skip_sha256)
    C.release_free_heap()
    platform = C.platform_info(device)
    cands = C.candidates(model, device)

    # "Last segment available": every preprocessed segment of the recording is in host memory.
    store = C.SegmentStore()
    segments = [store.eeg(key) for key in keys]
    raw_samples = [np.asarray(store.pair(key)["sample"], dtype=np.float32) for key in keys]
    labels = {store.label(key) for key in keys}
    if len(labels) != 1:
        raise RuntimeError(f"inconsistent labels within {participant}: {labels}")
    label = labels.pop()

    warm_keys = [key for key in fold_keys if C.subject_of(key) != participant]
    with torch.inference_mode():
        for key in warm_keys[: args.warmup_segments]:
            C.segment_score(model, store.eeg(key).to(device), cands)

    runs = []
    C.release_free_heap()
    with C.PeakMemory(device) as peak:
        for repeat in range(args.repeats):
            C.sync(device)
            started = time.perf_counter()
            with torch.inference_mode():
                scores = [C.segment_score(model, tensor.to(device), cands)[0]
                          for tensor in segments]
            participant_score = sum(scores) / len(scores)
            decision = int(participant_score >= 0.5)
            fields = C.participant_features(raw_samples)
            report = C.report_text(fields, decision)
            C.sync(device)
            seconds = time.perf_counter() - started
            runs.append({"seconds": seconds, "per_segment_ms": seconds * 1000.0 / len(segments),
                         "participant_score": participant_score, "decision": decision,
                         "report": report})
            C.log(f"repeat {repeat + 1}/{args.repeats}: {seconds:.2f} s "
                  f"(score {participant_score:.4f}, decision {decision})")

    index_of = {key: i for i, key in enumerate(fold_keys)}
    reference_scores = [reference["scores"][index_of[key]] for key in keys]
    segment_differences = [abs(a - b) for a, b in zip(scores, reference_scores)]
    reference_subject_score = float(selection.get("reference_row", {}).get("score", "nan")) \
        if not args.max_segments and not args.participant else \
        sum(reference_scores) / len(reference_scores)
    reference_decision = int(reference_subject_score >= 0.5)
    seconds = [run["seconds"] for run in runs]
    result = {
        "kind": "m4",
        "status": "complete" if not args.max_segments else "smoke_truncated",
        "created_utc": C.utc_now(),
        "script_sha256": C.sha256_file(Path(__file__)),
        "common_sha256": C.sha256_file(Path(C.__file__)),
        "config": {"checkpoint": CHECKPOINT, "device": args.device, "precision": precision,
                   "threads": threads, "repeats": args.repeats,
                   "warmup_segments": args.warmup_segments, "tag": args.tag},
        "platform": platform,
        "checkpoint": checkpoint_info,
        "participant": {"id": participant, "label": label, "n_segments": len(keys),
                        "recording_seconds": 5 * len(keys), "selection": selection},
        "headline": {
            "M4_per_participant_seconds": C.mean_sd(seconds),
            "M4_per_segment_ms_within_M4": statistics.fmean(r["per_segment_ms"] for r in runs),
            "timed_region": "scoring of all segments + mean-score aggregation + threshold 0.5 "
                            "+ deterministic report fields + template filling",
            "excluded": "model loading; raw OpenBCI preprocessing (filtering, segmentation); "
                        "reading the LMDB",
            "peak_memory_mib": peak.peak_mib,
            "peak_memory_method": peak.method,
        },
        "consistency_vs_rtx4090": {
            "participant_score_laptop": runs[-1]["participant_score"],
            "participant_score_rtx4090": reference_subject_score,
            "participant_score_abs_diff": abs(runs[-1]["participant_score"]
                                              - reference_subject_score),
            "decision_laptop": runs[-1]["decision"],
            "decision_rtx4090": reference_decision,
            "decision_agrees": runs[-1]["decision"] == reference_decision,
            "segment_score_max_abs_diff": max(segment_differences),
            "segment_score_mean_abs_diff": statistics.fmean(segment_differences),
        },
        "report_example": runs[-1]["report"],
        "report_fields": fields,
        "runs": runs,
        "segment_keys": keys,
        "segment_scores_last_run": scores,
    }
    C.write_json(output, result, overwrite=args.overwrite)
    C.log(f"M4 {statistics.fmean(seconds):.2f} s per participant ({len(keys)} segments); "
          f"decision agrees with RTX 4090: {result['consistency_vs_rtx4090']['decision_agrees']}"
          f" -> {output}")


if __name__ == "__main__":
    main()
