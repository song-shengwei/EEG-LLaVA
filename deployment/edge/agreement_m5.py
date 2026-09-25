#!/usr/bin/env python3
"""M5: decision agreement with the RTX 4090 on all 1,404 Protocol 1 test segments.

Re-scores every Protocol 1 test segment with the reported seed-42 checkpoint on this machine and
compares segment by segment with the loss-based scores saved by the RTX 4090 run
(data/reference/p1_seed42_test_rtx4090.json; Table 3: 76.6% BAcc, 0.829 AUC).

Scoring is the per-segment arithmetic of train_fold_clean.py::evaluate_loss_based (batch 1, two
template passes, s_g = softmax([-L_h, -L_g])[1]); the segment decision rule is score > 0.5 as in
that function.  Reported: decision agreement, number of flipped segments, maximum / mean
absolute score difference, and BAcc / AUC recomputed from the laptop scores.

Progress is appended to results/partial/*.jsonl after every segment; rerun with --resume after
an interruption (sleep, power loss) to continue where it stopped.

Optional --generated also runs the discrete (generated-text) mode with max_new_tokens=32, exactly
as train_fold_clean.py::evaluate, and compares with the saved RTX 4090 generated decisions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import common as C
import numpy as np
import torch

CHECKPOINT = "p1_seed42"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "int8"], default=None,
                        help="default: fp32 on cpu, bf16 on cuda")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--generated", action="store_true",
                        help="also compare generated-text decisions (much slower on CPU)")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke tests only: score the first N segments")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    precision = args.precision or ("fp32" if args.device == "cpu" else "bf16")
    output = args.output or C.default_output("m5", CHECKPOINT, args.device, precision, args.tag)
    if output.exists() and not args.overwrite:
        parser.error(f"output exists: {output} (use --tag or --overwrite)")
    partial = C.RESULTS_DIR / "partial" / (output.stem + ".jsonl")

    reference = C.load_reference(CHECKPOINT)
    keys = reference["keys"]
    if args.limit:
        keys = keys[: args.limit]
    threads = C.setup_threads(args.threads)
    config = {"checkpoint": CHECKPOINT, "device": args.device, "precision": precision,
              "threads": threads, "generated": bool(args.generated), "n_segments": len(keys),
              "tag": args.tag}

    done: dict[int, dict] = {}
    if partial.exists():
        if not args.resume:
            parser.error(f"partial results exist: {partial} (use --resume, or delete it)")
        lines = partial.read_text().splitlines()
        header = json.loads(lines[0])
        if header.get("config") != config:
            parser.error(f"--resume with a different configuration than {partial}")
        for line in lines[1:]:
            if line.strip():
                row = json.loads(line)
                done[row["i"]] = row
        C.log(f"resuming: {len(done)}/{len(keys)} segments already scored")

    device = C.resolve_device(args.device)
    C.log(f"M5 | {len(keys)} Protocol 1 test segments, device={device} precision={precision} "
          f"threads={threads}")
    model, checkpoint_info = C.load_model(CHECKPOINT, device, precision,
                                          verify_sha256=not args.skip_sha256)
    C.release_free_heap()
    platform = C.platform_info(device)
    cands = C.candidates(model, device)
    store = C.SegmentStore()

    partial.parent.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        partial.write_text(json.dumps({"config": config, "created_utc": C.utc_now(),
                                       "checkpoint_sha256": checkpoint_info["sha256"]}) + "\n")

    with torch.inference_mode():  # warm-up, not recorded
        for key in keys[:3]:
            C.segment_score(model, store.eeg(key).to(device), cands)

    started = time.perf_counter()
    todo = [i for i in range(len(keys)) if i not in done]
    with partial.open("a") as handle, torch.inference_mode():
        for count, i in enumerate(todo, start=1):
            key = keys[i]
            eeg = store.eeg(key).to(device)
            label = store.label(key)
            if label != reference["labels"][i]:
                raise RuntimeError(f"label mismatch for {key}: LMDB {label} vs reference")
            C.sync(device)
            t0 = time.perf_counter()
            score, loss_h, loss_g = C.segment_score(model, eeg, cands)
            C.sync(device)
            row = {"i": i, "key": key, "label": label, "score": score, "loss_h": loss_h,
                   "loss_g": loss_g, "ms": (time.perf_counter() - t0) * 1000.0}
            if args.generated:
                text = model.generate(eeg, C.PROMPT, max_new_tokens=32)
                row["generated_text"] = text
                row["generated_decision"] = C.generated_decision(text)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[i] = row
            if count % 50 == 0 or count == len(todo):
                elapsed = time.perf_counter() - started
                C.log(f"{len(done)}/{len(keys)} scored  "
                      f"(~{elapsed / count * (len(todo) - count) / 60:.1f} min left)")

    rows = [done[i] for i in range(len(keys))]
    laptop_scores = np.array([row["score"] for row in rows], dtype=np.float64)
    reference_scores = np.array(reference["scores"][: len(keys)], dtype=np.float64)
    labels = [row["label"] for row in rows]
    laptop_decisions = laptop_scores > 0.5
    reference_decisions = reference_scores > 0.5
    differences = np.abs(laptop_scores - reference_scores)
    flips = [
        {"key": row["key"], "label": row["label"], "score_rtx4090": float(ref),
         "score_this_machine": float(row["score"])}
        for row, ref, a, b in zip(rows, reference_scores, laptop_decisions, reference_decisions)
        if a != b
    ]
    laptop_metrics = C.binary_metrics(labels, laptop_scores, ">")
    reference_metrics = C.binary_metrics(labels, reference_scores, ">")
    headline = {
        "M5_decision_agreement": float((laptop_decisions == reference_decisions).mean()),
        "n_segments": len(rows),
        "n_flipped_decisions": len(flips),
        "max_abs_score_diff": float(differences.max()),
        "mean_abs_score_diff": float(differences.mean()),
        "median_abs_score_diff": float(np.median(differences)),
        "p99_abs_score_diff": float(np.percentile(differences, 99)),
        "mean_signed_score_diff_this_minus_rtx4090": float((laptop_scores - reference_scores).mean()),
        "reference_segments_within_0.01_of_threshold": int((np.abs(reference_scores - 0.5)
                                                            < 0.01).sum()),
        "balanced_accuracy_this_machine": laptop_metrics["balanced_accuracy"],
        "balanced_accuracy_rtx4090": reference_metrics["balanced_accuracy"],
        "roc_auc_this_machine": laptop_metrics["roc_auc"],
        "roc_auc_rtx4090": reference_metrics["roc_auc"],
        "scoring_latency_per_segment_ms": C.summarize(row["ms"] for row in rows),
    }
    if args.generated:
        generated = [row["generated_decision"] for row in rows]
        reference_generated = reference["preds_generated"][: len(keys)]
        headline["generated_decision_agreement"] = statistics.fmean(
            int(a == b) for a, b in zip(generated, reference_generated))
        headline["generated_decision_metrics_this_machine"] = C.binary_metrics(
            labels, generated, ">")
    result = {
        "kind": "m5",
        "status": "complete" if not args.limit else "smoke_truncated",
        "created_utc": C.utc_now(),
        "script_sha256": C.sha256_file(Path(__file__)),
        "common_sha256": C.sha256_file(Path(C.__file__)),
        "config": config,
        "platform": platform,
        "checkpoint": checkpoint_info,
        "headline": headline,
        "metrics_this_machine": laptop_metrics,
        "metrics_rtx4090_recomputed": reference_metrics,
        "flipped_segments": flips,
        "partial_file": str(partial.relative_to(C.PKG_ROOT)),
    }
    C.write_json(output, result, overwrite=args.overwrite)
    C.log(f"M5 agreement {headline['M5_decision_agreement'] * 100:.2f}% "
          f"({len(flips)} flips of {len(rows)}), max |diff| {headline['max_abs_score_diff']:.2e}, "
          f"BAcc {laptop_metrics['balanced_accuracy'] * 100:.1f}% vs "
          f"{reference_metrics['balanced_accuracy'] * 100:.1f}% -> {output}")


if __name__ == "__main__":
    main()
