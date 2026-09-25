#!/usr/bin/env python3
"""M1-M3: batch-1 latency and peak memory of the final 55-token EEG-LLaVA on this machine.

Two protocols run in the same process, on the same loaded model:

  A. rtx4090_protocol -- identical to code/reference/profile_r1a.py, the script behind Fig. 17
     and E09: one real test segment, per-component repeats (20 repeats after 5 warm-ups; single
     forward and loss-based score 10/2; 20-token generation 5/2).  Provides the E09 component
     breakdown (encoder / projector / single forward) for the json.
  B. plan_protocol -- the headline numbers of doc 03 section 1: warm up on 20 distinct test
     segments, then time 200 further distinct segments one at a time (batch 1):
        M1 = two-template loss-based score per segment (mean +- SD, ms)
        M2 = model.generate(max_new_tokens=20) per segment (mean +- SD, ms)
        M3 = peak memory while running M1 and M2 (CPU: process peak RSS; CUDA: allocator peak)

Examples
  python code/edge/profile_m1_m3.py --checkpoint p2_r1a_fold0 --device cpu  --precision fp32
  python code/edge/profile_m1_m3.py --checkpoint p1_seed42    --device cuda --precision bf16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common as C
import torch


def timed(function, device, repeats: int, warmup: int) -> dict:
    """Same procedure as profile_r1a.timed, with the CUDA synchronisation made device-aware."""
    with torch.inference_mode():
        for _ in range(warmup):
            function()
    C.sync(device)
    values = []
    with torch.inference_mode():
        for _ in range(repeats):
            C.sync(device)
            started = time.perf_counter()
            function()
            C.sync(device)
            values.append((time.perf_counter() - started) * 1000.0)
    summary = C.summarize(values)
    return {"mean_ms": summary["mean_ms"], "std_ms": summary["std_ms"],
            "min_ms": summary["min_ms"], "max_ms": summary["max_ms"],
            "repeats": repeats, "warmup": warmup}


def time_once(function, device) -> tuple[float, object]:
    C.sync(device)
    started = time.perf_counter()
    output = function()
    C.sync(device)
    return (time.perf_counter() - started) * 1000.0, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", choices=sorted(C.CHECKPOINTS), default="p2_r1a_fold0")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "int8"], default=None,
                        help="default: fp32 on cpu, bf16 on cuda (the RTX 4090 setting)")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch intra-op threads (default: $EEG_EDGE_THREADS or physical cores)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--legacy-repeats", type=int, default=20)
    parser.add_argument("--legacy-warmup", type=int, default=5)
    parser.add_argument("--skip-sha256", action="store_true",
                        help="skip re-hashing the 1.2 GB checkpoint (check_env.py already did)")
    parser.add_argument("--tag", default=None, help="suffix for the output file name")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    precision = args.precision or ("fp32" if args.device == "cpu" else "bf16")
    if args.warmup < 2 or args.n < 3 or args.legacy_repeats < 5 or args.legacy_warmup < 2:
        parser.error("requires --warmup >= 2, --n >= 3, --legacy-repeats >= 5, --legacy-warmup >= 2")

    output = args.output or C.default_output("m123", args.checkpoint, args.device, precision,
                                             args.tag)
    if output.exists() and not args.overwrite:
        parser.error(f"output exists: {output} (use --tag or --overwrite)")

    threads = C.setup_threads(args.threads)
    device = C.resolve_device(args.device)
    C.log(f"M1-M3 | checkpoint={args.checkpoint} device={device} precision={precision} "
          f"threads={threads}")
    memory_before_load = C.memory_snapshot(device)
    model, checkpoint_info = C.load_model(args.checkpoint, device, precision,
                                          verify_sha256=not args.skip_sha256)
    trimmed = C.release_free_heap()
    memory_after_load = C.memory_snapshot(device)
    C.log(f"model loaded in {checkpoint_info['load_seconds']:.1f} s; "
          f"RSS {memory_after_load['process_rss_mib']:.0f} MiB")
    platform = C.platform_info(device)

    reference = C.load_reference(args.checkpoint)
    keys = reference["keys"]
    if args.warmup + args.n > len(keys):
        parser.error(f"--warmup + --n exceeds the {len(keys)} reference test segments")
    warm_keys = keys[: args.warmup]
    measured_keys = keys[args.warmup: args.warmup + args.n]
    store = C.SegmentStore()
    warm = [store.eeg(key).to(device) for key in warm_keys]
    measured = [store.eeg(key).to(device) for key in measured_keys]
    cands = C.candidates(model, device)
    healthy = cands[0]
    counter = C.GenerationCounter(model.llm)

    # ---------------- A. RTX 4090 protocol (profile_r1a.py, unchanged logic) ----------------
    C.log("protocol A (identical to the RTX 4090 profile_r1a.py procedure)")
    sample_key = keys[0]
    eeg = store.eeg(sample_key).to(device)
    with torch.inference_mode():
        main_features = model.eeg_encoder(eeg).reshape(1, 30, 200)
        aux_features = model.aux_encoder.features(eeg)

    def measure(function, repeats=None, warmup=None):
        return timed(function, device,
                     repeats=args.legacy_repeats if repeats is None else repeats,
                     warmup=args.legacy_warmup if warmup is None else warmup)

    components = {
        "cbramod_encoder": measure(lambda: model.eeg_encoder(eeg)),
        "cbramod_projector": measure(lambda: model.projector(main_features)),
        "aux_transformer": measure(lambda: model.aux_encoder.features(eeg)),
        "aux_projector": measure(lambda: model.aux_proj(aux_features)),
        "spectral_branch": measure(lambda: model.spectral(eeg)),
        "combined_encode_eeg": measure(lambda: model.encode_eeg(eeg)),
        "single_template_forward": measure(
            lambda: model(eeg, *healthy),
            repeats=max(5, args.legacy_repeats // 2),
            warmup=max(2, args.legacy_warmup // 2)),
    }
    components["loss_based_score_two_passes"] = measure(
        lambda: C.loss_based_two_passes(model, eeg, cands),
        repeats=max(5, args.legacy_repeats // 2),
        warmup=max(2, args.legacy_warmup // 2))
    components["generate_20_tokens"] = measure(
        lambda: model.generate(eeg, C.PROMPT, max_new_tokens=args.max_new_tokens),
        repeats=5, warmup=2)
    C.log(f"protocol A: loss-based {components['loss_based_score_two_passes']['mean_ms']:.1f} ms, "
          f"generate {components['generate_20_tokens']['mean_ms']:.1f} ms")

    legacy_memory = {}
    for name, function in (
        ("single_template_forward_peak_mib", lambda: model(eeg, *healthy)),
        ("loss_based_score_peak_mib", lambda: C.loss_based_two_passes(model, eeg, cands)),
        ("generate_20_tokens_peak_mib",
         lambda: model.generate(eeg, C.PROMPT, max_new_tokens=args.max_new_tokens)),
    ):
        C.release_free_heap()
        with torch.inference_mode(), C.PeakMemory(device) as peak:
            function()
        legacy_memory[name] = peak.peak_mib
    legacy_memory["method"] = peak.method

    # ---------------- B. Plan protocol (headline M1, M2, M3) ----------------
    C.log(f"protocol B: warm-up on {len(warm)} segments")
    with torch.inference_mode():
        for tensor in warm:
            C.loss_based_two_passes(model, tensor, cands)
            model.generate(tensor, C.PROMPT, max_new_tokens=args.max_new_tokens)

    def progress(label, index, values, started):
        if (index + 1) % 20 == 0 or index + 1 == len(measured):
            elapsed = time.perf_counter() - started
            remaining = elapsed / (index + 1) * (len(measured) - index - 1)
            C.log(f"{label} {index + 1}/{len(measured)}  mean {sum(values) / len(values):.1f} ms"
                  f"  (~{remaining / 60:.1f} min left)")

    loss_ms = []
    C.release_free_heap()
    started = time.perf_counter()
    with torch.inference_mode(), C.PeakMemory(device) as peak_m1:
        for index, tensor in enumerate(measured):
            elapsed, _ = time_once(lambda: C.loss_based_two_passes(model, tensor, cands), device)
            loss_ms.append(elapsed)
            progress("M1 loss-based", index, loss_ms, started)

    generate_ms, new_tokens, texts = [], [], []
    C.release_free_heap()
    started = time.perf_counter()
    with torch.inference_mode(), C.PeakMemory(device) as peak_m2:
        for index, tensor in enumerate(measured):
            elapsed, text = time_once(
                lambda: model.generate(tensor, C.PROMPT, max_new_tokens=args.max_new_tokens),
                device)
            generate_ms.append(elapsed)
            new_tokens.append(counter.last_new_tokens)
            texts.append(text)
            progress("M2 generate", index, generate_ms, started)

    index_of = {key: i for i, key in enumerate(keys)}
    generated = [C.generated_decision(text) for text in texts]
    reference_generated = [reference["preds_generated"][index_of[key]] for key in measured_keys]
    agreement = sum(a == b for a, b in zip(generated, reference_generated)) / len(generated)

    m1 = C.summarize(loss_ms)
    m2 = C.summarize(generate_ms)
    m3 = max(peak_m1.peak_mib, peak_m2.peak_mib)
    headline = {
        "M1_loss_based_score_per_segment_ms": m1,
        "M2_generate_max20_per_segment_ms": m2,
        "M3_peak_memory_mib": m3,
        "M3_method": peak_m1.method,
        "M3_scope": ("CUDA allocator peak (same metric as the RTX 4090 1,305 MiB)"
                     if device.type == "cuda" else
                     "peak resident set size of the whole Python process while running "
                     "M1 and M2 (model weights + activations + runtime)"),
        "segments_per_second": 1000.0 / m1["mean_ms"],
        "score_time_as_fraction_of_5s_segment": m1["mean_ms"] / 5000.0,
        "faster_than_segment_duration": bool(m1["mean_ms"] < 5000.0),
        "generated_new_tokens": {
            "mean": sum(new_tokens) / len(new_tokens),
            "min": min(new_tokens), "max": max(new_tokens),
            "stopped_at_eos_before_limit": sum(t < args.max_new_tokens for t in new_tokens),
        },
    }
    result = {
        "kind": "m123",
        "status": "complete",
        "created_utc": C.utc_now(),
        "script_sha256": C.sha256_file(Path(__file__)),
        "common_sha256": C.sha256_file(Path(C.__file__)),
        "config": {"checkpoint": args.checkpoint, "device": args.device,
                   "precision": precision, "threads": threads, "batch_size": 1,
                   "warmup_segments": args.warmup, "measured_segments": args.n,
                   "max_new_tokens": args.max_new_tokens,
                   "legacy_repeats": args.legacy_repeats, "legacy_warmup": args.legacy_warmup,
                   "tag": args.tag},
        "platform": platform,
        "checkpoint": checkpoint_info,
        "headline": headline,
        "rtx4090_protocol": {
            "sample_key": sample_key,
            "model_load_time_s": checkpoint_info["load_seconds"],
            "component_latency_bs1": components,
            "memory": legacy_memory,
        },
        "plan_protocol": {
            "warmup_keys": warm_keys,
            "measured_keys": measured_keys,
            "loss_based_ms": loss_ms,
            "generate_ms": generate_ms,
            "generated_new_tokens": new_tokens,
            "generated_text_examples": texts[:3],
            "generated_decision_agreement_vs_rtx4090_on_measured_segments": agreement,
            "peak_memory_mib": {"m1_phase": peak_m1.peak_mib, "m2_phase": peak_m2.peak_mib},
        },
        "memory": {"before_load": memory_before_load, "after_load": memory_after_load,
                   "peak_rss_including_model_loading_mib": memory_after_load["process_vmhwm_mib"],
                   "malloc_trim_after_load": trimmed,
                   "end_of_run": C.memory_snapshot(device)},
    }
    C.write_json(output, result, overwrite=args.overwrite)
    C.log(f"M1 {m1['mean_ms']:.1f} +- {m1['std_ms']:.1f} ms | M2 {m2['mean_ms']:.1f} +- "
          f"{m2['std_ms']:.1f} ms | M3 {m3:.0f} MiB | generated-decision agreement "
          f"{agreement:.3f} -> {output}")


if __name__ == "__main__":
    main()
