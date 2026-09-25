#!/usr/bin/env python3
"""Step 0: environment, package integrity and one-segment parity check before M1-M5.

  1. Installed package versions against env/requirements-lock.txt.
  2. SHA-256 of every packaged file against MANIFEST.sha256 (--quick: only files < 50 MB are
     hashed, larger ones are only checked for existence and size).
  3. Machine description: CPU, RAM, OS, power source, background CPU load; GPU name, compute
     capability, driver, native bf16, and whether this torch build has kernels for the GPU.
  4. Parity: load the Protocol 2 checkpoint, score the segment used by the RTX 4090 profile
     (P_0604_P5_l_..._0) on CPU fp32 (and on CUDA if usable), compare with the saved RTX 4090
     score of that segment, and generate one response.
Writes results/check_env.json; scripts/run_all.sh reads "gpu_row_precision" from it.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as metadata
import os
import re
import time
from pathlib import Path

import common as C
import psutil
import torch

LARGE_FILE_BYTES = 50 * 1024 * 1024


def check_versions() -> dict:
    pins = {}
    for line in (C.PKG_ROOT / "env" / "requirements-lock.txt").read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
        if match:
            pins[match.group(1)] = match.group(2)
    rows, mismatches = {}, []
    for name, wanted in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        ok = installed is not None and installed.split("+")[0] == wanted
        rows[name] = {"pinned": wanted, "installed": installed, "ok": ok}
        if not ok:
            mismatches.append(f"{name}: pinned {wanted}, installed {installed}")
    python_ok = C.platform.python_version() == C.PINNED["python"]
    if not python_ok:
        mismatches.append(f"python: pinned {C.PINNED['python']}, running "
                          f"{C.platform.python_version()}")
    return {"packages": rows, "python_ok": python_ok, "mismatches": mismatches}


def check_manifest(quick: bool) -> dict:
    manifest = C.PKG_ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        return {"status": "missing MANIFEST.sha256"}
    failures, hashed, skipped = [], 0, 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*").strip()
        path = C.PKG_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if quick and path.stat().st_size > LARGE_FILE_BYTES:
            skipped += 1
            continue
        hashed += 1
        if C.sha256_file(path) != expected:
            failures.append(f"sha256 mismatch: {relative}")
    return {"status": "pass" if not failures else "FAIL", "hashed": hashed,
            "large_files_not_hashed": skipped, "failures": failures}


def parity(device_name: str, precision: str) -> dict:
    device = C.resolve_device(device_name)
    reference = C.load_reference("p2_r1a_fold0")
    index = reference["keys"].index(C.PROFILE_SAMPLE_KEY)
    started = time.perf_counter()
    model, info = C.load_model("p2_r1a_fold0", device, precision, verify_sha256=False)
    load_seconds = time.perf_counter() - started
    eeg = C.SegmentStore().eeg(C.PROFILE_SAMPLE_KEY).to(device)
    cands = C.candidates(model, device)
    with torch.inference_mode():
        C.segment_score(model, eeg, cands)  # warm-up
        C.sync(device)
        t0 = time.perf_counter()
        score, loss_h, loss_g = C.segment_score(model, eeg, cands)
        C.sync(device)
        score_ms = (time.perf_counter() - t0) * 1000.0
        text = model.generate(eeg, C.PROMPT, max_new_tokens=20)
    reference_score = reference["scores"][index]
    difference = abs(score - reference_score)
    same_decision = (score > 0.5) == (reference_score > 0.5)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "device": str(device), "precision": precision, "sample_key": C.PROFILE_SAMPLE_KEY,
        "score_this_machine": score, "score_rtx4090": reference_score,
        "abs_diff": difference, "same_decision": bool(same_decision),
        "loss_healthy": loss_h, "loss_glaucoma": loss_g,
        "generated_text": text, "one_segment_score_ms": score_ms,
        "model_load_seconds": load_seconds,
        "verdict": "ok" if same_decision and difference < 0.05 else "CHECK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="do not hash files > 50 MB")
    parser.add_argument("--no-parity", action="store_true", help="skip the model parity test")
    parser.add_argument("--no-gpu", action="store_true", help="ignore any GPU")
    parser.add_argument("--output", type=Path, default=C.RESULTS_DIR / "check_env.json")
    args = parser.parse_args()
    threads = C.setup_threads(None)
    warnings = []

    C.log("1/4 package versions")
    versions = check_versions()
    for mismatch in versions["mismatches"]:
        warnings.append(f"version mismatch -> {mismatch}")

    C.log("2/4 file integrity (MANIFEST.sha256)" + (" [quick]" if args.quick else ""))
    integrity = check_manifest(args.quick)
    if integrity.get("status") != "pass":
        warnings.append(f"integrity: {integrity}")

    C.log("3/4 machine description")
    probe_gpu = not args.no_gpu and torch.cuda.is_available()
    machine = C.platform_info(probe_gpu=probe_gpu)
    machine["background_cpu_percent_3s"] = psutil.cpu_percent(interval=3.0)
    if machine["background_cpu_percent_3s"] > 15:
        warnings.append(f"background CPU load {machine['background_cpu_percent_3s']:.0f}% -- "
                        "close other programs before timing")
    if machine["power"]["ac_online"] is False:
        warnings.append("running on battery -- plug in the charger (doc 03: measure on AC power)")
    if machine["ram_total_gib"] < 12:
        warnings.append(f"only {machine['ram_total_gib']} GiB RAM; fp32 model loading peaked "
                        "at 4,562 MiB on the server, close other programs")

    gpu_row_precision = None
    if probe_gpu:
        gpu = machine.get("gpu", {})
        if not gpu.get("arch_supported_by_this_torch_build"):
            warnings.append(
                f"GPU {gpu.get('name')} (sm_{''.join(map(str, gpu.get('capability', [])))}) has "
                f"no kernels in torch {torch.__version__} (arch list {gpu.get('torch_arch_list')}); "
                "the GPU row cannot use this environment -- run the CPU row only, or see "
                "env/README_ENV.md section 6")
        else:
            gpu_row_precision = "bf16" if gpu.get("native_bf16") else "fp32"
            if gpu_row_precision == "fp32":
                warnings.append("GPU has no native bf16 (compute capability < 8.0): the GPU row "
                                "will use fp32 and is not directly comparable to the bf16 RTX 4090 row")

    parity_results = []
    if not args.no_parity:
        C.log("4/4 parity on the RTX 4090 profile segment: CPU fp32")
        parity_results.append(parity("cpu", "fp32"))
        if gpu_row_precision:
            C.log(f"4/4 parity: CUDA {gpu_row_precision}")
            parity_results.append(parity("cuda", gpu_row_precision))
        for row in parity_results:
            C.log(f"   {row['device']} {row['precision']}: score {row['score_this_machine']:.5f} "
                  f"vs RTX 4090 {row['score_rtx4090']:.5f} (|diff| {row['abs_diff']:.2e}); "
                  f"generated: {row['generated_text']!r}")
            if row["verdict"] != "ok":
                warnings.append(f"parity check failed on {row['device']} {row['precision']}: "
                                f"{row}")

    result = {
        "kind": "check_env",
        "created_utc": C.utc_now(),
        "threads": threads,
        "versions": versions,
        "integrity": integrity,
        "machine": machine,
        "gpu_row_precision": gpu_row_precision,
        "parity": parity_results,
        "warnings": warnings,
        "status": "pass" if not warnings else "pass_with_warnings",
    }
    C.write_json(args.output, result, overwrite=True)
    print()
    print(f"CPU   : {machine['cpu_model']} | {machine['cpu_physical_cores']} physical / "
          f"{machine['cpu_logical_cores']} logical cores | RAM {machine['ram_total_gib']} GiB | "
          f"AC power: {machine['power']['ac_online']}")
    if probe_gpu and machine.get("gpu"):
        gpu = machine["gpu"]
        print(f"GPU   : {gpu['name']} | sm_{gpu['capability'][0]}{gpu['capability'][1]} | "
              f"{gpu['total_memory_mib']} MiB | GPU row precision: {gpu_row_precision}")
    print(f"torch : {torch.__version__} (CUDA build {torch.version.cuda}) | threads {threads}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"status: {result['status']} -> {args.output}")
    if integrity.get("status") not in (None, "pass") and integrity.get("failures"):
        raise SystemExit(2)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
