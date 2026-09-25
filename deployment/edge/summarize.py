#!/usr/bin/env python3
"""Collect results/*.json into results/E09b_profile_laptop.json and results/tab_edge_rows.md.

E09b keeps the E09 field names used by regen_fig17.py / check_consistency.py
(model_load_time_s; component_latency_bs1.{encoder, projector, full_forward, generate_20tok,
loss_based_two_passes}.{mean_ms, std_ms}; memory.{forward_peak_mb, generate_peak_mb,
model_loaded_mb}) for every profiled configuration, with the same component mapping as E09
(encoder = CBraMod + auxiliary Transformer + spectral branch, std combined in quadrature;
projector = CBraMod projector + auxiliary projector), and adds platform, precision and the
headline M1-M5 values.  Only untagged runs with status "complete" are used unless --tag is given.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import common as C


def load_runs(tag: str | None) -> dict:
    runs: dict[str, list] = {"m123": [], "m4": [], "m5": []}
    for path in sorted(C.RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        kind = data.get("kind")
        if kind not in runs or data.get("status") != "complete":
            continue
        if data.get("config", {}).get("tag") != tag:
            continue
        data["_file"] = path.name
        runs[kind].append(data)
    for kind in runs:  # keep the newest run per configuration
        latest = {}
        for data in sorted(runs[kind], key=lambda d: d["created_utc"]):
            config = data["config"]
            latest[(config.get("checkpoint"), config["device"], config["precision"])] = data
        runs[kind] = list(latest.values())
    return runs


def e09_view(run: dict) -> dict:
    components = run["rtx4090_protocol"]["component_latency_bs1"]

    def combine(*names):
        return {"mean_ms": sum(components[n]["mean_ms"] for n in names),
                "std_ms": math.sqrt(sum(components[n]["std_ms"] ** 2 for n in names))}

    memory = run["rtx4090_protocol"]["memory"]
    after_load = run["memory"]["after_load"]
    return {
        "model_load_time_s": run["rtx4090_protocol"]["model_load_time_s"],
        "component_latency_bs1": {
            "encoder": combine("cbramod_encoder", "aux_transformer", "spectral_branch"),
            "projector": combine("cbramod_projector", "aux_projector"),
            "full_forward": combine("single_template_forward"),
            "generate_20tok": combine("generate_20_tokens"),
            "loss_based_two_passes": combine("loss_based_score_two_passes"),
        },
        "memory": {
            "forward_peak_mb": memory["single_template_forward_peak_mib"],
            "generate_peak_mb": memory["generate_20_tokens_peak_mib"],
            "model_loaded_mb": after_load.get("cuda_allocated_mib",
                                              after_load["process_rss_mib"]),
            "method": memory["method"],
        },
    }


def platform_label(run: dict) -> str:
    platform = run["platform"]
    if run["config"]["device"] == "cuda":
        return f"Laptop GPU {platform.get('gpu', {}).get('name', '?')}"
    return f"Laptop CPU {platform['cpu_model']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="summarise runs with this tag instead")
    args = parser.parse_args()
    runs = load_runs(args.tag)
    reference = json.loads((C.REFERENCE_DIR / "E09_profile_final55token.json").read_text())

    configurations = []
    for run in sorted(runs["m123"], key=lambda r: (r["config"]["device"],
                                                   r["config"]["precision"],
                                                   r["config"]["checkpoint"])):
        config = run["config"]
        m4 = next((r for r in runs["m4"] if r["config"]["device"] == config["device"]
                   and r["config"]["precision"] == config["precision"]), None)
        m5 = next((r for r in runs["m5"] if r["config"]["device"] == config["device"]
                   and r["config"]["precision"] == config["precision"]), None)
        headline = run["headline"]
        configurations.append({
            "platform": platform_label(run),
            "device": config["device"],
            "precision": config["precision"],
            "checkpoint": config["checkpoint"],
            "checkpoint_sha256": run["checkpoint"]["sha256"],
            "threads": config["threads"],
            "source_files": [run["_file"]] + ([m4["_file"]] if m4 else [])
                            + ([m5["_file"]] if m5 else []),
            "machine": {k: run["platform"].get(k) for k in
                        ("os", "kernel", "cpu_model", "cpu_physical_cores",
                         "cpu_logical_cores", "ram_total_gib", "power", "torch",
                         "torch_cuda_build", "gpu")},
            **e09_view(run),
            "M1_loss_based_score_ms": {k: headline["M1_loss_based_score_per_segment_ms"][k]
                                       for k in ("mean_ms", "std_ms", "n")},
            "M2_generate_20_ms": {k: headline["M2_generate_max20_per_segment_ms"][k]
                                  for k in ("mean_ms", "std_ms", "n")},
            "M2_generated_new_tokens": headline["generated_new_tokens"],
            "M3_peak_memory_mib": headline["M3_peak_memory_mib"],
            "M3_method": headline["M3_method"],
            "M4_per_participant_s": (m4["headline"]["M4_per_participant_seconds"]
                                     if m4 else None),
            "M4_participant": ({"id": m4["participant"]["id"],
                                "n_segments": m4["participant"]["n_segments"],
                                "decision_agrees_with_rtx4090":
                                    m4["consistency_vs_rtx4090"]["decision_agrees"]}
                               if m4 else None),
            "M5": ({k: m5["headline"][k] for k in
                    ("M5_decision_agreement", "n_segments", "n_flipped_decisions",
                     "max_abs_score_diff", "mean_abs_score_diff",
                     "balanced_accuracy_this_machine", "balanced_accuracy_rtx4090",
                     "roc_auc_this_machine", "roc_auc_rtx4090")} if m5 else None),
        })

    summary = {
        "_schema": "E09 field names per configuration (see code/edge/summarize.py docstring) "
                   "plus platform/precision and headline M1-M5 (README_METRICS.md)",
        "created_utc": C.utc_now(),
        "reference_rtx4090_E09": {k: v for k, v in reference.items() if not k.startswith("_")},
        "configurations": configurations,
    }
    C.write_json(C.RESULTS_DIR / "E09b_profile_laptop.json", summary, overwrite=True)

    lines = ["| Platform | Precision | Checkpoint | Score / seg. (ms) | 20-token gen. (ms) | "
             "Peak mem. (MiB) | Per-participant (s) | Agreement |",
             "|---|---|---|---:|---:|---:|---:|---:|",
             "| RTX 4090 (reference) | bf16 | p2_r1a_fold0 | 94.4 ± 1.1 | 330.2 ± 1.4 | 1,305 | "
             "(run participant_m4.py on the 4090) | --- |"]
    for row in configurations:
        m1, m2 = row["M1_loss_based_score_ms"], row["M2_generate_20_ms"]
        m4, m5 = row["M4_per_participant_s"], row["M5"]
        primary = row["checkpoint"] == "p2_r1a_fold0"  # M4/M5 columns belong to the primary row
        m4_cell = "{:.2f} ± {:.2f}".format(m4["mean"], m4["sd"]) if m4 and primary else "—"
        m5_cell = "{:.2f}%".format(m5["M5_decision_agreement"] * 100) if m5 and primary else "—"
        lines.append(
            f"| {row['platform']} | {row['precision']} | {row['checkpoint']} | "
            f"{m1['mean_ms']:.1f} ± {m1['std_ms']:.1f} | {m2['mean_ms']:.1f} ± {m2['std_ms']:.1f} | "
            f"{row['M3_peak_memory_mib']:,.0f} | {m4_cell} | {m5_cell} |")
    text = "\n".join(lines)
    placeholders = []
    cpu = next((r for r in configurations if r["device"] == "cpu" and r["precision"] == "fp32"
                and r["checkpoint"] == "p2_r1a_fold0"), None)
    gpu = next((r for r in configurations if r["device"] == "cuda"
                and r["checkpoint"] == "p2_r1a_fold0"), None)
    if cpu:
        machine = cpu["machine"]
        placeholders += [
            f"【CPU 型号，核数】 = {machine['cpu_model']}, {machine['cpu_physical_cores']} cores "
            f"({machine['cpu_logical_cores']} threads); torch threads used = {cpu['threads']}",
            f"【内存】 = {machine['ram_total_gib']} GiB",
            f"【fp32/bf16】 = {cpu['precision']}",
            f"【M1】 = {cpu['M1_loss_based_score_ms']['mean_ms']:.0f} ms "
            f"(± {cpu['M1_loss_based_score_ms']['std_ms']:.0f}, n = {cpu['M1_loss_based_score_ms']['n']})",
            f"【M2】 = {cpu['M2_generate_20_ms']['mean_ms']:.0f} ms "
            f"(mean {cpu['M2_generated_new_tokens']['mean']:.1f} tokens actually generated)",
            f"【M3】 = {cpu['M3_peak_memory_mib']:,.0f} MiB ({cpu['M3_method']})",
        ]
        if cpu["M4_per_participant_s"]:
            placeholders.append(f"【N】 = {cpu['M4_participant']['n_segments']} segments "
                                f"({cpu['M4_participant']['id']}); 【M4】 = "
                                f"{cpu['M4_per_participant_s']['mean']:.1f} s")
        if cpu["M5"]:
            m5 = cpu["M5"]
            placeholders += [
                f"【M5】 = {m5['M5_decision_agreement'] * 100:.1f}% "
                f"({m5['n_flipped_decisions']} of {m5['n_segments']} segments flipped)",
                f"【x】 = {m5['max_abs_score_diff']:.2g} (max |score diff|)",
                f"BAcc on this machine = {m5['balanced_accuracy_this_machine'] * 100:.1f}% "
                f"(RTX 4090: {m5['balanced_accuracy_rtx4090'] * 100:.1f}%); AUC "
                f"{m5['roc_auc_this_machine']:.3f} (RTX 4090: {m5['roc_auc_rtx4090']:.3f})",
            ]
    if gpu:
        gpu_name = (gpu["machine"].get("gpu") or {}).get("name", "?")
        placeholders.append(f"【有独显则写型号】 = {gpu_name}")
    markdown = ("# tab:edge rows (generated by summarize.py)\n\n" + text
                + "\n\n## Placeholders in doc 03 section 2.1\n\n"
                + "\n".join(f"- {line}" for line in placeholders) + "\n")
    (C.RESULTS_DIR / "tab_edge_rows.md").write_text(markdown)
    print(markdown)
    print(f"-> {C.RESULTS_DIR / 'E09b_profile_laptop.json'}")


if __name__ == "__main__":
    main()
