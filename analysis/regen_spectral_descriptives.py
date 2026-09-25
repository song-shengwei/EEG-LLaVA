#!/usr/bin/env python3
"""Regenerate current-dataset spectral descriptive panels (Pre_03--Pre_06).

All panels use the locked 8,861-segment LMDB and the manuscript's /100 input
scale. Welch values are used for PSD/SSVEP panels; cross-band and Cohen-d
panels retain the earlier FFT-band definition but are explicitly descriptive
at the segment level. No segment-level p-values are produced.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

CHANNELS = ["PO3", "POz", "PO4", "O1", "Oz", "O2"]
BANDS = {
    "Delta (0.5–4 Hz)": (0.5, 4.0),
    "Theta (4–8 Hz)": (4.0, 8.0),
    "SSVEP (8–11.8 Hz)": (8.0, 11.8),
    "Alpha (8–13 Hz)": (8.0, 13.0),
    "Beta (13–30 Hz)": (13.0, 30.0),
}
FS, NPERSEG, NOVERLAP = 200, 200, 100
BLUE, LBLUE = "#1565C0", "#90CAF9"
RED, LRED = "#C62828", "#EF9A9A"
GOLD, GRAY = "#E65100", "#546E7A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, format=suffix, bbox_inches="tight", pad_inches=0.06,
                    dpi=220 if suffix == "png" else None)
    plt.close(fig)


def compute(data_dir: Path, batch_size: int = 256) -> dict:
    env = lmdb.open(str(data_dir), readonly=True, lock=False, readahead=False,
                    meminit=False)
    with env.begin(write=False) as txn:
        split_keys = pickle.loads(txn.get(b"__keys__"))
        keys = split_keys["train"] + split_keys["val"] + split_keys["test"]
    if len(keys) != len(set(keys)):
        raise ValueError("LMDB split key lists contain duplicates")
    n = len(keys)
    labels = np.empty(n, dtype=np.int8)
    band_values = {name: np.empty((n, 6), dtype=np.float64) for name in BANDS}
    fft_freq = np.fft.rfftfreq(1000, d=1 / FS)
    band_masks = {name: (fft_freq >= lo) & (fft_freq < hi)
                  for name, (lo, hi) in BANDS.items()}
    welch_freq = np.fft.rfftfreq(NPERSEG, d=1 / FS)
    welch_sums = {0: np.zeros((6, len(welch_freq))),
                  1: np.zeros((6, len(welch_freq)))}
    counts = {0: 0, 1: 0}

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        samples, batch_labels = [], []
        with env.begin(write=False) as txn:
            for key in keys[start:stop]:
                pair = pickle.loads(txn.get(key.encode()))
                samples.append(np.asarray(pair["sample"], dtype=np.float32)
                               .reshape(6, 1000) / 100.0)
                batch_labels.append(int(pair["label"]))
        signal = np.stack(samples)
        y = np.asarray(batch_labels, dtype=np.int8)
        labels[start:stop] = y

        fft_power = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
        for name, mask in band_masks.items():
            band_values[name][start:stop] = fft_power[:, :, mask].mean(axis=-1)

        freq, psd = welch(signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
                          axis=-1)
        if not np.array_equal(freq, welch_freq):
            raise AssertionError("unexpected Welch frequency grid")
        for label in (0, 1):
            selected = psd[y == label]
            welch_sums[label] += selected.sum(axis=0, dtype=np.float64)
            counts[label] += len(selected)
    env.close()

    if n != 8861 or counts != {0: 4174, 1: 4687}:
        raise AssertionError(f"unexpected locked counts: n={n}, groups={counts}")
    welch_means = {label: welch_sums[label] / counts[label] for label in (0, 1)}

    bands = {}
    for name, values in band_values.items():
        healthy, glaucoma = values[labels == 0], values[labels == 1]
        n_h, n_g = len(healthy), len(glaucoma)
        h_mean, g_mean = healthy.mean(axis=0), glaucoma.mean(axis=0)
        pooled_sd = np.sqrt(((n_h - 1) * healthy.var(axis=0, ddof=1)
                             + (n_g - 1) * glaucoma.var(axis=0, ddof=1))
                            / (n_h + n_g - 2))
        signed_d = (g_mean - h_mean) / (pooled_sd + 1e-12)
        h_all, g_all = healthy.mean(), glaucoma.mean()
        bands[name] = {
            "fft_rule": f"mean abs(rfft)^2 where {BANDS[name][0]} <= f < {BANDS[name][1]} Hz",
            "healthy_mean": float(h_all),
            "glaucoma_mean": float(g_all),
            "reduction_pct": float((1 - g_all / h_all) * 100),
            "per_channel": {
                channel: {
                    "healthy_mean": float(h_mean[index]),
                    "glaucoma_mean": float(g_mean[index]),
                    "cohen_d_glaucoma_minus_healthy": float(signed_d[index]),
                    "abs_cohen_d": float(abs(signed_d[index])),
                }
                for index, channel in enumerate(CHANNELS)
            },
        }

    return {
        "status": "complete",
        "dataset": {"segments_total": n, "healthy": counts[0], "glaucoma": counts[1]},
        "input_scale": 0.01,
        "fs_hz": FS,
        "segment_dependence_note": "FFT band summaries and Cohen d are descriptive at the segment level; no segment-unit p-values are used for participant inference.",
        "welch": {
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
            "frequencies_hz": welch_freq.tolist(),
            "healthy_channel_psd_mean": welch_means[0].tolist(),
            "glaucoma_channel_psd_mean": welch_means[1].tolist(),
        },
        "fft_bands": bands,
    }


def draw(audit: dict, output: Path) -> None:
    freq = np.asarray(audit["welch"]["frequencies_hz"])
    h_psd = np.asarray(audit["welch"]["healthy_channel_psd_mean"])
    g_psd = np.asarray(audit["welch"]["glaucoma_channel_psd_mean"])
    band_names = list(BANDS)

    d_matrix = np.asarray([
        [audit["fft_bands"][band]["per_channel"][ch]["abs_cohen_d"]
         for ch in CHANNELS] for band in band_names
    ])
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    image = ax.imshow(d_matrix, cmap="Blues", aspect="auto", vmin=0,
                      vmax=max(0.08, float(d_matrix.max())))
    ax.set_xticks(range(6), CHANNELS)
    ax.set_yticks(range(5), ["Delta\n0.5–4", "Theta\n4–8", "SSVEP\n8–11.8",
                             "Alpha\n8–13", "Beta\n13–30"])
    for row in range(5):
        for col in range(6):
            ax.text(col, row, f"{d_matrix[row, col]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if d_matrix[row, col] > 0.055 else "#263238")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    bar.set_label("|Cohen's d|")
    save(fig, output, "Pre_03_cohend_heatmap")

    mask = (freq >= 8.0) & (freq <= 11.8)
    h_band = h_psd[:, mask].mean(axis=1).reshape(2, 3)
    g_band = g_psd[:, mask].mean(axis=1).reshape(2, 3)
    reduction = (1 - g_band / h_band) * 100
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6))
    vmax = max(float(h_band.max()), float(g_band.max()))
    for ax, values, title in ((axes[0], h_band, "Healthy Welch power"),
                              (axes[1], g_band, "Glaucoma Welch power")):
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        for row in range(2):
            for col in range(3):
                index = row * 3 + col
                ax.text(col, row, f"{CHANNELS[index]}\n{values[row, col]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if values[row, col] > vmax * 0.45 else "black")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes[:2], fraction=0.025, pad=0.02)
    cbar.set_label("μV²/Hz", fontsize=10)
    im_r = axes[2].imshow(reduction, cmap="OrRd", vmin=75, vmax=95, aspect="equal")
    for row in range(2):
        for col in range(3):
            index = row * 3 + col
            axes[2].text(col, row, f"{CHANNELS[index]}\n{reduction[row, col]:.1f}%",
                         ha="center", va="center", fontsize=9,
                         color="white" if reduction[row, col] > 87 else "black")
    axes[2].set_title("Reduction", fontsize=11)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    cbar_r = fig.colorbar(im_r, ax=axes[2], fraction=0.05, pad=0.02)
    cbar_r.set_label("%", fontsize=10)
    save(fig, output, "Pre_04_topomap_ssvep")

    show = (freq >= 1) & (freq <= 50)
    h_curve, g_curve = h_psd.mean(axis=0), g_psd.mean(axis=0)
    pooled_reduction = (1 - g_curve[mask].mean() / h_curve[mask].mean()) * 100
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.axvspan(8.0, 11.8, alpha=0.12, color="gold", label="SSVEP (8–11.8 Hz)")
    ax.semilogy(freq[show], h_curve[show], color=BLUE, lw=2.2,
                label="Healthy (n=4,174)")
    ax.semilogy(freq[show], g_curve[show], color=RED, lw=2.2,
                label="Glaucoma (n=4,687)")
    ax.text(9.9, max(h_curve[mask]) * 1.8, f"−{pooled_reduction:.1f}%",
            color=RED, ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (μV²/Hz, log)")
    ax.set_xlim(1, 50)
    ax.grid(alpha=0.2, which="both", lw=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, output, "Pre_05_psd_occipital")

    labels = ["δ", "θ", "SSVEP", "α", "β"]
    h_values = [audit["fft_bands"][band]["healthy_mean"] for band in band_names]
    g_values = [audit["fft_bands"][band]["glaucoma_mean"] for band in band_names]
    reductions = [audit["fft_bands"][band]["reduction_pct"] for band in band_names]
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 3.0), gridspec_kw={"hspace": 0.60})
    x = np.arange(5)
    width = 0.36
    axes[0].bar(x - width / 2, np.log10(h_values), width, color=LBLUE,
                edgecolor=BLUE, lw=0.8, label="Healthy")
    axes[0].bar(x + width / 2, np.log10(g_values), width, color=LRED,
                edgecolor=RED, lw=0.8, label="Glaucoma")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel(r"$\log_{10}$ Power")
    axes[0].legend(ncol=2, fontsize=9, loc="upper right")
    axes[0].grid(axis="y", alpha=0.2, lw=0.5)
    colors = [RED if value > 90 else GOLD if value > 85 else GRAY
              for value in reductions]
    bars = axes[1].bar(x, reductions, color=colors, edgecolor="white", lw=0.8)
    for bar, value in zip(bars, reductions):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.25,
                     f"{value:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Reduction (%)")
    axes[1].set_ylim(75, 98)
    axes[1].grid(axis="y", alpha=0.2, lw=0.5)
    save(fig, output, "Pre_06_freq_band_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error(f"refusing to mix with existing files: {args.output_dir}")
    audit = compute(args.data_dir)
    draw(audit, args.output_dir)
    (args.output_dir / "spectral_descriptive_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    print(f"wrote current-dataset spectral audit and panels to {args.output_dir}")


if __name__ == "__main__":
    main()
