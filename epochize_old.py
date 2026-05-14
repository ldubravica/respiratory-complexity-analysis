import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import loadmat, savemat

# ---------------------------
# Configuration
# ---------------------------

DEFAULT_INPUT_DIR = "data_khodadad2018_200Hz"
DEFAULT_EPOCH_LENGTH_SEC = 120.0
DEFAULT_INPUT_PATTERN = ""
DEFAULT_DROP_BAD_EPOCHS = True
DEFAULT_SAVE_DROPPED_PLOTS = False
DEFAULT_SHOW_DROPPED_PLOTS = True

# Respiratory epoch QC values
MIN_STD = 0.01
MIN_ROBUST_AMP = 0.1
MIN_PEAK_DISTANCE_SEC = 0.8
BPM_MIN = 3.0
BPM_MAX = 60.0
MAX_IBI_CV = 1
MAX_CLIP_FRAC = 0.10
MAX_DIFF_MAD_RATIO = 10.0


def parse_args():
    parser = argparse.ArgumentParser(description=("Split filtered respiratory pre/during segments into epochs."))
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing preprocessed .mat files")
    parser.add_argument("--epoch-length-sec", type=float, default=DEFAULT_EPOCH_LENGTH_SEC, help="Epoch length in seconds")
    parser.add_argument("--input-pattern", default=DEFAULT_INPUT_PATTERN, help="Input file names' pattern to match (e.g., 'khodadad2018-200Hz')")
    parser.add_argument("--keep-all-epochs", dest="drop_bad_epochs", action="store_false", default=DEFAULT_DROP_BAD_EPOCHS, help="Skip quality control and dropping bad epochs")
    parser.add_argument("--save-plots", dest="save_dropped_plots", action="store_true", default=DEFAULT_SAVE_DROPPED_PLOTS, help="Save rejected-epoch plots")
    parser.add_argument("--skip-plots", dest="show_dropped_epoch_plots", action="store_false", default=DEFAULT_SHOW_DROPPED_PLOTS, help="Do not display rejected-epoch plots interactively")
    return parser.parse_args()

# ---------------------------
# Tools
# ---------------------------

def cell_array_to_list(cell_array):
    """Unpack a MATLAB cell array (object array) back into a list of 1D segments."""
    cell_array = np.asarray(cell_array, dtype=object).reshape(-1)
    segments = []
    for cell in cell_array:
        segment = np.asarray(cell, dtype=float).reshape(-1)
        segments.append(segment)
    return segments


def _epoch_qc_metrics(epoch, fs):
    """Compute simple respiratory quality-control metrics for one epoch.

    The heuristics are intentionally transparent:
    - flatline / low variance: reject if the signal barely varies
    - std: close to 0 means the epoch is nearly constant.
    - robust_amp: very small means there is almost no respiratory movement.
    - clipping: reject if a large fraction of samples sit near the extremes
    - spikeiness: reject if successive differences are unusually large
    - breath timing: detect peaks and reject impossible breathing rates or very irregular intervals
    """
    metrics = {
        "std": np.nan,
        "robust_amp": np.nan,
        "clip_frac": np.nan,
        "diff_mad_ratio": np.nan,
        "n_peaks": 0,
        "bpm": np.nan,
        "ibi_cv": np.nan,
    }

    n_samples = epoch.size
    if n_samples == 0:
        return metrics

    # Standard deviation of the epoch, a measure of overall variability. Very low values may indicate flatline or near-constant signal.
    metrics["std"] = float(np.std(epoch))

    # Robust amplitude: difference between 5th and 95th percentiles, which is less sensitive to outliers than the full range.
    q05, q95 = np.percentile(epoch, [5, 95])
    metrics["robust_amp"] = float(q95 - q05)

    # Approximate clipping/saturation: too many samples are stuck near the low/high ends.
    clip_eps = max(metrics["robust_amp"] * 0.01, 1e-12)
    near_low = np.mean(np.abs(epoch - np.min(epoch)) <= clip_eps)
    near_high = np.mean(np.abs(epoch - np.max(epoch)) <= clip_eps)
    metrics["clip_frac"] = float(max(near_low, near_high))

    # Median absolute deviation of successive differences divided by amplitude scale.
    # Higher values mean more jumpy or spiky behavior.
    diffs = np.diff(epoch)
    diff_mad = np.median(np.abs(diffs - np.median(diffs)))
    amp_scale = max(metrics["robust_amp"], 1e-12)
    metrics["diff_mad_ratio"] = float(diff_mad / amp_scale) if np.isfinite(diff_mad) else np.nan

    # Breath detection: peaks in the respiratory trace.
    # Use a minimum distance so we do not count noise as breaths.
    min_peak_distance = max(1, int(round(MIN_PEAK_DISTANCE_SEC * fs)))
    peak_distance = max(1, min_peak_distance)
    peak_prominence = max(metrics["robust_amp"] * 0.10, 1e-12)
    peaks, _ = find_peaks(epoch, distance=peak_distance, prominence=peak_prominence)
    metrics["n_peaks"] = int(peaks.size)

    # Estimate breathing rate in breaths per minute (BPM).
    duration_sec = n_samples / float(fs)
    if duration_sec > 0:
        metrics["bpm"] = float(60.0 * peaks.size / duration_sec)
    
    # Coefficient of variation of inter-breath intervals (IBI), a measure of breath timing regularity. 
    # Higher values indicate more irregular breathing.
    ibis = np.diff(peaks) / float(fs)
    if ibis.size >= 2 and np.mean(ibis) > 0:
        metrics["ibi_cv"] = float(np.std(ibis) / np.mean(ibis))

    return metrics


def drop_bad_epochs(epochs, fname, fs, args):
    """Return only epochs that pass simple respiratory QC.

    Parameters
    ----------
    epochs : array-like or list
        Epochs shaped like (n_epochs, n_samples) or a list of 1D arrays.
    fs : float
        Sampling rate in Hz

    Returns
    -------
    kept_epochs : np.ndarray
        Epochs that passed QC.
    dropped_epochs : list[dict]
        Rejected epochs with the original epoch samples included for plotting.
    report : list[dict]
        Per-epoch metrics and rejection reason.
    """
    if epochs.size == 0:
        return np.empty((0, 0), dtype=float), [], []

    kept_epochs = []
    dropped_epochs = []
    report = []

    for epoch_idx, epoch in enumerate(epochs):
        metrics = _epoch_qc_metrics(epoch, fs)
        reasons = []

        if not np.isfinite(metrics["std"]) or metrics["std"] < MIN_STD:
            reasons.append(f"STD < {MIN_STD:g}")
        if not np.isfinite(metrics["robust_amp"]) or metrics["robust_amp"] < MIN_ROBUST_AMP:
            reasons.append(f"Robust Amplitude < {MIN_ROBUST_AMP:g}")
        if np.isfinite(metrics["clip_frac"]) and metrics["clip_frac"] > MAX_CLIP_FRAC:
            reasons.append(f"Clip Fraction > {MAX_CLIP_FRAC:.2f}")
        if np.isfinite(metrics["diff_mad_ratio"]) and metrics["diff_mad_ratio"] > MAX_DIFF_MAD_RATIO:
            reasons.append(f"Diff MAD Ratio > {MAX_DIFF_MAD_RATIO:.1f}")

        # Peak-based rules only apply when we have enough detected breaths.
        if metrics["n_peaks"] < 2:
            reasons.append("too_few_peaks")
        else:
            if not np.isfinite(metrics["bpm"]) or metrics["bpm"] < BPM_MIN or metrics["bpm"] > BPM_MAX:
                reasons.append(f"BPM not in [{BPM_MIN:.1f},{BPM_MAX:.1f}]")
            if np.isfinite(metrics["ibi_cv"]) and metrics["ibi_cv"] > MAX_IBI_CV:
                reasons.append(f"IBI CV > {MAX_IBI_CV:.2f}")

        is_bad = len(reasons) > 0
        report.append({
            "epoch_index": epoch_idx,
            "is_bad": is_bad,
            "reasons": reasons,
            **metrics,
        })

        if not is_bad:
            kept_epochs.append(np.asarray(epoch, dtype=float).reshape(-1))
        else:
            dropped_epoch = {
                "fname": fname,
                "epoch_index": epoch_idx,
                "epoch": np.asarray(epoch, dtype=float).reshape(-1),
                "reasons": reasons,
                **metrics,
            }
            dropped_epochs.append(dropped_epoch)

            if args.show_dropped_epoch_plots or args.save_dropped_plots:
                out_path = None
                if args.save_dropped_plots:
                    plot_dir = os.path.join(args.out_directory, f"{os.path.splitext(fname)[0]}-{args.epochizing_info}-dropped-epochs")
                    os.makedirs(plot_dir, exist_ok=True)
                    out_path = os.path.join(plot_dir, f"pre_epoch_{epoch_idx:04d}.png") if args.save_dropped_plots else None
                _plot_rejected_epoch(dropped_epoch, fs, out_path=out_path, show=args.show_dropped_epoch_plots)

    if not kept_epochs:
        return np.empty((0, epochs.shape[1]), dtype=float), dropped_epochs, report

    return np.stack(kept_epochs, axis=0), dropped_epochs, report


def _plot_rejected_epoch(row, fs, out_path=None, show=True):
    """Create a quick-look figure for a rejected epoch."""
    epoch = np.asarray(row["epoch"], dtype=float).reshape(-1)
    t = np.arange(epoch.size) / float(fs)
    peaks, _ = find_peaks(
        epoch,
        distance=max(1, int(round(MIN_PEAK_DISTANCE_SEC * fs))),
        prominence=max(row.get("robust_amp", 0.0) * 0.10, 1e-12),
    )

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    fig.suptitle(
        f"{row['fname']}: Dropped epoch {row['epoch_index']} - reasons: {', '.join(row['reasons']) or 'n/a'}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    axes.plot(t, epoch, color="#1f77b4", linewidth=1.0)
    if peaks.size:
        axes.plot(t[peaks], epoch[peaks], "ro", markersize=4, label="Detected peaks")
        axes.legend(loc="best", fontsize=8)
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Amplitude")
    axes.grid(alpha=0.25)

    text_lines = [
        f"Standard Deviation={row['std']:.4g}",
        f"Robust Amplitude={row['robust_amp']:.4g}",
        f"Clip Fraction={row['clip_frac']:.3f}",
        f"Difference MAD Ratio={row['diff_mad_ratio']:.3f}",
        f"Number of Peaks={row['n_peaks']}",
        f"Breathing Rate (BPM)={row['bpm']:.3f}" if np.isfinite(row.get('bpm', np.nan)) else "Breathing Rate (BPM)=nan",
        f"Inter-Breath Interval - Variation Coefficient (IBI CV)={row['ibi_cv']:.3f}" if np.isfinite(row.get('ibi_cv', np.nan)) else "IBI CV=nan",
    ]
    axes.text(
        0.99,
        0.02,
        "\n".join(text_lines),
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.75"),
    )

    if out_path: fig.savefig(out_path, dpi=180)
    if show: plt.show()

    plt.close(fig)


def _write_qc_report_csv(report_rows, out_path):
    """Write a flat CSV report for all epochs examined in one input file."""
    fieldnames = [
        "segment_type",
        "epoch_index",
        "is_bad",
        "reasons",
        "std",
        "robust_amp",
        "clip_frac",
        "diff_mad_ratio",
        "n_peaks",
        "bpm",
        "ibi_cv",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            flat_row = {key: row.get(key, "") for key in fieldnames}
            flat_row["reasons"] = "; ".join(row.get("reasons", []))
            writer.writerow(flat_row)


def epochize_segments(segments, epoch_length_samples):
    if epoch_length_samples < 1:
        return np.empty((0, epoch_length_samples), dtype=float)

    epochs = []
    for segment in segments:
        n_epochs = len(segment) // epoch_length_samples
        print(f"    {n_epochs} n_epochs = {len(segment)} samples // {epoch_length_samples} samples/epoch")
        if n_epochs == 0:
            continue

        trimmed = np.asarray(segment[: n_epochs * epoch_length_samples], dtype=float)
        segment_epochs = trimmed.reshape(n_epochs, epoch_length_samples)
        for epoch in segment_epochs:
            epochs.append(epoch)

    if not epochs:
        return np.empty((0, epoch_length_samples), dtype=float)

    return np.stack(epochs, axis=0)

# ---------------------------
# Main loop
# ---------------------------

def main():
    args = parse_args()

    preprocessing_info = args.input_dir[5:]  # to remove "data_" prefix
    epochizing_info = f"{int(round(args.epoch_length_sec))}s"
    epoch_category = "_filtered" if args.drop_bad_epochs else "_all"
    out_directory = f"data_{preprocessing_info}_{epochizing_info}{epoch_category}"
    os.makedirs(out_directory, exist_ok=True)
    
    args.epochizing_info = epochizing_info
    args.out_directory = out_directory

    files = sorted(os.listdir(args.input_dir))
    # files = files[:2]  # TESTING

    for fname in files:
        if not fname.lower().endswith(".mat"):
            continue

        if args.input_pattern not in fname:
            continue

        path = os.path.join(args.input_dir, fname)
        print(f"\nProcessing {fname}...")

        # 1) load filtered, downsampled, and segmented data
        mat_data = loadmat(path)
        fs = float(mat_data["fs"].item())
        pre_segments = cell_array_to_list(mat_data["pre_segments"])
        dur_segments = cell_array_to_list(mat_data["dur_segments"])

        # 2) determine epoch length in samples
        epoch_length_samples = int(round(args.epoch_length_sec * fs))
        if epoch_length_samples < 1:
            print("  Skipping: invalid epoch length.")
            continue

        # 3) epochize segments
        print("\n  Epochizing pre segments:", [cell.shape for cell in mat_data["pre_segments"].reshape(-1)])
        pre_epochs = epochize_segments(pre_segments, epoch_length_samples)
        print("  Epochizing dur segments:", [cell.shape for cell in mat_data["dur_segments"].reshape(-1)])
        dur_epochs = epochize_segments(dur_segments, epoch_length_samples)

        if pre_epochs.shape[0] == 0 and dur_epochs.shape[0] == 0:
            print(f"  Skipping: no complete {args.epoch_length_sec}s epochs found.")
            continue

        # 4) drop bad epochs
        if args.drop_bad_epochs:
            print("  Screening pre epochs for artifacts...")
            pre_epochs, pre_dropped, pre_report = drop_bad_epochs(pre_epochs, fname, fs, args)
            print("  Screening dur epochs for artifacts...")
            dur_epochs, dur_dropped, dur_report = drop_bad_epochs(dur_epochs, fname, fs, args)

            n_pre_bad = sum(row["is_bad"] for row in pre_report)
            n_dur_bad = sum(row["is_bad"] for row in dur_report)
            print(f"  Rejected {n_pre_bad} pre-epochs and {n_dur_bad} dur-epochs using QC rules.")

            if pre_epochs.shape[0] == 0 and dur_epochs.shape[0] == 0:
                print("  Skipping: all epochs were rejected by QC.")
                continue

            # Save a flat CSV report for inspection.
            qc_report_rows = []
            for row in pre_report:
                qc_report_rows.append({"segment_type": "pre", **row})
            for row in dur_report:
                qc_report_rows.append({"segment_type": "dur", **row})

            report_name = f"{os.path.splitext(fname)[0]}-{epochizing_info}-qc-report.csv"
            report_path = os.path.join(out_directory, report_name)
            _write_qc_report_csv(qc_report_rows, report_path)
            print(f"  Saved QC report: {report_path}")

        # 4) store epochized data
        out_name = f"{os.path.splitext(fname)[0]}-{epochizing_info}.mat"
        out_path = os.path.join(out_directory, out_name)

        savemat(
            out_path,
            {
                "pre_epochs": pre_epochs,
                "dur_epochs": dur_epochs,
                "fs": fs,
                "epoch_length_sec": float(args.epoch_length_sec),
            },
        )

        print(f"\n  Saved {pre_epochs.shape[0]} pre-epochs and {dur_epochs.shape[0]} dur-epochs to {out_path}")


if __name__ == "__main__":
    main()
    print()
