import argparse
import glob
import os
import random

import antropy as ant
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ---------------------------
# Configuration
# ---------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# DEFAULT_FS_TARGETS = [3, 6, 9, 12, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 600, 700, 800, 900, 1000]
DEFAULT_FS_TARGETS = [3, 6, 9, 12, 15]
DEFAULT_WINDOW_SEC = 120  # test out also 120
# DEFAULT_SAMPLE_SIZE = 147000
DEFAULT_SAMPLE_SIZE = 15
DEFAULT_PATTERN = "*.mat"
DEFAULT_INPUT_DIR = os.path.abspath(os.path.join(script_dir, "..", "data_khodadad2018_200Hz"))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, "..", "figures", "complexity_sensitivity_frequency"))
DEFAULT_RANDOM_SEED = 42
DEFAULT_WINDOWS_PER_SEGMENT = 5
# DEFAULT_WINDOWS_PER_SEGMENT = 3
DEFAULT_FS = 500.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run respiratory LZC sensitivity analysis over sampling rates. "
            "Expects .mat files output from prep.py with pre_segments, dur_segments, and fs fields."
        )
    )
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT_DIR,
        help="Directory containing cleaned and segmented .mat files",
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN,
        help="Pattern to match .mat files",
    )
    parser.add_argument(
        "--fs", type=int, default=DEFAULT_FS,
        help="Sampling rate of the data",
    )
    parser.add_argument(
        "--fs-targets", type=float, nargs="+", default=DEFAULT_FS_TARGETS,
        help="Sampling rates to test (example: --fs-targets 100 200 300)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Number of participant files to analyze",
    )
    parser.add_argument(
        "--window", type=float, default=DEFAULT_WINDOW_SEC,
        help="Window sizes in seconds to test (example: --window 10)",
    )
    parser.add_argument(
        "--windows-per-segment", type=int, default=DEFAULT_WINDOWS_PER_SEGMENT,
        help="Number of random windows to draw from each eligible segment per window size",
    )
    parser.add_argument(
        "--stability-threshold", type=float, default=0.02,
        help="Relative change threshold used to identify stabilization",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures and CSV summaries",
    )
    parser.add_argument(
        "--skip-saving-figure", action="store_false",
        help="Flag to not save the generated figures after showing them",
    )
    return parser.parse_args()

# ---------------------------
# Phases & Tools
# ---------------------------

def discover_files(input_dir, pattern):
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    return files


def cell_array_to_list(cell_array):
    """Unpack a MATLAB cell array (object array) back into a list of 1D segments."""
    cell_array = np.asarray(cell_array, dtype=object).reshape(-1)
    segments = []
    for cell in cell_array:
        segment = np.asarray(cell, dtype=float).reshape(-1)
        segments.append(segment)
    return segments


def group_by_sampling_rate(results):
    summary = (
        results.groupby("FS")
        .agg(
            N_Segments=("Name", "nunique"),
            LZC_Mean_Mean=("LZC_Mean", "mean"),
            LZC_Mean_Std=("LZC_Mean", "std"),
            LZC_Mean_Median=("LZC_Mean", "median"),
            Binary_Balance_Mean=("Binary_Balance_Mean", "mean"),
            Binary_Balance_Std=("Binary_Balance_Mean", "std"),
        )
        .reset_index()
        .sort_values("FS")
    )

    summary["LZC_Abs_Delta"] = summary["LZC_Mean_Mean"].diff().abs()
    summary["LZC_Rel_Change"] = summary["LZC_Abs_Delta"] / summary["LZC_Mean_Mean"].shift(1).abs()
    summary["Binary_Balance_Delta_From_0.5"] = (summary["Binary_Balance_Mean"] - 0.5).abs()

    return summary


def estimate_stabilization_fs(summary, threshold):
    rel_change = summary["LZC_Rel_Change"].to_numpy(dtype=float)
    fs_vals = summary["FS"].to_numpy(dtype=float)

    # Find 1st sampling rate where the relative change is below the threshold.
    # If next value exists, check if it's below the threshold to reduce false positives.
    for idx in range(1, len(fs_vals)):
        current = rel_change[idx]
        if current < threshold:
            if idx + 1 < len(fs_vals):
                next_val = rel_change[idx + 1]
                if next_val < threshold:
                    return float(fs_vals[idx])
            else:
                return float(fs_vals[idx])

    return None


def process_segment(segment, window_sec, n_windows_per_segment, fs, fs_targets, rng):
    window_points = int(round(float(window_sec) * fs))
    min_required_len = n_windows_per_segment * window_points

    if len(segment) < min_required_len:  # maybe segment.size?
        return pd.DataFrame()

    segment_fs_info = []
    for fs_target in fs_targets:  # maybe add track progress tqdm?
        n_window_samples = int(round(window_sec * fs))
        lzc_values = []
        balance_values = []

        # Draw a random non-overlapping tiling of windows by selecting one offset,
        # then placing fixed-size windows back-to-back.
        max_offset = len(segment) - (n_windows_per_segment * n_window_samples)
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        starts = [offset + (k * n_window_samples) for k in range(n_windows_per_segment)]

        for start in starts:
            window = segment[start : start + n_window_samples]

            window_ds = window
            if fs_target != fs:  # Resample the window to the target frequency
                window_ds = np.asarray(
                    nk.signal_resample(window, sampling_rate=fs, desired_sampling_rate=fs_target), dtype=float)

            binary = window_ds > np.median(window_ds).item()
            lzc = ant.lziv_complexity(binary, normalize=True)
            lzc_values.append(lzc)
            balance_values.append(float(np.mean(binary)))

        lzc_values = np.asarray(lzc_values, dtype=float)
        balance_values = np.asarray(balance_values, dtype=float)
        segment_fs_info.append({
            "FS": float(fs_target),
            "LZC_Mean": float(np.mean(lzc_values)),
            "LZC_Std": float(np.std(lzc_values)),
            "LZC_CV": float(np.std(lzc_values) / np.mean(lzc_values)) if np.mean(lzc_values) > 0 else np.nan,
            "Binary_Balance_Mean": float(np.mean(balance_values)),
            "Binary_Balance_Std": float(np.std(balance_values))
        })

    return pd.DataFrame(segment_fs_info)

# ---------------------------
# Summaries & Plots
# ---------------------------

def print_summary_overview(segments_pd, segments_aggregate_pd, stabilization_fs, threshold):
    print("\n" + "=" * 72)
    print("LZC COMPLEXITY WINDOW SENSITIVITY")
    print("=" * 72)
    print(f"Segments analyzed: {segments_pd['Name'].nunique()}")
    print(f"Sampling rates tested: {', '.join(str(int(fs)) for fs in segments_aggregate_pd['FS'])}")
    print(f"Stability threshold: {threshold:.3f} relative change")

    print("\nPer-fs summary across sampled segments:")
    for _, row in segments_aggregate_pd.iterrows():
        rel_change = row["LZC_Rel_Change"]
        rel_change_str = f"{rel_change:.4f}" if np.isfinite(rel_change) else "-.----"
        print(
            f"  Sampling Rate {row['FS']:.0f} Hz | "
            f"LZC={row['LZC_Mean_Mean']:.4f} ± {row['LZC_Mean_Std']:.4f} | "
            f"CV={row['LZC_Mean_Std'] / row['LZC_Mean_Mean'] if row['LZC_Mean_Mean'] > 0 else np.nan:.4f} | "
            f"rel_change={rel_change_str} | "
            f"binary_balance={row['Binary_Balance_Mean']:.4f} | segments={int(row['N_Segments'])}"
        )

    if stabilization_fs is not None:
        print(f"\nEstimated stabilization sampling rate: ~{stabilization_fs:.0f} Hz")
    else:
        print("\nNo clear stabilization sampling rate found under the current heuristic.")

    print("=" * 72 + "\n")


def plot_results(segments_pd, segments_aggregate_pd, stabilization_fs, threshold, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)

    # Top: individual file curves + mean curve
    ax1 = axes[0]
    for segment_name, grp in segments_pd.groupby("Name"):
        ax1.plot(
            grp["FS"],
            grp["LZC_Mean"],
            color="0.75",
            linewidth=1.0,
            alpha=0.7,
        )

    ax1.errorbar(
        segments_aggregate_pd["FS"],
        segments_aggregate_pd["LZC_Mean_Mean"],
        yerr=segments_aggregate_pd["LZC_Mean_Std"],
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#1f77b4",
        label="Mean across sampled segments",
    )
    if stabilization_fs is not None:
        ax1.axvline(stabilization_fs, color="red", linestyle="--", label=f"Stabilization (<= {threshold:.0%}) ~{stabilization_fs:.0f} Hz")
    ax1.set_ylabel("Normalized LZC")
    ax1.set_title("Respiratory LZC Stability Across Sampling Rates")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    # Bottom: relative change between adjacent sampling rates
    ax2 = axes[1]
    ax2.plot(
        segments_aggregate_pd["FS"],
        segments_aggregate_pd["LZC_Rel_Change"],
        marker="o",
        color="#ff7f0e",
        linewidth=2.0,
        label="Relative change from previous sampling rate",
    )
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax2.axhline(float(threshold), color="red", linestyle="--", linewidth=1.0, alpha=0.7, label=f"{threshold:.0%} threshold")
    if stabilization_fs is not None:
        ax2.axvline(stabilization_fs, color="red", linestyle="--")
    ax2.set_ylabel("Relative Change")
    ax2.set_title("Relative Change in LZC Across Sampling Rates")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()

    save_path = save_path + ".png"
    fig.savefig(save_path, dpi=180)
    print(f"Saved figure: {save_path}")
    plt.show()


def point_transition_colors(segments_aggregate_pd, threshold):
    colors = []
    lzc_values = segments_aggregate_pd["LZC_Mean_Mean"].to_numpy(dtype=float)

    for idx, value in enumerate(lzc_values):
        if idx == 0:
            colors.append("black")
            continue

        prev = lzc_values[idx - 1]
        if prev == 0:
            colors.append("red")
            continue

        rel_change = abs(value - prev) / abs(prev)
        colors.append("green" if rel_change <= threshold else "red")

    return colors


def plot_relative_lzc_figure(segments_aggregate_pd, stabilization_fs, threshold, save_path):
    fig, ax = plt.subplots(figsize=(20, 6))

    x = segments_aggregate_pd["FS"].to_numpy(dtype=float)
    y = segments_aggregate_pd["LZC_Mean_Mean"].to_numpy(dtype=float)
    colors = point_transition_colors(segments_aggregate_pd, threshold)

    ax.plot(x, y, color="#1f77b4", linewidth=2.2, label="Normalized LZC")
    ax.scatter(x, y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.6)

    if stabilization_fs is not None:
        ax.axvline(stabilization_fs, color="red", linestyle="--", label=f"Stabilization ~{stabilization_fs:.0f} Hz")

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=2.2, label="Normalized LZC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label=f"<= {threshold:.0%} change"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label=f"> {threshold:.0%} change"),
    ]
    if stabilization_fs is not None:
        legend_items.append(Line2D([0], [0], color="red", linestyle="--", lw=1.5, label=f"Stabilization ~{stabilization_fs:.0f} Hz"))

    ax.set_title("Mean Respiratory LZC Across Sampling Rates", fontsize=14)
    ax.set_xlabel("Sampling Rate [Hz]")
    ax.set_ylabel("Normalized LZC")
    ax.grid(alpha=0.25)
    ax.legend(handles=legend_items, loc="best")
    fig.tight_layout()

    save_path = save_path + "_relative.png"
    fig.savefig(save_path, dpi=180)
    print(f"Saved figure: {save_path}")
    plt.show()
    plt.close(fig)

# ---------------------------
# Main loop
# ---------------------------

def main():
    args = parse_args()
    rng = random.Random(DEFAULT_RANDOM_SEED)
    min_segment_length_samples = args.window * args.windows_per_segment * args.fs
    print(f"\nMinimum segment length required for analysis: {min_segment_length_samples:.0f} samples ({min_segment_length_samples / args.fs:.1f} seconds)")

    # OBTAIN NECESSARY SEGMENTS

    files = discover_files(args.input_dir, args.pattern)
    if not files:
        raise FileNotFoundError(
            f"No files matched {os.path.join(args.input_dir, args.pattern)}"
        )

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    rng.shuffle(files)
    pre_segments = {}
    dur_segments = {}

    print("\nFiles discovered for analysis:")
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        mat_data = loadmat(path)
        file_pre_segments = cell_array_to_list(mat_data.get("pre_segments", np.empty((0, 0), dtype=object)))
        file_dur_segments = cell_array_to_list(mat_data.get("dur_segments", np.empty((0, 0), dtype=object)))

        print(f"  - {name} | pre-segments: {len(file_pre_segments)} | dur-segments: {len(file_dur_segments)}")

        if len(pre_segments) < args.sample_size:
            for idx, pre_seg in enumerate(file_pre_segments):
                print(f"    - Pre-segment {idx} | {len(pre_seg)} samples | {min_segment_length_samples:.0f} required")
                if len(pre_seg) >= min_segment_length_samples:
                    pre_segments[name + f"_{idx}"] = pre_seg
                    print(f"      -> ADDED")

        if len(dur_segments) < args.sample_size:
            for idx, dur_seg in enumerate(file_dur_segments):
                print(f"    - Dur-segment {idx} | {len(dur_seg)} samples | {min_segment_length_samples:.0f} required")
                if len(dur_seg) >= min_segment_length_samples:
                    dur_segments[name + f"_{idx}"] = dur_seg
                    print(f"      -> ADDED")

        if len(pre_segments) >= args.sample_size and len(dur_segments) >= args.sample_size:
            break

    print(f"\nFound {len(pre_segments)} pre-segments to analyze:")
    for path in set(list(pre_segments.keys())):
        print(f"  - {os.path.basename(path)}")
    
    print(f"\nFound {len(dur_segments)} dur-segments to analyze:")
    for path in set(list(dur_segments.keys())):
        print(f"  - {os.path.basename(path)}")

    segments = {**pre_segments, **dur_segments}

    # PROCESS EACH SEGMENT

    segments_fs_info = []
    for segment_name, segment_data in segments.items():
        print(f"\nProcessing {segment_name}...")
        segment_info = process_segment(segment_data, args.window, args.windows_per_segment, args.fs, args.fs_targets, rng)
        segment_info.insert(0, "Name", segment_name)
        segments_fs_info.append(segment_info)

    # SUMMARIZE & PLOT RESULTS

    segments_pd = pd.concat(segments_fs_info, ignore_index=True)
    # Aggregate results by sampling rate across all segments
    segments_fs_pd = group_by_sampling_rate(segments_pd)
    # Estimate stabilization sampling rate based on relative change heuristic
    stabilization_fs = estimate_stabilization_fs(segments_fs_pd, args.stability_threshold)

    input_folder = os.path.basename(os.path.normpath(args.input_dir))
    method_name = input_folder[5:]  # to remove "data_" prefix
    output_stem = f"lzc_frequency_{method_name}_n{len(segments)}_{args.window:.0f}s"
    save_path = os.path.join(args.output_dir, output_stem)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = save_path + "_summary.csv"
    segments_fs_pd.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    print_summary_overview(segments_pd, segments_fs_pd, stabilization_fs, args.stability_threshold)
    plot_results(segments_pd, segments_fs_pd, stabilization_fs, args.stability_threshold, save_path)
    plot_relative_lzc_figure(segments_fs_pd, stabilization_fs, args.stability_threshold, save_path)
    print("\n")


if __name__ == "__main__":
    main()
