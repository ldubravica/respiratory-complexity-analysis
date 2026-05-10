import argparse
import glob
import os
import random

import antropy as ant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ---------------------------
# Configuration
# ---------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# DEFAULT_WINDOWS_SEC = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
DEFAULT_WINDOWS_SEC = [10, 20, 30]
# DEFAULT_SAMPLE_SIZE = 147000
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_PATTERN = "*.mat"
DEFAULT_INPUT_DIR = os.path.abspath(os.path.join(script_dir, "..", "data_khodadad2018_200Hz"))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, "..", "figures", "complexity_sensitivity_window"))
DEFAULT_RANDOM_SEED = 42
DEFAULT_WINDOWS_PER_SEGMENT = 5
DEFAULT_FS = 200.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run respiratory LZC sensitivity analysis over window sizes. "
            "Expects .mat files output from prep.py with pre_segments, n2o_segments, and fs fields."
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
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Number of participant files to analyze",
    )
    parser.add_argument(
        "--windows", type=float, nargs="+", default=DEFAULT_WINDOWS_SEC,
        help="Window sizes in seconds to test (example: --windows 10 20 30 60)",
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


def group_by_window_size(results):
    summary = (
        results.groupby("Window_Sec")
        .agg(
            N_Segments=("Name", "nunique"),
            LZC_Mean_Mean=("LZC_Mean", "mean"),
            LZC_Mean_Std=("LZC_Mean", "std"),
            LZC_Mean_Median=("LZC_Mean", "median"),
            Binary_Balance_Mean=("Binary_Balance_Mean", "mean"),
            Binary_Balance_Std=("Binary_Balance_Mean", "std"),
        )
        .reset_index()
        .sort_values("Window_Sec")
    )

    summary["LZC_Abs_Delta"] = summary["LZC_Mean_Mean"].diff().abs()
    summary["LZC_Rel_Change"] = summary["LZC_Abs_Delta"] / summary["LZC_Mean_Mean"].shift(1).abs()
    summary["Binary_Balance_Delta_From_0.5"] = (summary["Binary_Balance_Mean"] - 0.5).abs()

    return summary


def estimate_stabilization_window(summary, threshold):
    rel_change = summary["LZC_Rel_Change"].to_numpy(dtype=float)
    windows = summary["Window_Sec"].to_numpy(dtype=float)

    # Find 1st window size where the relative change is below the threshold.
    # If next value exists, check if it's below the threshold to reduce false positives.
    for idx in range(1, len(windows)):
        current = rel_change[idx]
        if current < threshold:
            if idx + 1 < len(windows):
                next_val = rel_change[idx + 1]
                if next_val < threshold:
                    return float(windows[idx])
            else:
                return float(windows[idx])

    return None


def process_segment(segment, window_sizes_sec, n_windows_per_segment, fs, rng):
    max_window_points = int(round(float(max(window_sizes_sec)) * fs))
    min_required_len = n_windows_per_segment * max_window_points

    if len(segment) < min_required_len:  # maybe segment.size?
        return pd.DataFrame()

    segment_windows_info = []
    for window_sec in window_sizes_sec:  # maybe add track progress tqdm?
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
            binary = window > np.median(window).item()
            lzc = ant.lziv_complexity(binary, normalize=True)
            lzc_values.append(lzc)
            balance_values.append(float(np.mean(binary)))

        lzc_values = np.asarray(lzc_values, dtype=float)
        balance_values = np.asarray(balance_values, dtype=float)
        segment_windows_info.append({
            "Window_Sec": float(window_sec),
            "LZC_Mean": float(np.mean(lzc_values)),
            "LZC_Std": float(np.std(lzc_values)),
            "LZC_CV": float(np.std(lzc_values) / np.mean(lzc_values)) if np.mean(lzc_values) > 0 else np.nan,
            "Binary_Balance_Mean": float(np.mean(balance_values)),
            "Binary_Balance_Std": float(np.std(balance_values))
        })

    return pd.DataFrame(segment_windows_info)

# ---------------------------
# Summaries & Plots
# ---------------------------

def print_summary_overview(segments_pd, segments_windows_pd, stabilization_window, threshold):
    print("\n" + "=" * 72)
    print("LZC COMPLEXITY WINDOW SENSITIVITY")
    print("=" * 72)
    print(f"Segments analyzed: {segments_pd['Name'].nunique()}")
    print(f"Window sizes tested: {', '.join(str(int(w)) for w in segments_windows_pd['Window_Sec'])}")
    print(f"Stability threshold: {threshold:.3f} relative change")

    print("\nPer-window summary across sampled segments:")
    for _, row in segments_windows_pd.iterrows():
        rel_change = row["LZC_Rel_Change"]
        rel_change_str = f"{rel_change:.4f}" if np.isfinite(rel_change) else "-.----"
        print(
            f"  Window {row['Window_Sec']:.0f}s | "
            f"LZC={row['LZC_Mean_Mean']:.4f} ± {row['LZC_Mean_Std']:.4f} | "
            f"CV={row['LZC_Mean_Std'] / row['LZC_Mean_Mean'] if row['LZC_Mean_Mean'] > 0 else np.nan:.4f} | "
            f"rel_change={rel_change_str} | "
            f"binary_balance={row['Binary_Balance_Mean']:.4f} | segments={int(row['N_Segments'])}"
        )

    if stabilization_window is not None:
        print(f"\nEstimated stabilization window: ~{stabilization_window:.0f} seconds")
    else:
        print("\nNo clear stabilization window found under the current heuristic.")

    print("=" * 72 + "\n")


def plot_results(segments_pd, segments_windows_pd, stabilization_window, threshold, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

    # Top: individual file curves + mean curve
    ax1 = axes[0]
    for segment_name, grp in segments_pd.groupby("Name"):
        ax1.plot(
            grp["Window_Sec"],
            grp["LZC_Mean"],
            color="0.75",
            linewidth=1.0,
            alpha=0.7,
        )

    ax1.errorbar(
        segments_windows_pd["Window_Sec"],
        segments_windows_pd["LZC_Mean_Mean"],
        yerr=segments_windows_pd["LZC_Mean_Std"],
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#1f77b4",
        label="Mean across sampled segments",
    )
    if stabilization_window is not None:
        ax1.axvline(stabilization_window, color="red", linestyle="--", label=f"Stabilization (<= {threshold:.0%}) ~{stabilization_window:.0f}s")
    ax1.set_ylabel("Normalized LZC")
    ax1.set_title("Respiratory LZC Stability Across Window Sizes")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    # Middle: relative change between adjacent window sizes
    ax2 = axes[1]
    ax2.plot(
        segments_windows_pd["Window_Sec"],
        segments_windows_pd["LZC_Rel_Change"],
        marker="o",
        color="#ff7f0e",
        linewidth=2.0,
        label="Relative change from previous window",
    )
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax2.axhline(float(threshold), color="red", linestyle="--", linewidth=1.0, alpha=0.7, label=f"{threshold:.0%} threshold")
    if stabilization_window is not None:
        ax2.axvline(stabilization_window, color="red", linestyle="--")
    ax2.set_ylabel("Relative Change")
    ax2.set_title("Relative Change in LZC Across Window Sizes")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    # Bottom: binary balance diagnostic
    ax3 = axes[2]
    ax3.errorbar(
        segments_windows_pd["Window_Sec"],
        segments_windows_pd["Binary_Balance_Mean"],
        yerr=segments_windows_pd["Binary_Balance_Std"],
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#2ca02c",
        label="Mean binary balance across sampled segments",
    )
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="Ideal balance = 0.5")
    if stabilization_window is not None:
        ax3.axvline(stabilization_window, color="red", linestyle="--")
    ax3.set_xlabel("Window Length [s]")
    ax3.set_ylabel("Binary Balance")
    ax3.set_title("Binarization Quality Check")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")

    fig.tight_layout()

    save_path = save_path + ".png"
    fig.savefig(save_path, dpi=180)
    print(f"Saved figure: {save_path}")
    plt.show()


def point_transition_colors(segments_windows_pd, threshold):
    colors = []
    lzc_values = segments_windows_pd["LZC_Mean_Mean"].to_numpy(dtype=float)

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


def plot_relative_lzc_figure(segments_windows_pd, stabilization_window, threshold, save_path):
    fig, ax = plt.subplots(figsize=(20, 6))

    x = segments_windows_pd["Window_Sec"].to_numpy(dtype=float)
    y = segments_windows_pd["LZC_Mean_Mean"].to_numpy(dtype=float)
    colors = point_transition_colors(segments_windows_pd, threshold)

    ax.plot(x, y, color="#1f77b4", linewidth=2.2, label="Normalized LZC")
    ax.scatter(x, y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.6)

    if stabilization_window is not None:
        ax.axvline(stabilization_window, color="red", linestyle="--", label=f"Stabilization ~{stabilization_window:.0f}s")

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=2.2, label="Normalized LZC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label=f"<= {threshold:.0%} change"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label=f"> {threshold:.0%} change"),
    ]
    if stabilization_window is not None:
        legend_items.append(Line2D([0], [0], color="red", linestyle="--", lw=1.5, label=f"Stabilization ~{stabilization_window:.0f}s"))

    ax.set_title("Mean Respiratory LZC Across Window Sizes", fontsize=14)
    ax.set_xlabel("Window Length [s]")
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
    min_segment_length_samples = max(args.windows) * args.windows_per_segment * args.fs
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
    n2o_segments = {}

    print("\nFiles discovered for analysis:")
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        mat_data = loadmat(path)
        file_pre_segments = cell_array_to_list(mat_data.get("pre_segments", np.empty((0, 0), dtype=object)))
        file_n2o_segments = cell_array_to_list(mat_data.get("n2o_segments", np.empty((0, 0), dtype=object)))

        print(f"  - {name} | pre-segments: {len(file_pre_segments)} | n2o-segments: {len(file_n2o_segments)}")

        if len(pre_segments) < args.sample_size:
            for idx, pre_seg in enumerate(file_pre_segments):
                print(f"    - Pre-segment {idx} | {len(pre_seg)} samples | {min_segment_length_samples:.0f} required")
                if len(pre_seg) >= min_segment_length_samples:
                    pre_segments[name + f"_{idx}"] = pre_seg
                    print(f"      -> ADDED")

        if len(n2o_segments) < args.sample_size:
            for idx, n2o_seg in enumerate(file_n2o_segments):
                print(f"    - N2O-segment {idx} | {len(n2o_seg)} samples | {min_segment_length_samples:.0f} required")
                if len(n2o_seg) >= min_segment_length_samples:
                    n2o_segments[name + f"_{idx}"] = n2o_seg
                    print(f"      -> ADDED")

        if len(pre_segments) >= args.sample_size and len(n2o_segments) >= args.sample_size:
            break

    print(f"\nFound {len(pre_segments)} pre-segments to analyze:")
    for path in set(list(pre_segments.keys())):
        print(f"  - {os.path.basename(path)}")
    
    print(f"\nFound {len(n2o_segments)} n2o-segments to analyze:")
    for path in set(list(n2o_segments.keys())):
        print(f"  - {os.path.basename(path)}")

    segments = {**pre_segments, **n2o_segments}

    # PROCESS EACH SEGMENT

    segments_windows_info = []
    for segment_name, segment_data in segments.items():
        print(f"\nProcessing {segment_name}...")
        segment_info = process_segment(segment_data, args.windows, args.windows_per_segment, args.fs, rng)
        segment_info.insert(0, "Name", segment_name)
        segments_windows_info.append(segment_info)

    # SUMMARIZE & PLOT RESULTS

    segments_pd = pd.concat(segments_windows_info, ignore_index=True)
    # Aggregate results by window size across all segments
    segments_windows_pd = group_by_window_size(segments_pd)
    # Estimate stabilization window based on relative change heuristic
    stabilization_window = estimate_stabilization_window(segments_windows_pd, args.stability_threshold)

    input_folder = os.path.basename(os.path.normpath(args.input_dir))
    method_name = input_folder[5:]  # to remove "data_" prefix
    output_stem = f"lzc_window_{method_name}_n{len(segments)}"
    save_path = os.path.join(args.output_dir, output_stem)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = save_path + "_summary.csv"
    segments_windows_pd.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    print_summary_overview(segments_pd, segments_windows_pd, stabilization_window, args.stability_threshold)
    plot_results(segments_pd, segments_windows_pd, stabilization_window, args.stability_threshold, save_path)
    plot_relative_lzc_figure(segments_windows_pd, stabilization_window, args.stability_threshold, save_path)
    print("\n")


if __name__ == "__main__":
    main()
