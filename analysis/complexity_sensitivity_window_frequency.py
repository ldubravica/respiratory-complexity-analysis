import argparse
import glob
import os
import random
import time

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

DEFAULT_WINDOWS_SEC = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
# DEFAULT_WINDOWS_SEC = [10, 20, 30]
DEFAULT_FS_TARGETS = [3, 6, 9, 12, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500]
# DEFAULT_FS_TARGETS = [6, 100, 200]
DEFAULT_SAMPLE_SIZE = 147000
# DEFAULT_SAMPLE_SIZE = 3
DEFAULT_PATTERN = "*.mat"
DEFAULT_INPUT_DIR = os.path.abspath(os.path.join(script_dir, "..", "data_khodadad2018_200Hz"))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, "..", "figures", "complexity_sensitivity_window_frequency"))
DEFAULT_RANDOM_SEED = 42
DEFAULT_WINDOWS_PER_SEGMENT = 5
# DEFAULT_WINDOWS_PER_SEGMENT = 3
DEFAULT_FS = 500.0  # TODO - add automatic determination


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run respiratory LZC sensitivity analysis over window sizes. "
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
        "--fs-targets", type=int, nargs="+", default=DEFAULT_FS_TARGETS,
        help="Target sampling rates in Hz to test (example: --fs-targets 6 100 200)",
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


def aggregate(results):
    summary = (
        results.groupby(["Window_Sec", "FS"])
        .agg(
            N_Segments=("Name", "nunique"),
            LZC_Mean_Mean=("LZC_Mean", "mean"),
            LZC_Mean_Std=("LZC_Mean", "std"),
            LZC_Mean_Median=("LZC_Mean", "median"),
            Binary_Balance_Mean=("Binary_Balance_Mean", "mean"),
            Binary_Balance_Std=("Binary_Balance_Mean", "std"),
        )
        .reset_index()
        .sort_values(by=["Window_Sec", "FS"])
    )

    # Compute relative changes within each window size (across FS) and within each FS (across window sizes)
    summary["LZC_Abs_Delta"] = summary["LZC_Mean_Mean"].diff().abs()
    summary["LZC_Rel_Change"] = summary["LZC_Abs_Delta"] / summary["LZC_Mean_Mean"].shift(1).abs()
    
    # Per-window delta across FS values
    summary["LZC_Delta_Within_Window"] = summary.groupby("Window_Sec")["LZC_Mean_Mean"].diff().abs()
    summary["LZC_Rel_Change_Within_Window"] = summary["LZC_Delta_Within_Window"] / summary.groupby("Window_Sec")["LZC_Mean_Mean"].shift(1).abs()
    
    # Per-FS delta across window sizes
    summary["LZC_Delta_Within_FS"] = summary.groupby("FS")["LZC_Mean_Mean"].diff().abs()
    summary["LZC_Rel_Change_Within_FS"] = summary["LZC_Delta_Within_FS"] / summary.groupby("FS")["LZC_Mean_Mean"].shift(1).abs()

    # 2D average relative change at each (window, fs) pair
    summary["LZC_Rel_Change_Avg_2D"] = summary[
        ["LZC_Rel_Change_Within_Window", "LZC_Rel_Change_Within_FS"]
    ].mean(axis=1, skipna=True)
    
    summary["Binary_Balance_Delta_From_0.5"] = (summary["Binary_Balance_Mean"] - 0.5).abs()

    return summary


def estimate_stabilization(summary, threshold):
    windows = sorted(summary["Window_Sec"].unique())
    fs_values = sorted(summary["FS"].unique())

    rel_change_window = summary["LZC_Rel_Change_Within_Window"].to_numpy(dtype=float)
    rel_change_fs = summary["LZC_Rel_Change_Within_FS"].to_numpy(dtype=float)

    stabilization_points = []

    # this can be computationally improved by calculating it backwards

    for w_idx in range(1, len(windows)):
        for f_idx in range(1, len(fs_values)):
            current_window_change = rel_change_window[w_idx]
            current_fs_change = rel_change_fs[f_idx]
            if current_window_change <= threshold and current_fs_change <= threshold:
                
                next_window_stable = False
                if w_idx + 1 < len(windows):
                    next_window_change = rel_change_window[w_idx + 1]
                    if next_window_change <= threshold:
                        next_window_stable = True
                else:
                    next_window_stable = True

                next_fs_stable = False
                if f_idx + 1 < len(fs_values):
                    next_fs_change = rel_change_fs[f_idx + 1]
                    if next_fs_change <= threshold:
                        next_fs_stable = True
                else:
                    next_fs_stable = True

                stability = sum([next_window_stable, next_fs_stable])
                point = {"window_sec": float(windows[w_idx]), "fs": float(fs_values[f_idx]), "stability": stability}
                stabilization_points.append(point)

            elif current_window_change <= threshold or current_fs_change <= threshold:
                point = {"window_sec": float(windows[w_idx]), "fs": float(fs_values[f_idx]), "stability": -1}
                stabilization_points.append(point)

    # sort stabilization points by stability score (descending), window size (ascending) and FS (ascending)
    stabilization_points = sorted(
        stabilization_points,
        key=lambda x: (-x["stability"], x["window_sec"], x["fs"]),
    )

    return stabilization_points


def process_segment(segment, window_sizes_sec, n_windows_per_segment, fs, fs_targets, rng):
    max_window_points = int(round(float(max(window_sizes_sec)) * fs))
    min_required_len = n_windows_per_segment * max_window_points

    if len(segment) < min_required_len:  # maybe segment.size?
        return pd.DataFrame()

    segment_info = []
    for window_sec in window_sizes_sec:  # maybe add track progress tqdm?
        n_window_samples = int(round(window_sec * fs))

        # Draw a random non-overlapping tiling of windows by selecting one offset,
        # then placing fixed-size windows back-to-back.
        max_offset = len(segment) - (n_windows_per_segment * n_window_samples)
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        starts = [offset + (k * n_window_samples) for k in range(n_windows_per_segment)]

        for start in starts:
            window = segment[start : start + n_window_samples]
            for fs_target in fs_targets:
                
                window_ds = window
                if fs_target != fs:  # Resample the window to the target frequency
                    window_ds = np.asarray(
                        nk.signal_resample(window, sampling_rate=fs, desired_sampling_rate=fs_target), dtype=float)
                
                binary = window_ds > np.median(window_ds).item()
                lzc = ant.lziv_complexity(binary, normalize=True)

                segment_info.append({
                    "Window_Sec": float(window_sec),
                    "FS": float(fs_target),
                    "LZC": float(lzc),
                    "Binary_Balance": float(np.mean(binary))
                })

    # create LZC_Mean, LZC_Std, LZC_CV, Binary_Balance_Mean, Binary_Balance_Std for each window size and fs target combination
    segment_info_pd = pd.DataFrame(segment_info)
    segment_info_summary = (
        segment_info_pd.groupby(["Window_Sec", "FS"])
        .agg(
            LZC_Mean=("LZC", "mean"),
            LZC_Std=("LZC", "std"),
            Binary_Balance_Mean=("Binary_Balance", "mean"),
            Binary_Balance_Std=("Binary_Balance", "std"),
        )
        .reset_index()
    )
    segment_info_summary["LZC_CV"] = segment_info_summary["LZC_Std"] / segment_info_summary["LZC_Mean"].replace(0, np.nan)

    return segment_info_summary

# ---------------------------
# Summaries & Plots
# ---------------------------

def print_summary_overview(segments_pd, segments_windows_pd, stabilizations, threshold):
    print("\n" + "=" * 72)
    print("LZC COMPLEXITY WINDOW × FREQUENCY SENSITIVITY")
    print("=" * 72)
    print(f"Segments analyzed: {segments_pd['Name'].nunique()}")
    windows_unique = sorted(segments_windows_pd['Window_Sec'].unique())
    fs_unique = sorted(segments_windows_pd['FS'].unique())
    print(f"Window sizes tested: {', '.join(str(int(w)) for w in windows_unique)}")
    print(f"Sampling rates tested: {', '.join(str(int(f)) for f in fs_unique)}")
    print(f"Stability threshold: {threshold:.3f} relative change")

    print("\nPer (Window, FS) summary across sampled segments:")
    for _, row in segments_windows_pd.iterrows():
        rel_change_window = row.get("LZC_Rel_Change_Within_Window", np.nan)
        rel_change_fs = row.get("LZC_Rel_Change_Within_FS", np.nan)
        rel_change_window_str = f"{rel_change_window:.4f}" if np.isfinite(rel_change_window) else "-.----"
        rel_change_fs_str = f"{rel_change_fs:.4f}" if np.isfinite(rel_change_fs) else "-.----"
        print(
            f"  Window {row['Window_Sec']:.0f}s, FS {row['FS']:.0f}Hz | "
            f"LZC={row['LZC_Mean_Mean']:.4f} ± {row['LZC_Mean_Std']:.4f} | "
            f"Δ_window={rel_change_window_str} | Δ_fs={rel_change_fs_str} | "
            f"balance={row['Binary_Balance_Mean']:.4f}"
        )

    if stabilizations:
        print("\nEstimated stabilization points (sorted by stability score):")
        for stab in stabilizations:
            stab_window = stab["window_sec"]
            stab_fs = stab["fs"]
            stab_score = stab["stability"]
            stab_desc = "fully stable" if stab_score == 2 else ("stable" if stab_score >= 0 else "partially stable")
            print(f"  Stabilization candidate: Window ~{stab_window:.0f}s × FS ~{stab_fs:.0f}Hz ({stab_desc})")
    else:
        print("\nNo clear stabilization points found under the current heuristic.")

    print("=" * 72 + "\n")


def plot_results(segments_pd, segments_aggregate_pd, stabilizations, threshold, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=False)

    # Top: LZC across window sizes for each FS
    ax1 = axes[0]
    for segment_name, grp in segments_pd.groupby("FS"):
        grp = grp.sort_values("Window_Sec")
        ax1.plot(
            grp["Window_Sec"],
            grp["LZC_Mean"],
            color="0.75",
            linewidth=1.0,
            alpha=0.7,
        )

    # Aggregate across FS to get one error bar per window size
    window_summary = segments_aggregate_pd.groupby("Window_Sec").agg({
        "LZC_Mean_Mean": "mean",
        "LZC_Mean_Std": "mean",
    }).reset_index()
    
    ax1.errorbar(
        window_summary["Window_Sec"],
        window_summary["LZC_Mean_Mean"],
        yerr=window_summary["LZC_Mean_Std"],
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#1f77b4",
        label="Mean across sampled segments",
    )

    if stabilizations:
        first_stabilization_label = f"Full Stabilization ~ {stabilizations[0]['window_sec']:.0f}s & {stabilizations[0]['fs']:.0f}Hz"
        ax1.axvline(stabilizations[0]["window_sec"], color="red", linestyle="--", linewidth=2, label=first_stabilization_label)
    ax1.set_ylabel("Normalized LZC")
    ax1.set_xlabel("Window Length [s]")
    ax1.set_title("LZC Across Window Sizes (each line = different FS)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    # Bottom: LZC across FS for each window
    ax2 = axes[1]
    for window_sec in sorted(segments_aggregate_pd["Window_Sec"].unique()):
        window_data = segments_aggregate_pd[segments_aggregate_pd["Window_Sec"] == window_sec].sort_values("FS")
        ax2.plot(
            window_data["FS"],
            window_data["LZC_Mean_Mean"],
            marker="o",
            linewidth=1.5,
            alpha=0.7,
            label=f"Window={window_sec:.0f}s",
        )

    if stabilizations:
        first_stabilization_label = f"Full Stabilization ~ {stabilizations[0]['fs']:.0f}Hz & {stabilizations[0]['window_sec']:.0f}s"
        ax2.axvline(stabilizations[0]["fs"], color="red", linestyle="--", linewidth=2, label=first_stabilization_label)
    ax2.set_ylabel("Normalized LZC")
    ax2.set_xlabel("Sampling Rate [Hz]")
    ax2.set_title("LZC Across Sampling Rates (each line = different window size)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    save_path_png = save_path + ".png"
    fig.savefig(save_path_png, dpi=180)
    print(f"Saved figure: {save_path_png}")
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


def plot_lzc_relativity(segments_aggregate_pd, stabilizations, threshold, save_path):
    """Create heatmap-style visualization of 2D (window × FS) LZC landscape."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    windows = sorted(segments_aggregate_pd["Window_Sec"].unique())
    fs_vals = sorted(segments_aggregate_pd["FS"].unique())
    
    lzc_matrix = np.zeros((len(windows), len(fs_vals)))
    rel_change_window_matrix = np.zeros((len(windows), len(fs_vals)))
    rel_change_fs_matrix = np.zeros((len(windows), len(fs_vals)))
    
    for i, w in enumerate(windows):
        for j, f in enumerate(fs_vals):
            row = segments_aggregate_pd[(segments_aggregate_pd["Window_Sec"] == w) & (segments_aggregate_pd["FS"] == f)]
            if len(row) > 0:
                lzc_matrix[i, j] = row["LZC_Mean_Mean"].values[0]
                rel_change_window_matrix[i, j] = row.get("LZC_Rel_Change_Within_Window", np.nan).values[0] if "LZC_Rel_Change_Within_Window" in row else np.nan
                rel_change_fs_matrix[i, j] = row.get("LZC_Rel_Change_Within_FS", np.nan).values[0] if "LZC_Rel_Change_Within_FS" in row else np.nan
    
    # LZC heatmap
    im1 = ax1.imshow(lzc_matrix, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_xticks(range(len(fs_vals)))
    ax1.set_yticks(range(len(windows)))
    ax1.set_xticklabels([f"{int(f)}" for f in fs_vals], rotation=45)
    ax1.set_yticklabels([f"{int(w)}" for w in windows])
    ax1.set_xlabel("Sampling Rate [Hz]")
    ax1.set_ylabel("Window Length [s]")
    ax1.set_title("Normalized LZC Complexity")
    plt.colorbar(im1, ax=ax1)
    
    if stabilizations is not None and len(stabilizations) > 0:
        for stab in stabilizations:
            stab_w_idx = np.argmin(np.abs(np.array(windows) - stab["window_sec"]))
            stab_f_idx = np.argmin(np.abs(np.array(fs_vals) - stab["fs"]))

            marker_style = "ro"  # default for partially stable
            if stab["stability"] == 2:
                marker_style = "r*"
            elif stab["stability"] >= 0:
                marker_style = "r^"

            ax1.plot(stab_f_idx, stab_w_idx, marker_style, markersize=20)

        # Add legend for stabilization points
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="r*", color="w", label="Full Stabilization", markersize=15),
            Line2D([0], [0], marker="r^", color="w", label="Stable in one dimension", markersize=15),
            Line2D([0], [0], marker="ro", color="w", label="Partially Stable", markersize=15),
        ]
        ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    # Relative change: max of both dimensions
    rel_change_max = np.maximum(np.abs(rel_change_window_matrix), np.abs(rel_change_fs_matrix))
    im2 = ax2.imshow(rel_change_max <= threshold, aspect="auto", origin="lower", cmap="RdYlGn")
    ax2.set_xticks(range(len(fs_vals)))
    ax2.set_yticks(range(len(windows)))
    ax2.set_xticklabels([f"{int(f)}" for f in fs_vals], rotation=45)
    ax2.set_yticklabels([f"{int(w)}" for w in windows])
    ax2.set_xlabel("Sampling Rate [Hz]")
    ax2.set_ylabel("Window Length [s]")
    ax2.set_title(f"Stabilized Region (relative change <= {threshold:.1%})")
    
    fig.tight_layout()
    save_path_png = save_path + "_2d_landscape.png"
    fig.savefig(save_path_png, dpi=180)
    print(f"Saved figure: {save_path_png}")
    plt.show()
    plt.close(fig)


def plot_heatmap(segments_windows_pd, threshold, save_path):
    """Plot a thresholded 2D heatmap for average relative change over (window, fs)."""
    windows = sorted(segments_windows_pd["Window_Sec"].unique())
    fs_vals = sorted(segments_windows_pd["FS"].unique())

    value_matrix = np.full((len(windows), len(fs_vals)), np.nan, dtype=float)
    color_matrix = np.zeros((len(windows), len(fs_vals)), dtype=int)

    for i, w in enumerate(windows):
        for j, f in enumerate(fs_vals):
            row = segments_windows_pd[
                (segments_windows_pd["Window_Sec"] == w) & (segments_windows_pd["FS"] == f)
            ]
            if len(row) == 0:
                continue

            avg_rel_change = float(row["LZC_Rel_Change_Avg_2D"].iloc[0])
            value_matrix[i, j] = avg_rel_change
            if np.isfinite(avg_rel_change) and avg_rel_change <= threshold:
                color_matrix[i, j] = 1  # green
            else:
                color_matrix[i, j] = 0  # red (or missing)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#d73027", "#1a9850"])  # red, green
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(color_matrix, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(fs_vals)))
    ax.set_yticks(range(len(windows)))
    ax.set_xticklabels([str(int(f)) for f in fs_vals], rotation=45, ha="right")
    ax.set_yticklabels([str(int(w)) for w in windows])
    ax.set_xlabel("Sampling Rate [Hz]")
    ax.set_ylabel("Window Length [s]")
    ax.set_title(f"Average Relative Change (green <= {threshold:.1%}, red > {threshold:.1%})")

    for i in range(len(windows)):
        for j in range(len(fs_vals)):
            val = value_matrix[i, j]
            label = "NA" if not np.isfinite(val) else f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    save_path_png = save_path + "_threshold_heatmap.png"
    fig.savefig(save_path_png, dpi=180)
    print(f"Saved figure: {save_path_png}")
    plt.show()
    plt.close(fig)


# ---------------------------
# Main loop
# ---------------------------

def main():
    print(f"\n{'='*50}WORK IN PROGRESS{'='*50}\n")
    return  # to prevent accidental execution while still in development

    time_start = time.time()
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

    segments_info = []
    for segment_name, segment_data in segments.items():
        print(f"\nProcessing {segment_name}...")
        segment_pd = process_segment(segment_data, args.windows, args.windows_per_segment, args.fs, args.fs_targets, rng)
        segment_pd.insert(0, "Name", segment_name)
        segments_info.append(segment_pd)

    # SUMMARIZE & PLOT RESULTS

    segments_pd = pd.concat(segments_info, ignore_index=True)
    # Aggregate results by window size across all segments
    segments_aggregate_pd = aggregate(segments_pd)
    # Estimate stabilization window-fs pairs based on relative change heuristic
    stabilizations = estimate_stabilization(segments_aggregate_pd, args.stability_threshold)

    input_folder = os.path.basename(os.path.normpath(args.input_dir))
    method_name = input_folder[5:]  # to remove "data_" prefix
    output_stem = f"lzc_window_frequency_{method_name}_n{len(segments)}"
    save_path = os.path.join(args.output_dir, output_stem)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = save_path + "_summary.csv"
    segments_aggregate_pd.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    print(f"\nTotal execution time: {(time.time() - time_start) / 60:.2f} minutes")

    print_summary_overview(segments_pd, segments_aggregate_pd, stabilizations, args.stability_threshold)
    
    plot_results(segments_pd, segments_aggregate_pd, stabilizations, args.stability_threshold, save_path)
    plot_lzc_relativity(segments_aggregate_pd, stabilizations, args.stability_threshold, save_path)
    plot_heatmap(segments_aggregate_pd, args.stability_threshold, save_path)
    print("\n")


if __name__ == "__main__":
    main()
