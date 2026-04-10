import argparse
import glob
import os
import random

import antropy as ant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.io import loadmat


DEFAULT_WINDOW_SEC = 120.0
DEFAULT_TARGET_RATES = [6, 9, 10, 12, 15, 18, 20, 24, 25, 30, 40, 50, 60, 90, 100, 120, 150, 180, 200]
DEFAULT_SAMPLE_SIZE = 147  # 147 is all
DEFAULT_INPUT_DIR = "data_clean_segmented"
DEFAULT_PATTERN = "*-khodadad2018-200Hz.mat"
DEFAULT_OUTPUT_DIR = "figures/complexity_sensitivity_frequency"
DEFAULT_RANDOM_SEED = 123


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run respiratory LZC sensitivity over sampling frequency using "
            "cleaned and segmented 200 Hz source files."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing cleaned and segmented .mat files",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for input files inside the input directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of participant files to analyze",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample files instead of taking the first N sorted files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used when sampling files and windows",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help="Fixed window length in seconds used for all rate tests",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_RATES,
        help="Sampling rates to test in Hz",
    )
    parser.add_argument(
        "--windows-per-segment",
        type=int,
        default=4,
        help="Number of non-overlapping windows per eligible segment",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.02,
        help="Relative change threshold used to identify stabilization",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures and CSV summaries",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional explicit main figure path. If omitted, a default name is used.",
    )
    return parser.parse_args()


def discover_files(input_dir, pattern):
    search_pattern = os.path.join(input_dir, pattern)
    return sorted(glob.glob(search_pattern))


def pattern_to_method_label(pattern):
    base = os.path.basename(pattern)
    if base.startswith("*-") and base.endswith(".mat"):
        base = base[2:-4]
    elif base.endswith(".mat"):
        base = base[:-4]
    base = base.replace("*", "all")
    safe = []
    for char in base:
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    label = "".join(safe).strip("_")
    return label or "unknown"


def load_segment_matrix(mat, key_segments, key_lengths):
    segments = np.asarray(mat.get(key_segments, np.empty((0, 0))), dtype=float)
    lengths = np.asarray(mat.get(key_lengths, np.empty((0,))), dtype=int).reshape(-1)

    if segments.size == 0:
        return []

    if segments.ndim == 1:
        segments = segments.reshape(1, -1)

    if lengths.size == 0:
        lengths = np.asarray([np.sum(np.isfinite(row)) for row in segments], dtype=int)

    cleaned_segments = []
    n_rows = min(segments.shape[0], lengths.shape[0])
    for i in range(n_rows):
        seg_len = int(lengths[i])
        if seg_len <= 0:
            continue
        seg_len = min(seg_len, segments.shape[1])
        seg = np.asarray(segments[i, :seg_len], dtype=float)
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            cleaned_segments.append(seg)

    return cleaned_segments


def load_clean_segmented_file(path):
    mat = loadmat(path)

    fs_raw = np.asarray(mat.get("fs", np.array([[np.nan]])), dtype=float).reshape(-1)
    fs = float(fs_raw[0]) if fs_raw.size else np.nan
    if not np.isfinite(fs) or fs <= 0:
        fs = 50.0

    pre_segments = load_segment_matrix(mat, "pre_segments", "pre_segment_lengths")
    n2o_segments = load_segment_matrix(mat, "n2o_segments", "n2o_segment_lengths")
    segments = pre_segments + n2o_segments

    return {
        "fs": fs,
        "segments": segments,
        "pre_count": len(pre_segments),
        "n2o_count": len(n2o_segments),
    }


def normalized_lzc(binary_signal):
    binary_signal = np.asarray(binary_signal, dtype=bool)
    n = binary_signal.size
    if n < 2:
        return np.nan
    return float(ant.lziv_complexity(binary_signal, normalize=True))


def choose_non_overlapping_starts(segment_len, n_points, n_windows_per_segment, rng):
    if n_points < 2 or n_windows_per_segment <= 0:
        return []
    if segment_len < (n_windows_per_segment * n_points):
        return []

    max_offset = segment_len - (n_windows_per_segment * n_points)
    offset = rng.randint(0, max_offset) if max_offset > 0 else 0
    return [offset + (k * n_points) for k in range(n_windows_per_segment)]


def compute_window_lzc_at_rate(window_50hz, fs_source, fs_target):
    if fs_target <= 0 or fs_target > fs_source:
        return np.nan, np.nan

    if fs_target == fs_source:
        down = np.asarray(window_50hz, dtype=float)
    else:
        down = np.asarray(
            nk.signal_resample(
                window_50hz,
                sampling_rate=fs_source,
                desired_sampling_rate=fs_target,
            ),
            dtype=float,
        )

    if down.size < 2 or not np.all(np.isfinite(down)):
        return np.nan, np.nan

    binary = down > np.median(down)
    lzc = normalized_lzc(binary)
    balance = float(np.mean(binary))
    return lzc, balance


def file_level_rate_summary(file_data, rates_hz, window_sec, n_windows_per_segment, rng):
    fs_source = file_data["fs"]
    segments = file_data["segments"]

    n_points_source = int(round(window_sec * fs_source))
    if n_points_source < 2:
        return pd.DataFrame()

    min_required_len = n_windows_per_segment * n_points_source
    valid_segments = [seg for seg in segments if seg.size >= min_required_len]

    rows = []
    for rate in rates_hz:
        lzc_values = []
        balance_values = []
        segments_used = 0

        for seg in valid_segments:
            starts = choose_non_overlapping_starts(seg.size, n_points_source, n_windows_per_segment, rng)
            if not starts:
                continue

            used_this_segment = 0
            for start in starts:
                window_50hz = seg[start : start + n_points_source]
                lzc, balance = compute_window_lzc_at_rate(window_50hz, fs_source, rate)
                if np.isfinite(lzc):
                    lzc_values.append(lzc)
                    balance_values.append(balance)
                    used_this_segment += 1

            if used_this_segment > 0:
                segments_used += 1

        if lzc_values:
            lzc_values = np.asarray(lzc_values, dtype=float)
            balance_values = np.asarray(balance_values, dtype=float)
            rows.append(
                {
                    "Sampling_Rate_Hz": float(rate),
                    "LZC_Mean": float(np.mean(lzc_values)),
                    "LZC_Std": float(np.std(lzc_values)),
                    "LZC_CV": float(np.std(lzc_values) / np.mean(lzc_values)) if np.mean(lzc_values) > 0 else np.nan,
                    "Binary_Balance_Mean": float(np.mean(balance_values)),
                    "Binary_Balance_Std": float(np.std(balance_values)),
                    "N_Samples": int(lzc_values.size),
                    "N_Segments_Used": int(segments_used),
                    "Window_Sec": float(window_sec),
                }
            )
        else:
            rows.append(
                {
                    "Sampling_Rate_Hz": float(rate),
                    "LZC_Mean": np.nan,
                    "LZC_Std": np.nan,
                    "LZC_CV": np.nan,
                    "Binary_Balance_Mean": np.nan,
                    "Binary_Balance_Std": np.nan,
                    "N_Samples": 0,
                    "N_Segments_Used": int(segments_used),
                    "Window_Sec": float(window_sec),
                }
            )

    return pd.DataFrame(rows)


def aggregate_across_files(file_summaries):
    summary = (
        file_summaries.groupby("Sampling_Rate_Hz")
        .agg(
            LZC_Mean_Mean=("LZC_Mean", "mean"),
            LZC_Mean_Std=("LZC_Mean", "std"),
            LZC_Median=("LZC_Mean", "median"),
            Binary_Balance_Mean=("Binary_Balance_Mean", "mean"),
            Binary_Balance_Std=("Binary_Balance_Mean", "std"),
            N_Files=("File", "nunique"),
            Mean_Samples=("N_Samples", "mean"),
        )
        .reset_index()
        .sort_values("Sampling_Rate_Hz", ascending=True)
    )

    summary["LZC_Abs_Delta"] = summary["LZC_Mean_Mean"].diff().abs()
    summary["LZC_Rel_Change"] = summary["LZC_Abs_Delta"] / summary["LZC_Mean_Mean"].shift(1).abs()
    summary["Binary_Balance_Delta_From_0.5"] = (summary["Binary_Balance_Mean"] - 0.5).abs()
    return summary


def estimate_stabilization_rate(summary, threshold):
    if summary.empty or "LZC_Rel_Change" not in summary:
        return None

    rel_change = summary["LZC_Rel_Change"].to_numpy(dtype=float)
    rates = summary["Sampling_Rate_Hz"].to_numpy(dtype=float)

    for idx in range(1, len(rates)):
        current = rel_change[idx]
        if not np.isfinite(current):
            continue
        if current < threshold:
            if idx + 1 < len(rates):
                next_val = rel_change[idx + 1]
                if np.isfinite(next_val) and next_val < threshold:
                    return float(rates[idx])
            else:
                return float(rates[idx])

    return None


def point_transition_colors(aggregated, threshold):
    colors = []
    lzc_values = aggregated["LZC_Mean_Mean"].to_numpy(dtype=float)

    for idx, value in enumerate(lzc_values):
        if idx == 0 or not np.isfinite(value):
            colors.append("black")
            continue

        prev = lzc_values[idx - 1]
        if not np.isfinite(prev) or prev == 0:
            colors.append("red")
            continue

        rel_change = abs(value - prev) / abs(prev)
        colors.append("green" if rel_change <= threshold else "red")

    return colors


def print_summary_overview(file_summaries, aggregated, stabilization_rate, threshold, window_sec):
    print("\n" + "=" * 72)
    print("LZC COMPLEXITY SAMPLING-FREQUENCY SENSITIVITY")
    print("=" * 72)
    print(f"Files analyzed: {file_summaries['File'].nunique()}")
    print(f"Fixed window size: {window_sec:.1f} seconds")
    print(
        "Sampling rates tested: "
        + ", ".join(
            str(int(r)) if float(r).is_integer() else str(r)
            for r in aggregated["Sampling_Rate_Hz"].to_numpy(dtype=float)
        )
    )
    print(f"Stability threshold: {threshold:.3f} relative change")

    print("\nPer-rate summary across sampled files:")
    for _, row in aggregated.iterrows():
        rel_change = row["LZC_Rel_Change"]
        rel_change_str = "nan" if not np.isfinite(rel_change) else f"{rel_change:.4f}"
        cv = row["LZC_Mean_Std"] / row["LZC_Mean_Mean"] if row["LZC_Mean_Mean"] > 0 else np.nan
        print(
            f"  Rate {row['Sampling_Rate_Hz']:.0f}Hz | "
            f"LZC={row['LZC_Mean_Mean']:.4f} +- {row['LZC_Mean_Std']:.4f} | "
            f"CV={cv:.4f} | rel_change={rel_change_str} | "
            f"binary_balance={row['Binary_Balance_Mean']:.4f} | files={int(row['N_Files'])}"
        )

    if stabilization_rate is not None:
        print(f"\nEstimated stabilization rate: ~{stabilization_rate:.0f} Hz")
    else:
        print("\nNo clear stabilization rate found under the current heuristic.")

    print("=" * 72)


def plot_primary_lzc_figure(aggregated, stabilization_rate, threshold, output_dir, save_path=None):
    fig, ax = plt.subplots(figsize=(20, 6))

    x = aggregated["Sampling_Rate_Hz"].to_numpy(dtype=float)
    y = aggregated["LZC_Mean_Mean"].to_numpy(dtype=float)
    colors = point_transition_colors(aggregated, threshold)

    ax.plot(x, y, color="#1f77b4", linewidth=2.2, label="Normalized LZC")
    ax.scatter(x, y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.6)

    if stabilization_rate is not None:
        ax.axvline(stabilization_rate, color="red", linestyle="--", label=f"Stabilization ~{stabilization_rate:.0f}Hz")

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=2.2, label="Normalized LZC"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label=f"<= {threshold:.0%} change"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label=f"> {threshold:.0%} change"),
    ]
    if stabilization_rate is not None:
        legend_items.append(Line2D([0], [0], color="red", linestyle="--", lw=1.5, label=f"Stabilization ~{stabilization_rate:.0f}Hz"))

    ax.set_title("Mean Respiratory LZC Across Sampling Rates", fontsize=14)
    ax.set_xlabel("Sampling Rate [Hz]")
    ax.set_ylabel("Normalized LZC")
    ax.grid(alpha=0.25)
    ax.legend(handles=legend_items, loc="best")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(output_dir, "complexity_sampling_primary_lzc.png")
    fig.savefig(save_path, dpi=180)
    print(f"Saved figure: {save_path}")
    plt.show()
    plt.close(fig)


def plot_results(file_summaries, aggregated, stabilization_rate, threshold, output_dir, output_stem, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

    # Top: individual file curves + mean curve
    ax1 = axes[0]
    for _, grp in file_summaries.groupby("File"):
        grp_sorted = grp.sort_values("Sampling_Rate_Hz", ascending=True)
        ax1.plot(
            grp_sorted["Sampling_Rate_Hz"],
            grp_sorted["LZC_Mean"],
            color="0.75",
            linewidth=1.0,
            alpha=0.7,
        )

    ax1.errorbar(
        aggregated["Sampling_Rate_Hz"],
        aggregated["LZC_Mean_Mean"],
        yerr=aggregated["LZC_Mean_Std"],
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#1f77b4",
        label="Mean across sampled files",
    )
    if stabilization_rate is not None:
        ax1.axvline(stabilization_rate, color="red", linestyle="--", label=f"Stabilization ~{stabilization_rate:.0f}Hz")
    ax1.set_ylabel("Normalized LZC")
    ax1.set_title("Respiratory LZC Stability Across Sampling Rates (Window = 120s)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    # Middle: relative change between adjacent rates
    ax2 = axes[1]
    ax2.plot(
        aggregated["Sampling_Rate_Hz"],
        aggregated["LZC_Rel_Change"],
        marker="o",
        color="#ff7f0e",
        linewidth=2.0,
        label="Relative change from previous rate",
    )
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label=f"{threshold:.0%} threshold")
    if stabilization_rate is not None:
        ax2.axvline(stabilization_rate, color="red", linestyle="--")
    ax2.set_ylabel("Relative Change")
    ax2.set_title("Where the Curve Stops Changing Across Sampling Rates")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    # Bottom: binary balance diagnostic
    ax3 = axes[2]
    ax3.errorbar(
        aggregated["Sampling_Rate_Hz"],
        aggregated["Binary_Balance_Mean"],
        yerr=aggregated["Binary_Balance_Std"],
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#2ca02c",
        label="Binary balance (mean of thresholded window)",
    )
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1.0, label="Ideal balance = 0.5")
    if stabilization_rate is not None:
        ax3.axvline(stabilization_rate, color="red", linestyle="--")
    ax3.set_xlabel("Sampling Rate [Hz]")
    ax3.set_ylabel("Binary Balance")
    ax3.set_title("Binarization Quality Check")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(output_dir, f"{output_stem}.png")
    fig.savefig(save_path, dpi=180)
    print(f"Saved figure: {save_path}")
    plt.show()

    base, ext = os.path.splitext(save_path)
    primary_save_path = f"{base}_primary{ext or '.png'}"
    plot_primary_lzc_figure(
        aggregated=aggregated,
        stabilization_rate=stabilization_rate,
        threshold=threshold,
        output_dir=output_dir,
        save_path=primary_save_path,
    )


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Keep rates unique while preserving order.
    deduped_rates = []
    seen = set()
    for rate in args.rates:
        if rate not in seen:
            deduped_rates.append(rate)
            seen.add(rate)
    rates_hz = sorted(deduped_rates)

    files = discover_files(args.input_dir, args.pattern)
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(args.input_dir, args.pattern)}")

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    if args.sample_size < len(files):
        if args.random_sample:
            files = rng.sample(files, args.sample_size)
            files = sorted(files)
        else:
            files = files[: args.sample_size]
    else:
        files = sorted(files)

    print(f"Found {len(files)} file(s) to analyze.")
    print("Selected files:")
    for path in files:
        print(f"  - {os.path.basename(path)}")

    file_rows = []
    for path in files:
        file_name = os.path.basename(path)
        print(f"\nProcessing {file_name}...")
        file_data = load_clean_segmented_file(path)
        print(
            f"  fs={file_data['fs']:.3f} Hz | pre_segments={file_data['pre_count']} | "
            f"n2o_segments={file_data['n2o_count']} | total_segments={len(file_data['segments'])}"
        )

        if not file_data["segments"]:
            print("  Skipping: no usable segments found.")
            continue

        file_summary = file_level_rate_summary(
            file_data=file_data,
            rates_hz=rates_hz,
            window_sec=args.window_sec,
            n_windows_per_segment=args.windows_per_segment,
            rng=rng,
        )
        if file_summary.empty:
            print("  No valid windows could be sampled for this file.")
            continue

        file_summary.insert(0, "File", file_name)
        file_rows.append(file_summary)

    if not file_rows:
        raise RuntimeError("No valid data were found for the requested file sample.")

    file_summaries = pd.concat(file_rows, ignore_index=True)
    aggregated = aggregate_across_files(file_summaries)
    stabilization_rate = estimate_stabilization_rate(aggregated, args.stability_threshold)

    method_label = pattern_to_method_label(args.pattern)
    output_stem = f"complexity_sampling_sensitivity_{method_label}_n{len(files)}_win{int(round(args.window_sec))}s"

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{output_stem}_summary.csv")
    file_summaries.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    print_summary_overview(file_summaries, aggregated, stabilization_rate, args.stability_threshold, args.window_sec)
    plot_results(
        file_summaries,
        aggregated,
        stabilization_rate,
        args.stability_threshold,
        args.output_dir,
        output_stem,
        args.save_path,
    )


if __name__ == "__main__":
    main()
