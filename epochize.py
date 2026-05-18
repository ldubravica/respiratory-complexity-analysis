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
DEFAULT_DROP_BAD_EPOCHS = False
DEFAULT_SAVE_DROPPED_PLOTS = False
DEFAULT_SHOW_DROPPED_PLOTS = True

LOW_STD_RUN_SEC = 30.0
LOW_STD_FRACTION_THRESH = 0.05
HIGH_STD_FACTOR = 2
HIGH_STD_FRACTION_THRESH = 0.50

def parse_args():
    parser = argparse.ArgumentParser(description=("Split filtered respiratory pre/during segments into epochs."))
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing preprocessed .mat files")
    parser.add_argument("--method", default=None, help="Pipeline method")
    parser.add_argument("--epoch-length-sec", type=float, default=DEFAULT_EPOCH_LENGTH_SEC, help="Epoch length in seconds")
    parser.add_argument("--input-pattern", default=DEFAULT_INPUT_PATTERN, help="Input file names' pattern to match (e.g., 'khodadad2018-200Hz')")
    parser.add_argument("--drop-bad-epochs", action="store_true", default=DEFAULT_DROP_BAD_EPOCHS, help="Drop bad epochs")
    parser.add_argument("--save-plots", dest="save_dropped_plots", action="store_true", default=DEFAULT_SAVE_DROPPED_PLOTS, help="Save rejected-epoch plots")
    parser.add_argument("--skip-plots", dest="show_dropped_epoch_plots", action="store_false", default=DEFAULT_SHOW_DROPPED_PLOTS, help="Do not display rejected-epoch plots interactively")
    parser.add_argument("--standardize", action="store_true", help="Standardize each epoch before saving")

    args = parser.parse_args()
    if args.method is not None:
        args.input_dir = f"data_{args.method}"

    return args

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


def standardize_epoch(epoch):
    epoch = np.asarray(epoch, dtype=float).reshape(-1)
    epoch_mean = float(np.mean(epoch))
    epoch_std = float(np.std(epoch))

    if epoch_std == 0.0:
        return epoch - epoch_mean

    return (epoch - epoch_mean) / epoch_std


def drop_bad_epochs(epochs, file_std, fs, fname, args):
    """Filter epochs using file-level std-based rules plus existing heuristics.

    Reject when either:
    1) there exists a continuous period longer than `flat_run_sec` seconds where
       the local (1s window) STD < `flat_thresh_fraction * file_std` (flatline), or
    2) the fraction of the epoch where local STD > `high_std_factor * file_std`
       exceeds `high_std_fraction_thresh` (excessive variability).

    Returns: (kept_epochs, dropped_epochs, report)
    """

    kept_epochs = []
    dropped_epochs = []
    report = []

    for epoch_idx, epoch in enumerate(epochs):
        epoch = np.asarray(epoch, dtype=float).reshape(-1)
        n = epoch.size
        reasons = []

        # Now the file-std based local-window rules
        if file_std > 0 and n > 0:
            # 1-second moving window (1s resolution)
            win = max(1, int(round(fs)))
            if n >= win:
                kernel = np.ones(win, dtype=float) / float(win)
                mean = np.convolve(epoch, kernel, mode="valid")
                mean_sq = np.convolve(epoch * epoch, kernel, mode="valid")
                var = mean_sq - mean * mean
                var[var < 0] = 0.0
                mov_std = np.sqrt(var)

                low_thresh = LOW_STD_FRACTION_THRESH * file_std
                high_thresh = HIGH_STD_FACTOR * file_std

                low_windows = mov_std < low_thresh
                high_windows = mov_std > high_thresh

                # expand window booleans to per-sample mask
                low_mask = np.zeros(n, dtype=bool)
                high_mask = np.zeros(n, dtype=bool)
                for i, v in enumerate(low_windows):
                    if v:
                        low_mask[i : i + win] = True
                for i, v in enumerate(high_windows):
                    if v:
                        high_mask[i : i + win] = True

                # longest continuous low-std run
                if low_mask.any():
                    # compute run lengths
                    dif = np.diff(np.concatenate(([0], low_mask.view(np.int8), [0])))
                    starts = np.where(dif == 1)[0]
                    ends = np.where(dif == -1)[0]
                    run_lengths = ends - starts
                    max_run = run_lengths.max() if run_lengths.size else 0
                else:
                    max_run = 0

                if max_run / float(fs) >= float(LOW_STD_RUN_SEC):
                    reasons.append(f"long_flat>={int(LOW_STD_RUN_SEC)}s")

                # high-std coverage
                high_frac = float(np.mean(high_mask))
                if high_frac > HIGH_STD_FRACTION_THRESH:
                    reasons.append(f"high_std_frac>{HIGH_STD_FRACTION_THRESH:.2f}")

        is_bad = len(reasons) > 0
        row = {"epoch_index": epoch_idx, "is_bad": is_bad, "reasons": reasons}
        report.append(row)

        if not is_bad:
            kept_epochs.append(epoch)
        else:
            dropped = {
                "fname": fname, 
                "epoch_index": epoch_idx, 
                "epoch": epoch, 
                "reasons": reasons,
                "std": np.std(epoch),
                "std_file": file_std,
            }
            dropped_epochs.append(dropped)
            if args.show_dropped_epoch_plots or args.save_dropped_plots:
                out_path = None
                if args.save_dropped_plots:
                    plot_dir = os.path.join(args.out_directory, f"{os.path.splitext(fname)[0]}-{args.epochizing_info}-dropped-epochs")
                    os.makedirs(plot_dir, exist_ok=True)
                    out_path = os.path.join(plot_dir, f"epoch_{epoch_idx:04d}.png")

                # build context: up to 2 epochs on each side
                n_neighbour_epochs = 5
                start_idx = max(0, epoch_idx - n_neighbour_epochs)
                end_idx = min(len(epochs) - 1, epoch_idx + n_neighbour_epochs)
                context = [epochs[i] for i in range(start_idx, end_idx + 1)]
                center_idx = epoch_idx - start_idx

                _plot_rejected_epoch(dropped, fs, out_path=out_path, show=args.show_dropped_epoch_plots, context_epochs=context, center_idx=center_idx)

    if not kept_epochs:
        return np.empty((0, epochs.shape[1]), dtype=float), dropped_epochs, report

    return np.stack(kept_epochs, axis=0), dropped_epochs, report


def _plot_rejected_epoch(row, fs, out_path=None, show=True, context_epochs=None, center_idx=0):
    """Create a quick-look figure for a rejected epoch, optionally with context.

    Parameters
    - row: dict with epoch info (must contain 'epoch')
    - context_epochs: list of 1D arrays to plot in sequence (may include the central epoch)
    - center_idx: index in context_epochs corresponding to the epoch in `row`
    """
    # Build concatenated signal (use context if provided)
    if context_epochs:
        segs = [np.asarray(s, dtype=float).reshape(-1) for s in context_epochs]
        concat = np.concatenate(segs)
        epoch_len = segs[center_idx].size
        total_samples = concat.size
        t = np.arange(total_samples) / float(fs)
        plot_signal = concat
    else:
        plot_signal = np.asarray(row["epoch"], dtype=float).reshape(-1)
        epoch_len = plot_signal.size
        t = np.arange(plot_signal.size) / float(fs)

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    title_fname = row.get('fname', '')
    fig.suptitle(f"{title_fname}: Dropped epoch {row['epoch_index']} - reasons: {', '.join(row['reasons']) or 'n/a'}", fontsize=11)

    ax.plot(t, plot_signal, color="#1f77b4", linewidth=1.0)

    # highlight the central epoch boundaries if context used
    if context_epochs:
        main_start = float(center_idx * epoch_len) / float(fs)
        main_end = float((center_idx + 1) * epoch_len) / float(fs)
        ax.axvline(main_start, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(main_end, color="k", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

    text_lines = [
        f"Epoch STD={row['std']:.4g}",
        f"File STD={row['std_file']:.4g}",
    ]
    ax.text(0.99, 0.02, "\n".join(text_lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.75"))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if out_path:
        fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


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
    out_directory = f"data_{preprocessing_info}_{epochizing_info}{epoch_category}{'_std' if args.standardize else ''}"
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
        file_std = float(mat_data["std"].item())
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
            pre_epochs, pre_dropped, pre_report = drop_bad_epochs(pre_epochs, file_std, fs, fname, args)
            print("  Screening dur epochs for artifacts...")
            dur_epochs, dur_dropped, dur_report = drop_bad_epochs(dur_epochs, file_std, fs, fname, args)

            n_pre_bad = sum(row["is_bad"] for row in pre_report)
            n_dur_bad = sum(row["is_bad"] for row in dur_report)
            print(f"  Rejected {n_pre_bad} pre-epochs and {n_dur_bad} dur-epochs using QC rules.")

            if pre_epochs.shape[0] == 0 and dur_epochs.shape[0] == 0:
                print("  Skipping: all epochs were rejected by QC.")
                continue

        # optional epoch-level standardization
        if args.standardize:
            pre_epochs = np.asarray([standardize_epoch(epoch) for epoch in pre_epochs], dtype=float)
            dur_epochs = np.asarray([standardize_epoch(epoch) for epoch in dur_epochs], dtype=float)

        # 5) store epochized data
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
