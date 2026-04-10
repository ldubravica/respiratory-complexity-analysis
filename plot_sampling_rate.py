import argparse
import os

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw respiratory signal at original sampling rate and compare "
            "two downsampled versions on the same time axis."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data_raw",
        help="Directory containing participant .mat files",
    )
    parser.add_argument(
        "--file",
        default="P003-1-trimmed.mat",
        help="Participant .mat file name (default: P003-1-trimmed.mat)",
    )
    parser.add_argument(
        "--var-name",
        default="data",
        help="Variable name inside .mat file",
    )
    parser.add_argument(
        "--fs-orig",
        type=float,
        default=2000.0,
        help="Original sampling rate in Hz",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[50.0, 25.0],
        help="One or more downsampled rates in Hz (example: --rates 100 50 25)",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time (seconds) of the display window",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=1000.0,
        help="Duration (seconds) of the display window",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Line transparency for overlays",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=0.9,
        help="Line width for overlays",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional output image path. If omitted, figure is only displayed.",
    )
    parser.add_argument(
        "--plot-original",
        action="store_true",
        help="Include original 2000 Hz signal in the overlay (default: off)",
    )
    return parser.parse_args()


def load_resp_channel(mat_path, var_name):
    mat = loadmat(mat_path)
    if var_name not in mat:
        raise KeyError(f"Variable '{var_name}' not found in {mat_path}")

    arr = np.asarray(mat[var_name])
    if arr.ndim < 2 or arr.shape[1] < 2:
        raise ValueError(
            f"Variable '{var_name}' must have at least 2 columns; got shape {arr.shape}"
        )

    resp = np.asarray(arr[:, 1], dtype=float)
    if resp.size == 0:
        raise ValueError("Respiratory channel is empty")

    finite_mask = np.isfinite(resp)
    if not np.all(finite_mask):
        resp = resp[finite_mask]
        if resp.size == 0:
            raise ValueError("Respiratory channel contains no finite samples")

    return resp


def downsample_signal(signal, fs_orig, fs_target):
    if fs_target <= 0:
        raise ValueError(f"Target sampling rate must be positive; got {fs_target}")
    if fs_target > fs_orig:
        raise ValueError(
            f"Target sampling rate ({fs_target}) cannot exceed original ({fs_orig})"
        )

    return np.asarray(
        nk.signal_resample(
            signal,
            sampling_rate=fs_orig,
            desired_sampling_rate=fs_target,
        ),
        dtype=float,
    )


def clip_window(signal, fs, start_sec, duration_sec):
    n = signal.size
    if n == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    total_sec = n / fs
    start_sec = max(0.0, min(start_sec, total_sec))
    end_sec = max(start_sec, min(start_sec + duration_sec, total_sec))

    idx_start = int(round(start_sec * fs))
    idx_end = int(round(end_sec * fs))
    idx_start = max(0, min(idx_start, n))
    idx_end = max(idx_start, min(idx_end, n))

    y = signal[idx_start:idx_end]
    t = np.arange(idx_start, idx_end, dtype=float) / fs
    return t, y


def main():
    args = parse_args()

    if len(args.rates) == 0:
        raise ValueError("Please provide at least one rate with --rates")

    unique_rates = []
    seen = set()
    for r in args.rates:
        if r not in seen:
            unique_rates.append(r)
            seen.add(r)
    if len(unique_rates) < len(args.rates):
        print("Warning: duplicate rates detected in --rates; duplicates were removed.")

    args.rates = unique_rates

    mat_path = os.path.join(args.input_dir, args.file)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Input file not found: {mat_path}")

    resp_raw = load_resp_channel(mat_path, args.var_name)

    downsampled = {}
    for rate in args.rates:
        downsampled[rate] = downsample_signal(resp_raw, args.fs_orig, rate)

    t_raw, y_raw = clip_window(resp_raw, args.fs_orig, args.start_sec, args.duration_sec)
    clipped = {}
    for rate in args.rates:
        clipped[rate] = clip_window(downsampled[rate], rate, args.start_sec, args.duration_sec)

    if all(clipped[rate][1].size == 0 for rate in args.rates):
        raise ValueError("Selected display window has no samples")

    print("Comparison summary")
    print(f"  File: {mat_path}")
    print(f"  Original fs: {args.fs_orig} Hz, samples: {resp_raw.size}")
    for rate in args.rates:
        print(f"  Downsampled fs: {rate} Hz, samples: {downsampled[rate].size}")
    print(
        f"  Window: {args.start_sec:.2f}s to "
        f"{args.start_sec + args.duration_sec:.2f}s"
    )

    fig, ax = plt.subplots(figsize=(22, 6))
    if args.plot_original:
        ax.plot(t_raw, y_raw, label=f"Original ({args.fs_orig:.0f} Hz)", linewidth=args.linewidth, alpha=args.alpha)

    for rate in args.rates:
        t_rate, y_rate = clipped[rate]
        ax.plot(t_rate, y_rate, label=f"Downsampled ({rate:g} Hz)", linewidth=args.linewidth, alpha=args.alpha)

    ax.set_title(f"Respiratory Signal Sampling Comparison - {args.file}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if args.save_path:
        out_dir = os.path.dirname(args.save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.save_path, dpi=180)
        print(f"Saved figure: {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()
