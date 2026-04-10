import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


DEFAULT_METHODS = ["biosppy", "downsample", "khodadad2018", "manual_nk2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare cleaned+segmented respiratory outputs across multiple methods "
            "for each participant/session prefix (e.g., P003-1)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data_clean_segmented",
        help="Directory containing files like P003-1-biosppy.mat",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to compare (must match filename suffixes before .mat)",
    )
    parser.add_argument(
        "--participants",
        nargs="*",
        default=None,
        help="Optional list of participant-session prefixes, e.g. P001-1 P003-2",
    )
    parser.add_argument(
        "--output-dir",
        default="Figures/clean_segment_compare",
        help="Directory to save comparison plots",
    )
    return parser.parse_args()


def discover_files(input_dir, methods):
    """Return mapping: prefix -> {method: filepath} for available files."""
    patt = re.compile(r"^(P\d{3}-\d)-([A-Za-z0-9_]+)\.mat$")
    grouped = defaultdict(dict)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".mat"):
            continue
        m = patt.match(fname)
        if not m:
            continue

        prefix, method = m.group(1), m.group(2)
        if method not in methods:
            continue

        grouped[prefix][method] = os.path.join(input_dir, fname)

    return grouped


def flatten_valid_segments(seg_2d, lengths):
    """Convert NaN-padded segment matrix + lengths into one concatenated 1D signal."""
    if seg_2d.size == 0 or lengths.size == 0:
        return np.empty((0,), dtype=float)

    chunks = []
    n_rows = min(seg_2d.shape[0], lengths.shape[0])
    for i in range(n_rows):
        seg_len = int(lengths[i])
        if seg_len <= 0:
            continue
        seg_len = min(seg_len, seg_2d.shape[1])
        chunk = np.asarray(seg_2d[i, :seg_len], dtype=float)
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size > 0:
            chunks.append(chunk)

    if not chunks:
        return np.empty((0,), dtype=float)
    return np.concatenate(chunks)


def load_method_data(path):
    mat = loadmat(path)

    pre_segments = np.asarray(mat.get("pre_segments", np.empty((0, 0))), dtype=float)
    n2o_segments = np.asarray(mat.get("n2o_segments", np.empty((0, 0))), dtype=float)

    pre_lengths = np.asarray(mat.get("pre_segment_lengths", np.empty((0,))), dtype=int).reshape(-1)
    n2o_lengths = np.asarray(mat.get("n2o_segment_lengths", np.empty((0,))), dtype=int).reshape(-1)

    fs_raw = np.asarray(mat.get("fs", np.array([[np.nan]]))).reshape(-1)
    fs = float(fs_raw[0]) if fs_raw.size else np.nan

    n2o_start_raw = np.asarray(mat.get("n2o_start_sec", np.array([[np.nan]]))).reshape(-1)
    n2o_start_sec = float(n2o_start_raw[0]) if n2o_start_raw.size else np.nan

    pre_flat = flatten_valid_segments(pre_segments, pre_lengths)
    n2o_flat = flatten_valid_segments(n2o_segments, n2o_lengths)

    return {
        "fs": fs,
        "n2o_start_sec": n2o_start_sec,
        "pre_segments": pre_segments,
        "n2o_segments": n2o_segments,
        "pre_lengths": pre_lengths,
        "n2o_lengths": n2o_lengths,
        "pre_flat": pre_flat,
        "n2o_flat": n2o_flat,
    }


def summarize_signal(x):
    if x.size == 0:
        return {
            "samples": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "rms": np.nan,
            "p2p": np.nan,
            "nan_frac": np.nan,
        }

    finite = np.asarray(x[np.isfinite(x)], dtype=float)
    if finite.size == 0:
        return {
            "samples": int(x.size),
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "rms": np.nan,
            "p2p": np.nan,
            "nan_frac": 1.0,
        }

    q25, q75 = np.percentile(finite, [25, 75])
    return {
        "samples": int(x.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "iqr": float(q75 - q25),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "p2p": float(np.ptp(finite)),
        "nan_frac": float(1.0 - (finite.size / x.size)),
    }


def print_general_metrics(prefix, d):
    pre_stats = summarize_signal(d["pre_flat"])
    n2o_stats = summarize_signal(d["n2o_flat"])

    pre_dur = pre_stats["samples"] / d["fs"] if np.isfinite(d["fs"]) and d["fs"] > 0 else np.nan
    n2o_dur = n2o_stats["samples"] / d["fs"] if np.isfinite(d["fs"]) and d["fs"] > 0 else np.nan

    print(f"  [{prefix}] General Overview")
    print(
        f"    fs={d['fs']:.3f} Hz | n2o_start={d['n2o_start_sec']:.3f}s | "
        f"pre_segments={d['pre_segments'].shape[0]} | n2o_segments={d['n2o_segments'].shape[0]}"
    )
    print(f"    PRE: samples={pre_stats['samples']}, duration={pre_dur:.2f}s")
    print(f"    N2O: samples={n2o_stats['samples']}, duration={n2o_dur:.2f}s")


def print_method_metrics(prefix, method, d):
    pre_stats = summarize_signal(d["pre_flat"])
    n2o_stats = summarize_signal(d["n2o_flat"])

    print(f"  [{prefix}] Method: {method}")
    print(
        f"    PRE: mean={pre_stats['mean']:.5f}, std={pre_stats['std']:.5f}, "
        f"rms={pre_stats['rms']:.5f}, iqr={pre_stats['iqr']:.5f}, p2p={pre_stats['p2p']:.5f}"
    )
    print(
        f"    N2O: mean={n2o_stats['mean']:.5f}, std={n2o_stats['std']:.5f}, "
        f"rms={n2o_stats['rms']:.5f}, iqr={n2o_stats['iqr']:.5f}, p2p={n2o_stats['p2p']:.5f}"
    )


def safe_corr(a, b):
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    a2 = np.asarray(a[:n], dtype=float)
    b2 = np.asarray(b[:n], dtype=float)
    mask = np.isfinite(a2) & np.isfinite(b2)
    if np.sum(mask) < 3:
        return np.nan
    a3 = a2[mask]
    b3 = b2[mask]
    if np.std(a3) == 0 or np.std(b3) == 0:
        return np.nan
    return float(np.corrcoef(a3, b3)[0, 1])


def print_pairwise_similarity(prefix, method_data, methods):
    present = [m for m in methods if m in method_data]
    if len(present) < 2:
        return

    print(f"  [{prefix}] Pairwise correlations (aligned by truncated samples):")
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            m1, m2 = present[i], present[j]
            c_pre = safe_corr(method_data[m1]["pre_flat"], method_data[m2]["pre_flat"])
            c_n2o = safe_corr(method_data[m1]["n2o_flat"], method_data[m2]["n2o_flat"])
            print(f"    {m1} vs {m2}: PRE corr={c_pre:.4f}, N2O corr={c_n2o:.4f}")


def plot_prefix_comparison(prefix, method_data, methods, output_dir):
    present = [m for m in methods if m in method_data]
    if not present:
        return

    fig, axes = plt.subplots(2, 1, figsize=(24, 8), sharex=False)

    for method in present:
        d = method_data[method]
        fs = d["fs"] if np.isfinite(d["fs"]) and d["fs"] > 0 else 25.0

        pre = d["pre_flat"]
        if pre.size > 0:
            t_pre = np.arange(pre.size) / fs
            axes[0].plot(t_pre, pre, linewidth=0.8, alpha=0.9, label=method)

        n2o = d["n2o_flat"]
        if n2o.size > 0:
            t_n2o = np.arange(n2o.size) / fs
            axes[1].plot(t_n2o, n2o, linewidth=0.8, alpha=0.9, label=method)

    axes[0].set_title(f"{prefix} - PRE session comparison")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].set_title(f"{prefix} - N2O session comparison")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle(f"Respiratory cleaning comparison: {prefix}", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}-compare.png")
    fig.savefig(out_path, dpi=160)
    plt.show()
    plt.close(fig)
    print(f"  [{prefix}] Saved comparison chart: {out_path}")


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    grouped = discover_files(args.input_dir, set(args.methods))
    if not grouped:
        print("No matching .mat files found.")
        return

    prefixes = sorted(grouped.keys())
    if args.participants:
        wanted = set(args.participants)
        prefixes = [p for p in prefixes if p in wanted]

    if not prefixes:
        print("No participant/session prefixes matched your selection.")
        return

    print(f"Found {len(prefixes)} participant/session groups.")
    print(f"Methods requested: {', '.join(args.methods)}")

    for prefix in prefixes:
        files_by_method = grouped[prefix]
        present = [m for m in args.methods if m in files_by_method]
        missing = [m for m in args.methods if m not in files_by_method]

        print(f"\nProcessing {prefix}...")
        print(f"  Present methods: {', '.join(present) if present else 'none'}")
        if missing:
            print(f"  Missing methods: {', '.join(missing)}")

        d = load_method_data(files_by_method[present[0]])
        print_general_metrics(prefix, d)

        method_data = {}
        for method in present:
            path = files_by_method[method]
            d = load_method_data(path)
            method_data[method] = d
            print_method_metrics(prefix, method, d)

        if method_data:
            print_pairwise_similarity(prefix, method_data, args.methods)
            plot_prefix_comparison(prefix, method_data, args.methods, args.output_dir)

        print()


if __name__ == "__main__":
    main()
