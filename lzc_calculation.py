import argparse
import glob
import os

import antropy as ant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon


DEFAULT_INPUT_DIR = "data_epochized_120s"
DEFAULT_PATTERN = "*.mat"
DEFAULT_OUTPUT_DIR = "Figures/lzc_calculation"
DEFAULT_CSV_NAME = "lzc_epoch_values.csv"
DEFAULT_FIG_NAME = "lzc_pre_vs_n2o.png"
DEFAULT_BOX_FIG_NAME = "lzc_pre_vs_n2o_boxplot.png"
DEFAULT_STATS_CSV_NAME = "lzc_stats_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized epoch-level LZC from epochized respiratory files, "
            "compare pre vs n2o, and report statistical significance."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory with epochized .mat files")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for epochized files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV and figure")
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help="CSV file name")
    parser.add_argument("--stats-csv-name", default=DEFAULT_STATS_CSV_NAME, help="Stats summary CSV file name")
    parser.add_argument("--fig-name", default=DEFAULT_FIG_NAME, help="Figure file name")
    parser.add_argument("--box-fig-name", default=DEFAULT_BOX_FIG_NAME, help="Box plot figure file name")
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Skip LZC recomputation and load existing CSV for plotting/statistics",
    )
    parser.add_argument(
        "--reuse-stats",
        action="store_true",
        help="Skip statistical recomputation and load existing stats summary CSV",
    )
    parser.add_argument(
        "--binarize",
        choices=["median", "mean"],
        default="median",
        help="Thresholding method before LZC",
    )
    return parser.parse_args()


def discover_files(input_dir, pattern):
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def ensure_2d(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def threshold_binary(epoch, method="median"):
    x = np.asarray(epoch, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.array([], dtype=bool)

    if method == "median":
        thr = np.median(x)
    else:
        thr = np.mean(x)

    return x > thr


def normalized_lzc_from_epoch(epoch, binarize_method="median"):
    binary = threshold_binary(epoch, method=binarize_method)
    if binary.size < 2:
        return np.nan

    # If thresholding gives a constant sequence, complexity is minimal.
    if np.all(binary == binary[0]):
        return 0.0

    try:
        return float(ant.lziv_complexity(binary, normalize=True))
    except Exception:
        return np.nan


def load_epochs_from_file(path):
    mat = loadmat(path)

    pre_epochs = ensure_2d(mat.get("pre_epochs", np.empty((0, 0))))
    n2o_epochs = ensure_2d(mat.get("n2o_epochs", np.empty((0, 0))))

    fs_raw = np.asarray(mat.get("fs", np.array([[np.nan]])), dtype=float).reshape(-1)
    fs = float(fs_raw[0]) if fs_raw.size else np.nan

    epoch_len_raw = np.asarray(mat.get("epoch_length_sec", np.array([[np.nan]])), dtype=float).reshape(-1)
    epoch_length_sec = float(epoch_len_raw[0]) if epoch_len_raw.size else np.nan

    return pre_epochs, n2o_epochs, fs, epoch_length_sec


def describe(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    q25, q75 = np.percentile(vals, [25, 75])
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "median": float(np.median(vals)),
        "q25": float(q25),
        "q75": float(q75),
    }


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_sd = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled_sd == 0:
        return np.nan
    return float((np.mean(b) - np.mean(a)) / pooled_sd)


def run_stats(df):
    pre = df.loc[df["session"] == "pre", "lzc"].to_numpy(dtype=float)
    n2o = df.loc[df["session"] == "n2o", "lzc"].to_numpy(dtype=float)

    pre = pre[np.isfinite(pre)]
    n2o = n2o[np.isfinite(n2o)]

    stats_out = {
        "welch_t": np.nan,
        "welch_p": np.nan,
        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,
        "cohen_d": cohen_d(pre, n2o),
        "paired_t": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_w": np.nan,
        "wilcoxon_p": np.nan,
        "n_paired_files": 0,
    }

    if pre.size > 1 and n2o.size > 1:
        t_stat, t_p = ttest_ind(pre, n2o, equal_var=False, nan_policy="omit")
        u_stat, u_p = mannwhitneyu(pre, n2o, alternative="two-sided")
        stats_out["welch_t"] = float(t_stat)
        stats_out["welch_p"] = float(t_p)
        stats_out["mannwhitney_u"] = float(u_stat)
        stats_out["mannwhitney_p"] = float(u_p)

    # Optional paired analysis across per-file means.
    per_file = (
        df.groupby(["file", "session"]) ["lzc"]
        .mean()
        .unstack("session")
        .dropna(subset=["pre", "n2o"], how="any")
    )
    if not per_file.empty and per_file.shape[0] > 1:
        pre_m = per_file["pre"].to_numpy(dtype=float)
        n2o_m = per_file["n2o"].to_numpy(dtype=float)
        t_rel_stat, t_rel_p = ttest_rel(pre_m, n2o_m, nan_policy="omit")
        w_stat, w_p = wilcoxon(pre_m, n2o_m, alternative="two-sided", zero_method="wilcox")
        stats_out["paired_t"] = float(t_rel_stat)
        stats_out["paired_t_p"] = float(t_rel_p)
        stats_out["wilcoxon_w"] = float(w_stat)
        stats_out["wilcoxon_p"] = float(w_p)
        stats_out["n_paired_files"] = int(per_file.shape[0])

    return stats_out


def build_stats_row(pre_desc, n2o_desc, stats_out, files_count, binarize, reuse_csv, reuse_stats):
    return {
        "files_processed": int(files_count),
        "binarization": str(binarize),
        "reuse_csv_mode": bool(reuse_csv),
        "reuse_stats_mode": bool(reuse_stats),
        "pre_n": pre_desc["n"],
        "pre_mean": pre_desc["mean"],
        "pre_std": pre_desc["std"],
        "pre_median": pre_desc["median"],
        "pre_q25": pre_desc["q25"],
        "pre_q75": pre_desc["q75"],
        "n2o_n": n2o_desc["n"],
        "n2o_mean": n2o_desc["mean"],
        "n2o_std": n2o_desc["std"],
        "n2o_median": n2o_desc["median"],
        "n2o_q25": n2o_desc["q25"],
        "n2o_q75": n2o_desc["q75"],
        "mean_delta_n2o_minus_pre": n2o_desc["mean"] - pre_desc["mean"],
        "welch_t": stats_out["welch_t"],
        "welch_p": stats_out["welch_p"],
        "mannwhitney_u": stats_out["mannwhitney_u"],
        "mannwhitney_p": stats_out["mannwhitney_p"],
        "cohen_d_n2o_minus_pre": stats_out["cohen_d"],
        "paired_t": stats_out["paired_t"],
        "paired_t_p": stats_out["paired_t_p"],
        "wilcoxon_w": stats_out["wilcoxon_w"],
        "wilcoxon_p": stats_out["wilcoxon_p"],
        "n_paired_files": stats_out["n_paired_files"],
    }


def make_box_plot(df, out_path):
    pre_vals = df.loc[df["session"] == "pre", "lzc"].to_numpy(dtype=float)
    n2o_vals = df.loc[df["session"] == "n2o", "lzc"].to_numpy(dtype=float)
    pre_vals = pre_vals[np.isfinite(pre_vals)]
    n2o_vals = n2o_vals[np.isfinite(n2o_vals)]

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [pre_vals, n2o_vals]
    labels = ["Pre", "N2O"]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6},
        medianprops={"color": "red", "linewidth": 1.5},
    )
    bp["boxes"][0].set(facecolor="none", edgecolor="black")
    bp["boxes"][1].set(facecolor="none", edgecolor="black")

    # Light jittered points to show sample density.
    if pre_vals.size > 0:
        x_pre = 1 + np.random.uniform(-0.08, 0.08, size=pre_vals.size)
        ax.scatter(x_pre, pre_vals, s=10, alpha=0.45, color="green")
    if n2o_vals.size > 0:
        x_n2o = 2 + np.random.uniform(-0.08, 0.08, size=n2o_vals.size)
        ax.scatter(x_n2o, n2o_vals, s=10, alpha=0.45, color="blue")

    ax.set_title("Normalized LZC Distribution: Pre vs N2O")
    ax.set_ylabel("Normalized LZC")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    stats_csv_path = os.path.join(args.output_dir, args.stats_csv_name)
    fig_path = os.path.join(args.output_dir, args.fig_name)
    box_fig_path = os.path.join(args.output_dir, args.box_fig_name)

    fs_values = []
    epoch_len_values = []

    if args.reuse_csv:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found for reuse mode: {csv_path}")
        df = pd.read_csv(csv_path)
        required_cols = {"file", "session", "epoch_index", "lzc"}
        if not required_cols.issubset(df.columns):
            raise ValueError("CSV is missing required columns: file, session, epoch_index, lzc")
        print(f"Loaded existing CSV without recomputation: {csv_path}")
        files_count = int(df["file"].nunique()) if "file" in df.columns else 0
    else:
        files = discover_files(args.input_dir, args.pattern)
        if not files:
            raise FileNotFoundError(f"No files matched {os.path.join(args.input_dir, args.pattern)}")

        rows = []
        total_files = len(files)
        for file_idx, path in enumerate(files, start=1):
            file_name = os.path.basename(path)
            pre_epochs, n2o_epochs, fs, epoch_length_sec = load_epochs_from_file(path)

            total_epochs_file = pre_epochs.shape[0] + n2o_epochs.shape[0]
            print(f"[{file_idx}/{total_files}] {file_name}: pre={pre_epochs.shape[0]}, n2o={n2o_epochs.shape[0]}, total={total_epochs_file}")

            if np.isfinite(fs):
                fs_values.append(fs)
            if np.isfinite(epoch_length_sec):
                epoch_len_values.append(epoch_length_sec)

            for i, epoch in enumerate(pre_epochs):
                lzc_val = normalized_lzc_from_epoch(epoch, binarize_method=args.binarize)
                rows.append(
                    {
                        "file": file_name,
                        "session": "pre",
                        "epoch_index": int(i),
                        "lzc": lzc_val,
                    }
                )

            for i, epoch in enumerate(n2o_epochs):
                lzc_val = normalized_lzc_from_epoch(epoch, binarize_method=args.binarize)
                rows.append(
                    {
                        "file": file_name,
                        "session": "n2o",
                        "epoch_index": int(i),
                        "lzc": lzc_val,
                    }
                )

        if not rows:
            raise RuntimeError("No epochs were found to process.")

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        files_count = len(files)

    if args.reuse_stats:
        if not os.path.exists(stats_csv_path):
            raise FileNotFoundError(f"Stats CSV not found for reuse mode: {stats_csv_path}")
        stats_df = pd.read_csv(stats_csv_path)
        if stats_df.empty:
            raise ValueError(f"Stats CSV is empty: {stats_csv_path}")
        stats_row = stats_df.iloc[0].to_dict()
        print(f"Loaded existing stats without recomputation: {stats_csv_path}")

        pre_desc = {
            "n": int(stats_row.get("pre_n", 0)),
            "mean": float(stats_row.get("pre_mean", np.nan)),
            "std": float(stats_row.get("pre_std", np.nan)),
            "median": float(stats_row.get("pre_median", np.nan)),
            "q25": float(stats_row.get("pre_q25", np.nan)),
            "q75": float(stats_row.get("pre_q75", np.nan)),
        }
        n2o_desc = {
            "n": int(stats_row.get("n2o_n", 0)),
            "mean": float(stats_row.get("n2o_mean", np.nan)),
            "std": float(stats_row.get("n2o_std", np.nan)),
            "median": float(stats_row.get("n2o_median", np.nan)),
            "q25": float(stats_row.get("n2o_q25", np.nan)),
            "q75": float(stats_row.get("n2o_q75", np.nan)),
        }
        stats_out = {
            "welch_t": float(stats_row.get("welch_t", np.nan)),
            "welch_p": float(stats_row.get("welch_p", np.nan)),
            "mannwhitney_u": float(stats_row.get("mannwhitney_u", np.nan)),
            "mannwhitney_p": float(stats_row.get("mannwhitney_p", np.nan)),
            "cohen_d": float(stats_row.get("cohen_d_n2o_minus_pre", np.nan)),
            "paired_t": float(stats_row.get("paired_t", np.nan)),
            "paired_t_p": float(stats_row.get("paired_t_p", np.nan)),
            "wilcoxon_w": float(stats_row.get("wilcoxon_w", np.nan)),
            "wilcoxon_p": float(stats_row.get("wilcoxon_p", np.nan)),
            "n_paired_files": int(stats_row.get("n_paired_files", 0)),
        }
        mean_delta = float(stats_row.get("mean_delta_n2o_minus_pre", np.nan))
    else:
        pre_desc = describe(df.loc[df["session"] == "pre", "lzc"].to_numpy(dtype=float))
        n2o_desc = describe(df.loc[df["session"] == "n2o", "lzc"].to_numpy(dtype=float))
        stats_out = run_stats(df)
        mean_delta = n2o_desc["mean"] - pre_desc["mean"]

        stats_row = build_stats_row(
            pre_desc=pre_desc,
            n2o_desc=n2o_desc,
            stats_out=stats_out,
            files_count=files_count,
            binarize=args.binarize,
            reuse_csv=args.reuse_csv,
            reuse_stats=args.reuse_stats,
        )
        pd.DataFrame([stats_row]).to_csv(stats_csv_path, index=False)

    print("\n" + "=" * 72)
    print("LZC PRE VS N2O SUMMARY")
    print("=" * 72)
    print(f"Files processed: {files_count}")
    if fs_values:
        print(f"Sampling rate range: {np.min(fs_values):.3f} to {np.max(fs_values):.3f} Hz")
    if epoch_len_values:
        print(f"Epoch length range: {np.min(epoch_len_values):.3f} to {np.max(epoch_len_values):.3f} sec")
    print(f"Binarization: {args.binarize}")
    print(f"Reuse CSV mode: {args.reuse_csv}")
    print(f"Reuse stats mode: {args.reuse_stats}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Stats CSV: {stats_csv_path}")
    print(f"Saved Scatter Figure: {fig_path}")
    print(f"Saved Box Figure: {box_fig_path}")

    print("\nDescriptive statistics")
    print(
        f"  Pre: n={pre_desc['n']}, mean={pre_desc['mean']:.5f}, std={pre_desc['std']:.5f}, "
        f"median={pre_desc['median']:.5f}, IQR=[{pre_desc['q25']:.5f}, {pre_desc['q75']:.5f}]"
    )
    print(
        f"  N2O: n={n2o_desc['n']}, mean={n2o_desc['mean']:.5f}, std={n2o_desc['std']:.5f}, "
        f"median={n2o_desc['median']:.5f}, IQR=[{n2o_desc['q25']:.5f}, {n2o_desc['q75']:.5f}]"
    )

    print(f"  Mean delta (N2O - Pre): {mean_delta:.5f}")

    print("\nSignificance tests")
    print(f"  Welch t-test: t={stats_out['welch_t']:.5f}, p={stats_out['welch_p']:.6g}")
    print(f"  Mann-Whitney U: U={stats_out['mannwhitney_u']:.5f}, p={stats_out['mannwhitney_p']:.6g}")
    print(f"  Effect size (Cohen d, N2O-Pre): d={stats_out['cohen_d']:.5f}")

    if stats_out["n_paired_files"] > 1:
        print("\nPaired file-level tests (using per-file mean LZC)")
        print(f"  Paired t-test: t={stats_out['paired_t']:.5f}, p={stats_out['paired_t_p']:.6g}")
        print(f"  Wilcoxon signed-rank: W={stats_out['wilcoxon_w']:.5f}, p={stats_out['wilcoxon_p']:.6g}")
        print(f"  Paired files used: {stats_out['n_paired_files']}")

    print("=" * 72)

    make_box_plot(df, box_fig_path)


if __name__ == "__main__":
    main()
