import argparse
import glob
import os

import antropy as ant
import numpy as np
import pandas as pd
from scipy.io import loadmat

# ---------------------------
# Configuration
# ---------------------------

FILE_PATTERN = "*.mat"
INPUT_DIR = "data_khodadad2018_200Hz_120s"
OUTPUT_DIR = f"lzc_{INPUT_DIR[5:]}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized epoch-level LZC from epochized respiratory files, "
            "compare pre vs n2o, and report statistical significance."
        )
    )
    parser.add_argument("--pattern", default=FILE_PATTERN, help="Glob pattern for epochized files")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Directory with epochized .mat files")
    return parser.parse_args()

# ---------------------------
# Main loop
# ---------------------------

def main():
    args = parse_args()
    print("\n")

    # OBTAIN FILES TO PROCESS

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    method_name = args.input_dir[5:] # to remove "data_" prefix
    csv_path = os.path.join(OUTPUT_DIR, f"lzc_{method_name}.csv")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    files_count = len(files)
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(args.input_dir, args.pattern)}")

    files = files[10:20]  # TESTING

    # CALCULATE LZC FOR ALL EPOCHS IN ALL FILES

    rows = []
    for file_idx, path in enumerate(files, start=1):
        file_name = os.path.splitext(os.path.basename(path))[0]
        mat = loadmat(path)
        pre_epochs = mat.get("pre_epochs", np.empty((0, 0)))
        n2o_epochs = mat.get("n2o_epochs", np.empty((0, 0)))

        total_epochs_file = pre_epochs.shape[0] + n2o_epochs.shape[0]
        print(f"[{file_idx}/{files_count}] {file_name}: pre = {pre_epochs.shape[0]}\t| n2o = {n2o_epochs.shape[0]}\t| total = {total_epochs_file}")

        for i, epoch in enumerate(pre_epochs):
            binary = epoch > np.median(epoch).item()
            lzc_val = ant.lziv_complexity(binary, normalize=True)
            rows.append(
                {
                    "file": file_name,
                    "session": "pre",
                    "epoch_index": int(i),
                    "lzc": lzc_val,
                }
            )

        for i, epoch in enumerate(n2o_epochs):
            binary = epoch > np.median(epoch).item()
            lzc_val = ant.lziv_complexity(binary, normalize=True)
            rows.append(
                {
                    "file": file_name,
                    "session": "n2o",
                    "epoch_index": int(i),
                    "lzc": lzc_val,
                }
            )

    # STORE RAW LZC VALUES

    if not rows:
        raise RuntimeError("No epochs were found to process.")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # CALCULATE AVERAGE VALUES PER FILE

    summary_df = (
        df.groupby(["file", "session"])["lzc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "lzc_mean", "std": "lzc_std", "count": "epoch_count"})
        .sort_values(by=["file", "session"])
    )

    summary_csv_path = os.path.join(OUTPUT_DIR, f"lzc_{method_name}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n")
    # print(summary_df)
    summary_file_df = (
        summary_df.pivot(index="file", columns="session", values=["lzc_mean", "lzc_std", "epoch_count"])
        .reset_index()
    )
    summary_file_df.columns = [
        "file",
        "pre_lzc_mean",
        "n2o_lzc_mean",
        "pre_lzc_std",
        "n2o_lzc_std",
        "pre_lzc_epochs",
        "n2o_lzc_epochs",
    ]
    print(summary_file_df)

    # PRINT SUMMARY

    print("\n" + "=" * 72)
    print("LZC SUMMARY")
    print("=" * 72)
    print(f"Files processed: {files_count}\n")

    for _, row in summary_file_df.iterrows():
        print(f"{row['file']}\t| Pre LZC: {row['pre_lzc_mean']:.4f} ± {row['pre_lzc_std']:.4f} ({row['pre_lzc_epochs']})\t| N2O LZC: {row['n2o_lzc_mean']:.4f} ± {row['n2o_lzc_std']:.4f} ({row['n2o_lzc_epochs']})")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Summary CSV: {summary_csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
