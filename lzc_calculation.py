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
OUTPUT_DIR = f"data_{INPUT_DIR[5:]}_lzc"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized epoch-level LZC from epochized respiratory files, "
            "compare pre vs dur, and report statistical significance."
        )
    )
    parser.add_argument("--pattern", default=FILE_PATTERN, help="Glob pattern for epochized files")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Directory with epochized .mat files")
    parser.add_argument("--skip-calculation", action="store_true", help="Skip LZC calculation and only print summary from existing CSV")
    parser.add_argument("--csv-filename", default=None, help="Optional custom filename for output CSV (default: lzc_<input_dir>.csv)")
    
    args = parser.parse_args()
    if args.csv_filename:
        args.skip_calculation = True
    
    return args

# ---------------------------
# Main loop
# ---------------------------

def process_files(args):
    print()

    # OBTAIN FILES TO PROCESS

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    method_name = args.input_dir[5:] # to remove "data_" prefix
    if args.csv_filename:
        csv_path = os.path.join(OUTPUT_DIR, args.csv_filename)
    else:
        csv_path = os.path.join(OUTPUT_DIR, f"lzc_{method_name}.csv")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    files_count = len(files)
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(args.input_dir, args.pattern)}")

    # files = files[10:20]  # TESTING

    # CALCULATE LZC FOR ALL EPOCHS IN ALL FILES

    rows = []
    for file_idx, path in enumerate(files, start=1):
        file_name = os.path.splitext(os.path.basename(path))[0]
        mat = loadmat(path)
        pre_epochs = mat.get("pre_epochs", np.empty((0, 0)))
        dur_epochs = mat.get("dur_epochs", np.empty((0, 0)))

        total_epochs_file = pre_epochs.shape[0] + dur_epochs.shape[0]
        print(f"[{file_idx}/{files_count}] {file_name}: pre = {pre_epochs.shape[0]}\t| dur = {dur_epochs.shape[0]}\t| total = {total_epochs_file}")

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

        for i, epoch in enumerate(dur_epochs):
            binary = epoch > np.median(epoch).item()
            lzc_val = ant.lziv_complexity(binary, normalize=True)
            rows.append(
                {
                    "file": file_name,
                    "session": "dur",
                    "epoch_index": int(i),
                    "lzc": lzc_val,
                }
            )

    # STORE RAW LZC VALUES

    if not rows:
        raise RuntimeError("No epochs were found to process.")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return df


def main():
    args = parse_args()
    method_name = args.input_dir[5:]  # to remove "data_" prefix
    if args.csv_filename:
        csv_path = os.path.join(OUTPUT_DIR, args.csv_filename)
    else:
        csv_path = os.path.join(OUTPUT_DIR, f"lzc_{method_name}.csv")

    # IF --skip-calculation, READ EXISTING CSV INSTEAD OF RE-COMPUTING LZC

    if args.skip_calculation:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = process_files(args)

    files_count = df["file"].nunique()

    # CALCULATE AVERAGE VALUES PER FILE

    summary_df = (
        df.groupby(["file", "session"])["lzc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "lzc_mean", "std": "lzc_std", "count": "epoch_count"})
        .sort_values(by=["file", "session"])
    )

    if args.csv_filename:
        summary_csv_path = os.path.join(OUTPUT_DIR, args.csv_filename.replace(".csv", "_summary.csv"))
    else:
        summary_csv_path = os.path.join(OUTPUT_DIR, f"lzc_{method_name}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # print("\n")
    # print(summary_df)
    summary_file_df = (
        summary_df.pivot(index="file", columns="session", values=["lzc_mean", "lzc_std", "epoch_count"])
        .reset_index()
    )
    # print(summary_file_df)
    summary_file_df.columns = [
        "file",
        "dur_lzc_mean",
        "pre_lzc_mean",
        "dur_lzc_std",
        "pre_lzc_std",
        "dur_epoch_count",
        "pre_epoch_count",
    ]
    # print(summary_file_df)

    # PRINT SUMMARY

    print("\n" + "=" * 72)
    print("LZC SUMMARY")
    print("=" * 72)
    print(f"Files processed: {files_count}\n")

    for _, row in summary_file_df.iterrows():
        pre_is_nan = np.isnan(row["pre_lzc_mean"])
        dur_is_nan = np.isnan(row["dur_lzc_mean"])
        
        pre_lzc = f"{row['pre_lzc_mean']:.4f} ± {row['pre_lzc_std']:.4f} ({int(row['pre_epoch_count'])})" if not pre_is_nan else "-.---- ± -.----    "
        dur_lzc = f"{row['dur_lzc_mean']:.4f} ± {row['dur_lzc_std']:.4f} ({int(row['dur_epoch_count'])})" if not dur_is_nan else "-.---- ± -.----    "

        print(f"{row['file']}\t| Pre LZC: {pre_lzc}\t| Dur LZC: {dur_lzc}")

    files_with_pre = summary_file_df[~summary_file_df["pre_lzc_mean"].isna()]["file"].tolist()
    files_with_dur = summary_file_df[~summary_file_df["dur_lzc_mean"].isna()]["file"].tolist()

    print(f"\nAverage Pre LZC: {summary_file_df['pre_lzc_mean'].mean():.4f} ± {summary_file_df['pre_lzc_mean'].std():.4f} \t({len(files_with_pre)}/{files_count} files | {int(summary_file_df['pre_epoch_count'].sum())} epochs)")
    print(f"Average Dur LZC: {summary_file_df['dur_lzc_mean'].mean():.4f} ± {summary_file_df['dur_lzc_mean'].std():.4f} \t({len(files_with_dur)}/{files_count} files | {int(summary_file_df['dur_epoch_count'].sum())} epochs)")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Summary CSV: {summary_csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
