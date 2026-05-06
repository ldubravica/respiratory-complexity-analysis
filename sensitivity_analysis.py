import antropy as ant
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from scipy.io import loadmat, savemat
from tqdm import tqdm  # Progress bar


# ---------------------------
# Configuration
# ---------------------------


INPUT_DIR = "data_khodadad2018_200Hz"
INPUT_PATTERN = ""

WINDOW_SIZES = [10, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]


# ---------------------------
# Phases & Tools
# ---------------------------


def cell_array_to_list(cell_array):
    """Unpack a MATLAB cell array (object array) back into a list of 1D segments."""
    cell_array = np.asarray(cell_array, dtype=object).reshape(-1)
    segments = []
    for cell in cell_array:
        segment = np.asarray(cell, dtype=float).reshape(-1)
        segments.append(segment)
    return segments


def plot_stability(df, title):
    plt.figure(figsize=(12, 6))
    for label, grp in df.groupby('File'):
        plt.plot(grp['Window_Sec'], grp['LZC'], marker='o', alpha=0.7, label=label)
    
    plt.axvline(x=60, color='red', linestyle='--', label='Suggested Start (60s)')
    plt.title(f"LZC {title} Stability Across Window Sizes (Respiratory Data)")
    plt.xlabel("Window Length (Seconds)")
    plt.ylabel("Normalized LZC")
    # plt.xscale('log') # Log scale helps see 1s vs 300s better # MAYBE REMOVE
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ---------------------------
# Main loop
# ---------------------------


def main():
    results = []

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".mat"):
            continue

        if INPUT_PATTERN not in fname:
            continue

        # 0) isolate a test sample
        if not any(p in fname for p in ['P005-2', 'P006-1', 'P046-1', 'P077-1', 'P079-1']):
            continue

        path = os.path.join(INPUT_DIR, fname)
        print(f"\nProcessing {fname}...")

        # 1) load filtered, downsampled, and segmented data
        mat_data = loadmat(path)
        fs = float(mat_data["fs"])
        pre_segment = cell_array_to_list(mat_data["pre_segments"])[0]
        n2o_segment = cell_array_to_list(mat_data["n2o_segments"])[0]

        # 2) binarize using median
        pre_binary = pre_segment > np.median(pre_segment)
        n2o_binary = n2o_segment > np.median(n2o_segment)

        # 3) calculate LZC for multiple random windows of each segment for each window size
        for window in tqdm(WINDOW_SIZES, desc=f"Testing windows for {fname}", leave=False):

            vals = []

            n_points = int(window * fs)

            for i in range(0, len(pre_binary), n_points):  # Iterate over pre_binary in steps of window's size

                lzc_antropy = ant.lziv_complexity(pre_binary[i : i + n_points], normalize=True)
                vals.append(lzc_antropy)

            for i in range(0, len(n2o_binary), n_points):  # Iterate over n2o_binary in steps of window's size

                lzc_antropy = ant.lziv_complexity(n2o_binary[i : i + n_points], normalize=True)
                vals.append(lzc_antropy)

            results.append({
                'File': fname,
                'Window_Sec': window,
                'LZC': np.mean(vals)
            })

    df_antropy = pd.DataFrame(results)
    df_antropy.to_csv('respiratory_lzc_sensitivity.csv', index=False)

    # 3) plot data for this sample
    plot_stability(df_antropy, title="Antropy")


if __name__ == "__main__":
    main()
    print("\n")
