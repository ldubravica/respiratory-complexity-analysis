import argparse
import antropy as ant
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import time

from scipy.io import loadmat, savemat
from tqdm import tqdm


# ---------------------------
# Configuration
# ---------------------------

INPUT_DIR = "data_khodadad2018_200Hz"

WINDOW_SIZES = [10, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]
WINDOW_SIZES = [5, 10, 15, 20]
MIN_WINDOWS_PER_SIGNAL = 3
N_SEGMENTS_PER_CONDITION = 5

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


def plot_stability(df):
    plt.figure(figsize=(12, 6))
    for label, grp in df.groupby('Name'):
        plt.plot(grp['Window_Sec'], grp['LZC'], marker='o', alpha=0.7, label=label)
    
    plt.title("LZC Stability Across Window Sizes (Respiratory Data)")
    plt.xlabel("Window Length (Seconds)")
    plt.ylabel("Normalized LZC")
    # plt.xscale('log') # Log scale helps see 1s vs 300s better # MAYBE REMOVE
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def print_summary_statistics(df):
    """Print summary statistics for the sensitivity analysis"""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    # Group by window size and calculate statistics
    summary = df.groupby(['Window_Sec']).agg({
        'LZC': ['mean', 'std', 'count'],
        'Binary_Balance': ['mean', 'std']
    }).round(4)
    
    print("\nWindow Size Statistics:")
    print(summary)
    
    # Calculate coefficient of variation for stability
    cv_stats = df.groupby(['Window_Sec']).apply(
        lambda x: x['LZC'].std() / x['LZC'].mean() if x['LZC'].mean() > 0 else float('inf')
    ).round(4)
    
    print("\nCoefficient of Variation (Lower = More Stable):")
    print(cv_stats)
    
    # Find most stable window sizes
    stable_windows = cv_stats.groupby('Window_Sec').mean().sort_values().head(3)
    print(f"\nTop 3 Most Stable Window Sizes:")
    for window, cv in stable_windows.items():
        print(f"  {window}s: CV = {cv:.4f}")


def plot_improved_stability(df):
    """Create improved stability plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean LZC vs Window Size (with error bars)
    ax1 = axes[0, 0]
    stats = df.groupby('Window_Sec')['LZC'].agg(['mean', 'std'])
    ax1.errorbar(stats.index, stats['mean'], yerr=stats['std'], 
                marker='o', capsize=5)
    
    ax1.set_xlabel("Window Length (Seconds)")
    ax1.set_ylabel("Normalized LZC")
    ax1.set_title("LZC vs Window Size (with Error Bars)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of Variation
    ax2 = axes[0, 1]
    cv_data = df.groupby(['Window_Sec']).apply(
        lambda x: x['LZC'].std() / x['LZC'].mean() if x['LZC'].mean() > 0 else float('inf')
    )
    
    for method in cv_data.columns:
        ax2.plot(cv_data.index, cv_data[method], marker='o', label=method)
    
    ax2.set_xlabel("Window Length (Seconds)")
    ax2.set_ylabel("Coefficient of Variation")
    ax2.set_title("Stability Analysis (Lower CV = More Stable)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Binary Balance Quality
    ax3 = axes[1, 0]
    stats = df.groupby('Window_Sec')['Binary_Balance'].agg(['mean', 'std'])
    ax3.errorbar(stats.index, stats['mean'], yerr=stats['std'], marker='o', capsize=5)
    
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Ideal (0.5)')
    ax3.set_xlabel("Window Length (Seconds)")
    ax3.set_ylabel("Binary Balance (Mean of Binary Signal)")
    ax3.set_title("Binarization Quality Analysis")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample Size Effect
    ax4 = axes[1, 1]
    sample_counts = df.groupby('Window_Sec').size()
    ax4.bar(sample_counts.index, sample_counts.values)
    ax4.set_xlabel("Window Length (Seconds)")
    ax4.set_ylabel("Number of Samples")
    ax4.set_title("Sample Size per Window Length")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ---------------------------
# LZC Calculations
# ---------------------------

def calculate_lzc_per_window(binary_signal, n_points, method='antropy'):

    vals = []
    
    for i in range(0, len(binary_signal), n_points):  # Iterate over binary_signal in steps of window's size

        # start_time = time.time()
        if method == 'antropy':
            lzc = ant.lziv_complexity(binary_signal[i : i + n_points], normalize=True)
        elif method == 'nk2':  # Seemingly same results, just significantly slower
            lzc, info = nk.complexity_lempelziv(binary_signal[i : i + n_points])
        else:
            raise ValueError(f"Unknown method: {method}! TIP: add implementation in calculate_lzc_per_window()")
        vals.append(lzc)
        # end_time = time.time()
        # print(f"LZC {method}: {lzc:.10f} (computed in {end_time - start_time:.3f} seconds)")

    return np.mean(vals)

# ---------------------------
# Main loop
# ---------------------------

def main():
    pre_segments = {}
    n2o_segments = {}
    fs = 200

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".mat"):
            continue

        path = os.path.join(INPUT_DIR, fname)
        name = os.path.splitext(fname)[0]

        # 1) load filtered, downsampled, and segmented data
        mat_data = loadmat(path)
        fs = float(mat_data["fs"])
        pre_segments_list = cell_array_to_list(mat_data["pre_segments"])
        n2o_segments_list = cell_array_to_list(mat_data["n2o_segments"])

        if len(pre_segments) < N_SEGMENTS_PER_CONDITION:
            for pre_seg in pre_segments_list:
                if len(pre_seg) >= MIN_WINDOWS_PER_SIGNAL * fs * max(WINDOW_SIZES):
                    pre_segments[f"{name}-pre"] = pre_seg

        if len(n2o_segments) < N_SEGMENTS_PER_CONDITION:
            for n2o_seg in n2o_segments_list:
                if len(n2o_seg) >= MIN_WINDOWS_PER_SIGNAL * fs * max(WINDOW_SIZES):
                    n2o_segments[f"{name}-n2o"] = n2o_seg

        if len(pre_segments) == N_SEGMENTS_PER_CONDITION and len(n2o_segments) == N_SEGMENTS_PER_CONDITION:
            break


    segments = {**pre_segments, **n2o_segments}
    print(f"\nCollected {len(pre_segments)} pre segments and {len(n2o_segments)} n2o segments for analysis.")
    for seg_name in segments:
        print(f"  {seg_name}: {len(segments[seg_name])} samples ({len(segments[seg_name]) / fs:.2f} seconds)")


    results = []
    for segment_name, segment_data in segments.items():
        print(f"\nProcessing segment: {segment_name}...")
        
        # binarize using median
        binary_signal = segment_data > np.median(segment_data).item()

        # calculate LZC for multiple random windows of each segment for each window size
        for window in tqdm(WINDOW_SIZES, desc=f"Testing windows for {segment_name}", leave=False):
            # print(f"  Window size: {window} seconds")
            n_points = int(window * fs)
            lzc = calculate_lzc_per_window(binary_signal, n_points)
            results.append({
                'Name': segment_name,
                'Window_Sec': window,
                'LZC': lzc,
                'Variance': np.var(segment_data),
                'Binary_Balance': np.mean(binary_signal)
            })

    # store results and plot data
    df = pd.DataFrame(results)
    print_summary_statistics(df)
    df.to_csv(f'respiratory_lzc_sensitivity.csv', index=False)
    plot_stability(df)
    plot_improved_stability(df)


if __name__ == "__main__":
    main()
    print("\n")
