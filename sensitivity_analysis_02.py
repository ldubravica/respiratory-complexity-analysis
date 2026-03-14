import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import antropy as ant
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# --- CONFIGURATION ---
DATA_DIR = 'N2O-Study2-BIOPAC'
FILE_PATTERN = '*.mat'
SAMPLING_RATE = 2000  # 0.5ms per point = 2000 Hz

# Define window lengths to test (in seconds)
WINDOW_TESTS = [2, 4, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300] 

# --- IMPROVED LZC FUNCTIONS ---
def normalized_lzc(binary_signal):
    """
    Calculate normalized Lempel-Ziv complexity
    Normalizes by log2(n) to get values between 0 and 1
    """
    n = len(binary_signal)
    if n == 0:
        return 0
    
    raw_lzc = ant.lziv_complexity(binary_signal)
    normalized = raw_lzc / (n / np.log2(n))  # Normalization factor
    return normalized

def custom_lzc(binary_signal):
    """
    Custom LZC implementation for better control
    """
    # Convert binary signal to string for processing
    binary_str = ''.join(['1' if x else '0' for x in binary_signal])
    
    # Simple LZC implementation
    n = len(binary_str)
    if n == 0:
        return 0
    
    vocabulary = set()
    lzc = 0
    i = 0
    
    while i < n:
        # Find the longest prefix not in vocabulary
        longest = ""
        for j in range(i+1, n+1):
            substring = binary_str[i:j]
            if substring not in vocabulary:
                longest = substring
            else:
                break
        
        if longest:
            vocabulary.add(longest)
            lzc += 1
            i += len(longest)
        else:
            vocabulary.add(binary_str[i])
            lzc += 1
            i += 1
    
    # Normalize
    return lzc / (n / np.log2(n))

# --- IMPROVED SENSITIVITY ANALYSIS ---
def run_improved_sensitivity():
    all_files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    
    # Use more files for better statistics
    sample_files = all_files[:10]  # Increased from 5 to 10
    
    # Number of random windows per file per window size
    N_WINDOWS_PER_FILE = 5
    
    results = []

    print(f"Starting Improved Sensitivity Analysis on {len(sample_files)} files...")
    print(f"Testing {N_WINDOWS_PER_FILE} random windows per file per window size")

    for file_path in sample_files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")

        try:
            # Load MATLAB file
            mat_data = sio.loadmat(file_path)['data']
            resp_signal = mat_data[:, 1].astype(np.float32)
            
            # Use longer segment for more window options
            start_idx = 2 * 60 * SAMPLING_RATE 
            # Use 20 minutes instead of 15 for more window options
            test_segment = resp_signal[start_idx : start_idx + (20 * 60 * SAMPLING_RATE)]
            
            segment_length = len(test_segment)
            max_window_points = max(WINDOW_TESTS) * SAMPLING_RATE
            
            if segment_length < max_window_points:
                print(f"Warning: {file_name} segment too long for largest window")
                continue

            for window in tqdm(WINDOW_TESTS, desc=f"Testing windows for {file_name}", leave=False):
                n_points = int(window * SAMPLING_RATE)
                
                # Test multiple random windows per file
                window_results = []
                
                for w_idx in range(N_WINDOWS_PER_FILE):
                    # Random start position (ensure we don't go out of bounds)
                    max_start = segment_length - n_points
                    start_pos = random.randint(0, max_start)
                    
                    # Extract the window
                    slice_data = test_segment[start_pos : start_pos + n_points]
                    
                    # Multiple binarization methods
                    methods = {
                        'median': slice_data > np.median(slice_data),
                        'mean': slice_data > np.mean(slice_data),
                        'adaptive': slice_data > (np.mean(slice_data) + 0.1 * np.std(slice_data))
                    }
                    
                    for method_name, binary_sig in methods.items():
                        # Calculate both normalized and custom LZC
                        lzc_norm = normalized_lzc(binary_sig)
                        lzc_custom = custom_lzc(binary_sig)
                        
                        # Calculate signal quality metrics
                        signal_quality = {
                            'variance': np.var(slice_data),
                            'signal_to_noise': np.mean(slice_data) / np.std(slice_data) if np.std(slice_data) > 0 else 0,
                            'binary_balance': np.mean(binary_sig)  # Should be around 0.5 for good balance
                        }
                        
                        results.append({
                            'File': file_name,
                            'Window_Sec': window,
                            'Window_Index': w_idx,
                            'Binarization': method_name,
                            'LZC_Normalized': lzc_norm,
                            'LZC_Custom': lzc_custom,
                            'Variance': signal_quality['variance'],
                            'SNR': signal_quality['signal_to_noise'],
                            'Binary_Balance': signal_quality['binary_balance']
                        })
                
            # Free up memory
            del mat_data, resp_signal, test_segment
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('improved_respiratory_lzc_sensitivity.csv', index=False)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    return df

def print_summary_statistics(df):
    """Print summary statistics for the sensitivity analysis"""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    # Group by window size and calculate statistics
    summary = df.groupby(['Window_Sec', 'Binarization']).agg({
        'LZC_Normalized': ['mean', 'std', 'count'],
        'LZC_Custom': ['mean', 'std'],
        'Binary_Balance': ['mean', 'std']
    }).round(4)
    
    print("\nWindow Size Statistics:")
    print(summary)
    
    # Calculate coefficient of variation for stability
    cv_stats = df.groupby(['Window_Sec', 'Binarization']).apply(
        lambda x: x['LZC_Normalized'].std() / x['LZC_Normalized'].mean() if x['LZC_Normalized'].mean() > 0 else float('inf')
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
    for method in df['Binarization'].unique():
        method_data = df[df['Binarization'] == method]
        stats = method_data.groupby('Window_Sec')['LZC_Normalized'].agg(['mean', 'std'])
        ax1.errorbar(stats.index, stats['mean'], yerr=stats['std'], 
                    marker='o', capsize=5, label=f'{method} binarization')
    
    ax1.set_xlabel("Window Length (Seconds)")
    ax1.set_ylabel("Normalized LZC")
    ax1.set_title("LZC vs Window Size (with Error Bars)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of Variation
    ax2 = axes[0, 1]
    cv_data = df.groupby(['Window_Sec', 'Binarization']).apply(
        lambda x: x['LZC_Normalized'].std() / x['LZC_Normalized'].mean() if x['LZC_Normalized'].mean() > 0 else float('inf')
    ).unstack()
    
    for method in cv_data.columns:
        ax2.plot(cv_data.index, cv_data[method], marker='o', label=method)
    
    ax2.set_xlabel("Window Length (Seconds)")
    ax2.set_ylabel("Coefficient of Variation")
    ax2.set_title("Stability Analysis (Lower CV = More Stable)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Binary Balance Quality
    ax3 = axes[1, 0]
    for method in df['Binarization'].unique():
        method_data = df[df['Binarization'] == method]
        stats = method_data.groupby('Window_Sec')['Binary_Balance'].agg(['mean', 'std'])
        ax3.errorbar(stats.index, stats['mean'], yerr=stats['std'], 
                    marker='o', capsize=5, label=f'{method} binarization')
    
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

# Run the improved analysis
if __name__ == "__main__":
    df_results = run_improved_sensitivity()
    plot_improved_stability(df_results)
