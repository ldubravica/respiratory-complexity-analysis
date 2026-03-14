import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import antropy as ant
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

# --- CONFIGURATION ---
DATA_DIR = 'N2O-Study2-BIOPAC'
FILE_PATTERN = '*.mat'
SAMPLING_RATE = 2000  # 0.5ms per point = 2000 Hz

# Define window lengths to test (in seconds)
WINDOW_TESTS = [2, 4, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300] 

# --- SENSITIVITY ANALYSIS ---
def run_sensitivity():
    all_files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    
    # OPTION A: Automatic Sample (First 5 files)
    sample_files = all_files[:5] 
    
    # OPTION B: Custom Sample (Uncomment and edit to use specific files)
    # sample_files = [os.path.join(DATA_DIR, f) for f in ['session1.mat', 'session5.mat']]
    
    results = []

    print(f"Starting Sensitivity Analysis on {len(sample_files)} files...")

    for file_path in sample_files:
        file_name = os.path.basename(file_path)

        print(f"Processing {file_name}...")

        try:
            # Load MATLAB file & extract 'data' field (data points x 2 columns)
            mat_data = sio.loadmat(file_path)['data']
            
            # Take all rows (:), second column (1)
            resp_signal = mat_data[:, 1].astype(np.float32)
            
            # Obtaining 15 minute signal sample from 2 minutes onwards (skip noise/setup)
            start_idx = 2 * 60 * SAMPLING_RATE 
            test_segment = resp_signal[start_idx : start_idx + (15 * 60 * SAMPLING_RATE)]

            for window in tqdm(WINDOW_TESTS, desc=f"Testing windows for {file_name}", leave=False):
                n_points = int(window * SAMPLING_RATE)
                
                # Extract the window
                slice_data = test_segment[:n_points]  # ??????
                
                # Binarize using Median (Critical for respiration)
                binary_sig = slice_data > np.median(slice_data)
                
                # Calculate Normalized LZC
                # Antropy is incredibly fast here due to Numba JIT
                lzc_val = ant.lziv_complexity(binary_sig, normalize=True)
                
                results.append({
                    'File': file_name,
                    'Window_Sec': window,
                    'LZC': lzc_val
                })
                
            # Free up memory before next file
            del mat_data, resp_signal, test_segment
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('respiratory_lzc_sensitivity.csv', index=False)
    return df

# --- PLOTTING ---
def plot_stability(df):
    plt.figure(figsize=(12, 6))
    for label, grp in df.groupby('File'):
        plt.plot(grp['Window_Sec'], grp['LZC'], marker='o', alpha=0.7, label=label)
    
    plt.axvline(x=60, color='red', linestyle='--', label='Suggested Start (60s)')
    plt.title("LZC Stability Across Window Sizes (Respiratory Data)")
    plt.xlabel("Window Length (Seconds)")
    plt.ylabel("Normalized LZC")
    # plt.xscale('log') # Log scale helps see 1s vs 300s better # MAYBE REMOVE
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Run the process
df_results = run_sensitivity()
plot_stability(df_results)