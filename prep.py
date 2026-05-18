import argparse
import os
import re
import neurokit2 as nk
import numpy as np
from scipy.io import loadmat, savemat

# ---------------------------
# Configuration
# ---------------------------

INPUT_DIR = "data_raw"                  # raw .mat files; line noise removed; 0.05-5 Hz bandpass
VAR_NAME = "data"                       # variable name inside each .mat

FS_ORIG = 2000                          # original sampling rate
FS_TARGET = 200                         # target sampling rate

METHOD = "khodadad2018"                 # "manual", "khodadad2018", or "none" for no cleaning

LOWCUT = 0.05                           # bandpass low cutoff (Hz)
HIGHCUT = 5.0                           # bandpass high cutoff (Hz)
FILTER_ORDER = 4
FILTER_METHOD = "butterworth"


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Filter, downsample, and segment pre-inhalation and during-inhalation respiratory data.")
    )

    parser.add_argument("--input-dir", default=INPUT_DIR, help="Directory containing raw .mat files")

    parser.add_argument("--method", default=METHOD, help="Pipeline method")
    parser.add_argument("--lowcut", type=float, default=LOWCUT, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--highcut", type=float, default=HIGHCUT, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--filter-order", type=float, default=FILTER_ORDER, help="Filter order")
    parser.add_argument("--filter-method", default=FILTER_METHOD, help="Filter method")
    parser.add_argument("--fs", type=float, default=FS_ORIG, help="Original sampling rate")
    parser.add_argument("--fs-target", type=float, default=FS_TARGET, help="Target sampling rate")
    parser.add_argument("--standardize-file", action="store_true", help="Z-score the signal after downsampling before segmentation")
    parser.add_argument("--standardize-segment", action="store_true", help="Z-score each segment after segmentation (overrides --standardize-file)")

    return parser.parse_args()

# ---------------------------
# Phases & Tools
# ---------------------------

def filter_rsp(raw, method, lowcut, highcut, filter_order, filter_method, fs_orig):
    print(f"  Filtering using '{method}' method")

    if method == "manual":
        raw_vector = nk.as_vector(raw)
        rsp_filtered = nk.signal_filter(raw_vector, 
                                       sampling_rate=fs_orig, 
                                       lowcut=lowcut, 
                                       highcut=highcut, 
                                       order=filter_order, 
                                       method=filter_method)
    elif method in ["khodadad2018", "charlton2021", "biosppy", "hampel"]:
        # khodadad2018 method - lowcut=0.05, highcut=3, order=2
        rsp_filtered = nk.rsp_clean(raw, sampling_rate=fs_orig, method=method)
    else:
        print(f"  No filtering applied")
        rsp_filtered = raw  # no cleaning

    return rsp_filtered


def standardize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    signal_mean = float(np.mean(signal))
    signal_std = float(np.std(signal))

    if signal_std == 0.0:
        return signal - signal_mean

    return (signal - signal_mean) / signal_std


def segment_via_timestamps(data, fs, txt_path):
    print(f"  Segmenting using timestamps from {txt_path}...")

    n_samples = len(data)
    if n_samples == 0:
        return [], []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    timestamps = []
    for line in lines[2:]:
        row = line.strip()
        if not row:
            continue
        parts = row.split("\t")
        if len(parts) < 3:
            continue

        time_str = parts[1].strip()
        event_type = parts[2].strip()

        if event_type not in ("Append", "User Type 9"):
            continue

        time_num_str = re.sub(r"[A-Za-z\s]+", "", time_str)
        if not time_num_str:
            continue
        t_sec = float(time_num_str) * 60
        if t_sec is None:
            continue

        idx = int(round(t_sec * fs))
        idx = max(0, min(n_samples, idx))
        timestamps.append((idx, event_type))


    if len(timestamps) == 0:
        pre_segments = [np.asarray(data, dtype=float)]
        dur_segments = []
        return pre_segments, dur_segments
    
    first_ut9 = None
    second_ut9 = None
    pre_segments = []
    dur_segments = []

    idx_prev = 0
    for idx, event_type in timestamps:
        if idx <= idx_prev:
            continue
        if event_type == "User Type 9":
            if first_ut9 is None:
                first_ut9 = idx
            elif second_ut9 is None:
                second_ut9 = idx
                break
        seg = np.asarray(data[idx_prev:idx])
        print(f"  Found segment from {idx_prev} to {idx}")
        if first_ut9 is None:
            pre_segments.append(seg)
        elif second_ut9 is None:
            dur_segments.append(seg)
        idx_prev = idx

    return pre_segments, dur_segments


def list_to_cell_array(segments):
    """Pack variable-length 1D segments into a MATLAB cell array-compatible object array."""
    cell_array = np.empty((len(segments), 1), dtype=object)
    for i, seg in enumerate(segments):
        cell_array[i, 0] = np.asarray(seg, dtype=float)
    return cell_array

# ---------------------------
# Main loop
# ---------------------------

def main():
    args = parse_args()

    print(f"Preprocessing .mat files in {args.input_dir}...")

    std_ext = "_std_file" if args.standardize_file else ("_std_seg" if args.standardize_segment else "")
    out_directory = f"data_{args.method}_{int(args.fs_target)}Hz{std_ext}"
    os.makedirs(out_directory, exist_ok=True)

    for fname in os.listdir(args.input_dir):

        if not fname.lower().endswith(".mat"):
            continue
        
        fpath = os.path.join(args.input_dir, fname)
        print(f"\nProcessing {fname}...")

        mat = loadmat(fpath)
        if VAR_NAME not in mat:
            print(f"Skipping {fname}: variable '{VAR_NAME}' not found.")
            continue

        mat_data = np.asarray(mat[VAR_NAME])
        if mat_data.ndim < 2:
            print(f"Skipping {fname}: variable '{VAR_NAME}' not the right shape.")
            continue

        resp_data = mat_data[:, 1].astype(float)  # 2nd column
        if len(resp_data) == 0:
            print(f"Skipping {fname}: no data in respiratory channel.")
            continue

        # 1) filter + resample
        filtered_rsp = filter_rsp(resp_data, args.method, args.lowcut, args.highcut, args.filter_order, args.filter_method, args.fs)
        print(f"  Downsampling to '{args.fs_target}' Hz")
        resampled_rsp = nk.signal_resample(filtered_rsp, sampling_rate=FS_ORIG, desired_sampling_rate=args.fs_target)
        prep_rsp = np.asarray(resampled_rsp, dtype=float)

        # optional file-level standardization
        if args.standardize_file and not args.standardize_segment:
            prep_rsp = standardize_signal(prep_rsp)
            print("  Standardized the downsampled signal")
        
        # 2) splice into continuous segments using TXT timestamps
        txt_path = os.path.join(args.input_dir, os.path.splitext(fname)[0] + "-evt.txt")
        if os.path.exists(txt_path):
            pre_segments, dur_segments = segment_via_timestamps(prep_rsp, args.fs_target, txt_path)
        else:
            print(f"Skipping {fname}: TXT file not found.")
            continue

        # optional segment-level standardization
        if args.standardize_segment:
            for i, seg in enumerate(pre_segments):
                pre_segments[i] = standardize_signal(seg)
            for i, seg in enumerate(dur_segments):
                dur_segments[i] = standardize_signal(seg)

        # if pre_segments.shape[0] + dur_segments.shape[0] == 0:
        if len(pre_segments) + len(dur_segments) == 0:
            print(f"Skipping {fname}: no segments found after splicing.")
            continue

        # 3) store files
        out_name = f"{os.path.splitext(fname)[0][:6]}-{args.method}-{int(args.fs_target)}Hz.mat"
        out_path = os.path.join(out_directory, out_name)

        savemat(
            out_path,
            {
                "pre_segments": list_to_cell_array(pre_segments),
                "dur_segments": list_to_cell_array(dur_segments),
                "fs": args.fs_target,
                "std": np.std(prep_rsp),
            },
        )
        print(f"  Saved {fname} segments to {out_path} ({len(pre_segments)} pre & {len(dur_segments)} during)")

    print()


if __name__ == "__main__":
    main()
