import os
import re
import neurokit2 as nk
import numpy as np
import warnings
from scipy.io import loadmat, savemat

# ---------------------------
# Configuration
# ---------------------------

INPUT_DIR = "data_raw"                  # raw .mat files; line noise removed; 0.05-5 Hz bandpass
OUTPUT_DIR = "data_clean_segmented"        # preprocessed epoch files
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAR_NAME = "data"                       # variable name inside each .mat
FS_ORIG = 2000.0                        # original sampling rate
FS_TARGET = 200.0                        # target sampling rate

LOWCUT = 0.05                           # bandpass low cutoff (Hz)
HIGHCUT = 5.0                           # bandpass high cutoff (Hz)
FILTER_ORDER = 4

# NeuroKit2 cleaning parameters
RSP_CLEAN_METHOD = "khodadad2018"  # "manual_nk2", "khodadad2018", or "downsample" for no cleaning

# ---------------------------
# Helpers
# ---------------------------

def preprocess_rsp(raw, fs_orig, fs_target, method=RSP_CLEAN_METHOD):
    print(f"  Preprocessing: cleaning with method '{method}' and resampling from {fs_orig} Hz to {fs_target} Hz...")

    if method == "manual_nk2":
        raw_vector = nk.as_vector(raw)
        rsp_cleaned = nk.signal_filter(raw_vector, 
                                       sampling_rate=fs_orig, 
                                       lowcut=LOWCUT, 
                                       highcut=HIGHCUT, 
                                       order=FILTER_ORDER, 
                                       method="butterworth")
    elif method in ["khodadad2018", "charlton2021", "biosppy", "hampel"]:
        # khodadad2018 method - lowcut=0.05, highcut=3, order=2
        rsp_cleaned = nk.rsp_clean(raw, sampling_rate=fs_orig, method=method)
    else:
        rsp_cleaned = raw  # no cleaning

    # Resample to FS_TARGET (includes anti-aliasing)
    rsp_resampled = nk.signal_resample(rsp_cleaned,
                                       sampling_rate=fs_orig,
                                       desired_sampling_rate=fs_target)
    
    return np.asarray(rsp_resampled, dtype=float)


def segment_from_txt_timestamps(data, fs, txt_path):

    print(f"  Segmenting using TXT timestamps from {txt_path}...")

    n_samples = len(data)
    if n_samples == 0:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty, np.empty((0,), dtype=int), np.empty((0,), dtype=int), None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    markers = []
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
        markers.append((idx, event_type))

    if len(markers) == 0:
        pre_segments = [np.asarray(data, dtype=float)]
        n2o_segments = []
        n2o_start = None
        pre_matrix, pre_lengths = pack_segments_to_2d(pre_segments)
        n2o_matrix, n2o_lengths = pack_segments_to_2d(n2o_segments)
        return pre_matrix, n2o_matrix, pre_lengths, n2o_lengths, n2o_start
    
    first_ut9 = None
    second_ut9 = None
    pre_segments = []
    n2o_segments = []

    idx_prev = 0
    for idx, event_type in markers:
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
            n2o_segments.append(seg)
        idx_prev = idx

    n2o_start = first_ut9 / fs if first_ut9 is not None else None

    print(f"  Total segments found: {len(pre_segments)} pre-session, {len(n2o_segments)} during-session ({n2o_start}s).")

    pre_matrix, pre_lengths = pack_segments_to_2d(pre_segments)
    n2o_matrix, n2o_lengths = pack_segments_to_2d(n2o_segments)

    return pre_matrix, n2o_matrix, pre_lengths, n2o_lengths, n2o_start


def pack_segments_to_2d(segments):
    """Pack variable-length 1D segments into a NaN-padded 2D matrix (n_segments x max_len)."""
    if len(segments) == 0:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int)

    lengths = np.asarray([len(seg) for seg in segments], dtype=int)
    max_len = int(np.max(lengths))
    packed = np.full((len(segments), max_len), np.nan, dtype=float)

    for i, seg in enumerate(segments):
        packed[i, :lengths[i]] = np.asarray(seg, dtype=float)

    return packed, lengths


# ---------------------------
# Main loop
# ---------------------------

print(f"Starting preprocessing of .mat files in {INPUT_DIR}...")

for fname in os.listdir(INPUT_DIR):

    # TEST SAMPLE
    # if fname not in ['P001-1-trimmed.mat', 'P003-1-trimmed.mat']:
    #     continue

    if not fname.lower().endswith(".mat"):
        continue
    
    fpath = os.path.join(INPUT_DIR, fname)
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

    # 1) cleaning + bandpass + resampling to 25 Hz
    data_ds = preprocess_rsp(resp_data, FS_ORIG, FS_TARGET)
    
    # 2) splice into continuous segments using TXT timestamps
    txt_path = os.path.join(INPUT_DIR, os.path.splitext(fname)[0] + "-evt.txt")
    if os.path.exists(txt_path):
        pre_segments_2d, n2o_segments_2d, pre_lengths, n2o_lengths, n2o_start = segment_from_txt_timestamps(data_ds, FS_TARGET, txt_path)
    else:
        print(f"Skipping {fname}: TXT file not found.")
        continue
    
    if pre_segments_2d.shape[0] + n2o_segments_2d.shape[0] == 0:
        print(f"Skipping {fname}: no segments found after splicing.")
        continue
    
    # 4) store preprocessed segments into .mat file
    out_fname = os.path.splitext(fname)[0][:6] + f"{'-' if RSP_CLEAN_METHOD else ''}{RSP_CLEAN_METHOD}-{int(FS_TARGET)}Hz.mat"
    out_path = os.path.join(OUTPUT_DIR, out_fname)
    savemat(
        out_path,
        {
            "pre_segments": pre_segments_2d,
            "n2o_segments": n2o_segments_2d,
            "pre_segment_lengths": pre_lengths,
            "n2o_segment_lengths": n2o_lengths,
            "n2o_start_sec": n2o_start if n2o_start is not None else np.nan,
            "fs": FS_TARGET,
        },
    )
    
    print(f"\nSaved segments for {fname} to {out_path}\n")
