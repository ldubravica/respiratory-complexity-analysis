import os
import json
import neurokit2 as nk
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, decimate

# ---------------------------
# Configuration
# ---------------------------

INPUT_DIR = "x"                  # folder with .mat files
OUTPUT_DIR = "x_preprocessed"    # folder for output .mat files
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAR_NAME = "data"                # variable name inside each .mat
FS_ORIG = 2000.0                 # original sampling rate
FS_TARGET = 25.0                 # target sampling rate
EPOCH_LENGTH_SEC = 30.0          # epoch size

LOWCUT = 0.05                    # bandpass low cutoff (Hz)
HIGHCUT = 5.0                    # bandpass high cutoff (Hz)
FILTER_ORDER = 4

# NeuroKit2 cleaning parameters
RSP_CLEAN_METHOD = "khodadad2018" # default RSP method in NeuroKit2

# ---------------------------
# Helpers
# ---------------------------

def preprocess_rsp_neurokit(raw, fs_orig, fs_target):
    """
    1) Clean respiratory signal using NeuroKit2's rsp_clean (includes bandpass).
    2) Resample to target sampling rate.
    """
    # khodadad2018 method - lowcut=0.05, highcut=3, order=2
    rsp_cleaned = nk.rsp_clean(raw, sampling_rate=fs_orig, method=RSP_CLEAN_METHOD)

    # Manual NeuroKit2 alternative
    # raw_vector = as_vector(raw)
    # rsp_cleaned = nk.signal_filter(raw_vector, 
    #                                sampling_rate=fs_orig, 
    #                                lowcut=LOWCUT, 
    #                                highcut=HIGHCUT, 
    #                                order=FILTER_ORDER, 
    #                                method="butterworth")

    # Resample to FS_TARGET (includes anti-aliasing)
    rsp_resampled = nk.signal_resample(rsp_cleaned,
                                       sampling_rate=fs_orig,
                                       desired_sampling_rate=fs_target)
    return np.asarray(rsp_resampled, dtype=float)

def segment_from_json_timestamps(data, fs, json_info): # TODO - major changes
    """
    Placeholder: splice continuous segments based on external JSON timestamps.
    json_info: assumes:

      {
        "segments": [
          {"start": t_start_sec, "end": t_end_sec},
          ...
        ]
      }
    
    Expected behavior:
    - Convert each [start_time, end_time] (in seconds) into [start_idx, end_idx].
    - Return a list of 1D NumPy arrays (segments).
    """

    segments = []
    if "segments" not in json_info:
        return [data]

    for seg in json_info["segments"]:
        start_sec = seg["start"]
        end_sec = seg["end"]
        start_idx = int(round(start_sec * fs))
        end_idx = int(round(end_sec * fs))
        # clip to array bounds
        start_idx = max(0, min(start_idx, len(data)))
        end_idx = max(0, min(end_idx, len(data)))
        if end_idx > start_idx:
            segments.append(data[start_idx:end_idx])
    return segments

def is_bad_segment(segment, fs):
    """
    Basic sanity check for a continuous respiratory segment.
    You SHOULD refine based on your data.
    """
    if len(segment) < fs * 10:  # shorter than 10 s → too short to be useful
        return True
    
    # Flatline detection
    if np.std(segment) < 1e-6:
        return True
    
    # Extreme amplitude: adjust thresholds as needed
    if np.max(np.abs(segment)) > 10 * np.std(segment):
        # strong spike / artifact
        pass  # you may or may not treat this as bad

    # NeuroKit2 Check

    try:
        rsp_peaks, info = nk.rsp_peaks(segment, sampling_rate=fs)  # TODO - what to do with this?
        rsp_rate = nk.rsp_rate(segment, sampling_rate=fs, method="trough")
    except Exception:
        # if NeuroKit2 fails, treat as bad
        return True

    # respiratory rate sanity: remove impossible average RR
    # e.g., <3 bpm or >80 bpm for adults under your paradigm
    if len(rsp_rate) == 0:
        return True
    mean_rr = np.nanmean(rsp_rate)
    if np.isnan(mean_rr) or mean_rr < 3 or mean_rr > 80:
        return True
    
    return False

def split_into_epochs(segment, fs, epoch_length_sec):
    epoch_samples = int(round(epoch_length_sec * fs))
    n_epochs = len(segment) // epoch_samples
    if n_epochs == 0:
        return np.empty((0, epoch_samples))
    trimmed = segment[: n_epochs * epoch_samples]
    epochs = trimmed.reshape(n_epochs, epoch_samples)
    return epochs

def is_bad_epoch(epoch, fs):
    """
    Basic bad epoch detection. Again, refine for your data:
    - extremely low variance (flat)
    - huge spikes
    """
    if len(epoch) < fs * 5:  # <5s is too short for 30s epochs, but keep just in case
        return True

    if np.std(epoch) < 1e-6:
        return True
    if np.max(np.abs(epoch)) > 10 * np.std(epoch):
        # suspicious large spike
        return True
    
    try:
        rsp_rate = nk.rsp_rate(epoch, sampling_rate=fs, method="trough")
    except Exception:
        return True
    
    if len(rsp_rate) == 0:
        return True

    mean_rr = np.nanmean(rsp_rate)
    # Slightly stricter thresholds at epoch level
    if np.isnan(mean_rr) or mean_rr < 4 or mean_rr > 70:
        return True

    # Optional: reject epochs with extremely irregular RR
    if np.nanstd(rsp_rate) > 0.5 * mean_rr:  # very high variability
        # you might or might not want this; adjust to your paradigm
        pass
    
    return False

def find_minimal_common_length(segments):
    """
    Minimal common denominator across segments (here: minimal length).
    You can instead implement something more elaborate if desired.
    """
    if not segments:
        return [], None

    lengths = np.array([len(s) for s in segments])
    # Identify outlier segments (e.g., too short) via simple rule:
    median_len = np.median(lengths)
    mad = np.median(np.abs(lengths - median_len)) + 1e-9
    # Example: outliers shorter than median - 3*MAD
    min_acceptable_len = median_len - 3 * mad

    good_segments = [s for s in segments if len(s) >= min_acceptable_len]
    if len(good_segments) == 0:
        return [], None

    return good_segments, int(min_acceptable_len)

# ---------------------------
# Main loop
# ---------------------------

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".mat"):
        continue
    
    fpath = os.path.join(INPUT_DIR, fname)
    mat = loadmat(fpath)
    if VAR_NAME not in mat:
        print(f"Skipping {fname}: variable '{VAR_NAME}' not found.")
        continue
    
    raw = np.squeeze(mat[VAR_NAME]).astype(float)
    
    # 1) bandpass filtering at 2000 Hz
    filtered = bandpass_filter(raw, LOWCUT, HIGHCUT, FS_ORIG, order=FILTER_ORDER)
    
    # 2) downsample from 2000 Hz to 25 Hz
    data_ds = downsample(filtered, FS_ORIG, FS_TARGET)

    # 1–2) NeuroKit2: cleaning + bandpass + resampling to 25 Hz
    data_ds = preprocess_rsp_neurokit(raw, FS_ORIG, FS_TARGET)
    
    # 3) splice into continuous segments via JSON (placeholder)
    #    Here we assume a JSON with same basename as .mat, e.g., "file.json"
    json_path = os.path.join(INPUT_DIR, os.path.splitext(fname)[0] + ".json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f_json:
            json_info = json.load(f_json)
        segments = segment_from_json_timestamps(data_ds, FS_TARGET, json_info)
    else:
        # as fallback, use the whole signal as a single segment
        segments = [data_ds]
    
    if len(segments) == 0:
        print(f"No segments for {fname}, skipping.")
        continue
    
    # 4) find minimal common denominator across segments & drop outliers
    segments_good, min_len = find_minimal_common_length(segments)
    if len(segments_good) == 0:
        print(f"All segments dropped for {fname} after length filter.")
        continue
    
    # Optionally trim all good segments to same length (min_len)
    segments_trimmed = [s[:min_len] for s in segments_good]
    
    # 5) for each segment: reject if globally bad, then split into fixed 30 s epochs
    all_epochs = []
    epoch_samples = int(round(EPOCH_LENGTH_SEC * FS_TARGET))
    
    for seg in segments_trimmed:
        if is_bad_segment(seg, FS_TARGET):
            continue
        
        epochs = split_into_epochs(seg, FS_TARGET, EPOCH_LENGTH_SEC)
        for ep in epochs:
            if is_bad_epoch(ep, FS_TARGET):
                continue
            all_epochs.append(ep)
    
    if len(all_epochs) == 0:
        print(f"No valid epochs for {fname}.")
        continue
    
    epochs_array = np.stack(all_epochs, axis=0)  # shape: (n_epochs, epoch_samples)
    
    # 6) store preprocessed epochs into .mat file
    out_fname = os.path.splitext(fname)[0] + "_preprocessed.mat"
    out_path = os.path.join(OUTPUT_DIR, out_fname)
    savemat(out_path, {"epochs": epochs_array, "fs": FS_TARGET})
    
    print(f"Saved {epochs_array.shape[0]} epochs for {fname} to {out_path}")
