import argparse
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
VAR_NAME = "data"                       # variable name inside each .mat

FS_ORIG = 2000.0                        # original sampling rate
FS_TARGET = 200.0                       # target sampling rate

PIPELINE_METHOD = "khodadad2018"        # "manual", "khodadad2018", or "none" for no cleaning

LOWCUT = 0.05                           # bandpass low cutoff (Hz)
HIGHCUT = 5.0                           # bandpass high cutoff (Hz)
FILTER_ORDER = 4
FILTER_METHOD = "butterworth"

EPOCH_LENGTH_SEC = 120.0

# TO_FILTER = True
# TO_DOWNSAMPLE = True
# TO_SEGMENT = True
# TO_EPOCHIZE = True

EXPORT_SEGMENTS = True
EXPORT_EPOCHS = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Filter, downsample, segment, and epochize pre-inhalation and during-inhalation respiratory data.")
    )

    parser.add_argument("--pipeline-method", default=PIPELINE_METHOD, help="Pipeline method")

    parser.add_argument("--lowcut", type=float, default=LOWCUT, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--highcut", type=float, default=HIGHCUT, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--filter-order", type=float, default=FILTER_ORDER, help="Filter order")
    parser.add_argument("--filter-method", default=FILTER_METHOD, help="Filter method")

    parser.add_argument("--fs-target", type=float, default=FS_TARGET, help="Target sampling rate")
    parser.add_argument("--epoch-length-sec", type=float, default=EPOCH_LENGTH_SEC, help="Epoch length in seconds")

    # parser.add_argument("--to-filter", type=bool, default=TO_FILTER, help="Whether to filter data")
    # parser.add_argument("--to-downsample", type=bool, default=TO_DOWNSAMPLE, help="Whether to downsample data")
    # parser.add_argument("--to-segment", type=bool, default=TO_SEGMENT, help="Whether to segment data")
    # parser.add_argument("--to-epochize", type=bool, default=TO_EPOCHIZE, help="Whether to epochize data")

    parser.add_argument("--export-segments", type=bool, default=EXPORT_SEGMENTS, help="Whether to store intermediary segmented MATLAB files")
    parser.add_argument("--export-epochs", type=bool, default=EXPORT_EPOCHS, help="Whether to store epochized MATLAB files")
    
    return parser.parse_args()

# ---------------------------
# Phases & Tools
# ---------------------------

def filter_rsp(raw, pipeline_method, lowcut, highcut, order, method):
    print(f"  Filtering using '{pipeline_method}' method")

    if pipeline_method == "manual":
        raw_vector = nk.as_vector(raw)
        rsp_filtered = nk.signal_filter(raw_vector, 
                                       sampling_rate=FS_ORIG, 
                                       lowcut=lowcut, 
                                       highcut=highcut, 
                                       order=order, 
                                       method=method)
    elif pipeline_method in ["khodadad2018", "charlton2021", "biosppy", "hampel"]:
        # khodadad2018 method - lowcut=0.05, highcut=3, order=2
        rsp_filtered = nk.rsp_clean(raw, sampling_rate=FS_ORIG, method=pipeline_method)
    else:
        rsp_filtered = raw  # no cleaning

    return rsp_filtered


def segment_via_timestamps(data, fs, txt_path):
    print(f"  Segmenting using TXT timestamps from {txt_path}...")

    n_samples = len(data)
    if n_samples == 0:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty, np.empty((0,), dtype=int), np.empty((0,), dtype=int), None

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
        n2o_segments = []
        n2o_start = None
        pre_matrix, pre_lengths = list_to_matrix(pre_segments)
        n2o_matrix, n2o_lengths = list_to_matrix(n2o_segments)
        return pre_matrix, n2o_matrix, pre_lengths, n2o_lengths, n2o_start
    
    first_ut9 = None
    second_ut9 = None
    pre_segments = []
    n2o_segments = []

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
            n2o_segments.append(seg)
        idx_prev = idx

    n2o_start = first_ut9 / fs if first_ut9 is not None else None

    print(f"  Total segments found: {len(pre_segments)} pre-session, {len(n2o_segments)} during-session ({n2o_start}s).")

    pre_matrix, pre_lengths = list_to_matrix(pre_segments)
    n2o_matrix, n2o_lengths = list_to_matrix(n2o_segments)

    return pre_matrix, n2o_matrix, pre_lengths, n2o_lengths, n2o_start


def list_to_matrix(segments):
    """Pack variable-length 1D segments into a NaN-padded 2D matrix (n_segments x max_len)."""
    if len(segments) == 0:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int)

    lengths = np.asarray([len(seg) for seg in segments], dtype=int)
    max_len = int(np.max(lengths))
    matrix = np.full((len(segments), max_len), np.nan, dtype=float)

    for i, seg in enumerate(segments):
        matrix[i, :lengths[i]] = np.asarray(seg, dtype=float)

    return matrix, lengths


def epochize_segments(segments, epoch_length_sec):
    epoch_length_samples = int(round(epoch_length_sec * FS_TARGET))

    if epoch_length_samples < 1:
        return np.empty((0, epoch_length_samples), dtype=float)

    epochs = []
    for segment in segments:
        n_epochs = len(segment) // epoch_length_samples
        if n_epochs == 0:
            return np.empty((0, epoch_length_samples), dtype=float)

        trimmed = np.asarray(segment[: n_epochs * epoch_length_samples], dtype=float)
        segment_epochs = trimmed.reshape(n_epochs, epoch_length_samples)

        for epoch in segment_epochs:
            epochs.append(epoch)

    if not epochs:
        return np.empty((0, epoch_length_samples), dtype=float)

    return np.stack(epochs, axis=0)


# ---------------------------
# Main loop
# ---------------------------

def main():
    print(f"Starting preprocessing of .mat files in {INPUT_DIR}...")

    args = parse_args()

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
        filtered_rsp = filter_rsp(resp_data, args.pipeline_method, args.lowcut, args.highcut, args.order, args.filter_method)
        resampled_rsp = nk.signal_resample(filtered_rsp, sampling_rate=FS_ORIG, desired_sampling_rate=args.fs_target)
        prep_rsp = np.asarray(resampled_rsp, dtype=float)
        
        # 2) splice into continuous segments using TXT timestamps
        txt_path = os.path.join(INPUT_DIR, os.path.splitext(fname)[0] + "-evt.txt")
        if os.path.exists(txt_path):
            pre_segments, n2o_segments, pre_lengths, n2o_lengths, n2o_start = segment_via_timestamps(prep_rsp, args.fs_target, txt_path)
        else:
            print(f"Skipping {fname}: TXT file not found.")
            continue
        
        if pre_segments.shape[0] + n2o_segments.shape[0] == 0:
            print(f"Skipping {fname}: no segments found after splicing.")
            continue

        # 3) epochize
        pre_epochs = epochize_segments(pre_segments, args.epoch_length_sec)
        n2o_epochs = epochize_segments(n2o_segments, args.epoch_length_sec)

        if pre_epochs.shape[0] == 0 and n2o_epochs.shape[0] == 0:
            print("  Skipping: no complete 120s epochs found.")
            continue

        # 4) store files
        base = os.path.splitext(fname)[0][:6] + f"-{args.pipeline_method}-{int(args.fs_target)}Hz"

        if args.export_segments:
            out_directory = "fse_data_clean_segmented"
            os.makedirs(out_directory, exist_ok=True)
            out_name = f"{base}.mat"
            out_path = os.path.join(out_directory, out_name)

            savemat(
                out_path,
                {
                    "pre_segments": pre_segments,
                    "n2o_segments": n2o_segments,
                    "pre_segment_lengths": pre_lengths,
                    "n2o_segment_lengths": n2o_lengths,
                    "n2o_start_sec": n2o_start if n2o_start is not None else np.nan,
                    "fs": args.fs_target,
                },
            )
            print(f"\n  Saved {fname} segments to {out_path}: {pre_segments.shape[0]} pre & {n2o_segments.shape[0]} inhalation")

        if args.export_epochs:
            out_directory = f"fse_data_epochized_{int(round(args.epoch_length_sec))}s"
            os.makedirs(out_directory, exist_ok=True)
            out_name = f"{base}-epoch{int(round(args.epoch_length_sec))}s.mat"
            out_path = os.path.join(out_directory, out_name)

            savemat(
                out_path,
                {
                    "pre_epochs": pre_epochs,
                    "n2o_epochs": n2o_epochs,
                    "fs": args.fs_target,
                    "epoch_length_sec": float(args.epoch_length_sec),
                },
            )
            print(f"\n  Saved {fname} epochs to {out_path}: {pre_epochs.shape[0]} pre & {n2o_epochs.shape[0]} inhalation")

        print("\n")


if __name__ == "__main__":
    main()
