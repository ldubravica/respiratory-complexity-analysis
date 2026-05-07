import argparse
import os

import numpy as np
from scipy.io import loadmat, savemat


# ---------------------------
# Configuration
# ---------------------------

DEFAULT_INPUT_DIR = "data_khodadad2018_200Hz"
DEFAULT_EPOCH_LENGTH_SEC = 120.0
DEFAULT_INPUT_PATTERN = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Split filtered respiratory pre/during segments into epochs.")
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing preprocessed .mat files")
    parser.add_argument("--epoch-length-sec", type=float, default=DEFAULT_EPOCH_LENGTH_SEC, help="Epoch length in seconds")
    parser.add_argument("--input-pattern", default=DEFAULT_INPUT_PATTERN, help="Input file names' pattern to match (e.g., 'khodadad2018-200Hz')")
    return parser.parse_args()

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


def epochize_segments(segments, epoch_length_samples):
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
    args = parse_args()

    preprocessing_info = args.input_pattern.replace('-', '_')
    epochizing_info = f"epoch{int(round(args.epoch_length_sec))}s"
    out_directory = f"data_{preprocessing_info}_{epochizing_info}"
    os.makedirs(out_directory, exist_ok=True)

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(".mat"):
            continue

        if args.input_pattern not in fname:
            continue

        path = os.path.join(args.input_dir, fname)
        print(f"\nProcessing {fname}...")

        # 1) load filtered, downsampled, and segmented data
        mat_data = loadmat(path)
        fs = float(mat_data["fs"])
        pre_segments = cell_array_to_list(mat_data["pre_segments"])
        n2o_segments = cell_array_to_list(mat_data["n2o_segments"])

        # 2) determine epoch length in samples
        epoch_length_samples = int(round(args.epoch_length_sec * fs))
        if epoch_length_samples < 1:
            print("  Skipping: invalid epoch length.")
            continue

        # 3) epochize segments
        pre_epochs = epochize_segments(pre_segments, epoch_length_samples)
        n2o_epochs = epochize_segments(n2o_segments, epoch_length_samples)

        if pre_epochs.shape[0] == 0 and n2o_epochs.shape[0] == 0:
            print(f"  Skipping: no complete {args.epoch_length_sec}s epochs found.")
            continue

        # 4) store epochized data
        out_name = f"{os.path.splitext(fname)[0]}-{epochizing_info}.mat"
        out_path = os.path.join(out_directory, out_name)

        savemat(
            out_path,
            {
                "pre_epochs": pre_epochs,
                "n2o_epochs": n2o_epochs,
                "fs": fs,
                "epoch_length_sec": float(args.epoch_length_sec),
            },
        )

        print(f"  Saved {pre_epochs.shape[0]} pre epochs and {n2o_epochs.shape[0]} n2o epochs to {out_path}")


if __name__ == "__main__":
    main()
    print("\n")
