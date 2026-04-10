import argparse
import glob
import os
import random

import numpy as np
from scipy.io import loadmat, savemat


DEFAULT_INPUT_DIR = "data_clean_segmented"
DEFAULT_PATTERN = "*-khodadad2018-25Hz.mat"
DEFAULT_OUTPUT_DIR = "data_epochized_120s"
DEFAULT_SAMPLE_SIZE = 147
DEFAULT_EPOCH_LENGTH_SEC = 120.0
DEFAULT_RANDOM_SEED = 123


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split cleaned respiratory segments into 120s epochs while keeping "
            "pre-session and during-session epochs separate."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing cleaned segment .mat files")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for input files inside the input directory")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of participant files to analyze")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample files instead of taking the first N sorted files")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed used when sampling files")
    parser.add_argument("--epoch-length-sec", type=float, default=DEFAULT_EPOCH_LENGTH_SEC, help="Epoch length in seconds")
    parser.add_argument("--fs", type=float, default=None, help="Optional sampling rate override (Hz)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save epochized .mat files")
    return parser.parse_args()


def discover_files(input_dir, pattern):
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def load_segment_matrix(mat, key_segments, key_lengths):
    segments = np.asarray(mat.get(key_segments, np.empty((0, 0))), dtype=float)
    lengths = np.asarray(mat.get(key_lengths, np.empty((0,))), dtype=int).reshape(-1)

    if segments.size == 0:
        return []

    if segments.ndim == 1:
        segments = segments.reshape(1, -1)

    if lengths.size == 0:
        lengths = np.asarray([np.sum(np.isfinite(row)) for row in segments], dtype=int)

    cleaned_segments = []
    n_rows = min(segments.shape[0], lengths.shape[0])
    for i in range(n_rows):
        seg_len = int(lengths[i])
        if seg_len <= 0:
            continue
        seg_len = min(seg_len, segments.shape[1])
        seg = np.asarray(segments[i, :seg_len], dtype=float)
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            cleaned_segments.append(seg)

    return cleaned_segments


def load_clean_segment_file(path, fs_override=None):
    mat = loadmat(path)

    fs_raw = np.asarray(mat.get("fs", np.array([[np.nan]])), dtype=float).reshape(-1)
    fs = float(fs_raw[0]) if fs_raw.size else np.nan
    if fs_override is not None:
        fs = float(fs_override)
    if not np.isfinite(fs) or fs <= 0:
        fs = 50.0

    pre_segments = load_segment_matrix(mat, "pre_segments", "pre_segment_lengths")
    n2o_segments = load_segment_matrix(mat, "n2o_segments", "n2o_segment_lengths")
    return {
        "fs": fs,
        "pre_segments": pre_segments,
        "n2o_segments": n2o_segments,
    }


def build_output_name(input_name, epoch_length_sec):
    base = os.path.splitext(input_name)[0]
    if base.endswith("-khodadad2018-50Hz"):
        base = base[: -len("-khodadad2018-50Hz")]
    return f"{base}-epoch{int(round(epoch_length_sec))}s.mat"


def split_into_epochs(segment, epoch_length_samples):
    if epoch_length_samples < 1:
        return np.empty((0, 0), dtype=float)

    n_epochs = len(segment) // epoch_length_samples
    if n_epochs == 0:
        return np.empty((0, epoch_length_samples), dtype=float)

    trimmed = np.asarray(segment[: n_epochs * epoch_length_samples], dtype=float)
    return trimmed.reshape(n_epochs, epoch_length_samples)


def epochize_segments(segments, epoch_length_samples):
    epochs = []
    for segment in segments:
        segment_epochs = split_into_epochs(segment, epoch_length_samples)
        for epoch in segment_epochs:
            epochs.append(epoch)

    if not epochs:
        return np.empty((0, epoch_length_samples), dtype=float)

    return np.stack(epochs, axis=0)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    files = discover_files(args.input_dir, args.pattern)
    if not files:
        raise FileNotFoundError(f"No files matched {os.path.join(args.input_dir, args.pattern)}")

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    if args.sample_size < len(files):
        if args.random_sample:
            files = rng.sample(files, args.sample_size)
            files = sorted(files)
        else:
            files = files[: args.sample_size]
    else:
        files = sorted(files)

    print(f"Found {len(files)} file(s) to epochize.")
    print("Selected files:")
    for path in files:
        print(f"  - {os.path.basename(path)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for path in files:
        file_name = os.path.basename(path)
        print(f"\nProcessing {file_name}...")

        file_data = load_clean_segment_file(path, fs_override=args.fs)
        fs = file_data["fs"]
        epoch_length_samples = int(round(args.epoch_length_sec * fs))

        if epoch_length_samples < 1:
            print("  Skipping: invalid epoch length.")
            continue

        pre_epochs = epochize_segments(file_data["pre_segments"], epoch_length_samples)
        n2o_epochs = epochize_segments(file_data["n2o_segments"], epoch_length_samples)

        if pre_epochs.shape[0] == 0 and n2o_epochs.shape[0] == 0:
            print("  Skipping: no complete 120s epochs found.")
            continue

        out_name = build_output_name(file_name, args.epoch_length_sec)
        out_path = os.path.join(args.output_dir, out_name)

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
