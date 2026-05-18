import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = "data_raw"
DEFAULT_METHOD = "khodadad2018"
DEFAULT_LOWCUT = 0.05
DEFAULT_HIGHCUT = 5.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_FILTER_METHOD = "butterworth"
DEFAULT_FS = 2000.0
DEFAULT_FS_TARGET = 200.0
DEFAULT_EPOCH_LENGTH_SEC = 120.0
DEFAULT_DROP_BAD_EPOCHS = False
DEFAULT_SAVE_PLOTS = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run prep.py, epochize.py, lzc_calculation.py, and lzc_analysis.py in sequence."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing raw .mat files")
    parser.add_argument("--method", default=DEFAULT_METHOD, help="Preprocessing method")
    parser.add_argument("--lowcut", type=float, default=DEFAULT_LOWCUT, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--highcut", type=float, default=DEFAULT_HIGHCUT, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--filter-order", type=float, default=DEFAULT_FILTER_ORDER, help="Filter order")
    parser.add_argument("--filter-method", default=DEFAULT_FILTER_METHOD, help="Filter method")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="Original sampling rate")
    parser.add_argument("--fs-target", type=float, default=DEFAULT_FS_TARGET, help="Target sampling rate")
    parser.add_argument("--epoch-length-sec", type=float, default=DEFAULT_EPOCH_LENGTH_SEC, help="Epoch length in seconds")
    parser.add_argument("--drop-bad-epochs", action="store_true", default=DEFAULT_DROP_BAD_EPOCHS, help="Drop bad epochs during epochization")
    parser.add_argument("--save-plots", action="store_true", default=DEFAULT_SAVE_PLOTS, help="Save rejected-epoch plots during epochization")
    parser.add_argument("--standardize-file", action="store_true", help="Standardize the downsampled signal before segmentation in prep.py")
    parser.add_argument("--standardize-segment", action="store_true", help="Standardize each segment after segmentation in prep.py")
    parser.add_argument("--standardize-epoch", action="store_true", help="Standardize each epoch before saving in epochize.py")
    return parser.parse_args()


def run_step(script_name, args):
    script_path = ROOT / script_name
    command = [sys.executable, str(script_path), *args]
    print("\n" + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def int_label(value):
    return str(int(round(float(value))))


def prep_std_suffix(args):
    if args.standardize_segment:
        return "_std_seg"
    if args.standardize_file:
        return "_std_file"
    return ""


def main():
    args = parse_args()

    fs_target_label = int_label(args.fs_target)
    epoch_length_label = int_label(args.epoch_length_sec)
    epoch_suffix = "filtered" if args.drop_bad_epochs else "all"

    prep_args = [
        "--input-dir",
        args.input_dir,
        "--method",
        args.method,
        "--lowcut",
        str(args.lowcut),
        "--highcut",
        str(args.highcut),
        "--filter-order",
        str(args.filter_order),
        "--filter-method",
        args.filter_method,
        "--fs",
        str(args.fs),
        "--fs-target",
        str(args.fs_target),
    ]
    if args.standardize_file:
        prep_args.append("--standardize-file")
    if args.standardize_segment:
        prep_args.append("--standardize-segment")
    run_step("prep.py", prep_args)

    prep_output_dir = f"data_{args.method}_{fs_target_label}Hz{prep_std_suffix(args)}"
    epochize_args = [
        "--input-dir",
        prep_output_dir,
        "--epoch-length-sec",
        str(args.epoch_length_sec),
    ]
    if args.drop_bad_epochs:
        epochize_args.append("--drop-bad-epochs")
    if args.save_plots:
        epochize_args.append("--save-plots")
    if args.standardize_epoch:
        epochize_args.append("--standardize")
    run_step("epochize.py", epochize_args)

    epochized_method = f"{prep_output_dir[5:]}_{epoch_length_label}s_{epoch_suffix}"
    if args.standardize_epoch:
        epochized_method = f"{epochized_method}_std"
    epochized_dir = f"data_{epochized_method}"

    lzc_calculation_args = [
        "--input-dir",
        epochized_dir,
    ]
    run_step("lzc_calculation.py", lzc_calculation_args)

    lzc_analysis_args = [
        "--method",
        epochized_method,
    ]
    run_step("lzc_analysis.py", lzc_analysis_args)


if __name__ == "__main__":
    main()