# N2O Respiratory Complexity Analysis

## Setup

### Requirements

- Python 3.8+
- Conda (Anaconda or Miniconda)

### Installation

1. Clone or download this repository.

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate n2o-analysis
```

3. Prepare your data:
   - Place all raw MATLAB `.mat` files in the `data_raw` folder.
   - Each file should contain respiratory and ECG signals sampled at 2000 Hz.
   - Ensure corresponding metadata Excel file is present in `data_raw`.

## Full Pipeline (`full.py`)

Runs all 4 core pipeline components:
1. Filtering, downsampling, and segmentation (`prep.py`)
2. Splitting segments into epochs (`epochize.py`)
3. Calculating LZC per epoch (`lzc_calculation.py`)
4. Comparing LZC and building RLMs (`lzc_analysis.py`)

#### Usage

```bash
python full.py --input-dir data_raw
```

This example assumes the raw MATLAB files are provided in the folder data_raw.

#### Arguments Availale

- `--input-dir`: Path to the folder containing raw participant `.mat` files.
- `--method`: Preprocessing method name used by `prep.py` _(default: `khodadad2018`, also supports: `manual`, `charlton2021`, `biosppy`, `hampel`, and `none`)_.
- `--lowcut`: Bandpass low cutoff in Hz _(default: `0.05`, used if method set to `manual`)_.
- `--highcut`: Bandpass high cutoff in Hz _(default: `5.0`, used if method set to `manual`)_.
- `--filter-order`: Filter order _(default: `4`, used if method set to `manual`)_.
- `--filter-method`: Filtering method _(default: `butterworth`, used if method set to `manual`)_.
- `--fs`: Original sampling frequency in Hz _(default: `2000.0`)_.
- `--fs-target`: Target sampling frequency in Hz used for downsampling _(default: `200.0`)_.
- `--epoch-length-sec`: The target length of each epoch in seconds _(default: `120.0`)_.
- `--drop-bad-epochs`: If set, drops bad epochs during epochization _(WIP: not fully implemented)_.
- `--save-plots`: If set, saves rejected-epoch plots during epochization _(default: `False`)_.
- `--standardize-file`: Standardizes the downsampled signal before segmentation in `prep.py`.
- `--standardize-segment`: Standardizes each segment after segmentation in `prep.py`.
- `--standardize-epoch`: Standardizes each epoch before saving in `epochize.py`.

## Pipeline Components

### 1. Preprocessing (`prep.py`)

FILTERING: By default, _khodad2018_ (NeuroKit2 methdod inspired by [Khodadad et al. (2018)](https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta) is applied. It removes slow baseline drifts by applying a lowcut at .05Hz (preserves breathing rates higher than 3 breath per minute) and high frequency noise by applying a highcut at 3 Hz (preserves breathing rates slower than 180 breath per minute). It applies 2nd order butterworth method, the traditional and legacy implementation using "butterworth_ba".

DOWNSAMPLING: By default, the signal is resampled from 2000Hz to 200Hz.

SEGMENTATION: The signal for each participant is then split into continuous segments before inhalation and during inhalation, based on the timestamps provided for each participant's data.

OUTPUT: Filtered, downsample, and segmented .mat files are stored in `data_{pipeline_method}_{fs_target}Hz` folder.
If `--standardize-file` or `--standardize-segment` are used, the output folder gets a `_std_file` or `_std_seg` suffix respectively.

#### Usage

```bash
python full.py --input-dir data_raw
```

This example assumes the raw MATLAB files are provided in the folder data_raw.

#### Arguments Availale

- `--input-dir`: Directory containing raw `.mat` files _(default: `data_raw`)_.
- `--method`: Preprocessing method _(default: `khodadad2018`, also supports: `manual`, `charlton2021`, `biosppy`, `hampel`, and `none`)_.
- `--lowcut`: Bandpass low cutoff in Hz _(default: `0.05`, used if method set to `manual`)_.
- `--highcut`: Bandpass high cutoff in Hz _(default: `5.0`, used if method set to `manual`)_.
- `--filter-order`: Filter order _(default: `4`, used if method set to `manual`)_.
- `--filter-method`: Filtering method _(default: `butterworth`, used if method set to `manual`)_.
- `--fs`: Original sampling frequency in Hz _(default: `2000.0`)_.
- `--fs-target`: Target sampling frequency in Hz _(default: `200.0`)_.
- `--standardize-file`: Standardize the resampled signal before segmentation.
- `--standardize-segment`: Standardize each segment after segmentation.


### 2. Splitting into epochs (`epochize.py`)

Splits each pre- and dur-segment into fixed-length epochs. Optional QC rules can reject bad epochs, and rejected epochs can be plotted interactively or saved to disk.

#### Usage

```bash
python epochize.py --input-dir data_khodadad2018_200Hz
```

This example assumes the preprocessed MATLAB files are provided in the folder `data_khodadad2018_200Hz`.

#### Arguments Available

- `--input-dir`: Directory containing preprocessed `.mat` files _(default: `data_khodadad2018_200Hz`)_.
- `--method`: Pipeline method name; if set, the input directory becomes `data_<method>`.
- `--epoch-length-sec`: Epoch length in seconds _(default: `120.0`)_.
- `--drop-bad-epochs`: If set, drops epochs that fail QC rules.
- `--save-plots`: If set, saves plots of rejected epochs.
- `--skip-plots`: If set, does not display rejected-epoch plots interactively.
- `--standardize`: If set, standardizes each epoch before saving.

### 3. Calculating LZC (`lzc_calculation.py`)

Computes normalized Lempel-Ziv complexity for each epoch, then aggregates the results per file and condition and saves CSV summaries.

#### Usage

```bash
python lzc_calculation.py --input-dir data_khodadad2018_200Hz_120s_all
```

This example assumes the epochized MATLAB files are provided in the folder `data_khodadad2018_200Hz_120s_all`.

#### Arguments Available

- `--pattern`: Search pattern for epochized files _(default: `*.mat`)_.
- `--input-dir`: Directory containing epochized `.mat` files _(default: `data_khodadad2018_200Hz_120s_all`)_.
- `--method`: Pipeline method name; if set, the input directory becomes `data_<method>`.
- `--skip-calculation`: If set, skips LZC computation and reads the existing CSV instead.
- `--csv-filename`: Optional custom CSV filename to read/write.

### 4. Analysing LZC values (`lzc_analysis.py`)

Merges the LZC summaries with study metadata, runs Welch t-tests for key condition comparisons, and fits robust linear models for exploratory analysis.

#### Usage

```bash
python lzc_analysis.py --method khodadad2018_200Hz_120s_all
```

This example assumes the corresponding LZC CSV files are in `data_khodadad2018_200Hz_120s_all_lzc` and the study metadata Excel file is in `data_raw`.

#### Arguments Available

- `--method`: Method name used to locate the LZC output folder and CSV files _(default: `khodadad2018_200Hz_120s_all`)_.

### 5. Further LZC analysis - Python Notebook (`lzc_analysis.ipynb`)

Reuses aspects of `lzc_analysis.py` and builds upon it with additional box and bar plots in the cells at the end.

## Python Analysis Tools

### `complexity_sensitivity_frequency.py`

Tests how normalized LZC changes across different downsampling rates. 

#### Example usage

```bash
python analysis/complexity_sensitivity_frequency.py --input-dir data_khodadad2018_200Hz --fs-targets 3 6 9 12 15
```

### `complexity_sensitivity_window.py`

Tests how normalized LZC changes across different window sizes.

#### Example usage

```bash
python analysis/complexity_sensitivity_window.py --input-dir data_khodadad2018_200Hz --windows 10 20 30
```

### `plot_epoch.py`

Interactively plots epochs from `epochize.py` outputs and lets you move between files and epochs.

#### Example usage

```bash
python analysis/plot_epoch.py data_khodadad2018_200Hz_120s_all/P003-1-trimmed.mat --epoch-type pre --index 0
```

### `plot_segment.py`

Interactively plots pre- or dur-segments from `prep.py` outputs and lets you move between files and segments.

#### Example usage

```bash
python analysis/plot_segment.py data_khodadad2018_200Hz/P003-1-trimmed.mat --segment-type pre --index 0
```

### `plot_different_prep_methods.py`

Compares the different cleaning methods across participant files and plots them side by side.

#### Example usage

```bash
python analysis/plot_different_prep_methods.py --input-dir data_clean_segmented --methods khodadad2018 manual_nk2 biosppy
```

### `plot_different_sampling_rates.py`

Compares the raw respiratory trace against one or more downsampled versions.

#### Example usage

```bash
python analysis/plot_different_sampling_rates.py --input-dir data_raw --file P003-1-trimmed.mat --rates 50 25 --plot-original
```

### `print_unique_timestamp_values.py`

Inspects unique values in the timestamps data.

#### Example usage

```bash
python analysis/print_unique_timestamp_values.py
```

## MATLAB Analysis Tools

### `plot_resp_ecg.m`

Visualizes raw respiratory and ECG data after loading a raw MATLAB file.

#### Arguments Available

- `X`: N x 2 array of [resp, ecg] signals at 2000 Hz.
- `fsTarget`: Target sampling frequency in Hz for downsampling _(default: `25`)_.
- `fixedY`: Logical flag; if true, locks y-limits to global ranges for the entire signal _(default: `false`)_.

#### Example usage

```bash
plot_resp_ecg(data, 200, True)
```
