# N2O Respiratory Complexity Analysis

## Setup

**TODO** How to install the necessary libraries and prepare the environment

## Main Pipeline

### 1. Preprocessing (`prep.py`)

Default method preprocesses respiratory data with _khodadad2018_ NeuroKit2 method, inspired by ([Khodadad et al. (2018)](https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta)).

_khodad2018_ removes slow baseline drifts by applying a lowcut at .05Hz (preserves breathing rates higher than 3 breath per minute) and high frequency noise by applying a highcut at 3 Hz (preserves breathing rates slower than 180 breath per minute). It applies 2nd order butterworth method, the traditional and legacy implementation using "butterworth_ba".

After cleaning, the signal is resampled from 2000Hz to 200Hz.

The signal for each participant is then split into segments before inhalation and during inhalation, based on the timestamps provided for each participant's data.

Filtered, downsample, and segmented .mat files are stored in `data_{pipeline_method}_{fs_target}Hz`.

**TODO** - usage

### 2. Splitting into epochs (`epochize.py`) - TODO
### 3. Calculating LZC (`lzc_calculation.py`) - TODO
### 4. Analysing LZC values (`lzc_analysis.py`) - TODO

## Python Analysis Tools

**TODO** - usage

`complexity_sensitivity_frequency.py` tests how normalized LZC changes across different downsampling rates.

`complexity_sensitivity_window.py` tests how normalized LZC changes across different window sizes.

`plot_different_prep_methods.py` compares the different cleaning methods across participant files and plots them side by side.

`plot_different_sampling_rates.py` compares the raw respiratory trace against one or more downsampled versions.

`print_unique_timestamp_values.py` inspects unique values in the workspace data or output files.

## MATLAB Analysis Tools

`plot_resp_and_ecg_data.m` visualizes raw respiratory and ECG data after loading a raw file from `data_raw`.
