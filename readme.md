# N2O Respiratory Complexity Analysis

## Main Pipeline

1. `clean_segment.py`

Default method preprocesses respiratory data with _khodadad2018_ NeuroKit2 method, inspired by ([Khodadad et al. (2018)](https://iopscience.iop.org/article/10.1088/1361-6579/aad7e6/meta)).

_khodad2018_ removes slow baseline drifts by applying a lowcut at .05Hz (preserves breathing rates higher than 3 breath per minute) and high frequency noise by applying a highcut at 3 Hz (preserves breathing rates slower than 180 breath per minute). It applies 2nd order butterworth method, the traditional and legacy implementation using "butterworth_ba".

After cleaning, the signal is resampled from 2000Hz to 200Hz.

The signal for each participant is then split into segments before inhalation and during inhalation, based on the timestamps provided for each participant's data.

Preprocessed and split segments are stored into .mat files.

2. `epochize.py` - TODO
3. `lzc_calculation.py` - TODO
4. `lzc_analysis.py` - TODO

## Python Scripts

TODO - add all the ways in which each can be used

`clean_segment.py` cleans raw respiratory data, splits it into session segments, and saves 25 Hz segmented files.

`clean_segment_compare.py` compares the different cleaning methods across participant files and plots them side by side.

`complexity_sensitivity_window_size.py` tests how normalized LZC changes across different window sizes.

`complexity_sensitivity_sampling_frequency.py` tests how normalized LZC changes across different downsampling rates.

`complexity_sensitivity_analysis.py` is a broader LZC sensitivity script for cleaned respiratory segments.

`delete.py` deletes files in `data_clean_segmented` whose names contain `50Hz`.

`lzc_preprocessing.py` preprocesses raw respiratory data into cleaned pre-session and session segments for LZC analysis.

`plot_sampling_rate.py` compares the raw respiratory trace against one or more downsampled versions.

`rename.py` renames segmented files by replacing `50.0` with `50` in filenames.

`analyze_unique_values.py` inspects unique values in the workspace data or output files.

`sensitivity_analysis.py` is an older example LZC sensitivity script.

`sensitivity_analysis_01.py` is a revised LZC sensitivity example with extra diagnostics.

`sensitivity_analysis_02.py` is another sensitivity-analysis variant with multiple binarization checks.

## MATLAB Tools

`plot_resp_ecg.m` visualizes raw respiratory and ECG data after loading a raw file from `data_raw`.

## Legacy / Supporting Files

`lz76/setup.py` builds the LZ76 helper package used by the complexity analysis code.