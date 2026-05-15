# EEM561 Binary Tongue Coating Classification

This folder contains a reproducible experimentation pipeline for the binary tongue coating dataset.

## Run

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

The script fixes the random seed, creates an 80/10/10 stratified train/validation/test split, applies augmentation only to the training set, trains four MLP-based experiments, and writes all outputs under `outputs/`.

## Outputs

- `outputs/split.csv`: reproducible train/validation/test file list
- `outputs/metrics.csv`: accuracy, precision, and F-score for validation and test
- `outputs/details.json`: confusion matrices, feature counts, thresholds, and MLP training details
- `outputs/test_metrics.png`: test metric comparison plot
- `outputs/report.md`: editable report text
- `outputs/report.pdf`: formatted report for submission
