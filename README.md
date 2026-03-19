# Reproducibility Package for "Restoring the Forecasting Power of Google Trends with Statistical Preprocessing"

Candice Djorno, Mauricio Santillana, Shihao Yang

---

## Overview

This replication package contains all code and data necessary to reproduce the results of the manuscript *"Restoring the Forecasting Power of Google Trends with Statistical Preprocessing"*. The main contents of the repository are:

- `data/`: folder containing raw and processed data files
- `scripts/`: folder containing scripts for each preprocessing step, evaluation, forecasting and plotting
- `results/`: folder containing the prediction results of forecasting 
- `figures/`: folder containing generated figures for the manuscript
- `tables/`: folder containing generated tables for the manuscript

---

## Repository Structure

Follow the steps below **in order** to fully reproduce the results.

### Preprocessing

1. **Hierarchical Clustering**
   - Script: `hierarchical_clustering.ipynb` (Open and run cell by cell)
   - Input: `individual_merged_trends`: time series of Google Trends keywords for each location.
   - Performed clustering of search terms for each location to group similar terms based on correlations.
   - The optimal number of clusters was selected using the Elbow method.
   - Outputs: `hierarchical_clusters` and `individual_terms`, merged into `terms` for each location. 
   
*Note*: Because cluster selection used visual inspection of the Elbow plot, final cluster assignments per location used in the paper are provided in `hierarchical_clusters`. Running the notebook is optional and only demonstrates the clustering procedure. After identifying clusters, combined search volumes were collected for each location from Google Trends and merged into `cluster_gt.csv`.

2. **Denoising**
   - Scripts: `gt_denoising.R`, `ss_denoising.R`
   - Input: `cluster_gt.csv`
   - Smoothing splines were applied iteratively in a rolling window fashion to avoid look-ahead bias.
   - Output: `smooth_gt.csv`

**Evaluation of denoising**
   - Scripts: `gt_wt_denoising.R`, `wt_denoising.R`, `gt_ma_denoising.R`, `ma_denoising.R`, `gt_ssa_denoising.R`, `ssa_denoising.R`, `arimax_preprocessing.py`
   - Input: `cluster_gt.csv`
   - Outputs: Denoised datasets from the different methods: `smooth_wt_gt.csv`, `smooth_ma_gt.csv`, `smooth_ssa_gt.csv`; forecasting results from ARIMAX: `arimax_111_smooth_....csv`

> This step uses parallel computing via R's `parallel::mclapply`. It may take several hours on a standard laptop; it is recommended to run it on a multi-core machine.

3. **Detrending**
   - Script: `detrending.py`
   - Input: `smooth_gt.csv`
   - Linear detrending, quadratic detrending and differencing were applied to the smoothed data to remove long-term trends, based on the outcome from the ADF test.
   - Output: `detrend_gt.csv`

### Application

4. **Forecasting Models**
- Target: hospitalization data (`hospitalizations.csv`)
- Input:
    - Individual raw variables (`individual_merged_trends/`)
    - Topics only (`individual_merged_trends/`)
    - Clustered raw data (`cluster_gt.csv`)
    - Denoised data (`smooth_gt.csv`)
    - Detrended data (`detrend_gt.csv`)
- Models:
    - ARIMAX: `arimax.py`, `arimax_indiv.py`, `run_arimax.sh`, `run_arimax_indiv.sh`
    - SARIMAX: `sarimax.py`, `sarimax_indiv.py`, `run_sarimax.sh`, `run_sarimax_indiv.sh`
    - ARGO: `argo_predictions.R`, `argo_helper.R`, `argo_functions.R`, `argo_predictions_indiv.R`, `argo_helper_indiv.R`, `run_argo.sh`, `run_argo_indiv.sh`, `format_argo.py`
    - LightGBM: `lgbm_forecasting.py`, `lightgbm_indiv.py`, `run_lgbm.sh`, `run_lgbm_indiv.sh`
    - AdaBoost: `adaboost.py`, `adaboost_indiv.py`, `run_adaboost.sh`, `run_adaboost_indiv.sh`
- Output: `arimax_results`, `sarimax_results`, `argo_results`, `lgbm_results`, `adaboost_results`, `forecast_rmses`, `forecast_errors`
     
5. **Statistical Testing**
   - Script: `statistical_tests.R`
   - Input: RMSEs (`forecast_rmses`) and weekly errors (`forecast_errors`) for each method
   - Conducted statistical tests (one-sided paired Wilcoxon Signed-Rank test, panel Diebold-Mariano test, Fluctuation test) on forecast errors (RMSEs and weekly errors) to evaluate the significance of differences and analyzed time-varying forecast performance
   - Output: tables for Wilcoxon test and DM test (`tables/`), figures for Fluctuation test (`figures/...fluctuation_tests...`)

### Plotting
   - Script: `manuscript_plots.py`
   - Inputs: `data`, `results/`
   - Generate all figures and tables in the manuscript
   - Output: figures in `figures/` and tables in `tables/`

---

## Data

### 1. Hospitalization Data (`data/01_raw/hospitalizations.csv`)
- **Description:** Weekly flu hospitalization counts for all locations, used as the target variable.
- **Format:** CSV with a `date` column (weekly, Saturdays) and one column per location.
- **Source:** This data is publicly available through CDC FluSight GitHub repository http://github.com/cdcepi/FluSight-forecast-hub/blob/main/target-data/target-hospital-admissions.csv. To extend the training period for modeling, we follow the methodology of Meyer et al. (2025) to augment the dataset using ILINet data and FluSurv-Net data with transfer learning. 
- **Access:** Included directly in this replication kit.

### 2. Google Trends Data (`data/01_raw/`; `data/03_preprocessed/cluster_gt.csv`)

- **Description:** Weekly search volumes of search queries for all locations, used as exogenous variables.
- **Format:** CSV with a `date` column (weekly, Saturdays) and one column per keyword per location.
- **Source:** https://trends.google.com. An API key is required to retrieve this data.
- **Access:** Included directly in this replication kit. 

### 3. Intermediary Datasets (`data/02_intermediate/hierarchical_clusters`, `data/02_intermediate/individual_terms`, `data/02_intermediate/terms`)

The following intermediary files are included in the repository and can also be regenerated:

- **Description:** Terms and clusters for each location.
- **Format:** Text files with one search phrase per line.
- **Source:** Generated by `hierarchical_clustering.ipynb` notebook.
- **Access:** Included directly in this replication kit. 

### 4. Preprocessed Data (`data/03_preprocessed/smooth_gt.csv`, `data/03_preprocessed/detrend_gt.csv`)

The following preprocessed files are included in the repository and can also be regenerated:

- **Description:** Denoised and detrended weekly search volumes of Google Trends search queries, used as exogenous variables.
- **Format:** CSV with a `date` column (weekly, Saturdays) and one column per keyword per location.
- **Source:** Generated by `gt_denoising.R` and `detrending.py` scripts.
- **Access:** Included directly in this replication kit. 

---

## Results (`results/`)

### 1. Forecast Results for Each Method (`results/..._results`)
- **Description:** predictions for each week, location, horizon and model and RMSEs for each location, horizon and model.
- **Format:** CSV files.
- **Source:** Generated by `04_forecasting` scripts.
- **Access:** Included directly in this replication kit. 

### 2. Forecast Errors for Each Method (`results/forecast_errors`)
- **Description:** residuals for each week, location, horizon and model.
- **Format:** CSV files.
- **Source:** Generated by `format_errors.py` script.
- **Access:** Included directly in this replication kit. 

### 3. Forecast RMSE for Each Method (`results/forecast_rmses`)
- **Description:** RMSEs for each location, horizon and model.
- **Format:** CSV files.
- **Source:** Generated by `04_forecasting` scripts.
- **Access:** Included directly in this replication kit. 

---

## Figures and Tables

All figures from the manuscript are included in this replication kit (`figures/` and `tables/`) and generated by the scripts `scripts/05_statistical_testing/statistical_tests.R` and `scripts/06_plotting/manuscript_plots.py`.

---

## Computing Environment

### Execution Directory (Important)
Run all scripts from the repository root unless a script explicitly states otherwise. Many scripts use relative paths such as `data/...` and `scripts/...`, which assume your current working directory is the project root.

### Hardware
The experiments were run on a Linux-based HPC cluster. Parallel computing is used for denoising (R) and some forecasting scripts (R, Python). No GPU is required.

**Recommended specifications:**
- **CPU**: 8+ cores (for parallel denoising)
- **RAM**: 16 GB
- **OS**: Linux or macOS (Unix-like system required for `mclapply` and `joblib.Parallel`)

### Languages
- **Python** ≥ 3.8 (tested with 3.8.17 and with 3.11.4)
- **R** ≥ 4.0 (tested with 4.3.3)

### Dependency Lock Files

**Python dependencies (Conda multi-env workflow)**:
- `environment-py38.yml`: Defines `py38-main` for most Python scripts
- `environment-py311.yml`: Defines `py311` for `arimax_indiv.py` and `sarimax_indiv.py`
- `requirements-lock.txt`: Python package lock list consumed by `environment-py38.yml`

Create Python environments with:
```bash
conda env create -f environment-py38.yml
conda env create -f environment-py311.yml
```

Run via the per-script wrappers:

```bash
bash scripts/<step_folder>/run_<script_name>.sh
```

**R packages**: `environment-R.yml` lists the R packages used in the project. The `renv.lock` file contains the exact versions of these packages for reproducibility.

*Note*: The `argo` package functions are adapted for multiple-steps-ahead prediction and provided directly in `scripts/04_forecasting/argo_functions.R`.

### Expected Runtime

**Estimated times on recommended hardware:**
- Clustering: 3 minutes per location for 51 locations
- Denoising (R, parallel): 1-2 hours
- Detrending: 10 minutes
- Forecasting models (parallel): 1-3 hours each
   - ARIMAX: 3-4 hours
   - SARIMAX: 3-4 hours
   - ARGO: 2-3 hours
   - LightGBM: 1-2 hours
   - AdaBoost: 1-2 hours
- Statistical testing & plotting: 30 minutes
- **Total**: 13-21 hours (with parallelization)

Note: Denoising and some forecasting scripts use parallel computing.

---

## Special Setup

- Both **R** and **Python** are required (not optional)
- Use Conda environments for reproducibility:
   - Python (main): `conda env create -f environment-py38.yml`
   - Python (indiv): `conda env create -f environment-py311.yml`
   - R: `conda env create -f environment-R.yml`
- Execute Python scripts through `run_*.sh` wrappers so each script runs in its intended environment.
- `requirements-lock.txt` is part of the `py38-main` environment specification, not a standalone single-environment workflow.
- Scripts use relative paths and must be run from specific directories
- **Parallel computing:** 
   - The denoising scripts (`gt_denoising.R`, `ma_denoising.R`, `wt_denoising.R`, `ssa_denoising.R`) use R's `parallel::mclapply`. 
   - AdaBoost forecasting script (`adaboost.py`) uses `joblib.Parallel` for parallelizing model training across locations and horizons.
   These require a multi-core Unix/Linux environment. The number of cores used is set to `detectCores() - 1` in R scripts and `n_jobs=-1` in Python scripts, which uses all available cores except one to avoid overloading the system. Adjust these settings if you want to limit the number of cores used.
- **No GPU required.**

---

# Manual Workflow (Step-by-Step)

Run each script manually in the same order as `run_all.sh`.

## Prerequisites

- `conda` installed (default path used by scripts: `$HOME/miniconda3/bin/conda`)
- Run from repository root
- Scripts are executable (`chmod +x scripts/**/run_*.sh`)

---

## 1. Clustering Google Trends search terms

Run the `scripts/01_clustering/hierarchical_clustering.ipynb` notebook cell by cell to perform hierarchical clustering of search terms for each location based on the visual inspection of the Elbow plot.

## 2. Denoising Google Trends data

```bash
bash scripts/02_denoising/run_gt_denoising.sh
```

## Denoising evaluation

```bash
bash scripts/02_denoising/run_gt_wt_denoising.sh
bash scripts/02_denoising/run_gt_ma_denoising.sh
bash scripts/02_denoising/run_gt_ssa_denoising.sh
bash scripts/04_forecasting/run_arimax_preprocessing.sh
```

## 3. Detrending Google Trends data

```bash
bash scripts/03_detrending/run_detrend_gt.sh
```

## 4. Forecasting

```bash
bash scripts/04_forecasting/run_arimax.sh
bash scripts/04_forecasting/run_arimax_indiv.sh

bash scripts/04_forecasting/run_sarimax.sh
bash scripts/04_forecasting/run_sarimax_indiv.sh

bash scripts/04_forecasting/run_argo_predictions.sh
bash scripts/04_forecasting/run_argo_predictions_indiv.sh

bash scripts/04_forecasting/run_lgbm_forecasting.sh
bash scripts/04_forecasting/run_lightgbm_indiv.sh

bash scripts/04_forecasting/run_adaboost.sh
bash scripts/04_forecasting/run_adaboost_indiv.sh
```

## 5. Statistical testing

```bash
bash scripts/05_statistical_testing/run_statistical_tests.sh
```

## 6. Plotting

```bash
bash scripts/06_plotting/run_manuscript_plots.sh
```

---

## Optional: Run everything manually in one paste

```bash
bash scripts/02_denoising/run_gt_denoising.sh && \
bash scripts/03_detrending/run_detrend_gt.sh && \
bash scripts/02_denoising/run_gt_wt_denoising.sh && \
bash scripts/02_denoising/run_gt_ma_denoising.sh && \
bash scripts/02_denoising/run_gt_ssa_denoising.sh && \
bash scripts/04_forecasting/run_arimax_preprocessing.sh && \
bash scripts/04_forecasting/run_arimax.sh && \
bash scripts/04_forecasting/run_arimax_indiv.sh && \
bash scripts/04_forecasting/run_sarimax.sh && \
bash scripts/04_forecasting/run_sarimax_indiv.sh && \
bash scripts/04_forecasting/run_argo_predictions.sh && \
bash scripts/04_forecasting/run_argo_predictions_indiv.sh && \
bash scripts/04_forecasting/run_lgbm_forecasting.sh && \
bash scripts/04_forecasting/run_lightgbm_indiv.sh && \
bash scripts/04_forecasting/run_adaboost.sh && \
bash scripts/04_forecasting/run_adaboost_indiv.sh && \
bash scripts/05_statistical_testing/run_statistical_tests.sh && \
bash scripts/06_plotting/run_manuscript_plots.sh
```

---

**Last updated:** March 2026