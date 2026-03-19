# Results Directory README

This directory contains forecasting results from the replication package. All files are organized by forecasting method and data scenario.

## Directory Structure

```
results/
├── arimax_results/          # ARIMAX forecasts
├── sarimax_results/         # SARIMAX forecasts  
├── argo_results/            # ARGO forecasts
├── lgbm_results/            # LightGBM forecasts
├── adaboost_results/        # AdaBoost forecasts
├── denoising_results/       # Denoising evaluation outputs
├── forecast_rmses/          # RMSE (Root Mean Squared Error) by model/method
├── forecast_errors/         # Prediction errors (residuals) by model/method
```

## Data Scenarios

All forecasting methods are evaluated across six data preprocessing approaches:

1. **`noexog`**: No exogenous variables (hospitalization data only)
2. **`indiv`**: Individual non-preprocessed Google Trends search terms
3. **`topics`**: Topics from Google Trends (subset of individual terms)
4. **`clusters`**: Hierarchical clusters of search terms and individual terms and topics
5. **`smooth`**: Denoised (smoothed) cluster data
6. **`detrend`**: Detrended (denoised + detrended) cluster data

## File Naming Convention

### Forecast Results Files

**Format:** `{model}_{method}_{horizon}.csv`

Example: `arimax_clusters_h0.csv`

**Components:**
- `{model}`: Forecasting method (`arimax`, `sarimax`, `argo`, `lgbm`, `adaboost`)
- `{method}`: Data preprocessing (`noexog`, `indiv`, `clusters`, `smooth`, `detrend`, `topics`)
- `{horizon}`: Forecast horizon (`h0`, `h1`, `h2`, `h3` = 0, 1, 2, 3 weeks ahead)

**File Contents:** 
- Columns: `date`, location (e.g., `US-AL`, `US-CA`, etc.)
- Values: Predicted hospitalization counts
- Index: One row per week, one column per location

### RMSE Files

**Format:** `{model}_{method}_rmses.csv`

Example: `arimax_clusters_rmses.csv`

**File Contents:**
- Index: Geographic locations (states/territories)
- Columns: `h0`, `h1`, `h2`, `h3` (RMSE for each forecast horizon)
- Values: Root mean squared error between predictions and actual values

### Error Files  

**Format:** `{model}_{method}_{horizon}_errors.csv`

Example: `adaboost_clusters_h0_errors.csv`

**File Contents:**
- Columns: `date`, location (e.g., `US-AL`, `US-CA`, etc.)
- Values: Prediction errors (residuals) = actual - predicted
- Index: One row per week, one column per location

## Denoising Results

The `denoising_results/` directory contains comparative results from different denoising methods:

- `arimax_111_smooth_*.csv`: ARIMAX results using smoothing spline denoising
- `arimax_111_smooth_wt_*.csv`: ARIMAX results using wavelet denoising
- `arimax_111_smooth_ma_*.csv`: ARIMAX results using moving average denoising  
- `arimax_111_smooth_ssa_*.csv`: ARIMAX results using singular spectrum analysis

These files demonstrate the effectiveness of different denoising approaches in the context of forecasting.

## Notes

### Forecast Horizons

- **h0**: Nowcast
- **h1**: 1 week ahead
- **h2**: 2 weeks ahead
- **h3**: 3 weeks ahead

### Locations

Forecasts are provided for:
- 50 US states (e.g., `US-AL`, `US-AK`, ..., `US-WY`)
- District of Columbia (`US-DC`)
- Puerto Rico (`PR`) and US national-level (`US`) were excluded from the analyses, even though they are present in the data files.