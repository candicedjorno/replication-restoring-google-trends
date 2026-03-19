import pandas as pd
import numpy as np
import os
import us # for state names

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# Data Preparation
# ==============================================================================

# all locations
geo = pd.read_csv('data/01_raw/geo.txt', header = None)[0]
geo = geo[geo != "PR"].reset_index(drop=True) # excluding PR

# hospitalization data
hosp = pd.read_csv('data/01_raw/hospitalizations.csv')
min_date = '2022-10-17'
max_date = '2024-04-27'
hosp = hosp[(hosp['date'] >= min_date) & (hosp['date'] <= max_date)] 
hosp = hosp.reset_index(drop = True)

def get_state_name(state):
    state_code = state.split('-')[1] if '-' in state else state

    # Special case for 'US-DC'
    if state_code == 'DC':
        state_name = 'District of Columbia'
    else:
        state_name = us.states.lookup(state_code).name if us.states.lookup(state_code) else 'US'

    return state_name

# ==============================================================================
# Output error files
# ==============================================================================

# compute residuals
def errors_over_time(save_dir, model, method_names, h):
    errors_dict = {}  # Dictionary to store errors for each method
    for data in method_names:
        file = f'{save_dir}{model}_{data}_{h}.csv'
        df = pd.read_csv(file)  # Predictions
        # Convert date to datetime format first
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop any rows with NaN dates if present
        df = df.dropna(subset=['date'])
        # Order df columns by alphabetical order, keeping the date as the first column
        df = df.reindex(sorted(df.columns), axis=1)
        df = df[['date'] + [col for col in df.columns if col != 'date']]

        # Ensure hosp dates are also in datetime format
        if not pd.api.types.is_datetime64_dtype(hosp['date']):
            hosp['date'] = pd.to_datetime(hosp['date'])
            
        # Now compare dates safely
        # min_date = min(df['date'])
        hosp_sub = hosp[hosp['date'] >= min_date]
        hosp_sub = hosp_sub.reset_index(drop=True)

        df = df[(df['date'] >= min_date) & (df['date'] <= max_date)] 
        df = df.reset_index(drop = True)
        
        # Create a mapping for columns (skip 'date')
        if model == 'adaboost' or model == 'lgbm':
            new_columns = {
                col: get_state_name(col) if col != 'date' else 'date'
                for col in df.columns
            }
            # Rename columns
            df = df.rename(columns=new_columns)

        # Remove 'hosp_' prefix from hosp columns for alignment
        hosp_renamed = hosp_sub.rename(columns={col: col.replace('hosp_', '') for col in hosp_sub.columns if col != 'date'})
        # Find common location columns
        common_cols = [col for col in df.columns if col != 'date' and col in hosp_renamed.columns]
        # Align both dataframes
        df_aligned = df[['date'] + common_cols]
        hosp_aligned = hosp_renamed[['date'] + common_cols]
        # Compute errors for each location
        errors = []
        for col in common_cols:
            print(col)
            errors.append(df_aligned[col] - hosp_aligned[col])

        errors = pd.DataFrame(errors).T
        errors = pd.concat([df_aligned['date'], errors], axis=1)
        errors.columns = ['date'] + common_cols

        # Compute the errors for each row
        errors['date'] = pd.to_datetime(errors['date'], errors='coerce')  # Convert to datetime, handle invalid values
        errors.set_index('date', inplace=True)

        # remove Puerto Rico data
        errors = errors.loc[:, errors.columns != 'Puerto Rico']

        # Add the errors DataFrame to the dictionary
        errors_dict[data] = errors

    errors_noexog = errors_dict['noexog']
    errors_indiv = errors_dict['indiv']
    errors_topics = errors_dict['topics']
    errors_clusters = errors_dict['clusters']
    errors_smooth = errors_dict['smooth']
    errors_detrend = errors_dict['detrend']

    return errors_noexog, errors_indiv, errors_topics, errors_clusters, errors_smooth, errors_detrend

# ARIMA
save_dir = 'results/arimax_results/'  # directory where results are stored
model = "arimax_111"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

for h in horizons:
    errors = errors_over_time(save_dir, model, method_names, h)
    for name, df in zip(method_names, errors):
        df.to_csv(f'results/forecast_errors/{model}_{name}_{h}_errors.csv')

# SARIMA
save_dir = 'results/sarimax_results/'
model = "sarimax_010"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

for h in horizons:
    errors = errors_over_time(save_dir, model, method_names, h)
    for name, df in zip(method_names, errors):
        df.to_csv(f'results/forecast_errors/{model}_{name}_{h}_errors.csv')

# ARGO
save_dir = 'results/argo_results/' # directory where results are stored
model = "argo"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

for h in horizons:
    errors = errors_over_time(save_dir, model, method_names, h)
    for name, df in zip(method_names, errors):
        df.to_csv(f'results/forecast_errors/{model}_{name}_{h}_errors.csv')

# LightGBM
save_dir = 'results/lgbm_results/' # directory where results are stored
model = "lgbm"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

for h in horizons:
    errors = errors_over_time(save_dir, model, method_names, h)
    for name, df in zip(method_names, errors):
        df.to_csv(f'results/forecast_errors/{model}_{name}_{h}_errors.csv')

# AdaBoost
save_dir = 'results/adaboost_results/' # directory where results are stored
model = "adaboost"
horizons = ['h0', 'h1', 'h2', 'h3']
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

for h in horizons:
    errors = errors_over_time(save_dir, model, method_names, h)
    for name, df in zip(method_names, errors):
        df.to_csv(f'results/forecast_errors/{model}_{name}_{h}_errors.csv')
