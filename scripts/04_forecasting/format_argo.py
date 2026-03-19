# separating the predictions by horizon for ARGO/ARGO2
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

hosp = pd.read_csv('data/01_raw/hospitalizations.csv')
min_date = '2022-10-17'
max_date = '2024-04-27'
save_dir = "results/argo_results/"
horizons = [0, 1, 2, 3]
method_names = ['noexog', 'indiv', 'topics', 'clusters', 'smooth', 'detrend']

for data in method_names:
    argo = pd.read_csv(f"{save_dir}argo_{data}.csv")
    # argo = pd.read_csv(f"{save_dir}argo2_{data}.csv")
    
    # Convert target_end_date to datetime for filtering
    argo['target_end_date'] = pd.to_datetime(argo['target_end_date'])
    
    # Filter to match the date range used for hospitalization data
    argo = argo[(argo['target_end_date'] >= min_date) & (argo['target_end_date'] <= max_date)]
    
    for h in horizons:
        argo_sub = argo[['target_end_date', 'location', 'value']][argo['horizon'] == h]
        argo_sub = argo_sub.pivot(index='target_end_date', columns='location', values='value')
        argo_sub = argo_sub.reset_index(drop=False)
        argo_sub.rename(columns={'target_end_date': 'date'}, inplace=True)
        argo_sub.columns.name = None
        argo_sub.to_csv(f"{save_dir}argo_{data}_h{h}.csv", index=False)