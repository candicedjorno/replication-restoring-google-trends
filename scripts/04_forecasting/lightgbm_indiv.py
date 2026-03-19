from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import us # for state names
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import signal
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# hospitalization data
hosp = pd.read_csv('data/01_raw/hospitalizations.csv')
max_date = '2024-04-27'
hosp = hosp[hosp['date'] <= max_date] # end date for 2023-2024 flu season
cutoff = '2022-10-01' # cutoff between train and test sets
hosp_train = hosp[hosp['date'] <= cutoff]

# all locations
geo = pd.read_csv('data/01_raw/geo.txt', header = None)[0]
# geo = geo[geo != 'US']

save_dir = 'results/lgbm_results'
save_dir_rmse = 'results/forecast_rmses'

import os
# read each location's data
directory = 'data/01_raw/individual_merged_trends/'
# sorting file names alphabetically
files = sorted(os.listdir(directory))
files = sorted([file for file in files if file.endswith('.csv')])
print(len(files), 'csv files')

print("data loaded")

# getting the state name corresponding to its code
def get_state_name(state):
    state_code = state.split('-')[1] if '-' in state else state

    # Special case for 'US-DC'
    if state_code == 'DC':
        state_name = 'District of Columbia'
    else:
        state_name = us.states.lookup(state_code).name if us.states.lookup(state_code) else 'US'

    return state_name
   
# builds lag features and merges exogenous variables if provided
min_date = '2022-10-17'
# incorporate lagged exogenous variables 
# Lagged exogenous features allow the model to capture delayed relationships between Google Trends and hospitalizations
def create_lagged_features(Y_ts, X_ts, lags, exog_lags, horizon):
    df = Y_ts.copy()
    # df['y_target'] = df['y'].shift(horizon)
    df['y_target'] = df['y'].shift(-horizon)
    for lag in lags:
        df[f'y_lag{lag}'] = df['y'].shift(lag + horizon)

    feature_cols = [f'y_lag{lag}' for lag in lags]
    if X_ts is not None:
        X_exog = X_ts.copy()
        exog_cols = [col for col in X_exog.columns if col not in ['unique_id', 'ds']]
        # Add lagged exogenous features
        for col in exog_cols:
            for lag in exog_lags:
                X_exog[f'{col}_lag{lag}'] = X_exog[col].shift(lag)
        if horizon > 0:
            # Shift exogenous variables forward so they align with the target
            for col in exog_cols:
                X_exog[col] = X_exog[col].shift(horizon)
                for lag in exog_lags:
                    X_exog[f'{col}_lag{lag}'] = X_exog[f'{col}_lag{lag}'].shift(horizon)
        df = df.merge(X_exog, on=['unique_id', 'ds'], how='left')
        # Add lagged exogenous columns to feature_cols
        for col in exog_cols:
            feature_cols.append(col)
            for lag in exog_lags:
                feature_cols.append(f'{col}_lag{lag}')
    # df = df.dropna().reset_index(drop=True)
    return df, feature_cols


# this actually uses a rolling window, not an expanding window
def lgbm_forecast_expanding_window(X_ts, Y_ts, horizon, lags, exog_lags):
    df, feature_cols = create_lagged_features(Y_ts=Y_ts, X_ts=X_ts, lags=lags, horizon=horizon, exog_lags=exog_lags)
    test_mask = (df['ds'] > min_date) & (df['ds'] <= max_date)
    test_dates = df.loc[test_mask, 'ds']
    preds = []
    actuals = []
    test_ds = []
    for test_date in test_dates:
        train = df[(df['ds'] < test_date) & (df['ds'] > '2018-10-01')] # expanding window
        test_row = df[df['ds'] == test_date]
        lgbm = LGBMRegressor(random_state=42, n_jobs=1, n_estimators=300, 
                             learning_rate=0.5, boosting_type="goss", data_sample_strategy="goss",
                             max_depth=5, verbosity=-1, force_col_wise=True)
        lgbm.fit(train[feature_cols], train['y_target'])
        pred = lgbm.predict(test_row[feature_cols])
        preds.append(pred[0])
        actuals.append(test_row['y'].values[0])
        test_ds.append(test_date)
    res = pd.DataFrame({'ds': test_ds, 'LGBM': preds, 'actual': actuals})
    rmse = np.sqrt(mean_squared_error(res['actual'], res['LGBM']))
    mae = mean_absolute_error(res['actual'], res['LGBM'])
    print(f'Expanding window | RMSE: {rmse:.2f} | MAE: {mae:.2f}')
    return res, rmse


def run_lgbm_expanding_all_locations_indiv(
    geo,  
    lags, 
    exog_lags, 
    horizons, 
    save_dir='results/lgbm_results'
):
    results = {
        'indiv': []
    }

    preds_dict = {
        'indiv': {h: {} for h in horizons}
    }
    
    for i in range(len(geo)):
        state = geo[i]
        row_indiv = {'geo': state}
        # Prepare data once for each method
        filename = [file for file in files if file.startswith(state+'_')][0]
        print(filename)
        data = pd.read_csv(f'data/01_raw/individual_merged_trends/{filename}') # with topics    
        # removing columns that have all zeros
        all_zero_columns = data.columns[(data == 0).all()]
        data = data.drop(columns=all_zero_columns)
        
        data = data[data['date'] >= hosp['date'].min()]
        data = data[data['date'] <= hosp['date'].max()]
        data = data.reset_index(drop=True)
        data = data.drop(columns=['date'])
        
        state_code = state.split('-')[1] if '-' in state else state
        print(state_code)
        state_name = get_state_name(state)
        print(state_name)
        hosp_state = hosp[f'hosp_{state_name}']
        
        hosp_sub = hosp_state[:hosp_train.shape[0]]
        data_sub = data.iloc[:hosp_train.shape[0]]
        columns_to_keep = []
        correlations = data_sub.corrwith(hosp_sub)
        # percentile-based threshold based on correlation distribution
        corr_threshold = np.percentile(correlations.dropna(), 25)
        filtered_corrs = correlations[correlations >= corr_threshold]

        selected_columns = filtered_corrs.index.tolist()
        # computing the correlation matrix for the selected columns
        selected_corr_matrix = data_sub[selected_columns].corr().abs()
        
        # identifying highly correlated pairs (above 0.90) to avoid multicollinearity
        to_remove = set()
        for i in range(len(selected_columns)):
            for j in range(i + 1, len(selected_columns)):
                if selected_corr_matrix.iloc[i, j] > 0.90:
                    # keeping only one of the two variables
                    to_remove.add(selected_columns[j])  # removing the second one

                    
        final_columns = [col for col in selected_columns if col not in to_remove]
        corrs_25 = filtered_corrs.loc[final_columns].nlargest(25)
        final_25 = corrs_25.index.tolist()
        columns_to_keep.append(final_25)
        columns_to_keep_flat = [col for sublist in columns_to_keep for col in sublist]

        Y_ts = pd.DataFrame(hosp_state)
        Y_ts = Y_ts.rename(columns={hosp_state.name: 'y'})
        Y_ts.insert(0, 'ds', hosp['date'])
        Y_ts["ds"] = pd.to_datetime(Y_ts["ds"])
        Y_ts.insert(0, 'unique_id', '1')
        
        X_ts = data[columns_to_keep_flat].copy()
        X_ts.insert(0, 'ds', hosp['date']) # adding date from format of statsforecast
        X_ts.insert(0, 'unique_id', '1')
        X_ts["ds"] = pd.to_datetime(X_ts["ds"])
        X_ts[columns_to_keep_flat] = X_ts[columns_to_keep_flat].shift(1)

        print("number preds:", len(columns_to_keep_flat))

        # date_index = None  # To store the date index for predictions
        date_index = {h: None for h in horizons}

        for h in horizons:
            res_indiv, rmse_indiv = lgbm_forecast_expanding_window(
                X_ts=X_ts, Y_ts=Y_ts, horizon=h, lags=lags, exog_lags=exog_lags)
            row_indiv[f'rmse_lgbm_h{h}'] = rmse_indiv
            # row_indiv[f'mae_h{h}'] = mae_indiv
            preds_dict['indiv'][h][state] = res_indiv['LGBM'].values
            date_index[h] = res_indiv['ds']

        results['indiv'].append(row_indiv)

    # Convert results to DataFrames for easy analysis
    dfs = {k: pd.DataFrame(v) for k, v in results.items()}

    # Save each method's RMSEs to CSV
    for method in ['indiv']:
        cols = ['geo'] + [f'rmse_lgbm_h{h}' for h in horizons]
        df = dfs[method][cols]
        df.to_csv(f'{save_dir_rmse}/lgbm_{method}_rmses.csv', index=False)

    # Save predictions in wide format for each method and horizon
    for method in ['indiv']:
        for h in horizons:
            pred_df = pd.DataFrame(preds_dict[method][h])
            pred_df.insert(0, 'date', date_index[h])
            pred_df.to_csv(f'{save_dir}/lgbm_{method}_h{h}.csv', index=False)

    return dfs

print("starting")
dfs = run_lgbm_expanding_all_locations_indiv(
    geo, 
    lags=list(range(1,53)),
    exog_lags=[1,2,3,4],
    horizons=[0,1,2,3], 
    save_dir='results/lgbm_results'
)

def run_lgbm_expanding_all_locations_topics(
    geo,  
    lags, 
    exog_lags, 
    horizons, 
    save_dir='results/lgbm_results'
):
    results = {
        'topics': []
    }

    preds_dict = {
        'topics': {h: {} for h in horizons}
    }
    
    for i in range(len(geo)):
        state = geo[i]
        row_topics = {'geo': state}
        # Prepare data once for each method

        filename = [file for file in files if file.startswith(state+'_')][0]
        print(filename)
        data = pd.read_csv(f'data/01_raw/individual_merged_trends/{filename}') # with topics    
        # removing columns that have all zeros
        all_zero_columns = data.columns[(data == 0).all()]
        data = data.drop(columns=all_zero_columns)
        
        data = data[data['date'] >= hosp['date'].min()]
        data = data[data['date'] <= hosp['date'].max()]
        data = data.reset_index(drop=True)
        data = data.drop(columns=['date'])
        
        state_code = state.split('-')[1] if '-' in state else state
        print(state_code)
        state_name = get_state_name(state)
        print(state_name)
        hosp_state = hosp[f'hosp_{state_name}']
        
        columns_to_keep = []
        hosp_sub = hosp_state[:hosp_train.shape[0]]
        data_sub = data.iloc[:hosp_train.shape[0]]
        # filtering columns whose name contains "/m/" or "(TOPIC)" (topic columns)
        topic_cols = [col for col in data.columns if '/m/' in col or '(TOPIC)' in col]
        filtered_df = data_sub[topic_cols]
        correlations = filtered_df.corrwith(hosp_sub)
        # selecting only the highly correlated topics
        corr_threshold = np.percentile(correlations.dropna(), 25)
        filtered_corrs = correlations[correlations >= corr_threshold]
        # columns_to_keep.append(filtered_corrs.index.tolist())
        # columns_to_keep_flat = [col for sublist in columns_to_keep for col in sublist]

        selected_columns = filtered_corrs.index.tolist()
        # computing the correlation matrix for the selected columns
        selected_corr_matrix = data_sub[selected_columns].corr().abs()
        
        # identifying highly correlated pairs (above 0.90) to avoid multicollinearity
        to_remove = set()
        for i in range(len(selected_columns)):
            for j in range(i + 1, len(selected_columns)):
                if selected_corr_matrix.iloc[i, j] > 0.90:
                    # keeping only one of the two variables
                    to_remove.add(selected_columns[j])  # removing the second one

                    
        final_columns = [col for col in selected_columns if col not in to_remove]
        corrs_25 = filtered_corrs.loc[final_columns].nlargest(25)
        final_25 = corrs_25.index.tolist()
        columns_to_keep.append(final_25)
        columns_to_keep_flat = [col for sublist in columns_to_keep for col in sublist]

        Y_ts = pd.DataFrame(hosp_state)
        Y_ts = Y_ts.rename(columns={hosp_state.name: 'y'})
        Y_ts.insert(0, 'ds', hosp['date'])
        Y_ts["ds"] = pd.to_datetime(Y_ts["ds"])
        Y_ts.insert(0, 'unique_id', '1')
        
        X_ts = data[columns_to_keep_flat].copy()
        X_ts.insert(0, 'ds', hosp['date']) # adding date from format of statsforecast
        X_ts.insert(0, 'unique_id', '1')
        X_ts["ds"] = pd.to_datetime(X_ts["ds"])
        X_ts[columns_to_keep_flat] = X_ts[columns_to_keep_flat].shift(1)
        
        print("number preds:", len(columns_to_keep_flat))

        # date_index = None  # To store the date index for predictions
        date_index = {h: None for h in horizons}

        for h in horizons:
            res_topics, rmse_topics = lgbm_forecast_expanding_window(
                X_ts=X_ts, Y_ts=Y_ts, horizon=h, lags=lags, exog_lags=exog_lags)
            row_topics[f'rmse_lgbm_h{h}'] = rmse_topics
            # row_topics[f'mae_h{h}'] = mae_topics
            preds_dict['topics'][h][state] = res_topics['LGBM'].values
            date_index[h] = res_topics['ds']

        results['topics'].append(row_topics)

    # Convert results to DataFrames for easy analysis
    dfs = {k: pd.DataFrame(v) for k, v in results.items()}

    # Save each method's RMSEs to CSV
    for method in ['topics']:
        cols = ['geo'] + [f'rmse_lgbm_h{h}' for h in horizons]
        df = dfs[method][cols]
        df.to_csv(f'{save_dir_rmse}/lgbm_{method}_rmses.csv', index=False)

    # Save predictions in wide format for each method and horizon
    for method in ['topics']:
        for h in horizons:
            pred_df = pd.DataFrame(preds_dict[method][h])
            # pred_df.insert(0, 'date', date_index)
            pred_df.insert(0, 'date', date_index[h])
            pred_df.to_csv(f'{save_dir}/lgbm_{method}_h{h}.csv', index=False)

    return dfs

print("starting")
dfs = run_lgbm_expanding_all_locations_topics(
    geo, 
    lags=list(range(1,53)),
    exog_lags=[1,2,3,4],
    horizons=[0,1,2,3], 
    save_dir='results/lgbm_results'
)