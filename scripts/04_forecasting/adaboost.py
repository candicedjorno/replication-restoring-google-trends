from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import us # for state names
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import signal
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# hospitalization data
hosp = pd.read_csv('data/01_raw/hospitalizations.csv')
max_date = '2024-04-27'
hosp = hosp[hosp['date'] <= max_date] # end date for 2023-2024 flu season
cutoff = '2022-10-01' # cutoff between train and test sets
hosp_train = hosp[hosp['date'] <= cutoff]

# clusters
clusters = pd.read_csv('data/03_preprocessed/cluster_gt.csv')
clusters = clusters[clusters['date'] >= hosp['date'].min()]
clusters = clusters[clusters['date'] <= hosp['date'].max()]
clusters = clusters.reset_index(drop = True)

# denoised
smooth = pd.read_csv('data/03_preprocessed/smooth_gt.csv')
smooth = smooth[smooth['date'] >= hosp['date'].min()]
smooth = smooth[smooth['date'] <= hosp['date'].max()]
smooth = smooth.reset_index(drop = True)

# detrended
detrend = pd.read_csv('data/03_preprocessed/detrend_gt.csv')
detrend = detrend[detrend['date'] >= hosp['date'].min()]
detrend = detrend[detrend['date'] <= hosp['date'].max()]
detrend = detrend.reset_index(drop = True)

# all locations
geo = pd.read_csv('data/01_raw/geo.txt', header = None)[0]
# geo = geo[geo != 'US']

save_dir = 'results/adaboost_results'
save_dir_rmse = 'results/forecast_rmses'

print("data loaded")

def filter_corr(state, data, hosp_state, percentile_value=25): 
    hosp_sub = hosp_state[:hosp_train.shape[0]]
    data_sub = data.iloc[:hosp_train.shape[0]]
    columns_to_keep = []
    correlations = data_sub.corrwith(hosp_sub)
    
    # percentile-based threshold based on correlation distribution
    corr_threshold = np.percentile(correlations.dropna(), percentile_value)
    # corr_threshold = 0.3 # setting a fixed threshold for correlation

    filtered_corrs = correlations[correlations >= corr_threshold]
    selected_columns = filtered_corrs.index.tolist()
    # computing the correlation matrix for the selected columns
    selected_corr_matrix = data_sub[selected_columns].corr().abs()
    
    # identifying highly correlated pairs (above 0.99) to avoid multicollinearity
    to_remove = set()
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            if selected_corr_matrix.iloc[i, j] > 0.99:
                # keeping only one of the two variables
                to_remove.add(selected_columns[j])  # removing the second one

                
    final_columns = [col for col in selected_columns if col not in to_remove]
    corrs_35 = filtered_corrs.loc[final_columns].nlargest(25)
    final_35 = corrs_35.index.tolist()

    return final_35, filtered_corrs.loc[final_35], corr_threshold

# getting the state name corresponding to its code
def get_state_name(state):
    state_code = state.split('-')[1] if '-' in state else state

    # Special case for 'US-DC'
    if state_code == 'DC':
        state_name = 'District of Columbia'
    else:
        state_name = us.states.lookup(state_code).name if us.states.lookup(state_code) else 'US'

    return state_name
        
# preparing the data in the format of the statsforecast package
def data_prep(state, df):
    state_code = state.split('-')[1] if '-' in state else state
    print(state_code)
    state_name = get_state_name(state)
    print(state_name)

    # exogenous
    data = df[df.columns[df.columns.str.contains(state+'_')]]

    # hospitalizations
    hosp_state = hosp[f'hosp_{state_name}']
    
    cols, corrs, threshold = filter_corr(state, data, hosp_state)
    print('corr threshold:', threshold)
    print('num preds:', corrs.shape[0])

    # matching format of statsforecast
    Y_ts = pd.DataFrame(hosp_state)
    Y_ts = Y_ts.rename(columns={hosp_state.name: 'y'})
    Y_ts.insert(0, 'ds', hosp['date'])
    Y_ts["ds"] = pd.to_datetime(Y_ts["ds"])
    Y_ts.insert(0, 'unique_id', '1')
    
    X_ts = data[cols].copy()
    X_ts.insert(0, 'ds', df['date']) # adding date from format of statsforecast
    X_ts.insert(0, 'unique_id', '1')
    X_ts["ds"] = pd.to_datetime(X_ts["ds"])

    return X_ts, Y_ts, hosp_state, corrs, threshold

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

def ada_forecast_expanding_window(X_ts, Y_ts, horizon, lags, exog_lags):
    df, feature_cols = create_lagged_features(Y_ts=Y_ts, X_ts=X_ts, lags=lags, horizon=horizon, exog_lags=exog_lags)
    test_mask = (df['ds'] > min_date) & (df['ds'] <= max_date)
    test_dates = df.loc[test_mask, 'ds']
    preds = []
    actuals = []
    test_ds = []
    for test_date in test_dates:
        # train = df[df['ds'] < test_date]
        train = df[(df['ds'] < test_date) & (df['ds'] > '2018-10-01')] # expanding window
        train = train.dropna().reset_index(drop=True)

        test_row = df[df['ds'] == test_date]
        base_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        ada = AdaBoostRegressor(random_state=42, estimator=base_tree,
                                n_estimators=500, learning_rate=0.5)
        ada.fit(train[feature_cols], train['y_target'])
        pred = ada.predict(test_row[feature_cols])
        preds.append(pred[0])
        actuals.append(test_row['y'].values[0])
        test_ds.append(test_date)
    res = pd.DataFrame({'ds': test_ds, 'AdaBoost': preds, 'actual': actuals})
    rmse = np.sqrt(mean_squared_error(res['actual'], res['AdaBoost']))
    mae = mean_absolute_error(res['actual'], res['AdaBoost'])
    print(f'Expanding window | RMSE: {rmse:.2f} | MAE: {mae:.2f}')
    return res, rmse

def process_single_location(state, lags, exog_lags, horizons, clusters, smooth, detrend):
    """Process a single location with all methods (noexog, clusters, smooth, detrend)."""
    try:
        print(f"Processing {state}")
        row_noexog = {'geo': state}
        row_clusters = {'geo': state}
        row_smooth = {'geo': state}
        row_detrend = {'geo': state}
        
        # Prepare data once for each method
        X_ts_noexog, Y_ts, _, _, _ = data_prep(state, clusters)
        X_ts_clusters, _, _, _, _ = data_prep(state, clusters)
        X_ts_smooth, _, _, _, _ = data_prep(state, smooth)
        X_ts_detrend, _, _, _, _ = data_prep(state, detrend)
        
        location_preds = {
            'noexog': {h: None for h in horizons},
            'clusters': {h: None for h in horizons},
            'smooth': {h: None for h in horizons},
            'detrend': {h: None for h in horizons}
        }
        
        date_indices = {h: None for h in horizons}
        
        for h in horizons:
            # No exogenous
            res_noexog, rmse_noexog = ada_forecast_expanding_window(
                X_ts=None, Y_ts=Y_ts, horizon=h, lags=lags, exog_lags=exog_lags)
            row_noexog[f'rmse_adaboost_h{h}'] = rmse_noexog
            location_preds['noexog'][h] = res_noexog['AdaBoost'].values
            date_indices[h] = res_noexog['ds']
            
            # Clusters
            res_clusters, rmse_clusters = ada_forecast_expanding_window(
                X_ts=X_ts_clusters, Y_ts=Y_ts, horizon=h, lags=lags, exog_lags=exog_lags)
            row_clusters[f'rmse_adaboost_h{h}'] = rmse_clusters
            location_preds['clusters'][h] = res_clusters['AdaBoost'].values
            
            # Smooth
            res_smooth, rmse_smooth = ada_forecast_expanding_window(
                X_ts=X_ts_smooth, Y_ts=Y_ts, horizon=h, lags=lags, exog_lags=exog_lags)
            row_smooth[f'rmse_adaboost_h{h}'] = rmse_smooth
            location_preds['smooth'][h] = res_smooth['AdaBoost'].values
            
            # Detrend
            res_detrend, rmse_detrend = ada_forecast_expanding_window(
                X_ts=X_ts_detrend, Y_ts=Y_ts, horizon=h, lags=lags, exog_lags=exog_lags)
            row_detrend[f'rmse_adaboost_h{h}'] = rmse_detrend
            location_preds['detrend'][h] = res_detrend['AdaBoost'].values
        
        return {
            'noexog': row_noexog,
            'clusters': row_clusters,
            'smooth': row_smooth,
            'detrend': row_detrend,
            'preds': location_preds,
            'date_indices': date_indices,
            'state': state
        }
    except Exception as e:
        print(f"Error processing {state}: {str(e)}")
        return {'state': state, 'error': str(e)}

def run_ada_expanding_all_locations(
    geo,  
    lags, 
    exog_lags, 
    horizons, 
    save_dir='results/adaboost_results',
    n_jobs=-1  # -1 uses all available cores
):
    # Determine number of cores to use
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    print(f"Running with {n_jobs} parallel processes")
    
    # Process all locations in parallel
    location_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_location)(
            state, lags, exog_lags, horizons, clusters, smooth, detrend
        ) for state in geo
    )
    
    # Filter out any failed results
    location_results = [r for r in location_results if 'error' not in r]
    
    # Initialize results containers
    results = {
        'noexog': [],
        'clusters': [],
        'smooth': [],
        'detrend': []
    }
    
    preds_dict = {
        'noexog': {h: {} for h in horizons},
        'clusters': {h: {} for h in horizons},
        'smooth': {h: {} for h in horizons},
        'detrend': {h: {} for h in horizons}
    }
    
    # Any date index will work, since they should all be the same
    date_index = {h: None for h in horizons}
    
    # Collect results
    for result in location_results:
        state = result['state']
        results['noexog'].append(result['noexog'])
        results['clusters'].append(result['clusters'])
        results['smooth'].append(result['smooth'])
        results['detrend'].append(result['detrend'])
        
        # Get the predictions
        for method in ['noexog', 'clusters', 'smooth', 'detrend']:
            for h in horizons:
                preds_dict[method][h][state] = result['preds'][method][h]
        
        # Use first state's date indices as they should be the same for all
        if date_index[0] is None:
            date_index = result['date_indices']

    # Convert results to DataFrames for easy analysis
    dfs = {k: pd.DataFrame(v) for k, v in results.items()}

    # Save each method's RMSEs to CSV
    for method in ['noexog', 'clusters', 'smooth', 'detrend']:
        cols = ['geo'] + [f'rmse_adaboost_h{h}' for h in horizons]
        df = dfs[method][cols]
        df.to_csv(f'{save_dir_rmse}/adaboost_{method}_rmses.csv', index=False)

    # Save predictions in wide format for each method and horizon
    for method in ['noexog', 'clusters', 'smooth', 'detrend']:
        for h in horizons:
            pred_df = pd.DataFrame(preds_dict[method][h])
            pred_df.insert(0, 'date', date_index[h])
            pred_df.to_csv(f'{save_dir}/adaboost_{method}_h{h}.csv', index=False)

    return dfs

# lags=list(range(1,53)),
# lags=[1,2,3,4,5,6,7],
# lags=[1,2,3,4],
print("starting")
dfs = run_ada_expanding_all_locations(
    geo, 
    lags=[1,2,3,4],
    exog_lags=[1,2,3,4],
    horizons=[0,1,2,3], 
    save_dir='results/adaboost_results',
    n_jobs=-1  # Use all available cores
)