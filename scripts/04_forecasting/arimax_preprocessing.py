# Comparison of ARIMA and ARIMAX forecasts for horizons 0 to 3

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string
from statsforecast.models import ARIMA
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

# denoised with smoothing splines
smooth = pd.read_csv('data/03_preprocessed/smooth_gt.csv')
smooth = smooth[smooth['date'] >= hosp['date'].min()]
smooth = smooth[smooth['date'] <= hosp['date'].max()]
smooth = smooth.reset_index(drop = True)

# denoised with moving average
smooth_ma = pd.read_csv('results/denoising_results/smooth_ma_gt.csv')
smooth_ma = smooth_ma[smooth_ma['date'] >= hosp['date'].min()]
smooth_ma = smooth_ma[smooth_ma['date'] <= hosp['date'].max()]
smooth_ma = smooth_ma.reset_index(drop = True)

# denoised with singular spectrum analysis
smooth_ssa = pd.read_csv('results/denoising_results/smooth_ssa_gt.csv')
smooth_ssa = smooth_ssa[smooth_ssa['date'] >= hosp['date'].min()]
smooth_ssa = smooth_ssa[smooth_ssa['date'] <= hosp['date'].max()]
smooth_ssa = smooth_ssa.reset_index(drop = True)

# denoised with wavelet transform
smooth_wt = pd.read_csv('results/denoising_results/smooth_wt_gt.csv')
smooth_wt = smooth_wt[smooth_wt['date'] >= hosp['date'].min()]
smooth_wt = smooth_wt[smooth_wt['date'] <= hosp['date'].max()]
smooth_ssa = smooth_ssa.reset_index(drop = True)

# all locations
geo = pd.read_csv('data/01_raw/geo.txt', header = None)[0]

# definition of ARIMA(1,1,1) model
model = ARIMA(
    order=(1, 1, 1),           # ARIMA(1,1,1)
    include_mean=False,        # No intercept if the series is stationary
    include_drift=False,       # No drift component
    season_length=1,           # No seasonal effects
    seasonal_order=(0, 0, 0)   # Explicitly stating no seasonality
)

save_dir = 'results/denoising_results'

# computing RMSE and MAE
def rmse_mae(res, truth):
    rmse = np.sqrt(mean_squared_error(res['ARIMA'], truth.iloc[:,2]))
    mae = mean_absolute_error(res['ARIMA'], truth.iloc[:,2])
    print('rmse:', rmse)
    return rmse, mae

def filter_corr(state, data, hosp_state, percentile_value=25): 
    hosp_sub = hosp_state[:hosp_train.shape[0]]
    data_sub = data.iloc[:hosp_train.shape[0]]
    columns_to_keep = []
    correlations = data_sub.corrwith(hosp_sub)
    
    # percentile-based threshold based on correlation distribution
    corr_threshold = np.percentile(correlations.dropna(), percentile_value)
    
    filtered_corrs = correlations[correlations >= corr_threshold]
    selected_columns = filtered_corrs.index.tolist()
    # computing the correlation matrix for the selected columns
    selected_corr_matrix = data_sub[selected_columns].corr().abs()
    
    # identifying highly correlated pairs (above 0.99) to avoid multicollinearity
    to_remove = set()
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            if selected_corr_matrix.iloc[i, j] > 0.90:
                # keeping only one of the two variables
                to_remove.add(selected_columns[j])  # removing the second one

                
    final_columns = [col for col in selected_columns if col not in to_remove]
    corrs_35 = filtered_corrs.loc[final_columns].nlargest(35)
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

# ARIMAX forecast for one state
start = 104 # using rolling window of past 104 weeks
min_date = '2022-10-17'
horizons = (0, 1, 2, 3)
def arima_forecast(X_ts, Y_ts):

    all_results = []
    dates = X_ts['ds'].unique()
    dates = dates[dates > np.datetime64('2019-01-01')] # starts the test set on 2021-01-01

    for n in range(start, len(dates)):
        dtrain = dates[n-start:n] # rolling window of 104 weeks
        dtest = dates[n]
        
        Y_train = Y_ts.query('ds in @dtrain')
        Y_test = Y_ts.query('ds == @dtest')
        
        X_train = X_ts.query('ds in @dtrain')
        X_test = X_ts.query('ds == @dtest')

        # merging training data
        train = Y_train.merge(X_train, how='left', on=['unique_id', 'ds'])
        
        # creating a StatsForecast instance with weekly frequency (hospitalization date corresponds to ending Saturday)
        sf = StatsForecast(models=[model], freq='W-SAT', n_jobs=-1)

        # dynamically setting the forecast horizon 
        max_h = min(4, len(dates) - n)
        if max_h == 0:
            break  # No more dates to predict

        forecast_dates = dates[n:n + max_h]
        X_test_replicated = pd.concat([X_test]*max_h, ignore_index=True)
        X_test_replicated['ds'] = forecast_dates

        fcst = sf.forecast(df=train, h=max_h, X_df=X_test_replicated)

        result = pd.DataFrame(fcst)
        unique_dates = result['ds'].unique()
        horizon_mapping = {date: i for i, date in enumerate(unique_dates)}
        result.insert(1, 'horizon', result['ds'].map(horizon_mapping))
        result = result.sort_values(by='horizon')

        all_results.append(result)

    # concatenating all results
    res = pd.concat(all_results, ignore_index=True)

    # retrieving the predictions for each horizon and computing errors with truth
    results = {}
    rmses = {}
    maes = {}
    for h in horizons:
        offset = timedelta(days=7*h)
        res_h = res[(res['horizon'] == h) & (res['ds'] >= pd.to_datetime(min_date) + offset) & (res['ds'] <= max_date)].reset_index(drop=True)
        truth_h = Y_ts[Y_ts['ds'] >= pd.to_datetime(min_date) + offset].reset_index(drop=True)
        rmse, mae = rmse_mae(res_h, truth_h)
        results[f'res_h{h}'] = res_h
        rmses[f'rmse_h{h}'] = rmse
        maes[f'mae_h{h}'] = mae
    
    return results, rmses, maes

# ARIMA forecast for one state without exogenous variables
def arima_noexog(X_ts, Y_ts):

    all_results = []
    dates = X_ts['ds'].unique()
    dates = dates[dates > np.datetime64('2019-01-01')] # starts the test set on 2021-01-01

    for n in range(start, len(dates)):
        dtrain = dates[n-start:n] # rolling window of 104 weeks
        dtest = dates[n]
        Y_train = Y_ts.query('ds in @dtrain') 
        Y_test = Y_ts.query('ds == @dtest')       
        # creating a StatsForecast instance with weekly frequency (hospitalization date corresponds to ending Saturday)
        sf = StatsForecast(models=[model], freq='W-SAT', n_jobs=-1)

        # dynamically setting the forecast horizon 
        max_h = min(4, len(dates) - n)
        if max_h == 0:
            break  # No more dates to predict

        fcst = sf.forecast(df=Y_train, h=max_h)
        result = pd.DataFrame(fcst)
        unique_dates = result['ds'].unique()
        horizon_mapping = {date: i for i, date in enumerate(unique_dates)}
        result.insert(1, 'horizon', result['ds'].map(horizon_mapping))
        result = result.sort_values(by='horizon')

        all_results.append(result)

    # concatenating all results
    res = pd.concat(all_results, ignore_index=True)

    # retrieving the predictions for each horizon and computing errors with truth
    results = {}
    rmses = {}
    maes = {}
    for h in horizons:
        offset = timedelta(days=7*h)
        res_h = res[(res['horizon'] == h) & (res['ds'] >= pd.to_datetime(min_date) + offset) & (res['ds'] <= max_date)].reset_index(drop=True)
        truth_h = Y_ts[Y_ts['ds'] >= pd.to_datetime(min_date) + offset].reset_index(drop=True)
        rmse, mae = rmse_mae(res_h, truth_h)
        results[f'res_h{h}'] = res_h
        rmses[f'rmse_h{h}'] = rmse
        maes[f'mae_h{h}'] = mae
    
    return results, rmses, maes

# function to run ARIMAX for all locations
def run_arimax_all_locations(geo, data, save_dir, prefix, name):
    preds_h0 = pd.DataFrame()
    preds_h1 = pd.DataFrame()
    preds_h2 = pd.DataFrame()
    preds_h3 = pd.DataFrame()

    rmses_h0 = []
    rmses_h1 = []
    rmses_h2 = []
    rmses_h3 = []

    for i in range(len(geo)):
        state = geo[i]
        X_ts, Y_ts, hosp_state, corrs, threshold = data_prep(state, data)
        results, rmses, maes = arima_forecast(X_ts, Y_ts)

        res_h0 = results['res_h0']
        res_h1 = results['res_h1']
        res_h2 = results['res_h2']
        res_h3 = results['res_h3']
        
        rmses_h0.append(rmses['rmse_h0'])
        rmses_h1.append(rmses['rmse_h1'])
        rmses_h2.append(rmses['rmse_h2'])
        rmses_h3.append(rmses['rmse_h3'])
        preds_h0[f'{state}_pred'] = res_h0['ARIMA']
        preds_h1[f'{state}_pred'] = res_h1['ARIMA']
        preds_h2[f'{state}_pred'] = res_h2['ARIMA']
        preds_h3[f'{state}_pred'] = res_h3['ARIMA']

    # Format and save predictions for each horizon
    for preds, res, h in zip(
        [preds_h0, preds_h1, preds_h2, preds_h3],
        [res_h0, res_h1, res_h2, res_h3],
        ['h0', 'h1', 'h2', 'h3']
    ):
        preds.insert(0, 'date', res['ds'])
        preds.columns = preds.columns.str.replace('_pred', '')
        preds.columns = [get_state_name(col) if col != 'date' else col for col in preds.columns]
        preds.to_csv(f'{save_dir}/{prefix}_{name}_{h}.csv', index=False)

    # Return predictions and RMSEs as DataFrames
    rmses_df = pd.DataFrame({
        'geo': geo,
        f'rmse_{prefix}_h0': rmses_h0,
        f'rmse_{prefix}_h1': rmses_h1,
        f'rmse_{prefix}_h2': rmses_h2,
        f'rmse_{prefix}_h3': rmses_h3
    })
    rmses_df.to_csv(f'{save_dir}/{prefix}_{name}_rmses.csv', index=False)
    return preds_h0, preds_h1, preds_h2, preds_h3, rmses_df

# function to run ARIMA without exogenous information for all locations
def run_arima_all_locations(geo, data, save_dir, prefix):
    preds_h0 = pd.DataFrame()
    preds_h1 = pd.DataFrame()
    preds_h2 = pd.DataFrame()
    preds_h3 = pd.DataFrame()

    rmses_h0 = []
    rmses_h1 = []
    rmses_h2 = []
    rmses_h3 = []

    for i in range(len(geo)):
        state = geo[i]
        X_ts, Y_ts, hosp_state, corrs, threshold = data_prep(state, data)
        results, rmses, maes = arima_noexog(X_ts, Y_ts)

        res_h0 = results['res_h0']
        res_h1 = results['res_h1']
        res_h2 = results['res_h2']
        res_h3 = results['res_h3']
        
        rmses_h0.append(rmses['rmse_h0'])
        rmses_h1.append(rmses['rmse_h1'])
        rmses_h2.append(rmses['rmse_h2'])
        rmses_h3.append(rmses['rmse_h3'])
        preds_h0[f'{state}_pred'] = res_h0['ARIMA']
        preds_h1[f'{state}_pred'] = res_h1['ARIMA']
        preds_h2[f'{state}_pred'] = res_h2['ARIMA']
        preds_h3[f'{state}_pred'] = res_h3['ARIMA']

    # Format and save predictions for each horizon
    for preds, res, h in zip(
        [preds_h0, preds_h1, preds_h2, preds_h3],
        [res_h0, res_h1, res_h2, res_h3],
        ['h0', 'h1', 'h2', 'h3']
    ):
        preds.insert(0, 'date', res['ds'])
        preds.columns = preds.columns.str.replace('_pred', '')
        preds.columns = [get_state_name(col) if col != 'date' else col for col in preds.columns]
        preds.to_csv(f'{save_dir}/{prefix}_{h}.csv', index=False)

    # Return predictions and RMSEs as DataFrames
    rmses_df = pd.DataFrame({
        'geo': geo,
        f'rmse_{prefix}_h0': rmses_h0,
        f'rmse_{prefix}_h1': rmses_h1,
        f'rmse_{prefix}_h2': rmses_h2,
        f'rmse_{prefix}_h3': rmses_h3
    })
    rmses_df.to_csv(f'{save_dir}/{prefix}_rmses.csv', index=False)
    return preds_h0, preds_h1, preds_h2, preds_h3, rmses_df

# smooth clusters
preds_h0, preds_h1, preds_h2, preds_h3, rmses_df = run_arimax_all_locations(geo, smooth, save_dir, 'arimax_111', 'smooth')

# smooth MA clusters
preds_h0, preds_h1, preds_h2, preds_h3, rmses_df = run_arimax_all_locations(geo, smooth_ma, save_dir, 'arimax_111', 'smooth_ma')

# smooth SSA clusters
preds_h0, preds_h1, preds_h2, preds_h3, rmses_df = run_arimax_all_locations(geo, smooth_ssa, save_dir, 'arimax_111', 'smooth_ssa')

# smooth WT clusters
preds_h0, preds_h1, preds_h2, preds_h3, rmses_df = run_arimax_all_locations(geo, smooth_wt, save_dir, 'arimax_111', 'smooth_wt')
