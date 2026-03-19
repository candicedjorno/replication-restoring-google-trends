import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

cutoff = '2022-10-01' # cutoff between train and test sets
smooth = pd.read_csv('data/03_preprocessed/smooth_gt.csv')
smooth = smooth.reset_index(drop = True)
train = smooth.copy()
train = train[train['date'] < cutoff]
test = smooth.copy()
test = test[test['date'] >= cutoff]

smooth['date'] = pd.to_datetime(smooth['date'])
smooth.set_index('date', inplace=True)

# ADF test to determine whether a time series needs to be detrended
adf_c = []
adf_ct = []
adf_ctt = []
alpha = 0.05 # critical value
print("adf")
print("max index:", train.shape[1])
for i in range(1, train.shape[1]):
    kw = train.iloc[:,i]
    # constant only: check if time series is stationary
    adf1 = adfuller(kw, regression='c')[1]
    adf_c.append(adf1)
    # constant + trend: check if time series has deterministic trend
    adf2 = adfuller(kw, regression='ct')[1]
    adf_ct.append(adf2)
    # Constant, linear, and quadratic trend: check if time series has quadratic deterministic trend
    adf3 = adfuller(kw, regression='ctt')[1]
    adf_ctt.append(adf3)

# using ADF test to determine whether a keyword needs to be detrended and detrending that keyword
time = np.arange(len(smooth))
detrended_smooth = pd.DataFrame(index=smooth.index)
from sklearn.metrics import r2_score
r2s = []
detrended_cols = []
differenced_cols = []

print("detrending")
for i in range(smooth.shape[1]):
    data = smooth.iloc[:, i]
    if (adf_c[i] > alpha) & (adf_ct[i] < alpha): # linear
        train_size = len(train)
        train_time = time[:train_size]
        train_data = data[:train_size]
        
        # Fit the linear trend on the train set
        mean_time_train = np.mean(train_time)
        mean_data_train = np.mean(train_data)
        
        numerator = np.sum((train_time - mean_time_train) * (train_data - mean_data_train))
        denominator = np.sum((train_time - mean_time_train) ** 2)
        slope_train = numerator / denominator
        intercept_train = mean_data_train - slope_train * mean_time_train
        
        # Apply the trend to detrend the entire dataset
        trend_estimate = slope_train * time + intercept_train
        detrended_data = data / trend_estimate
        
        r2 = r2_score(data, trend_estimate)
        r2s.append(r2)
        detrended_smooth[data.name] = detrended_data
        detrended_cols.append(data.name)
        
    elif (adf_c[i] > alpha) & (adf_ct[i] > alpha) & (adf_ctt[i] < alpha): # quadratic
        train_size = len(train)
        train_time = time[:train_size]
        train_data = data[:train_size]
        
        # Fit the quadratic trend on the train set
        # Calculate the coefficients for a second-degree polynomial: ax^2 + bx + c
        coeffs = np.polyfit(train_time, train_data, deg=2)
        a, b, c = coeffs

        # Apply the quadratic trend to detrend the entire dataset
        trend_estimate = a * time**2 + b * time + c
        detrended_data = data / trend_estimate

        detrended_smooth[data.name] = detrended_data
        
        r2 = r2_score(data, trend_estimate)
        r2s.append(r2)
        detrended_cols.append(data.name)

    elif (adf_c[i] > alpha) & (adf_ct[i] > alpha) & (adf_ctt[i] > alpha): # stochastic
        detrended_data = smooth[data.name].diff() # differencing
        differenced_cols.append(data.name)
        detrended_smooth[data.name] = detrended_data

    else:
        detrended_smooth[data.name] = smooth[data.name]


print('Dim of detrended data', detrended_smooth.shape)

detrended_smooth = detrended_smooth.dropna() # because of differencing
detrended_smooth = detrended_smooth.reset_index()
detrended_smooth.to_csv('data/03_preprocessed/detrend_gt.csv', index = False)
