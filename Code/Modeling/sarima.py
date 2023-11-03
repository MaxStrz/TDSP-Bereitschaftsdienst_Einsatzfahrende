from IPython.display import display # Allows the use of display() for DataFrames
# getcwd
from os import getcwd

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pickle

import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

# Registriert Matplotlib-Konverter, um Pandas Zeitreihendaten problemlos zu verarbeiten
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters() # Konverter registrieren

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA # Autoregressive Integrated Moving Average Model
from statsmodels.tsa.statespace.sarimax import SARIMAX # Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors Model
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller unit root test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Autocorrelation and Partial Autocorrelation Plots
from time import time 
import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore') # ignoriere warnings

# immer eine gute Praxis, RANDOM_SEED zu setzen, um den Code mit den gleichen Ergebnissen reproduzierbar zu machen
RANDOM_SEED = np.random.seed(0)

current_directory = os.path.dirname(__file__) # absolute dir the script is in
filepath = os.path.join(current_directory, "../Analysis/df1.pkl")
df = pd.read_pickle(filepath)

def check_stationarity(ts):
    aug_dickey_fuller_test = adfuller(df['calls_reg_act_diff'])
    adf = aug_dickey_fuller_test[0]
    p_value = aug_dickey_fuller_test[1]
    critical_values = aug_dickey_fuller_test[4]['5%']
    if (adf < critical_values) and (p_value < 0.05):
        print('The time series is stationary')
    else:
        print('The time series is not stationary')

check_stationarity(df['calls_reg_act_diff'])

new_series = pd.Series(df['calls_reg_act_diff'].values, index=df['date'])

freq = pd.infer_freq(new_series.index)
new_series = new_series.asfreq(freq) # setze Frequenz auf 1 Tag bzw. taeglich

# plt.figure(figsize=(15, 3))
# plt.plot(new_series)
# plt.title('Notrufe minus Regressiontrend')
# plt.ylabel('Anzahl der Notrufe')

result = seasonal_decompose(new_series, model='additive')
#fig = result.plot()

# Autocorrelation-Plot erstellen und als jpg speichern
plot_acf(new_series, lags=70, zero=False)
plt.savefig('acf_60.jpg')

# Regression der Zeitreihe auf ihre Lags und auf Konstant.
plot_pacf(new_series, lags=70, zero=False)
plt.savefig('pacf_60.jpg')

new_series_train = new_series[:-46]
new_series_test = new_series[-46:]

my_order = (7, 0, 1)
model_arima = ARIMA(new_series_train, order=my_order)

# fit model
start = time()
model_arima_fitted = model_arima.fit()
end = time()
print('ARIMA model fitting time:', end - start)

# make prediction
# predictions = model_arima_fitted
# plt.figure(figsize=(15, 3))
# plt.plot(new_series, label='Original')
# plt.plot(predictions, color='red', label='Predicted')
# plt.legend()
# plt.title('ARIMA Model')
# plt.ylabel('Anzahl der Notrufe')
# plt.savefig('arima701.jpg') 

# P - Ein P=1 würde die erste saisonal versetzte Beobachtung im Modell verwenden, z.B. t-(m*1) oder t-12
# D - ein D von 1 würde eine saisonale Differenz erster Ordnung berechnen
# Q - Q=1 would use a first order errors in the model (e.g. moving average)
# s von 12 mit monatlichen Daten deutet auf ein jährliches seasonales Zyklus hin 

# my_seasonal_order = (1, 0, 1, 365)
# model_sarima = SARIMAX(new_series_train, order=my_order, seasonal_order=my_seasonal_order)

# # fit model
# start = time()
# model_sarima_fitted = model_sarima.fit(maxiter=10, disp=1, low_memory=True)
# end = time()
# print('SARIMA model fitting time:', end - start)
# with open('sarima_model.pkl', 'wb') as f:
#     pickle.dump(model_sarima_fitted, f)

# load model
with open('model_sarima.pkl', 'rb') as f: # 'rb' for read binary
    model_sarima_fitted = pickle.load(f)

print(model_sarima_fitted.summary())

# make prediction
z = len(new_series_test)
print(z)
predictions = model_sarima_fitted.forecast(z) # 46 Tage in der Zukunft 
predictions = pd.Series(predictions, index=new_series_test.index)
print(predictions)
print(type(predictions))
print(new_series_test.shape)
print(type(new_series_test))
residuals = new_series_test - predictions
print(residuals)
 
plt.figure(figsize=(15, 3))
plt.plot(residuals)
plt.plot(predictions, color='red', label='Predicted')
plt.plot(new_series_test, color='green', label='Original')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()