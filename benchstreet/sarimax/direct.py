from benchstreet.util import *

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')

df = getDataFrame()

df_monthly = df.resample('M').mean()

decompose = seasonal_decompose(df_monthly)
decompose.plot()

plt.show()

ord_diff = np.diff(df_monthly['Price'], n=1) # non stationary data warrants 1st order difference
adf_res = adfuller(ord_diff)  # stationality test (p < 0.5 is good)

print(f'p-value: {adf_res[1]}')

# can't use built in split b/c date unit is months, not days
train = df_monthly[:-24]
test = df_monthly[-24:]

# build model
model = SARIMAX(train,
                order = (2, 1, 3),
                seasonal_order = (0, 2, 2, 12),
                enforce_stationarity = False,
                enforce_invertibility = True, trend='c')

# fit model
result = model.fit(maxiter = 500, method = 'nm')

start = len(train)
end = len(train) + len(test) - 1

predictions = result.predict(start, end)
predictions_filled = predictions.resample('D').ffill() # monthly seasonal prediction, have to fill in daily values

print(predictions_filled)