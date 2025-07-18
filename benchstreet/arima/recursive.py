import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

test_size = 730

train_seq, test_seq, split_idx = util.split_train_test(price_series.values, test_size)

model = ARIMA(train_seq, order=(4, 1, 2), seasonal_order=(1, 0, 2, 5))
model_fit = model.fit()

predictions = []
predictions = model_fit.forecast(steps=test_size)

predictions = np.array(predictions)
mae = util.calculateMAE(test_seq, predictions)

# Create comparison graph
util.graph_comparison('S&P 500 ARIMA Multi-Step (Direct) Prediction', dataset, mae, price_series,
                      predictions, split_idx)
