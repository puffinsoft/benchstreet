import numpy as np

from prophet import Prophet

from benchstreet import util

df = util.getDataFrame()
prices = util.getPrice(df)

prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Price': 'y'})
train_seq, test_seq, split_idx = util.split_train_test(prices, 730)

sliced_df = prophet_df.iloc[0:split_idx]

# build model
m = Prophet(
    changepoint_prior_scale=0.05,
    interval_width=0.95
)

# fit model
m.fit(sliced_df)


future = m.make_future_dataframe(periods=730, freq="B")
forecast = m.predict(future)

predictions = np.array(forecast['yhat'])[-730:]
mae = util.calculateMAE(test_seq, predictions)

util.graph_comparison(
    'S&P 500 FBProphet Prediction',
    df,
    mae,
    df,
    predictions,
    split_idx
)
