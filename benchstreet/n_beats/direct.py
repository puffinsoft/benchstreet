from neuralforecast.models import NBEATS
from neuralforecast.core import NeuralForecast

from benchstreet import util

df = util.getDataFrame()

input_df = df[:-730]
train_seq, test_seq, split_idx = util.split_train_test(df, 730)

input_df = input_df.reset_index()
input_df['unique_id'] = 1
input_df = input_df.rename(columns={'Date': 'ds', 'Price': 'y'})

model = NBEATS(
    input_size=730,
    h=730,
    max_steps=1000,
    scaler_type='standard',
    stack_types=['trend', 'seasonality']
)

nf = NeuralForecast(models=[model], freq='ME')
nf.fit(input_df)

predictions = nf.predict()

mae = util.calculateMAE(test_seq, predictions)

util.graph_comparison('S&P 500 N-BEATS Prediction', df, mae, df,
                      predictions, split_idx)
