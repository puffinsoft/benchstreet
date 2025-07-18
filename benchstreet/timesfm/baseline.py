import numpy as np
import timesfm
from benchstreet import util

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=730,
        num_layers=50,
        use_positional_embedding=False,
        context_len=2048,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)

df = util.getDataFrame()
prices = util.getPrice(df)
timesfm_df = df.reset_index().rename(columns={'Date': 'ds', 'Price': 'y'})

train_seq, test_seq, split_idx = util.split_train_test(prices, 730)

sliced_df = timesfm_df.iloc[0:split_idx]
sliced_df['unique_id'] = 1

timesfm_forecast = tfm.forecast_on_df(
    inputs=sliced_df,
    freq="B",
    value_name="y",
    num_jobs=-1,
    verbose=1
)
timesfm_forecast.head()

prediction = np.array(timesfm_forecast["timesfm"])
mae = util.calculateMAE(prediction, test_seq)

util.graph_comparison(
    'S&P 500 TimesFM (Baseline) Prediction',
    df,
    mae,
    df,
    np.array(timesfm_forecast["timesfm"]),
    split_idx
)
