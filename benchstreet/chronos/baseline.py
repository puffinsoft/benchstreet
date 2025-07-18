import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline

from benchstreet import util

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small", 
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

df = util.getDataFrame()
price_series = util.getPrice(df)
train_seq, test_seq, split_idx = util.split_train_test(price_series, 730)

quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(train_seq),
    prediction_length=730,
    quantile_levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75,
                0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
)

predictions = mean[0].numpy()
mae = util.calculateMAE(test_seq, predictions)
util.graph_comparison('S&P 500 Chronos-T5-Small (Baseline) Prediction', df, mae, df,
                     predictions, split_idx)