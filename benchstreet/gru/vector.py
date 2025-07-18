import numpy as np

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 730
horizon = 730
epochs = 10
n_features = 1

# normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, horizon)

# split data
X, y = util.split_sequence(train_seq, window_size, horizon)
X = X.reshape((X.shape[0], X.shape[1], n_features))

# build model
model = Sequential()
model.add(GRU(64,
              activation='tanh',
              return_sequences=True,
              input_shape=(window_size, n_features),
              dropout=0.2,
              recurrent_dropout=0.2))
model.add(GRU(32,
              activation='tanh',
              dropout=0.2,
              recurrent_dropout=0.2))
model.add(Dropout(0.3))
model.add(Dense(horizon, activation='linear'))

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])

# fit model
model.fit(X, y, epochs=epochs, verbose=1)

test_input = np.array(train_seq[-window_size:]).reshape((1, window_size, n_features))
scaled_predictions = model.predict(test_input, verbose=1)
predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()

# SCALE ADJUSTMENT - fix the offset
last_train_value = price_series.iloc[split_idx-1]
first_pred_value = predictions[0]
offset = last_train_value - first_pred_value
predictions_corrected = predictions + offset

print("Predicted (unnormalized) prices:")
print(predictions_corrected)

actual = scaler.inverse_transform(test_seq.reshape(-1, 1)).flatten().reshape(1,-1).flatten()
mae = util.calculateMAE(actual, predictions_corrected)

util.graph_comparison('S&P 500 GRU Multi-Step (Vector) Prediction', price_series, mae, price_series,
                 predictions_corrected, split_idx)
