import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 730
horizon = 730
epochs = 50
n_features = 1

# normalize data
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, horizon)

# split data
X, y = util.split_sequence(train_seq, window_size, horizon)
X = X.reshape((X.shape[0], X.shape[1], n_features))

# build model
model = Sequential()
model.add(LSTM(100, activation='tanh', return_sequences=True, input_shape=(window_size, n_features)))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(horizon))
optimizer = Adam(learning_rate=0.005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mse')

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

# Calculate MAE using corrected predictions
actual = scaler.inverse_transform(test_seq.reshape(-1, 1)).flatten().reshape(1,-1).flatten()
mae = util.calculateMAE(actual, predictions_corrected)

print(f"MAE with scale adjustment: {mae}")

util.graph_comparison('S&P 500 LSTM Multi-Step (Vector) Prediction', price_series, mae, price_series,
                      predictions_corrected, split_idx)
