import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 730
horizon = 730
epochs = 200
n_features = 1

# normalize data
scaler = MinMaxScaler()
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, horizon)

# split data
X, y = util.split_sequence(train_seq, window_size, horizon)
X = X.reshape((X.shape[0], X.shape[1], n_features))

# build model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=window_size))
model.add(Dense(horizon))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=epochs, verbose=1)

# predict future prices
test_input = np.array(train_seq[-window_size:]).reshape((1, window_size, n_features))
scaled_predictions = model.predict(test_input, verbose=1)
predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()

print("Predicted (unnormalized) prices:")
print(predictions)

actual = scaler.inverse_transform(test_seq.reshape(-1, 1))
mae = util.calculateMAE(actual, predictions)

util.graph_comparison('S&P 500 MLP Multi-Step (Vector) Prediction', price_series, mae, price_series,
                      predictions, split_idx)
