import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 730
horizon = 730
epochs = 200
n_features = 1

# normalize data
scaler = MinMaxScaler(feature_range=(-2,2))
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, horizon)

# split data
X, y = util.split_sequence(train_seq, window_size, horizon)
X = X.reshape((X.shape[0], X.shape[1], n_features))

# convert targets to deltas
last_inputs = X[:, -1, 0].reshape(-1, 1)
y_deltas = y - last_inputs

# build model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(window_size, n_features)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(horizon))

optimizer = Adam()
model.compile(optimizer=optimizer, loss='mse')

# fit model
model.fit(X, y_deltas, epochs=epochs, verbose=1)

test_input = np.array(train_seq[-window_size:]).reshape((1, window_size, n_features))
last_val = test_input[0, -1, 0]

scaled_deltas = model.predict(test_input, verbose=1).flatten()
scaled_predictions = scaled_deltas + last_val 

predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()

actual = scaler.inverse_transform(test_seq.reshape(-1, 1)).flatten().reshape(1,-1).flatten()
mae = util.calculateMAE(actual, predictions)

util.graph_comparison(
    'S&P 500 1D-CNN Multi-Step (Vector) Prediction (Delta Mode)',
    price_series,
    mae,
    price_series,
    predictions,
    split_idx
)
