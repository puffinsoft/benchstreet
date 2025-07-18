import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 730
horizon = 1
epochs = 300
n_features = 1

# normalize data
scaler = MinMaxScaler(feature_range=(-2, 2))
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
# not a multi-step predictor so horizon must be custom set
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, 730)

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

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# fit model
model.fit(X, y_deltas, epochs=epochs, verbose=1)

predictions = []

test_input = train_seq[-window_size:].tolist()

for i in range(len(test_seq)):
    x_input = np.array(test_input[-window_size:]).reshape((1, window_size, 1))
    predicted_y = model.predict(x_input, verbose=0)
    last_val = test_input[-1]
    predicted_price = float(predicted_y[0, 0]) + last_val

    predictions.append(predicted_price)
    test_input.append(predicted_price)

    print(f"Step {i + 1}/{len(test_seq)} - Prediction {predicted_price}")

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actual = scaler.inverse_transform(test_seq.reshape(-1, 1)).flatten().reshape(1,-1).flatten()
mae = util.calculateMAE(actual, predictions)

util.graph_comparison(
    'S&P 500 1D-CNN Recursive Prediction (Delta Mode)',
    price_series,
    mae,
    price_series,
    predictions,
    split_idx
)
