import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 730
horizon = 1
epochs = 20
n_features = 1

# normalize data
scaler = MinMaxScaler()
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
# not a multi-step predictor so horizon must be custom set
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, 730)

# split data
X, y = util.split_sequence(train_seq, window_size, horizon)
X = X.reshape((X.shape[0], X.shape[1], n_features))

# build model
model = Sequential()
model.add(Input(shape=(window_size, 1)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=epochs, verbose=1)

predictions = []

test_input = train_seq[-window_size:].tolist()

for i in range(len(test_seq)):
    x_input = np.array(test_input[-window_size:]).reshape((1, window_size, 1))
    predicted_y = model.predict(x_input, verbose=0)
    predicted_price = float(predicted_y[0, 0])

    predictions.append(predicted_price)
    test_input.append(predicted_price)

    print(f"Step {i + 1}/{len(test_seq)} - Prediction {predicted_price}")

# Inverse transform predictions and actual test values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
print('predictions', predictions)
actual = scaler.inverse_transform(test_seq.reshape(-1, 1)).flatten().reshape(1,-1).flatten()
mae = util.calculateMAE(actual, predictions)

util.graph_comparison('S&P 500 LSTM Recursive Prediction', price_series, mae, price_series,
                      predictions, split_idx)
