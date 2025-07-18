import numpy as np

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from benchstreet import util

dataset = util.getDataFrame()
price_series = util.getPrice(dataset)

window_size = 10
horizon = 1
epochs = 20
n_features = 1

# normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_sequence = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
# not a multi-step predictor so horizon must be custom set
train_seq, test_seq, split_idx = util.split_train_test(scaled_sequence, 730)

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

predictions = []

test_input = train_seq[-window_size:].tolist()

for i in range(len(test_seq)):
    x_input = np.array(test_input[-window_size:]).reshape((1, window_size, 1))
    predicted_y = model.predict(x_input, verbose=0)
    predicted_price = float(predicted_y[0, 0])

    predictions.append(predicted_price)
    test_input.append(predicted_price)

    print(f"Step {i + 1}/{len(test_seq)} - Prediction {predicted_price}")

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# SCALE ADJUSTMENT - fix the offset
last_train_value = price_series.iloc[split_idx-1]
first_pred_value = predictions[0]
offset = last_train_value - first_pred_value
predictions_corrected = predictions + offset

actual = scaler.inverse_transform(test_seq.reshape(-1, 1)).flatten().reshape(1,-1).flatten()
mae = util.calculateMAE(actual, predictions_corrected)

util.graph_comparison('S&P 500 GRU Recursive Prediction', price_series, mae, price_series,
                      predictions_corrected, split_idx)
