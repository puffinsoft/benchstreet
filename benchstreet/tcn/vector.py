import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN

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
model.add(TCN(
    nb_filters=256,
    kernel_size=3,
    nb_stacks=1,
    dilations=[1, 2, 4, 8, 16, 32],
    padding='causal',
    use_skip_connections=True,
    dropout_rate=0.2,
    return_sequences=False,
    input_shape=(window_size, n_features)
))
model.add(Dense(horizon, activation='linear'))

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# fit model
model.fit(X, y, epochs=epochs, verbose=1, validation_split=0.1)

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

util.graph_comparison('S&P 500 TCN Multi-Step (Vector) Prediction', price_series, mae, price_series,
                      predictions_corrected, split_idx)
