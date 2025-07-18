import numpy as np

import pandas as pd
from pandas import DataFrame, Series

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

base_path = Path(__file__).parent
CONFIG = {
    'data_url': base_path / 'data/sp500_prices.csv',
    'date_col': 'Date',
    'price_col': 'Price'
}

def getDataFrame() -> DataFrame:
    dataset = pd.read_csv(CONFIG['data_url'], parse_dates=[CONFIG['date_col']], index_col=CONFIG['date_col']).sort_index()
    return dataset


def getPrice(dataframe: DataFrame) -> Series:
    return dataframe[CONFIG['price_col']]


def calculateMAE(y_true, y_pred):
    """
    Calculate the Mean Absolute Error between true and predicted values.
    
    Args:
        y_true: Array of actual/true values
        y_pred: Array of predicted values
        
    Returns:
        float: Mean Absolute Error value
        
    Raises:
        ValueError: If arrays are not 1-dimensional or have different lengths
    """
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Both arrays must be 1-dimensional.")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays must have the same length (y_true: {len(y_true)}, y_pred: {len(y_pred)})")

    return np.mean(np.abs(y_true - y_pred))


def split_sequence(sequence, window_size, horizon):
    """
    Split a time series sequence into input-output pairs for training.

    Args:
        sequence: List or array of sequential data points
        window_size: Number of timesteps to use as input features
        horizon: Number of timesteps to predict as output

    Returns:
        input_sequences: Array of input sequences, each of length window_size
        output_sequences: Array of corresponding output sequences, each of length horizon
    """
    
    X, y = [], []

    for start_index in range(len(sequence)):
        end_index = start_index + window_size
        output_end_index = end_index + horizon

        if output_end_index > len(sequence):
            break

        input_sequence = sequence[start_index:end_index]
        output_sequence = sequence[end_index:output_end_index]

        X.append(input_sequence)
        y.append(output_sequence)

    return np.array(X), np.array(y)


def split_train_test(raw_sequence, horizon):
    """
    Split a time series sequence into training and test sets.
    
    Args:
        raw_sequence: Complete time series sequence
        horizon: Number of timesteps to reserve for testing
        
    Returns:
        tuple: (train_seq, test_seq, split_idx) where split_idx is the index where split occurs
    """

    split_idx = len(raw_sequence) - horizon
    train_seq = raw_sequence[:split_idx]
    test_seq = raw_sequence[split_idx:]
    return train_seq, test_seq, split_idx


def graph_comparison(title, dataset, mae, original, predictions, split_idx):
    """
    Create visualization comparing original vs predicted values with two plots: full view and zoomed view.
    
    Args:
        title: Title for the plots
        dataset: DataFrame containing the original data with date index
        mae: Mean Absolute Error value to display on plots
        original: Array of original/true values
        predictions: Array of predicted values
        split_idx: Index where training/test split occurs
        
    Returns:
        None: Displays two matplotlib plots
    """
        
    date_index = dataset.index

    plt.figure(figsize=(15, 5))

    plt.plot(date_index, original, label='Original Price')

    test_dates = date_index[split_idx:]
    plt.plot(test_dates, predictions, label='Predicted Price')

    plt.axvline(x=date_index[split_idx], color='r', linestyle='--', label='Train/Test Split')

    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(title)

    start_date = date_index[len(date_index) - 3000]
    end_date = date_index[-1]
    plt.xlim(start_date, end_date)

    # Format x-axis to show only years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.text(0.02, 0.95, f'MAE: {mae:.2f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9),
             verticalalignment='top', fontsize=10, color='black')

    plt.legend()
    plt.show()

    # === ZOOMED IN PLOT ===

    plt.figure(figsize=(15, 5))

    plt.plot(date_index, original, label='Original Price')
    plt.plot(test_dates, predictions, label='Predicted Price')

    plt.axvline(x=date_index[split_idx], color='r', linestyle='--', label='Train/Test Split')

    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.title(title)

    start_date = date_index[len(date_index) - 1000]
    end_date = date_index[-1]
    plt.xlim(start_date, end_date)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.text(0.02, 0.95, f'MAE: {mae:.2f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9),
             verticalalignment='top', fontsize=10, color='black')

    plt.legend()
    plt.show()
