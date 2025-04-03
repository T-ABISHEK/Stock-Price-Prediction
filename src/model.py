import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(input_shape):
    """Creates and returns an LSTM model."""
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
