import pandas as pd
import numpy as np
import tensorflow as tf

def nn_model():
    input = tf.keras.Input(shape=(None,1), name="input_data")
    x = tf.keras.layers.LSTM(256, activation="tanh", return_sequences = True)(input)
    x = tf.keras.layers.LSTM(256, activation="tanh", return_sequences = True)(x)
    x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences = True)(x)
    x = tf.keras.layers.LSTM(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(input, output, name="lstm_model")
    # model.summary()
    return model