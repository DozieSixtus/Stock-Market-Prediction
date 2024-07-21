import pandas as pd
import numpy as np

import tensorflow as tf

from prep import FeaturePreProcessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Read training stock data files
bp_data = pd.read_excel(r'.\data\training\BP.L_filtered.xlsx')
dge_data = pd.read_excel(r'.\data\training\DGE.L_filtered.xlsx')
gsk_data = pd.read_excel(r'.\data\training\GSK.L_filtered.xlsx')
hsba_data = pd.read_excel(r'.\data\training\HSBA.L_filtered.xlsx')
ulvr_data = pd.read_excel(r'.\data\training\ULVR.L_filtered.xlsx')

# Read test stock data files
azn_data = pd.read_excel(r'.\data\testing\AZN.L_filtered.xlsx')
barc_data = pd.read_excel(r'.\data\testing\BARC.L_filtered.xlsx')
rr_data = pd.read_excel(r'.\data\testing\RR.L_filtered.xlsx')
tsco_data = pd.read_excel(r'.\data\testing\TSCO.L_filtered.xlsx')
vod_data = pd.read_excel(r'.\data\testing\VOD.L_filtered.xlsx')

feature_eng = FeaturePreProcessing()
lag_days = 5

bp_data = feature_eng(bp_data, lag_days=lag_days)

# Replace nan values with 0
bp_data = bp_data.fillna(0)


bp_y = bp_data['Close']
bp_x = bp_data.drop(columns=['Close','Date','year','High','Low','Adj Close'])

# train test split
bp_train_x, bp_test_x, bp_train_y, bp_test_y = train_test_split(bp_x, bp_y, shuffle=True, test_size=0.2, random_state=42)
#bp_train_x = bp_train_x.sample(random_state=42, ignore_index=True)

def nn_model():
    input = tf.keras.Input(shape=(None,1), name="input_data")
    x = tf.keras.layers.LSTM(256, activation="tanh", return_sequences = True)(input)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences = True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.MaxPooling2D(3)(x)
    x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences = True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #output = tf.keras.layers.GlobalMaxPooling2D()(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(input, output, name="lstm_model")
    model.summary()
    return model

model = nn_model()
tf.keras.utils.plot_model(model, "model_architecture.png", show_shapes=True)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.MeanSquaredError()],
)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, 
                                   verbose=0, mode='min', start_from_epoch=50, restore_best_weights=True)

history = model.fit(bp_train_x, bp_train_y, batch_size=32, callbacks=[es], epochs=200, validation_split=0.2)

test_scores = model.evaluate(bp_test_x, bp_test_y, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])