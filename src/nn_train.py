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

training_data = {'bp': bp_data
                 ,'dge': dge_data
                 ,'gsk': gsk_data
                 ,'hsba': hsba_data
                 ,'ulvr': ulvr_data
                }

# Read test stock data files
azn_data = pd.read_excel(r'.\data\testing\AZN.L_filtered.xlsx')
barc_data = pd.read_excel(r'.\data\testing\BARC.L_filtered.xlsx')
rr_data = pd.read_excel(r'.\data\testing\RR.L_filtered.xlsx')
tsco_data = pd.read_excel(r'.\data\testing\TSCO.L_filtered.xlsx')
vod_data = pd.read_excel(r'.\data\testing\VOD.L_filtered.xlsx')

test_data = {'azn': azn_data
             ,'barc': barc_data
             ,'rr': rr_data
             ,'tsco': tsco_data
             ,'vod': vod_data
            }

feature_eng = FeaturePreProcessing()
lag_days = 5

def nn_model():
    input = tf.keras.Input(shape=(None,1), name="input_data")
    x = tf.keras.layers.LSTM(256, activation="tanh", return_sequences = True)(input)
    x = tf.keras.layers.LSTM(256, activation="tanh", return_sequences = True)(x)
    #x = tf.keras.layers.MaxPooling2D(3)(x)
    x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences = True)(x)
    x = tf.keras.layers.LSTM(16, activation="relu")(x)
    #output = tf.keras.layers.GlobalMaxPooling2D()(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(input, output, name="lstm_model")
    model.summary()
    return model

for train_name, train in training_data.items():
    print(f"Training on {train_name} ...")
    train = train.copy()
    train = feature_eng(train, lag_days=lag_days)

    # Replace nan values with 0
    train = train.fillna(0)


    train_y = train['Close']
    train_x = train.drop(columns=['Close','Date','year','High','Low','Adj Close'])

    # train test split
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, shuffle=True, test_size=0.2, random_state=42)
    #train_x = train_x.sample(random_state=42, ignore_index=True)
    model = nn_model()
    tf.keras.utils.plot_model(model, "model_architecture.png", show_shapes=True)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['MAE'],
    )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, 
                                    verbose=0, mode='min', start_from_epoch=100, restore_best_weights=True)

    history = model.fit(train_x, train_y, batch_size=64, callbacks=[es], epochs=200, validation_split=0.2)

    test_scores = model.evaluate(test_x, test_y, verbose=2)
    print("Test mean squared error:", test_scores[0])
    print("Test mean absolute error:", test_scores[1])

    # Test models on test data
    print(f"{'*'*50} \nTesting on test datasets\n{'*'*50}")
    for test_name, test in test_data.items():
        print(f"Testing on {test_name}")

        test = test.copy()
        test = feature_eng(test, lag_days=lag_days)

        # Replace nan values with 0
        test = test.fillna(0)

        test_y = test['Close']
        test_x = test.drop(columns=['Close','Date','year','High','Low','Adj Close'])

        test_scores = model.evaluate(test_x, test_y, verbose=2)
        print("Test mean squared error:", test_scores[0])
        print("Test mean absolute error:", test_scores[1])