import numpy as np
import pandas as pd

from prep import FeaturePreProcessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

for train_name, train in training_data.items():
    print("="*50)
    print(f"Training on {train_name} ...")
    print("="*50)
    train = train.copy()
    train = feature_eng(train, lag_days=lag_days)

    # Replace nan values with 0
    train = train.fillna(0)


    train_y = train['Close']
    train_x = train.drop(columns=['Close','Date','year','High','Low','Adj Close'])

    # train test split
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, shuffle=True, test_size=0.2, random_state=42)
    #train_x = train_x.sample(random_state=42, ignore_index=True)

    rf_regr = RandomForestRegressor(n_estimators=100, random_state=0, verbose=0)
    rf_regr.fit(train_x, train_y)

    pred = np.abs(rf_regr.predict(test_x))
    error = root_mean_squared_error(test_y, pred)
    print("Inferencing with Random Forest model ...")
    print(f"The root mean squared error on the test data is {error}.")

    xgb_regr = GradientBoostingRegressor(n_estimators=100, random_state=0, verbose=0)
    xgb_regr.fit(train_x, train_y)

    pred = np.abs(xgb_regr.predict(test_x))
    error = root_mean_squared_error(test_y, pred)
    print("Inferencing with Gradient Boosting model ...")
    print(f"The root mean squared error on the test data is {error}.")

    linear_regr = LinearRegression()
    linear_regr.fit(train_x, train_y)

    pred = np.abs(linear_regr.predict(test_x))
    error = root_mean_squared_error(test_y, pred)
    print("Inferencing with Linear Regression model ...")
    print(f"The root mean squared error on the test data is {error}.")

    # Test models on test data
    print(f"{'*'*50} \nTesting on test datasets\n{'*'*50}")

    for test_name, test in test_data.items():
        print("-"*50)
        print(f"Testing on {test_name}")

        test = test.copy()
        test = feature_eng(test, lag_days=lag_days)

        # Replace nan values with 0
        test = test.fillna(0)

        test_y = test['Close']
        test_x = test.drop(columns=['Close','Date','year','High','Low','Adj Close'])

        pred = np.abs(rf_regr.predict(test_x))
        error = root_mean_squared_error(test_y, pred)
        print("Inferencing with Random Forest model ...")
        print(f"The root mean squared error on the test data is {error}.")

        pred = np.abs(xgb_regr.predict(test_x))
        error = root_mean_squared_error(test_y, pred)
        print("Inferencing with Gradient Boosting model ...")
        print(f"The root mean squared error on the test data is {error}.")

        pred = np.abs(linear_regr.predict(test_x))
        error = root_mean_squared_error(test_y, pred)
        print("Inferencing with Linear Regression model ...")
        print(f"The root mean squared error on the test data is {error}.")