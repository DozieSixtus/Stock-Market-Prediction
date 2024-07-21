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

rf_regr = RandomForestRegressor(n_estimators=100, random_state=0, verbose=0)
rf_regr.fit(bp_train_x, bp_train_y)

bp_pred = np.abs(rf_regr.predict(bp_test_x))
error = root_mean_squared_error(bp_test_y, bp_pred)
print("Inferencing with Random Forest model ...")
print(f"The root mean squared error on the test data is {error}.")

xgb_regr = GradientBoostingRegressor(n_estimators=100, random_state=0, verbose=0)
xgb_regr.fit(bp_train_x, bp_train_y)

bp_pred = np.abs(xgb_regr.predict(bp_test_x))
error = root_mean_squared_error(bp_test_y, bp_pred)
print("Inferencing with Gradient Boosting model ...")
print(f"The root mean squared error on the test data is {error}.")

linear_regr = LinearRegression()
linear_regr.fit(bp_train_x, bp_train_y)

bp_pred = np.abs(linear_regr.predict(bp_test_x))
error = root_mean_squared_error(bp_test_y, bp_pred)
print("Inferencing with Linear Regression model ...")
print(f"The root mean squared error on the test data is {error}.")

# Test models on test data
print(f"{'*'*50} \nTesting on test datasets\n{'*'*50}")
azn_data = feature_eng(azn_data, lag_days=lag_days)

# Replace nan values with 0
azn_data = azn_data.fillna(0)

azn_y = azn_data['Close']
azn_test = azn_data.drop(columns=['Close','Date','year','High','Low','Adj Close'])

azn_pred = np.abs(rf_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Random Forest model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(xgb_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Gradient Boosting model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(linear_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Linear Regression model ...")
print(f"The root mean squared error on the test data is {error}.")

print(f"{'*'*50} \nTesting on barclays datasets\n{'*'*50}")
azn_data = feature_eng(barc_data, lag_days=lag_days)

# Replace nan values with 0
azn_data = azn_data.fillna(0)

azn_y = azn_data['Close']
azn_test = azn_data.drop(columns=['Close','Date','year','High','Low','Adj Close'])

azn_pred = np.abs(rf_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Random Forest model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(xgb_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Gradient Boosting model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(linear_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Linear Regression model ...")
print(f"The root mean squared error on the test data is {error}.")

print(f"{'*'*50} \nTesting on rr datasets\n{'*'*50}")
azn_data = feature_eng(rr_data, lag_days=lag_days)

# Replace nan values with 0
azn_data = azn_data.fillna(0)

azn_y = azn_data['Close']
azn_test = azn_data.drop(columns=['Close','Date','year','High','Low','Adj Close'])

azn_pred = np.abs(rf_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Random Forest model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(xgb_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Gradient Boosting model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(linear_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Linear Regression model ...")
print(f"The root mean squared error on the test data is {error}.")

print(f"{'*'*50} \nTesting on tsco datasets\n{'*'*50}")
azn_data = feature_eng(tsco_data, lag_days=lag_days)

# Replace nan values with 0
azn_data = azn_data.fillna(0)

azn_y = azn_data['Close']
azn_test = azn_data.drop(columns=['Close','Date','year','High','Low','Adj Close'])

azn_pred = np.abs(rf_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Random Forest model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(xgb_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Gradient Boosting model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(linear_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Linear Regression model ...")
print(f"The root mean squared error on the test data is {error}.")

print(f"{'*'*50} \nTesting on vod datasets\n{'*'*50}")
azn_data = feature_eng(vod_data, lag_days=lag_days)

# Replace nan values with 0
azn_data = azn_data.fillna(0)

azn_y = azn_data['Close']
azn_test = azn_data.drop(columns=['Close','Date','year','High','Low','Adj Close'])

azn_pred = np.abs(rf_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Random Forest model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(xgb_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Gradient Boosting model ...")
print(f"The root mean squared error on the test data is {error}.")

azn_pred = np.abs(linear_regr.predict(azn_test))
error = root_mean_squared_error(azn_y, azn_pred)
print("Inferencing with Linear Regression model ...")
print(f"The root mean squared error on the test data is {error}.")