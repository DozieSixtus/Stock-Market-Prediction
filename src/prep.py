import pandas as pd
import numpy as np

class FeaturePreProcessing():
    def __init__(self):
        self.stat_params = ['min', 'max', 'mean', 'std']
        self.Date_cols = ['month','day','hour']

    # Get date components
    def get_date_components(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        #df['date'] = df['Date'].dt.date
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['day_of_year'] = df['Date'].dt.day_of_year
        #df['hour'] = df['Date'].dt.hour
        return df
    
    # Lag train data
    def lag_train_data(self, df, target,lag_days):    
        for i in range(1,lag_days+1):
            df_lag = df[target+['Date']]
            #df_lag['Date'] = df_lag['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f'))
            df_lag = df_lag.sort_values(['Date']).shift(i)
            df_lag.drop(columns='Date', inplace=True)

            # join lag table to original table
            if i==1:
                df_merged = df.join(df_lag.rename(columns=lambda x: x+f"_lag_{i}"))
            else:
                df_merged = df_merged.join(df_lag.rename(columns=lambda x: x+f"_lag_{i}"))
        #lagged_target = pd.concat([df[target].shift(), df[target].shift(2)], axis=1)
        return df_merged
    
    def __call__(self, train_data, lag_days=10):
        self.train_data = train_data.copy()
        self.lag_days = lag_days
        
        # Get date components
        self.train_data = self.get_date_components(self.train_data)
        
        # Lag train data by n days
        self.train_lagged = self.lag_train_data(self.train_data, 
                                                target=['Adj Close'],
                                                lag_days=self.lag_days)
        
        return self.train_lagged
    