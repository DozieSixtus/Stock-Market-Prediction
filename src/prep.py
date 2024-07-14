import pandas as pd
import numpy as np

class FeaturePreProcessing():
    def __init__(self):
        self.stat_params = ['min', 'max', 'mean', 'std']
        self.Date_cols = ['month','day','hour']

    def get_date_components(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        #df['date'] = df['Date'].dt.date
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['day_of_year'] = df['Date'].dt.day_of_year
        #df['hour'] = df['Date'].dt.hour
        return df
        
    # Compute range of columns in dataframe
    def column_range(self, df, groupby_cols=[], grad_cols=[]):
        df_grad = df.groupby(groupby_cols)[grad_cols].agg(np.ptp)
        df_grad = df_grad.rename_axis(groupby_cols).reset_index()
        df_grad.rename(columns={k: k+'_range' for k in grad_cols}, inplace=True)
        #df = df.merge(df_grad, on=groupby_cols, how='left')
        return df_grad
    
    # Calculate statistics parameters 
    def data_stats(self, df, stats_cols, groupby_cols=None):
        if 'Date' in list(df.columns):
            sorted_date = sorted(df['Date'].unique())
        elif 'date' in list(df.columns):
            sorted_date = sorted(df['date'].unique())
        
        data_stats = df.groupby(groupby_cols).agg({attr : self.stat_params for attr in stats_cols})
        data_stats = data_stats.reset_index()
        data_stats.columns = [' '.join(col).strip() for col in data_stats.columns.values]
        return data_stats        
    
    # Lag train data
    def lag_train_data(self, df, target,lag_days):    
        for i in range(1,lag_days+1):
            df_lag = df[target+['Date']]
            #df_lag['Date'] = df_lag['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f'))
            df_lag = df_lag.sort_values(['Date']).shift(i)
            df_lag.drop(columns='Date', inplace=True)
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

        # Calculate range from columns
        self.train_col_range = self.column_range(self.train_data, groupby_cols=['year'],
                                                 grad_cols=['Open','High','Low','Close','Adj Close','Volume'])
        
        # Calculate stats from fields
        self.train_stat = self.data_stats(self.train_data, 
                                          stats_cols=['Open','High','Low','Close','Adj Close','Volume'],
                                          groupby_cols=['year'])
        
        # Lag train data by n days
        self.train_lagged = self.lag_train_data(self.train_data, 
                                                target=['Open','High','Low','Close','Adj Close','Volume'],
                                                lag_days=self.lag_days)
        
        return self.train_lagged
    