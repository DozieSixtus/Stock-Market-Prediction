import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_predictions(data, train_data, test_data, model_name):
    data = data.copy()
    
    # Convert Date to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Set Date as the index
    data.set_index('Date', inplace=True)

    # Plotting the line chart for price trends
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['y_pred'], label='y_pred', alpha=0.6)
    plt.plot(data.index, data['y_true'], label='y_true', alpha=0.6)
    plt.title(f'Price predictions for {test_data} using {model_name} trained on {train_data}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'.\EDA visuals\pred_{test_data}_{model_name}_{train_data}.png')