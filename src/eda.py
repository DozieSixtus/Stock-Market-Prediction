import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
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

for data_name, data in training_data.items():
    print(f"Ploting chart for {data_name}...")

    # Convert Date to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Set Date as the index
    data.set_index('Date', inplace=True)

    # Plotting the line chart for price trends
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Open'], label='Open Price', alpha=0.6)
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.6)
    plt.plot(data.index, data['High'], label='High Price', alpha=0.6)
    plt.plot(data.index, data['Low'], label='Low Price', alpha=0.6)
    plt.plot(data.index, data['Adj Close'], label='Adjusted Close Price', alpha=0.6)
    plt.title(f'Price Trends Over Time ({data_name})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'.\EDA visuals\{data_name}_trend_chart.png')

    # Calculate the correlation matrix for BP dataset
    correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
    plt.title(f'Correlation Matrix ({data_name})')
    plt.savefig(f'.\EDA visuals\{data_name}_correlation_matrix.png')


    # Plotting the histogram for Adj Close
    plt.figure(figsize=(12, 8))
    plt.hist(data['Adj Close'].dropna(), bins=50, alpha=0.75, color='green', edgecolor='black')
    plt.title(f'Distribution of Adj Close ({data_name})')
    plt.xlabel('Adj Close')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'.\EDA visuals\{data_name}_histogram.png')