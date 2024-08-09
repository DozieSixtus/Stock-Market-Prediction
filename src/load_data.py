import yfinance as yf
import pandas as pd
import os

# Specify directories for saving data
training_dir = os.getcwd()+'\\data\\training'
testing_dir = os.getcwd()+'\\data\\testing'

# Create folders if they don't exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# List of stocks
training_stocks = ['BP.L', 'ULVR.L', 'GSK.L', 'HSBA.L', 'DGE.L']
testing_stocks = ['AZN.L', 'VOD.L', 'BARC.L', 'RR.L', 'TSCO.L']

# Function to download and save data to Excel
def download_and_save_data_to_excel(stock_list, start_date, end_date, folder_name):
    for stock in stock_list:
        print(f"Downloading data for {stock}")
        df = yf.download(stock, start=start_date, end=end_date)

        # Save to Excel file
        file_name = f"{folder_name}\{stock}.xlsx"
        df.to_excel(file_name, sheet_name=stock)

        print(f"Data for {stock} saved to {file_name}")

# Download data for training and testing and save to Excel
download_and_save_data_to_excel(training_stocks, '2014-01-01', '2023-12-31', training_dir)
download_and_save_data_to_excel(testing_stocks, '2024-01-01', '2024-03-31', testing_dir)

print("All data downloaded and saved to Excel files.")

# Load all the datasets
dge_file_path = r'.\data\training\DGE.L.xlsx'
bp_file_path = r'.\data\training\BP.L.xlsx'
gsk_file_path = r'.\data\training\GSK.L.xlsx'
ulvr_file_path = r'.\data\training\ULVR.L.xlsx'
hsba_file_path = r'.\data\training\HSBA.L.xlsx'

# Read the first sheet of each dataset
dge_data = pd.ExcelFile(dge_file_path)
bp_data = pd.ExcelFile(bp_file_path)
gsk_data = pd.ExcelFile(gsk_file_path)
ulvr_data = pd.ExcelFile(ulvr_file_path)
hsba_data = pd.ExcelFile(hsba_file_path)

dge_df = dge_data.parse(dge_data.sheet_names[0])
bp_df = bp_data.parse(bp_data.sheet_names[0])
gsk_df = gsk_data.parse(gsk_data.sheet_names[0])
ulvr_df = ulvr_data.parse(ulvr_data.sheet_names[0])
hsba_df = hsba_data.parse(hsba_data.sheet_names[0])

# Convert 'Date' columns to datetime for all datasets
dge_df['Date'] = pd.to_datetime(dge_df['Date'], errors='coerce')
bp_df['Date'] = pd.to_datetime(bp_df['Date'], errors='coerce')
gsk_df['Date'] = pd.to_datetime(gsk_df['Date'], errors='coerce')
ulvr_df['Date'] = pd.to_datetime(ulvr_df['Date'], errors='coerce')
hsba_df['Date'] = pd.to_datetime(hsba_df['Date'], errors='coerce')

# Find common dates
common_dates = set(dge_df['Date']).intersection(bp_df['Date'], gsk_df['Date'], ulvr_df['Date'], hsba_df['Date'])

# Filter each dataset by common dates
dge_filtered = dge_df[dge_df['Date'].isin(common_dates)]
bp_filtered = bp_df[bp_df['Date'].isin(common_dates)]
gsk_filtered = gsk_df[gsk_df['Date'].isin(common_dates)]
ulvr_filtered = ulvr_df[ulvr_df['Date'].isin(common_dates)]
hsba_filtered = hsba_df[hsba_df['Date'].isin(common_dates)]

# Save the filtered datasets
filtered_dfs = {
    'DGE.L_filtered.xlsx': dge_filtered,
    'BP.L_filtered.xlsx': bp_filtered,
    'GSK.L_filtered.xlsx': gsk_filtered,
    'ULVR.L_filtered.xlsx': ulvr_filtered,
    'HSBA.L_filtered.xlsx': hsba_filtered
}

# Save to new Excel files in Colab directory
for file_name, df in filtered_dfs.items():
    df.to_excel(fr'.\data\training\{file_name}', index=False)
    print(f'Saved {file_name}.')


# Load the data from each Excel file
azn_path = r'.\data\testing\AZN.L.xlsx'
vod_path = r'.\data\testing\VOD.L.xlsx'
barc_path = r'.\data\testing\BARC.L.xlsx'
rr_path = r'.\data\testing\RR.L.xlsx'
tsco_path = r'.\data\testing\TSCO.L.xlsx'

# Load data from Excel files
azn_df = pd.read_excel(azn_path)
vod_df = pd.read_excel(vod_path)
barc_df = pd.read_excel(barc_path)
rr_df = pd.read_excel(rr_path)
tsco_df = pd.read_excel(tsco_path)

# Display first few rows of each dataframe to understand the structure
azn_df.head(), vod_df.head(), barc_df.head(), rr_df.head(), tsco_df.head()

# Convert the 'Date' column to datetime for each DataFrame
azn_df['Date'] = pd.to_datetime(azn_df['Date'])
vod_df['Date'] = pd.to_datetime(vod_df['Date'])
barc_df['Date'] = pd.to_datetime(barc_df['Date'])
rr_df['Date'] = pd.to_datetime(rr_df['Date'])
tsco_df['Date'] = pd.to_datetime(tsco_df['Date'])

# Identify common dates
common_dates = set(azn_df['Date']) & set(vod_df['Date']) & set(barc_df['Date']) & set(rr_df['Date']) & set(tsco_df['Date'])

# Convert common_dates to list and sort it
common_dates = sorted(list(common_dates))

# Filter each DataFrame to include only the common dates
azn_filtered = azn_df[azn_df['Date'].isin(common_dates)].sort_values(by='Date').reset_index(drop=True)
vod_filtered = vod_df[vod_df['Date'].isin(common_dates)].sort_values(by='Date').reset_index(drop=True)
barc_filtered = barc_df[barc_df['Date'].isin(common_dates)].sort_values(by='Date').reset_index(drop=True)
rr_filtered = rr_df[rr_df['Date'].isin(common_dates)].sort_values(by='Date').reset_index(drop=True)
tsco_filtered = tsco_df[tsco_df['Date'].isin(common_dates)].sort_values(by='Date').reset_index(drop=True)

# Save the filtered data back to Excel files in Colab directory
azn_filtered_path = '.\\data\\testing\\AZN.L_filtered.xlsx'
vod_filtered_path = '.\\data\\testing\\VOD.L_filtered.xlsx'
barc_filtered_path = '.\\data\\testing\\BARC.L_filtered.xlsx'
rr_filtered_path = '.\\data\\testing\\RR.L_filtered.xlsx'
tsco_filtered_path = '.\\data\\testing\\TSCO.L_filtered.xlsx'

azn_filtered.to_excel(azn_filtered_path, index=False)
vod_filtered.to_excel(vod_filtered_path, index=False)
barc_filtered.to_excel(barc_filtered_path, index=False)
rr_filtered.to_excel(rr_filtered_path, index=False)
tsco_filtered.to_excel(tsco_filtered_path, index=False)