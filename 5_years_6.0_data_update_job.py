import os
import glob
import json
import datetime
import pandas as pd
import yfinance as yf
import subprocess

print("--- STEP 1: DOWNLOADING MISSING DATA ---")
metadata_file = 'dataset_metadata.json'
if not os.path.exists(metadata_file):
    print(f"Error: {metadata_file} not found! Please ensure it is generated.")
else:
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    last_date_str = metadata.get('last_updated', metadata.get('global_last_date', '2000-01-01'))
    last_date = datetime.datetime.strptime(last_date_str, '%Y-%m-%d').date()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    
    if last_date < yesterday:
        print(f"Data is out of date! Last date in dataset: {last_date}")
        print(f"Downloading ALL missing data from {last_date + datetime.timedelta(days=1)} to {yesterday}...")
        
        start_d = last_date + datetime.timedelta(days=1)
        end_d = datetime.date.today() # yfinance end date is exclusive
        
        # Download Nifty 50 delta
        nifty50 = yf.download("^NSEI", start=start_d, end=end_d, auto_adjust=False, progress=False)
        if isinstance(nifty50.columns, pd.MultiIndex): nifty50.columns = nifty50.columns.droplevel(1)
        
        nifty_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in nifty_cols:
            if c not in nifty50.columns: nifty50[c] = 0
        nifty50 = nifty50[nifty_cols]
        nifty50.rename(columns={'Open': 'Nifty50_Open', 'High': 'Nifty50_High', 'Low': 'Nifty50_Low', 'Close': 'Nifty50_Close', 'Volume': 'Nifty50_Volume'}, inplace=True)
        
        raw_dir = '5_year_data/raw'
        files = glob.glob(os.path.join(raw_dir, '*_data.csv'))
        
        updated_max_date = last_date_str
        
        for f_path in files:
            symbol = os.path.basename(f_path).replace('_data.csv', '')
            try:
                df_new = yf.download(f"{symbol}.NS", start=start_d, end=end_d, auto_adjust=False, progress=False)
                if df_new.empty: continue
                if isinstance(df_new.columns, pd.MultiIndex): df_new.columns = df_new.columns.droplevel(1)
                
                df_new['Stock Symbol'] = symbol
                
                df_old = pd.read_csv(f_path, nrows=1)
                sector = df_old['Sector'].iloc[0]
                sector_idx = df_old['Sector Index'].iloc[0]
                
                df_new['Sector'] = sector
                df_new['Sector Index'] = sector_idx
                df_new['Trading_Day_Of_Week'] = df_new.index.day_name()
                if 'Adj Close' not in df_new.columns: df_new['Adj Close'] = df_new['Close'] if 'Close' in df_new.columns else None
                
                df_new = df_new.join(nifty50, how='left')
                
                cols_needed = ['Stock Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sector', 'Sector Index', 'Nifty50_Open', 'Nifty50_High', 'Nifty50_Low', 'Nifty50_Close', 'Nifty50_Volume', 'Trading_Day_Of_Week']
                for c in cols_needed:
                    if c not in df_new.columns: df_new[c] = None
                df_new = df_new[cols_needed]
                
                df_new.index.name = 'Date'
                df_new.to_csv(f_path, mode='a', header=False)
                print(f"Updated {symbol}")
                
                new_max = df_new.index.max().strftime('%Y-%m-%d')
                if 'symbols' in metadata:
                    if symbol in metadata['symbols']:
                        metadata['symbols'][symbol]['last_date'] = new_max
                    else:
                        metadata['symbols'][symbol] = {'last_date': new_max}
                
                if new_max > updated_max_date:
                    updated_max_date = new_max
                    
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
        
        metadata['last_updated'] = updated_max_date
        if 'global_last_date' in metadata: metadata['global_last_date'] = updated_max_date
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"\n✅ All raw data updated! JSON Log updated to {updated_max_date}.")
    else:
        print("✅ Raw Data is already up to date! No downloading needed.")

print("\n--- STEP 2: CALCULATING FEATURES (Notebook 2) ---")
subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", "5_years_2_feature_generation.ipynb"], check=True)

print("\n--- STEP 3: CLEANING DATA (Notebook 3) ---")
subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", "5_years_3_data_cleaning.ipynb"], check=True)

print("\n✅ DATA UPDATE JOB COMPLETE!")
