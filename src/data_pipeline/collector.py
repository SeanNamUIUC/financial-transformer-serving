import os
import sqlite3
import yaml
import yfinance as yf
import pandas as pd

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:#read config file with utf-8 
        return yaml.safe_load(f)#text to dictionary structure

def create_market_table(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok = True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE if not exists daily_stock_prices (
            date TEXT,
            ticker TEXT, --stock code
            open REAL,  --opening price
            high REAL, --highest price
            low REAL, --lowest price
            close REAL, --closing price
            adj_close REAL, --dividened or stock split
            volume INTEGER, --total number of stocks
            PRIMARY KEY (date, ticker) --standard for no duplication 
        
        
        )

    ''')
    
    conn.commit()
    conn.close()

    print("finished db initialization")

def collect_store_data(config):
    tickers = config["data_pipeline"]["tickers"]
    start_date = config["data_pipeline"]["start_date"]
    end_date = config["data_pipeline"]["end_date"]
    db_path = config["data_pipeline"]['db_path']

    create_market_table(db_path)
    conn = sqlite3.connect(db_path)
    
    for ticker in tickers:
        print(f"Collecting data {ticker}... ({start_date} ~ {end_date})")

        try:
            df = yf.download(ticker, start = start_date , end = end_date)
            # print(df.columns.tolist()) # Close, High, Low, Open, Volume 다섯개의 특징들로 나옴. 
            # print(f"Columns: {df.columns}") 
            # print(f"df shape: {df.shape}")
            # print(df.head(2)) 
            # print(df.index)
            # print(df.columns)
            
            if df.empty:
                continue
            #if multiIndex -> only take level0 index and use for columns else -> pass
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            print(df.columns)

            clean_df = pd.DataFrame()
            clean_df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            clean_df['ticker'] = ticker
            clean_df['open'] = df['Open'].astype(float)
            clean_df['high'] = df['High'].astype(float)
            clean_df['low'] = df['Low'].astype(float)
            clean_df['close'] = df['Close'].astype(float)
            clean_df['adj_close'] = df['Close'].astype(float)
            clean_df['volume'] = df['Volume'].astype(int)

            # push into db (duplicate data: replace or ignore)
            clean_df.to_sql("daily_stock_prices", conn, if_exists="append", index=False)
            print("------------------------------------------------------------")
            print(f" {ticker} saved: length:{len(clean_df)} ")
            
        except Exception as e:
            print(f"failed to collect{ticker} : {e}")
            
    conn.close()
    print("Finished data collection pipeline")

# To handle two cases:
# 1. Executed directly as a standalone script (python collector.py)
# 2. Imported as a module/dependency inside main.py (avoids auto-execution)
if __name__ == "__main__": 
    # testing code in codespaces
    config = load_config()
    collect_store_data(config)


