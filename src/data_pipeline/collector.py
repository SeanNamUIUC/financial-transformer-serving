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
            df.columns = [col.lower() for col in df.columns]
            print(df.columns)

            clean_df = pd.DataFrame()
            clean_df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            clean_df['ticker'] = ticker
            clean_df['open'] = df['open'].astype(float)
            clean_df['high'] = df['high'].astype(float)
            clean_df['low'] = df['low'].astype(float)
            clean_df['close'] = df['close'].astype(float)
            clean_df['adj_close'] = df['close'].astype(float) # no adj close in current yfinance data
            clean_df['volume'] = df['volume'].astype(int)

            #data update
            cursor = conn.cursor()
            
            #upsert query
            upsert_query = '''
                INSERT OR REPLACE INTO daily_stock_prices (date, ticker, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            #data frame to data tuples -> ex) [('2016-01-04', 'TSLA' , 180.0 .......)]
            data_tuples = list(clean_df.itertuples(index=False, name=None))
            
            # 3.(Bulk Insert)
            cursor.executemany(upsert_query, data_tuples)
            conn.commit()
            # ==================================================================================
            
            print("------------------------------------------------------------")
            print(f" {ticker} saved/upserted finished: length: {len(clean_df)} ")
            
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
    
    print("\n" + "="*20 + " [DB Data " + "="*20)
    db_path = config["data_pipeline"]["db_path"]
    conn = sqlite3.connect(db_path)

    db_check_df = pd.read_sql("SELECT * FROM daily_stock_prices LIMIT 5", conn)
    print(db_check_df)
    
    # total stored rows
    total_rows = conn.execute("SELECT COUNT(*) FROM daily_stock_prices").fetchone()
    print(f"current total rows in db: {total_rows}개")
    print("="*50 + "\n")
    conn.close()
    # =====================================================================


