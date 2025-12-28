import csv
import requests
import time
from datetime import datetime, timedelta

# --- CONFIG ---
ASSET_CSV = "02_classify/output/assets_native_pre2025.csv"
OUTPUT_CSV = "03_getprice/output/all_native_prices_2025.csv"
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
START_DATE = "2025-01-01"
END_DATE = "2025-12-14"
INTERVAL = "1d"
MAX_RETRIES = 5
SLEEP_BETWEEN = 0.0  # seconds between requests

def read_symbols(asset_csv):
    symbols = []
    with open(asset_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["symbol_binance"]
            if symbol and row["exclude"].lower() != "true":
                symbols.append(symbol)
    return symbols

def fetch_binance_klines(symbol, start_str, end_str, interval="1d"):
    # Convert date strings to milliseconds
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    url = BINANCE_API_URL
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1000
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data
        except Exception as e:
            print(f"Error fetching {symbol} (attempt {attempt+1}): {e}")
            time.sleep(2)
    return []

def main():
    symbols = read_symbols(ASSET_CSV)
    print(f"Total symbols: {len(symbols)}")
    # Collect all dates for the period
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    all_dates = []
    dt = start_dt
    while dt <= end_dt:
        all_dates.append(dt.strftime("%Y-%m-%d"))
        dt += timedelta(days=1)

    # Prepare price table: {date: {symbol: close}}
    price_table = {date: {} for date in all_dates}

    for idx, symbol in enumerate(symbols):
        print(f"[{idx+1}/{len(symbols)}] Fetching {symbol} ...")
        klines = fetch_binance_klines(symbol, START_DATE, END_DATE, INTERVAL)
        for k in klines:
            # k[0]: open time (ms), k[4]: close price
            date_str = datetime.utcfromtimestamp(k[0] // 1000).strftime("%Y-%m-%d")
            close_price = k[4]
            if date_str in price_table:
                price_table[date_str][symbol] = close_price
        time.sleep(SLEEP_BETWEEN)

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["date"] + symbols
        writer.writerow(header)
        for date in all_dates:
            row = [date]
            for symbol in symbols:
                price = price_table[date].get(symbol, "")
                row.append(price)
            writer.writerow(row)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()