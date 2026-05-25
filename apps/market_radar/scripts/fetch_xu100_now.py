from market_radar.data_access import BorsapyMarketDataClient, DB_PATH
client = BorsapyMarketDataClient()
res = client.load_history_with_meta("XU100", db_path=DB_PATH, force=True)
print(f"XU100 fetched status: {res.ohlcv_cache_status}")
print(f"Latest date: {res.data_latest_date}")
print(f"Lag days: {res.data_lag_days}")
print(f"Rows: {res.history_rows}")
