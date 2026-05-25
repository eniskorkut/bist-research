from market_radar.data_access import BorsapyMarketDataClient, DB_PATH
client = BorsapyMarketDataClient()
for sym in ["THYAO", "ASELS", "EREGL"]:
    res = client.load_history_with_meta(sym, db_path=DB_PATH, force=True)
    print(f"{sym} | Latest: {res.data_latest_date} | Status: {res.ohlcv_cache_status}")
