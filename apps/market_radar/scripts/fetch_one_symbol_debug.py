import sys
from market_radar.data_access import BorsapyMarketDataClient

client = BorsapyMarketDataClient()
db_path = "/app/data/market_radar_cache.sqlite"

try:
    res = client.load_history_with_meta("THYAO", db_path=db_path, force=True)
    print(f"THYAO | Status: {res.ohlcv_cache_status} | Latest: {res.data_latest_date} | Source: {res.source}")
except Exception as e:
    print(f"Error fetching THYAO: {e}")
