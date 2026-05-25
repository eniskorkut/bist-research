import sqlite3
import json

db_path = "/data/market_radar_cache.sqlite"
conn = sqlite3.connect(db_path)
rows = conn.execute("SELECT symbol, payload_json FROM daily_ohlcv_cache").fetchall()
conn.close()

date_counts = {}
symbols_with_25 = []

for symbol, payload_json in rows:
    payload = json.loads(payload_json)
    records = payload.get("records") or []
    if records:
        # Get the latest date
        latest_date = records[-1].get("date")[:10]
        date_counts[latest_date] = date_counts.get(latest_date, 0) + 1
        if latest_date == "2026-05-25":
            symbols_with_25.append(symbol)

print(f"Total symbols in cache: {len(rows)}")
print(f"Latest date distribution: {date_counts}")
print(f"Number of symbols with 2026-05-25: {len(symbols_with_25)}")
