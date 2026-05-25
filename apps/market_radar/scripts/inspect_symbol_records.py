import sqlite3
import json
import os

def check_db(path, name):
    if not os.path.exists(path):
        print(f"{name} does not exist at {path}")
        return
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT payload_json FROM daily_ohlcv_cache WHERE symbol = 'THYAO'").fetchone()
    conn.close()
    if row:
        payload = json.loads(row[0])
        records = payload.get("records") or []
        print(f"{name} (path: {path}): Total records: {len(records)}")
        if records:
            print(f"  First: {records[0].get('date')[:10]} | Last: {records[-1].get('date')[:10]}")
    else:
        print(f"{name} (path: {path}): No THYAO record found.")

check_db("/app/data/market_radar_cache.sqlite", "App DB")
check_db("/data/market_radar_cache.sqlite", "Data DB")
