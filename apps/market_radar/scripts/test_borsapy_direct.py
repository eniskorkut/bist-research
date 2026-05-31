import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import datetime
import borsapy as bp


for sym in ["XU100", "THYAO"]:
    ticker = bp.Ticker(sym)
    start = datetime.datetime.now() - datetime.timedelta(days=30)
    try:
        df = ticker.history(start=start, interval="1d")
        print(f"{sym} Direct History:")
        if df is not None and not df.empty:
            print(df.tail(5))
        else:
            print("  Empty DataFrame returned.")
    except Exception as e:
        print(f"Error fetching {sym}: {e}")
