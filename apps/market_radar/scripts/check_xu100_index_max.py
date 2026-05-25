from market_radar.data_access import BorsapyMarketDataClient, DB_PATH

client = BorsapyMarketDataClient()
benchmark = client.load_history(
    "XU100",
    lookback_days=260,
    db_path=DB_PATH,
    force=False,
)
print(f"XU100 index max date: {benchmark.index.max()}")
print(f"XU100 tail:\n{benchmark.tail(5)}")
