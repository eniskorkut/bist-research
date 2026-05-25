from __future__ import annotations

import json
import sqlite3
from typing import Any
import pandas as pd
from market_radar.data_access import BorsapyMarketDataClient, DB_PATH
from market_radar.symbols import normalize_bist_symbol

class MarketRegimeDetector:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._xu100_df: pd.DataFrame | None = None

    def load_xu100_data(
        self,
        client: BorsapyMarketDataClient | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if self._xu100_df is not None and not force_refresh:
            return self._xu100_df

        if client is None:
            client = BorsapyMarketDataClient()

        # Load XU100 history
        try:
            # We want plenty of history for MA200 and MA50
            df = client.load_history(
                "XU100",
                lookback_days=700,
                db_path=self.db_path,
                force=force_refresh,
            )
        except Exception:
            # Fallback to direct cache read
            try:
                with sqlite3.connect(self.db_path) as conn:
                    row = conn.execute(
                        "SELECT payload_json FROM daily_ohlcv_cache WHERE symbol = 'XU100'"
                    ).fetchone()
                if row:
                    payload = json.loads(row[0])
                    records = payload.get("records") or []
                    df = pd.DataFrame.from_records(records)
                    if not df.empty and "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date")
                else:
                    df = pd.DataFrame()
            except Exception:
                df = pd.DataFrame()

        if df is None or df.empty:
            self._xu100_df = pd.DataFrame(columns=["close", "ma50", "ma200", "return_20d_pct"])
            return self._xu100_df

        # Clean history
        df = df.copy().sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="last")]

        # Calculate indicators timezone-naively
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()

        df["close_20d_ago"] = df["close"].shift(20)
        df["return_20d_pct"] = (df["close"] - df["close_20d_ago"]) / df["close_20d_ago"] * 100.0

        self._xu100_df = df
        return self._xu100_df

    def detect_regime(
        self,
        date_str: str,
        client: BorsapyMarketDataClient | None = None,
    ) -> dict[str, Any]:
        """
        Detects market regime as of a given date (YYYY-MM-DD), look-ahead-freely.
        """
        df = self.load_xu100_data(client=client)
        if df.empty:
            return {
                "regime_label": "Neutral",
                "return_20d_pct": 0.0,
                "close": None,
                "ma50": None,
                "ma200": None,
            }

        # Find the last trading day on or before target date
        target = pd.Timestamp(date_str).tz_localize(None)
        mask = df.index <= target
        if not mask.any():
            ref_idx = df.index.min()
        else:
            ref_idx = df.index[mask].max()

        row = df.loc[ref_idx]
        ret_20d = float(row.get("return_20d_pct", 0.0))
        label = "Bull" if ret_20d > 1.5 else "WeakOrNeutral"

        return {
            "regime_label": label,
            "return_20d_pct": ret_20d,
            "close": float(row.get("close")) if pd.notna(row.get("close")) else None,
            "ma50": float(row.get("ma50")) if pd.notna(row.get("ma50")) else None,
            "ma200": float(row.get("ma200")) if pd.notna(row.get("ma200")) else None,
        }
