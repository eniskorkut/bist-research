import pandas as pd

def compute_quality_threshold_score(out: pd.DataFrame) -> pd.DataFrame:
    """
    Computes quality_threshold_score for a dataframe of candidate features.
    It intentionally does not mutate special filter pass/fail flags.
    """
    if out.empty:
        return out

    def _get_col(col_name: str) -> pd.Series:
        if col_name in out.columns:
            return pd.to_numeric(out[col_name], errors="coerce")
        return pd.Series(float('nan'), index=out.index)

    rs = _get_col("relative_strength_20d_pct")
    rs_score = ((rs.clip(lower=0.0, upper=12.0) / 12.0) * 25.0).fillna(0.0)
    
    volp = _get_col("volume_ratio_3d_vs_20d")
    vol_score = (((volp - 1.0).clip(lower=0.0, upper=0.8) / 0.8) * 15.0).fillna(0.0)
    
    cmf = _get_col("cmf_20")
    mf_score = ((cmf.clip(lower=0.0, upper=0.20) / 0.20) * 15.0).fillna(0.0)
    
    rsi = _get_col("rsi_14")
    rsi_score = (((rsi - 50.0).clip(lower=0.0, upper=20.0) / 20.0) * 8.0).fillna(0.0)
    
    macd_score = (_get_col("macd_hist") > 0).astype(float) * 7.0
    
    slope = _get_col("ma20_slope_5d")
    slope_score = ((slope.clip(lower=0.0, upper=3.0) / 3.0) * 10.0).fillna(0.0)
    
    cp = _get_col("close_position")
    close_score = (((cp - 0.50).clip(lower=0.0, upper=0.35) / 0.35) * 10.0).fillna(0.0)
    
    r5 = _get_col("return_5d_pct")
    r10 = _get_col("return_10d_pct")
    d52 = _get_col("distance_from_52w_low_pct")
    
    penalty = (
        (r5 > 20).astype(float) * 4.0
        + (r5 > 25).astype(float) * 3.0
        + (r10 > 35).astype(float) * 4.0
        + (r10 > 45).astype(float) * 3.0
        + (d52 > 100).astype(float) * 3.0
    )
    safety_score = (10.0 - penalty).clip(lower=0.0)
    
    out["quality_threshold_score"] = (
        rs_score + vol_score + mf_score + rsi_score + macd_score + slope_score + close_score + safety_score
    ).clip(lower=0.0, upper=100.0)

    return out
