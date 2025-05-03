import numpy as np
import pandas as pd

def fill_gaps(series: pd.Series, freq: str) -> pd.Series:
    return series.asfreq(freq).ffill(limit=1)

def clean_intraday(series: pd.Series, z_thresh: float = 6.0) -> pd.Series:
    diff = series.diff().dropna()
    mad = np.median(np.abs(diff - np.median(diff)))
    z = 0.6745 * diff / mad
    mask = np.abs(z) < z_thresh
    series.loc[~mask] = np.nan
    return series.interpolate("time")
