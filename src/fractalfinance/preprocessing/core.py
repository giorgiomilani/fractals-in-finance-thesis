import numpy as np
import pandas as pd


def fill_gaps(series: pd.Series, freq: str) -> pd.Series:
    return series.asfreq(freq).ffill(limit=1)


def clean_intraday(series: pd.Series, z_thresh: float = 6.0) -> pd.Series:
    """Return a copy of *series* with extreme intraday moves removed.

    Parameters
    ----------
    series : pd.Series
        Input price series indexed by timestamp.
    z_thresh : float, default 6.0
        Observations with a modified z-score greater than this threshold
        are treated as outliers.

    Returns
    -------
    pd.Series
        Cleaned series with outliers interpolated. The original input is
        left unmodified.
    """
    s = series.copy()
    diff = s.diff().dropna()
    mad = np.median(np.abs(diff - np.median(diff)))
    z = 0.6745 * diff / mad
    mask = np.abs(z) < z_thresh
    s.loc[~mask] = np.nan
    return s.interpolate("time")
